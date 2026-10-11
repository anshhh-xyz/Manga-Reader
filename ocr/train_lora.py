# ocr/train_lora.py
"""
Phase 6 — OCR Adaptation (Track B): TrOCR LoRA Fine-Tuning
============================================================
Fine-tunes microsoft/trocr-base-printed on our aligned manga crops.
Training targets = gold label text (from aligned_boxes.json, high+medium tiers).

Run with:
    python -m ocr.train_lora [--epochs 3] [--batch 4] [--lr 5e-4] [--out outputs/ocr_lora]

After training, you can swap in the adapted model via ocr/engines.py.

Requirements:
    pip install transformers peft pillow torch torchvision
"""

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig, ViTImageProcessor, RobertaTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# PEFT compatibility fix for VisionEncoderDecoderModel
VisionEncoderDecoderConfig.vocab_size = property(
    lambda self: getattr(self.decoder, 'vocab_size', 50265) if hasattr(self, 'decoder') else 50265,
    lambda self, val: setattr(self.decoder, 'vocab_size', val) if hasattr(self, 'decoder') else None
)

ALIGNED_PATH = Path("outputs/alignment/aligned_boxes.json")
IMG_ROOT     = Path("dataset/development/images")
DEFAULT_OUT  = Path("outputs/ocr_lora")
MODEL_NAME   = "microsoft/trocr-base-printed"

SEED = 42
random.seed(SEED)


# ── Dataset ─────────────────────────────────────────────────────────────────

class MangaCropDataset(Dataset):
    """
    Loads (image_crop, label_text) pairs from aligned_boxes.json.
    Crop = bbox region of the page image.
    Label = gold label text (what the scorer expects).
    """

    def __init__(self, records, processor, max_label_len=128):
        self.records = records
        self.processor = processor
        self.max_label_len = max_label_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = IMG_ROOT / rec["seq"] / rec["img_file"]
        img = Image.open(img_path).convert("RGB")

        x1, y1, x2, y2 = [int(v) for v in rec["bbox"]]
        # Add a small margin
        W, H = img.size
        pad = 4
        crop = img.crop((max(0, x1-pad), max(0, y1-pad),
                         min(W, x2+pad), min(H, y2+pad)))

        pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            rec["label"], padding="max_length", max_length=self.max_label_len,
            truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)
        # Replace padding token id with -100 (ignore in loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_train_val(val_frac=0.15):
    """Load aligned crops, split into train/val by sequence."""
    labels_map = {}
    for line in open("dataset/development/labels.jsonl", "r", encoding="utf-8"):
        d = json.loads(line)
        labels_map[d["sequence_id"]] = d["pages"]

    raw = json.loads(ALIGNED_PATH.read_text(encoding="utf-8"))

    records = []
    for r in raw:
        tier = r.get("tier", r.get("match_tier"))
        if tier not in ("high", "medium"):
            continue
        if r.get("ambiguous"):
            continue
        seq   = r.get("sequence_id") or r.get("seq_id") or r.get("seq")
        page  = r.get("page", 0)
        lines = r.get("lines")
        bbox  = r.get("poly") or r.get("bbox") or r.get("box")

        if not lines or not bbox or not seq or seq not in labels_map:
            continue

        pages = labels_map[seq]
        if page >= len(pages):
            continue
        page_lines = pages[page]
        gold_texts = [page_lines[idx]["text"] for idx in lines if idx < len(page_lines)]
        if not gold_texts:
            continue
        label = " ".join(gold_texts).strip()

        # Resolve bbox to [x1,y1,x2,y2]
        if isinstance(bbox[0], (list, tuple)):
            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

        # Find image file
        seq_dir = IMG_ROOT / seq
        if not seq_dir.exists():
            continue
        imgs = sorted(seq_dir.glob("*"))
        if page >= len(imgs):
            continue

        records.append({
            "seq": seq,
            "img_file": imgs[page].name,
            "bbox": bbox,
            "label": label,
        })

    if not records:
        raise RuntimeError(
            "No training records found. Check ALIGNED_PATH and that images exist in dataset/development/images/."
        )

    # Split by sequence
    seqs = list({r["seq"] for r in records})
    random.shuffle(seqs)
    n_val = max(1, int(len(seqs) * val_frac))
    val_seqs = set(seqs[:n_val])
    train = [r for r in records if r["seq"] not in val_seqs]
    val   = [r for r in records if r["seq"] in val_seqs]
    print(f"Train: {len(train)} crops | Val: {len(val)} crops")
    return train, val


# ── LoRA setup ───────────────────────────────────────────────────────────────

def build_lora_model(model_name=MODEL_NAME):
    img_proc = ViTImageProcessor.from_pretrained(model_name)
    tok = RobertaTokenizer.from_pretrained(model_name)
    processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tok)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],   # attention projections in the decoder
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return processor, model


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_recs, val_recs = load_train_val()
    processor, model = build_lora_model()
    model.to(device)

    train_ds = MangaCropDataset(train_recs, processor)
    val_ds   = MangaCropDataset(val_recs, processor)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=(device.type == "cuda"))
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_dl), epochs=args.epochs
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)
            out = model(pixel_values=pixel_values, labels=labels)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                pixel_values = batch["pixel_values"].to(device)
                labels       = batch["labels"].to(device)
                out = model(pixel_values=pixel_values, labels=labels)
                val_loss += out.loss.item()
        val_loss /= max(len(val_dl), 1)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        log.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)})

    # Save adapter
    model.save_pretrained(str(out_dir / "adapter"))
    processor.save_pretrained(str(out_dir / "processor"))
    json.dump(log, open(out_dir / "train_log.json", "w"), indent=2)
    print(f"\nSaved LoRA adapter -> {out_dir}/adapter")
    print(f"Saved processor    -> {out_dir}/processor")
    print(f"Training log       -> {out_dir}/train_log.json")

    # Quick sanity decode on val
    print("\n--- Sample predictions ---")
    model.eval()
    for rec in val_recs[:3]:
        img = Image.open(IMG_ROOT / rec["seq"] / rec["img_file"]).convert("RGB")
        x1, y1, x2, y2 = [int(v) for v in rec["bbox"]]
        crop = img.crop((x1, y1, x2, y2))
        pv = processor(images=crop, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pv, max_new_tokens=64)
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
        print(f"  Gold: {rec['label']!r}")
        print(f"  Pred: {pred!r}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TrOCR LoRA fine-tuning on manga crops.")
    ap.add_argument("--epochs", type=int,   default=3,      help="Training epochs")
    ap.add_argument("--batch",  type=int,   default=4,      help="Batch size")
    ap.add_argument("--lr",     type=float, default=5e-4,   help="Learning rate")
    ap.add_argument("--out",    default=str(DEFAULT_OUT),   help="Output directory")
    ap.add_argument("--model",  default=MODEL_NAME,         help="Base TrOCR model")
    args = ap.parse_args()
    train(args)
