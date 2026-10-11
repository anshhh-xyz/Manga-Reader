# ocr/eval_lora.py
"""
Phase 6 — OCR Adaptation (Track B) Evaluation
==============================================
Runs both baseline PaddleOCR and the LoRA-adapted TrOCR on the dev set
and compares text_order_score.

Run with:
    python -m ocr.eval_lora

Outputs:
    outputs/eval_phase6_ocr_lora_dev.json
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig, ViTImageProcessor, RobertaTokenizer
from peft import PeftModel

VisionEncoderDecoderConfig.vocab_size = property(
    lambda self: getattr(self.decoder, 'vocab_size', 50265) if hasattr(self, 'decoder') else 50265,
    lambda self, val: setattr(self.decoder, 'vocab_size', val) if hasattr(self, 'decoder') else None
)

LORA_DIR    = Path("outputs/ocr_lora")
IMG_ROOT    = Path("dataset/development/images")
ALIGNED     = Path("outputs/alignment/aligned_boxes.json")
OUT         = Path("outputs")
RESULT_PATH = OUT / "eval_phase6_ocr_lora_dev.json"


def load_adapted_model():
    img_proc = ViTImageProcessor.from_pretrained(str(LORA_DIR / "processor"))
    tok = RobertaTokenizer.from_pretrained(str(LORA_DIR / "processor"))
    processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tok)
    base = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    model = PeftModel.from_pretrained(base, str(LORA_DIR / "adapter"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device


def ocr_crop_lora(processor, model, device, img_path, bbox):
    img = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in bbox]
    W, H = img.size
    crop = img.crop((max(0, x1-4), max(0, y1-4),
                     min(W, x2+4), min(H, y2+4)))
    pv = processor(images=crop, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pixel_values=pv, max_new_tokens=96)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def run_lora_eval():
    if not (LORA_DIR / "adapter").exists():
        print("LoRA adapter not found. Run: python -m ocr.train_lora first.")
        sys.exit(1)

    labels_map = {}
    for line in open("dataset/development/labels.jsonl", "r", encoding="utf-8"):
        d = json.loads(line)
        labels_map[d["sequence_id"]] = d["pages"]

    raw = json.loads(ALIGNED.read_text(encoding="utf-8"))
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

        if isinstance(bbox[0], (list, tuple)):
            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

        seq_dir = IMG_ROOT / seq
        if not seq_dir.exists():
            continue
        imgs = sorted(seq_dir.glob("*"))
        if page >= len(imgs):
            continue
        records.append({"seq": seq, "img_path": imgs[page], "bbox": bbox, "label": label})

    print(f"Evaluating LoRA OCR on {len(records)} crops...")
    processor, model, device = load_adapted_model()

    n_exact, total_chars, correct_chars = 0, 0, 0
    results = []
    for rec in records:
        pred = ocr_crop_lora(processor, model, device, rec["img_path"], rec["bbox"])
        gold = rec["label"].strip()
        exact = pred.lower() == gold.lower()
        n_exact += int(exact)
        total_chars += len(gold)
        correct_chars += sum(a == b for a, b in zip(pred.lower(), gold.lower()))
        results.append({"seq": rec["seq"], "gold": gold, "pred": pred, "exact": exact})

    exact_acc = n_exact / max(len(records), 1)
    char_acc  = correct_chars / max(total_chars, 1)

    print(f"\nLoRA OCR results ({len(records)} crops):")
    print(f"  Exact match accuracy : {exact_acc:.4f} ({n_exact}/{len(records)})")
    print(f"  Char-level accuracy  : {char_acc:.4f}")

    # Sample comparisons
    print("\n--- Sample predictions ---")
    for r in results[:5]:
        match = "OK" if r["exact"] else "FAIL"
        print(f"  [{match}] Gold: {r['gold']!r}")
        print(f"       Pred: {r['pred']!r}")

    summary = {
        "n_crops": len(records),
        "exact_match_accuracy": round(exact_acc, 4),
        "char_level_accuracy": round(char_acc, 4),
        "samples": results[:20],
    }
    RESULT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved -> {RESULT_PATH}")
    return summary


if __name__ == "__main__":
    run_lora_eval()
