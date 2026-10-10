# ocr/vlm_zeroshot.py
import argparse, json, re, time
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = """You are given three consecutive English manga pages, in order.
Transcribe ONLY the story text, in reading order, and say who speaks each line.

INCLUDE: dialogue, thoughts, narration boxes, spoken screams/grunts, punctuation-only balloons (e.g. "...", "?!").
EXCLUDE: sound effects, tiny reactions, titles, logos, credits, page numbers, ads, watermarks,
intro labels, scanlator notes, text on signs/clothes/objects, text inside letters/phone screens/documents.

Speaker rules: use "char1", "char2", ... and keep the SAME label for the same character across all three pages.
Use "NARRATION" only for narrator caption text. Do not invent new labels when unsure.
Do not include printed line-wrap hyphens. A page with no story text must be an empty list.

Return ONLY valid JSON, no explanation, exactly in this shape:
{"pages": [[{"speaker": "char1", "text": "..."}], [], [{"speaker": "char2", "text": "..."}]]}
The "pages" list must have exactly 3 entries (page 1, 2, 3)."""


def parse_pages(raw):
    """Strict parse. Returns (pages, ok). Falls back to 3 empty pages."""
    try:
        s, e = raw.index("{"), raw.rindex("}") + 1
        obj = json.loads(raw[s:e])
        pages = obj["pages"]
        assert isinstance(pages, list) and len(pages) == 3
        clean = []
        for pg in pages:
            lines = []
            for ln in pg:
                text = str(ln.get("text", "")).strip()
                spk = str(ln.get("speaker", "char1")).strip() or "char1"
                if text:
                    lines.append({"speaker": spk, "text": text})
            clean.append(lines)
        return clean, True
    except Exception:
        return [[], [], []], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="development", choices=["development", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=640 * 28 * 28,
                    help="per-page pixel budget; lower = less VRAM, worse small text")
    ap.add_argument("--no4bit", action="store_true")
    args = ap.parse_args()

    kw = {}
    if not args.no4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto", **kw).eval()
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels)

    img_root = ROOT / "dataset" / args.split / "images"
    seq_ids = sorted(p.name for p in img_root.iterdir() if p.is_dir())
    if args.limit:
        seq_ids = seq_ids[:args.limit]

    cache = ROOT / "outputs" / "cache" / "vlm_qwen3b" / args.split
    cache.mkdir(parents=True, exist_ok=True)
    out_path = ROOT / "outputs" / f"preds_vlm_qwen3b_{args.split}.jsonl"
    stats = {"parse_fail": 0, "peak_vram_gb": 0.0, "secs": []}

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sid in enumerate(seq_ids, 1):
            cf = cache / f"{sid}.json"
            if cf.exists():
                pages = json.loads(cf.read_text(encoding="utf-8"))
            else:
                files = sorted(p for p in (img_root / sid).iterdir() if p.suffix.lower() in IMG_EXT)
                assert len(files) == 3, sid
                imgs = [Image.open(p).convert("RGB") for p in files]
                content = []
                for n in range(3):
                    content += [{"type": "text", "text": f"Page {n+1}:"}, {"type": "image"}]
                content.append({"type": "text", "text": PROMPT})
                text = proc.apply_chat_template([{"role": "user", "content": content}],
                                                tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)

                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=1500, do_sample=False)
                stats["secs"].append(time.time() - t0)
                stats["peak_vram_gb"] = max(stats["peak_vram_gb"],
                                            torch.cuda.max_memory_allocated() / 1e9)
                raw = proc.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True)[0]
                pages, ok = parse_pages(raw)
                if not ok:
                    stats["parse_fail"] += 1
                    print("  parse failed:", raw[:120].replace("\n", " "))
                cf.write_text(json.dumps(pages), encoding="utf-8")
            f.write(json.dumps({"sequence_id": sid, "pages": pages}, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(seq_ids)}] {sid}")

    n = max(len(stats["secs"]), 1)
    print(f"parse failures: {stats['parse_fail']} | peak VRAM: {stats['peak_vram_gb']:.1f} GB "
          f"| avg {sum(stats['secs'])/n:.1f}s/seq")
    (ROOT / "outputs" / "vlm_stats.json").write_text(json.dumps(stats), encoding="utf-8")


if __name__ == "__main__":
    main()