"""This script checks how well the OCR boxes were aligned with ground-truth labels 
and generates sample images so you can inspect the matches visually
Usage:  python -m alignment.report
"""
import json
import random
import textwrap
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "alignment"
random.seed(42)

boxes = json.loads((OUT / "aligned_boxes.json").read_text(encoding="utf-8"))
unm = json.loads((OUT / "unmatched_lines.json").read_text(encoding="utf-8"))
cache = json.loads((ROOT / "outputs" / "ocr_boxes" / "development.json").read_text(encoding="utf-8"))
labels = {}
for line in (ROOT / "dataset" / "development" / "labels.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        labels[d["sequence_id"]] = d["pages"]

n = len(boxes)
tiers = Counter(b["tier"] for b in boxes)
tags = Counter(b["tag"] for b in boxes)
total_lines = sum(len(p) for pages in labels.values() for p in pages)
short = [b for b in boxes if b["n_tokens"] and b["n_tokens"] <= 2]
short_none = [b for b in short if b["tier"] == "none"]

print(f"OCR boxes: {n}")
print("tiers:", dict(tiers), {k: f"{v/n:.1%}" for k, v in tiers.items()})
print("tags:", dict(tags))
print("ambiguous boxes:", sum(b["ambiguous"] for b in boxes))
print(f"gold lines: {total_lines}, with no box: {len(unm)} ({len(unm)/total_lines:.1%})")
print(f"short boxes (<=2 tokens): {len(short)}, of which unmatched: {len(short_none)}")
print("unmatched lines by speaker type:", Counter(u["speaker"] if u["speaker"] in ("NARRATION", "UNKNOWN") else "character" for u in unm))
print("sample unmatched gold lines:")
for u in random.sample(unm, min(8, len(unm))):
    print("  ", u["sequence_id"], "p", u["page"], "|", u["text"][:70])

# ---- draw samples ----
(OUT / "samples").mkdir(parents=True, exist_ok=True)
font = ImageFont.load_default()


def draw(b, name):
    img = Image.open(cache[b["sequence_id"]]["images"][b["page"]]).convert("RGB")
    xs = [p[0] for p in b["poly"]]
    ys = [p[1] for p in b["poly"]]
    ImageDraw.Draw(img).polygon([tuple(p) for p in b["poly"]], outline=(255, 0, 0), width=3)
    pad = 120
    crop = img.crop((max(0, min(xs) - pad), max(0, min(ys) - pad),
                     min(img.width, max(xs) + pad), min(img.height, max(ys) + pad)))
    lines = [f"OCR  : {b['ocr_text']}", f"tier={b['tier']} prec={b['precision']} amb={b['ambiguous']}"]
    if b["lines"]:
        gold = " | ".join(labels[b["sequence_id"]][b["page"]][i]["text"] for i in b["lines"])
        lines.append(f"LABEL: {gold}")
        lines.append(f"speaker={b['speaker']}  order_index={b['order_index']}")
    wrapped = [w for ln in lines for w in textwrap.wrap(ln, 70) or [""]]
    canvas = Image.new("RGB", (max(crop.width, 520), crop.height + 14 * len(wrapped) + 10), "white")
    canvas.paste(crop, (0, 0))
    d = ImageDraw.Draw(canvas)
    for i, w in enumerate(wrapped):
        d.text((5, crop.height + 5 + 14 * i), w, fill="black", font=font)
    canvas.save(OUT / "samples" / name)


matched = [b for b in boxes if b["tier"] != "none"]
for i, b in enumerate(random.sample(matched, min(20, len(matched)))):
    draw(b, f"match_{i:02d}_{b['tier']}.png")
unmatched_boxes = [b for b in boxes if b["tier"] == "none" and b["n_tokens"] >= 2]
for i, b in enumerate(random.sample(unmatched_boxes, min(10, len(unmatched_boxes)))):
    draw(b, f"none_{i:02d}.png")
print("samples saved to", OUT / "samples")