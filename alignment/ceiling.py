"""
Usage:  python -m alignment.ceiling                 (tiers high+medium)
        python -m alignment.ceiling high            (high tier only)

This script creates the best-case baseline predictions using already-aligned OCR boxes. 
It exports OCR text and speaker labels for boxes whose alignment quality meets your chosen tier.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "alignment"
tiers = set((sys.argv[1] if len(sys.argv) > 1 else "high,medium").split(","))
UNKNOWN_LABEL = "UNKNOWN"   # how gold UNKNOWN lines are written out (see note below)

boxes = json.loads((OUT / "aligned_boxes.json").read_text(encoding="utf-8"))
by_page = defaultdict(list)
for b in boxes:
    if b["tier"] in tiers and b["lines"]:
        by_page[(b["sequence_id"], b["page"])].append(b)

seq_order = []
for line in (ROOT / "dataset" / "development" / "labels.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        seq_order.append(json.loads(line)["sequence_id"])

rows = []
for sid in seq_order:
    pages = []
    for p in range(3):
        bs = sorted(by_page.get((sid, p), []), key=lambda b: b["order_index"])
        grouped = defaultdict(list)
        first = {}
        for b in bs:
            ln = b["lines"][0]
            grouped[ln].append(b)
            first.setdefault(ln, b["order_index"])
        page_lines = []
        for ln in sorted(grouped, key=lambda l: first[l]):
            spk = grouped[ln][0]["speaker"]
            page_lines.append({"speaker": UNKNOWN_LABEL if spk == "UNKNOWN" else spk,
                               "text": " ".join(b["ocr_text"] for b in grouped[ln])})
        pages.append(page_lines)
    rows.append({"sequence_id": sid, "pages": pages})

name = "ceiling_dev.jsonl" if tiers == {"high", "medium"} else f"ceiling_dev_{'_'.join(sorted(tiers))}.jsonl"
with open(OUT / name, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT / name)