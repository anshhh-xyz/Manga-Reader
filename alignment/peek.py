import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
boxes = json.loads((ROOT / "outputs/alignment/aligned_boxes.json").read_text(encoding="utf-8"))
labels = {}
for line in (ROOT / "dataset/development/labels.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        labels[d["sequence_id"]] = d["pages"]

long_none = [b for b in boxes if b["tier"] == "none" and b["n_tokens"] >= 8]
print(len(long_none), "long unmatched boxes")
random.seed(1)
for b in random.sample(long_none, min(6, len(long_none))):
    gold = " / ".join(l["text"] for l in labels[b["sequence_id"]][b["page"]])
    print("\n", b["sequence_id"], "p", b["page"], "prec", b["precision"])
    print("  OCR :", b["ocr_text"][:160])
    print("  GOLD:", gold[:260])