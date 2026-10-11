"""Diagnose the worst sequence for the YOLO v3 pipeline."""
import json
from pathlib import Path

preds = {}
for line in Path("outputs/predictions_yolo_v3_val.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        preds.update(json.loads(line))

gt = {}
for line in Path("dataset/ref_val.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        obj = json.loads(line)
        gt[obj["sequence_id"]] = obj["pages"]

for sid in ["seq_a3bd36201a0cd61b", "seq_da71f7a842c91090"]:
    print(f"\n{'='*60}")
    print(f"SEQUENCE: {sid}")
    print("=== YOLO V3 PREDICTIONS ===")
    pred_pages = preds.get(sid, [])
    for pi, page in enumerate(pred_pages):
        print(f"  Page {pi+1}: {len(page)} lines")
        for item in page[:5]:
            spk = item["speaker"]
            txt = item["text"][:70]
            print(f"    [{spk}] {txt}")

    print("=== GROUND TRUTH ===")
    gt_pages = gt.get(sid, [])
    for pi, page in enumerate(gt_pages):
        print(f"  Page {pi+1}: {len(page)} lines")
        for item in page[:5]:
            spk = item["speaker"]
            txt = item["text"][:70]
            print(f"    [{spk}] {txt}")
