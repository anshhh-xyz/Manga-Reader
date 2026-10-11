"""Check why worst sequences produce zero YOLO detections."""
import json
from pathlib import Path
from evaluation.common import split_ids, image_paths
from detection.bubble_ocr import detect_bubbles

ROOT = Path(".")

for sid in ["seq_a3bd36201a0cd61b", "seq_da71f7a842c91090"]:
    imgs = image_paths(sid)
    print(f"\n{sid}")
    for pg, img in enumerate(imgs):
        bubbles = detect_bubbles(img, conf_thresh=0.35)
        print(f"  page{pg}: {len(bubbles)} bubbles @ conf>0.35")
        # Try lower conf
        bubbles_low = detect_bubbles(img, conf_thresh=0.10)
        print(f"  page{pg}: {len(bubbles_low)} bubbles @ conf>0.10")
        for b in bubbles_low[:3]:
            print(f"    conf={b['conf']:.3f}  bbox={[round(x) for x in b['bbox']]}")
