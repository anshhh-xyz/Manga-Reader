"""Quick smoke-test for the YOLO bubble detector."""
from evaluation.common import split_ids, image_paths
from detection.bubble_ocr import detect_bubbles

ids = split_ids("val")[:3]
for sid in ids:
    imgs = image_paths(sid)
    img = imgs[0]
    bubbles = detect_bubbles(img, conf_thresh=0.35)
    print(f"{sid} page0: {len(bubbles)} bubbles detected")
    for b in bubbles[:3]:
        bbox = b["bbox"]
        conf = b["conf"]
        print(f"  bbox={bbox}  conf={conf:.2f}")
    print()
