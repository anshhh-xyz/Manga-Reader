# detection/prepare_yolo_data.py
"""
Converts outputs/alignment/aligned_boxes.json into YOLOv8 dataset format
for training a dedicated manga speech bubble detector.
"""
import json
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ALIGNED_BOXES_PATH = ROOT / "outputs" / "alignment" / "aligned_boxes.json"
DEV_IMAGES_ROOT = ROOT / "dataset" / "development" / "images"
SPLITS_PATH = ROOT / "dataset" / "splits.json"

YOLO_ROOT = ROOT / "dataset" / "yolo_balloons"
TRAIN_IMG = YOLO_ROOT / "images" / "train"
VAL_IMG = YOLO_ROOT / "images" / "val"
TRAIN_LBL = YOLO_ROOT / "labels" / "train"
VAL_LBL = YOLO_ROOT / "labels" / "val"


def bbox_from_poly(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def main():
    print("=" * 60)
    print("Preparing YOLO Balloon Dataset from Aligned Boxes")
    print("=" * 60)

    for p in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
        p.mkdir(parents=True, exist_ok=True)

    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    train_seqs = set(splits.get("train", []))
    val_seqs = set(splits.get("val", []))
    print(f"Train sequences: {len(train_seqs)} | Val sequences: {len(val_seqs)}")

    raw_boxes = json.loads(ALIGNED_BOXES_PATH.read_text(encoding="utf-8"))

    # Group valid story dialogue balloons by (seq_id, page)
    balloons_by_page = {}
    total_valid = 0
    for b in raw_boxes:
        if b.get("tier") not in ("high", "medium") or b.get("tag") != "dialogue":
            continue
        poly = b.get("poly")
        if not poly or len(poly) < 4:
            continue
        seq_id = b["sequence_id"]
        page = b["page"]
        key = (seq_id, page)
        balloons_by_page.setdefault(key, []).append(bbox_from_poly(poly))
        total_valid += 1

    print(f"Total valid balloon boxes: {total_valid} across {len(balloons_by_page)} pages")

    # Image cache for sizes
    size_cache = {}

    processed_train = 0
    processed_val = 0
    boxes_train = 0
    boxes_val = 0

    for (seq_id, page), boxes in balloons_by_page.items():
        is_val = seq_id in val_seqs
        split_name = "val" if is_val else "train"
        img_dest_dir = VAL_IMG if is_val else TRAIN_IMG
        lbl_dest_dir = VAL_LBL if is_val else TRAIN_LBL

        # Locate original image
        seq_dir = DEV_IMAGES_ROOT / seq_id
        img_files = sorted(seq_dir.glob("*.png"))
        if not img_files:
            img_files = sorted(seq_dir.glob("*.jpg"))
        if page >= len(img_files):
            continue

        orig_img_path = img_files[page]
        img_filename = f"{seq_id}_p{page}.png"
        target_img_path = img_dest_dir / img_filename
        target_lbl_path = lbl_dest_dir / f"{seq_id}_p{page}.txt"

        if orig_img_path not in size_cache:
            with Image.open(orig_img_path) as im:
                size_cache[orig_img_path] = im.size
        w, h = size_cache[orig_img_path]

        # Copy image (symlink or copy)
        if not target_img_path.exists():
            shutil.copy2(orig_img_path, target_img_path)

        # Write YOLO labels: class_id x_center y_center width height (normalized)
        label_lines = []
        for x0, y0, x1, y1 in boxes:
            # Clamp coordinates to image boundaries
            x0 = max(0.0, min(float(x0), float(w)))
            y0 = max(0.0, min(float(y0), float(h)))
            x1 = max(0.0, min(float(x1), float(w)))
            y1 = max(0.0, min(float(y1), float(h)))
            bw = x1 - x0
            bh = y1 - y0
            if bw <= 4 or bh <= 4:
                continue

            cx = (x0 + x1) / 2.0 / w
            cy = (y0 + y1) / 2.0 / h
            nw = bw / w
            nh = bh / h

            label_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        target_lbl_path.write_text("\n".join(label_lines), encoding="utf-8")

        if is_val:
            processed_val += 1
            boxes_val += len(label_lines)
        else:
            processed_train += 1
            boxes_train += len(label_lines)

    # Write data.yaml
    data_yaml = f"""path: {YOLO_ROOT.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: balloon
"""
    yaml_path = YOLO_ROOT / "data.yaml"
    yaml_path.write_text(data_yaml, encoding="utf-8")

    print("\nDataset generation complete:")
    print(f"  Train: {processed_train} pages, {boxes_train} balloon labels -> {TRAIN_IMG}")
    print(f"  Val:   {processed_val} pages, {boxes_val} balloon labels -> {VAL_IMG}")
    print(f"  Config saved -> {yaml_path}")


if __name__ == "__main__":
    main()
