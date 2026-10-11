# detection/train_yolo.py
"""
Trains a YOLOv8 manga speech bubble detector on the prepared dataset.
Hardware target: NVIDIA GeForce RTX 5050 Laptop GPU (CUDA).
Includes CUDA-NMS compatibility patch for Windows PyTorch.
"""
import sys
from pathlib import Path
import torch
import torchvision

# Monkey-patch torchvision.ops.nms to handle CUDA tensors gracefully on Windows
_orig_nms = torchvision.ops.nms
def _safe_nms(boxes, scores, iou_threshold):
    if boxes.is_cuda:
        idx = _orig_nms(boxes.cpu(), scores.cpu(), iou_threshold)
        return idx.to(boxes.device)
    return _orig_nms(boxes, scores, iou_threshold)

torchvision.ops.nms = _safe_nms

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "dataset" / "yolo_balloons" / "data.yaml"
WEIGHTS_DIR = ROOT / "weights"


def main():
    print("=" * 60)
    print("Training YOLOv8 Manga Balloon Detector (with CUDA patch)")
    print("=" * 60)

    # Use yolov8s.pt or yolov8n.pt
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=35,
        imgsz=800,         # 800px provides strong detail with fast GPU convergence
        batch=8,           # efficient batch size for RTX 5050 GPU
        device=0,          # CUDA GPU
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        mosaic=0.2,        # mild mosaic
        degrees=0.0,       # orthogonal text
        flipud=0.0,
        fliplr=0.2,
        project=str(ROOT / "runs" / "bubble_yolo"),
        name="train_exp",
        exist_ok=True
    )

    best_pt = ROOT / "runs" / "bubble_yolo" / "train_exp" / "weights" / "best.pt"
    if best_pt.exists():
        target_pt = WEIGHTS_DIR / "bubble_yolo.pt"
        import shutil
        shutil.copy2(best_pt, target_pt)
        print(f"\nTraining Complete! Best model saved -> {target_pt}")


if __name__ == "__main__":
    main()
