# detection/char_detector.py
"""
Character Body Detector for Speaker Diarization Feature Augmentation
=====================================================================
Uses YOLOv8n (pretrained COCO, person class=0) to detect manga character
body/torso regions on each page. Results are cached to avoid re-running YOLO.

Provides:
  detect_chars(img_path, conf=0.25) -> List[Dict]   # [{bbox, conf}, ...]
  nearest_char_features(bubble_bbox, chars, W, H)   # (dist_norm, char_idx)
  precompute_char_cache(seq_id)                      # cache all 3 pages
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHAR_CACHE_DIR = ROOT / "outputs" / "cache" / "char_detections"
CHAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS = ROOT / "yolov8n.pt"   # pretrained COCO — person class = 0
PERSON_CLS = 0
CONF_THRESH = 0.20   # low threshold — manga chars are stylized, recall > precision

# ── CUDA NMS compatibility patch (Windows: torchvision CUDA NMS not available) ─
try:
    import torchvision
    _orig_nms = torchvision.ops.nms
    def _safe_nms(boxes, scores, iou_threshold):
        if boxes.is_cuda:
            import torch
            idx = _orig_nms(boxes.cpu(), scores.cpu(), iou_threshold)
            return idx.to(boxes.device)
        return _orig_nms(boxes, scores, iou_threshold)
    torchvision.ops.nms = _safe_nms
except Exception:
    pass

_YOLO_MODEL = None


def _get_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(str(YOLO_WEIGHTS))
    return _YOLO_MODEL


def detect_chars(img_path: Path, conf: float = CONF_THRESH) -> List[Dict]:
    """
    Run YOLOv8n on a single page image; return list of person bbox detections.
    Each detection: {"bbox": [x0,y0,x1,y1], "conf": float}
    """
    cache_key = CHAR_CACHE_DIR / f"{Path(img_path).stem}_chars.json"
    if cache_key.exists():
        return json.loads(cache_key.read_text(encoding="utf-8"))

    model = _get_model()
    results = model(str(img_path), classes=[PERSON_CLS], conf=conf, verbose=False, device="cpu")
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            detections.append({"bbox": [x0, y0, x1, y1], "conf": float(box.conf[0])})

    cache_key.write_text(json.dumps(detections, ensure_ascii=False), encoding="utf-8")
    return detections


def nearest_char_features(
    bubble_bbox: List[float],
    chars: List[Dict],
    W: float,
    H: float,
) -> Tuple[float, int]:
    """
    For a bubble bbox, compute:
      - dist_norm: normalised Euclidean distance to the nearest character centre
      - char_idx:  index of the nearest character detection (-1 if none)

    Returns (dist_norm, char_idx).
    """
    if not chars:
        return 1.0, -1   # max distance, no character

    bx = (bubble_bbox[0] + bubble_bbox[2]) / 2.0 / W
    by = (bubble_bbox[1] + bubble_bbox[3]) / 2.0 / H

    best_dist = float("inf")
    best_idx = -1
    for k, ch in enumerate(chars):
        cx = (ch["bbox"][0] + ch["bbox"][2]) / 2.0 / W
        cy = (ch["bbox"][1] + ch["bbox"][3]) / 2.0 / H
        d = float(np.hypot(bx - cx, by - cy))
        if d < best_dist:
            best_dist = d
            best_idx = k

    return min(best_dist, 1.0), best_idx


def precompute_char_cache(seq_id: str) -> List[List[Dict]]:
    """
    Run character detection on all 3 pages of a sequence and cache results.
    Returns list of 3 detection lists (one per page).
    """
    from evaluation.common import image_paths
    paths = image_paths(seq_id)
    results = []
    for p in paths:
        results.append(detect_chars(p))
    return results
