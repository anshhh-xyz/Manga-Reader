# detection/bubble_ocr.py
"""
Container-First Bubble OCR Module
===================================
Implements the YOLO → OCR → Speaker pipeline described in 'yolo approach.md'.

Instead of running OCR on full pages and merging fragments post-hoc, this module:
1. Detects speech bubble bounding boxes with the trained YOLOv8 balloon detector.
2. Sorts detected bubbles in manga right-to-left (RTL) reading order within panels.
3. Crops each bubble from the page image and runs OCR inside the crop.
4. Returns one complete utterance string per bubble — no fragmentation.

Usage (standalone):
    from detection.bubble_ocr import extract_bubble_utterances
    utterances = extract_bubble_utterances(img_path, seq_id="seq_xxx", page=0, panel_info=None)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUBBLE_YOLO_PATH = ROOT / "weights" / "bubble_yolo.pt"

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
    pass  # torchvision not installed or already patched

_YOLO_MODEL = None


def _load_yolo():
    """Lazy-load YOLOv8 bubble detector."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        if not BUBBLE_YOLO_PATH.exists():
            raise FileNotFoundError(
                f"Bubble YOLO model not found at {BUBBLE_YOLO_PATH}. "
                "Run: python -m detection.train_yolo"
            )
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(str(BUBBLE_YOLO_PATH))
    return _YOLO_MODEL


def detect_bubbles(img_path: Path, conf_thresh: float = 0.35) -> List[Dict]:
    """
    Run YOLOv8 on a manga page and return detected speech bubble bboxes.

    Returns:
        List of dicts: [{'bbox': [x0, y0, x1, y1], 'conf': float}, ...]
    """
    model = _load_yolo()
    results = model.predict(str(img_path), conf=conf_thresh, imgsz=800, verbose=False)[0]

    boxes = []
    for b in results.boxes:
        xyxy = b.xyxy[0].tolist()
        boxes.append({
            "bbox": [round(v, 1) for v in xyxy],
            "conf": float(b.conf[0]),
        })
    return boxes


def _find_panel_id(bbox: List[float], panels: List[List[float]]) -> int:
    """Assign a bubble to the panel whose centre is closest to the bubble centre."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    for k, (px0, py0, px1, py1) in enumerate(panels):
        if px0 <= cx <= px1 and py0 <= cy <= py1:
            return k
    dists = [
        (cx - (p[0] + p[2]) / 2) ** 2 + (cy - (p[1] + p[3]) / 2) ** 2
        for p in panels
    ]
    return int(np.argmin(dists))


def sort_bubbles_rtl(bubbles: List[Dict], panels: List[List[float]], page_size: Tuple[float, float]) -> List[Dict]:
    """
    Sort bubbles in manga right-to-left reading order.

    Strategy (mirrors reading_order.py):
    1. Group bubbles by assigned panel.
    2. Order panels top-to-bottom, right-to-left.
    3. Within each panel, sort bubbles right-to-left, top-to-bottom.
    """
    W, H = page_size

    # Assign panel IDs
    for b in bubbles:
        b["panel"] = _find_panel_id(b["bbox"], panels)

    # Sort panels themselves: top-to-bottom, right-to-left
    unique_panels = sorted(set(b["panel"] for b in bubbles))

    def panel_sort_key(pid: int):
        px0, py0, px1, py1 = panels[pid]
        return (round(py0 / H * 4), -(px0 + px1) / 2)

    ordered_panels = sorted(unique_panels, key=panel_sort_key)

    # Within each panel: right-to-left columns (RTL), top-to-bottom rows
    STRIPE = 0.15  # 15% of page width as column tolerance
    ordered_bubbles: List[Dict] = []
    for pid in ordered_panels:
        panel_bubbles = [b for b in bubbles if b["panel"] == pid]
        # RTL: sort by x-center descending, then y ascending
        panel_bubbles.sort(key=lambda b: (
            -round((b["bbox"][0] + b["bbox"][2]) / 2 / W / STRIPE),
            (b["bbox"][1] + b["bbox"][3]) / 2
        ))
        ordered_bubbles.extend(panel_bubbles)

    return ordered_bubbles


def preprocess_crop(crop: Image.Image, min_side: int = 64, target_side: int = 192) -> np.ndarray:
    """
    Preprocess a bubble crop before OCR:
      1. Upscale: if the shorter dimension < target_side, scale up using LANCZOS.
      2. Convert to grayscale.
      3. CLAHE contrast enhancement (no hard binarization — Otsu over-thresholds
         manga crops with screentone backgrounds or grey speech bubbles).
      4. Return as 3-channel RGB numpy array (EasyOCR expects RGB).

    Why no Otsu: manga bubbles are not always clean white-on-black. Grey-tinted
    bubbles, light screentones, or soft shadows cause Otsu to threshold text away.
    CLAHE alone provides enough contrast for EasyOCR to read manga fonts reliably.
    """
    import cv2

    # --- Step 1: Upscale to give EasyOCR enough pixels to resolve letter shapes ---
    w, h = crop.size
    short = min(w, h)
    if short < target_side:
        scale = target_side / max(short, 1)
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        crop = crop.resize((new_w, new_h), Image.LANCZOS)

    # --- Step 2: Grayscale ---
    gray = np.array(crop.convert("L"))  # uint8, 0-255

    # --- Step 3: CLAHE contrast enhancement (adaptive, no global threshold) ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # --- Step 4: Mild denoising to smooth CLAHE tile boundaries ---
    denoised = cv2.bilateralFilter(enhanced, d=3, sigmaColor=15, sigmaSpace=15)

    # --- Step 5: Return as RGB for EasyOCR ---
    rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    return rgb


def read_bubble_text(crop: Image.Image, ocr_fn) -> str:
    """
    Preprocess a bubble crop then run OCR on it.
    Joins all detected text lines top-to-bottom into one utterance.

    Args:
        crop: PIL Image of the cropped speech bubble.
        ocr_fn: Callable that accepts a numpy array and returns
                [(bbox, text, conf), ...] (EasyOCR-style).
    """
    # Preprocess: upscale + CLAHE + Otsu before OCR
    arr = preprocess_crop(crop)
    try:
        raw = ocr_fn(arr)
    except Exception:
        return ""

    if not raw:
        return ""

    # Sort by top-y of first point, then join
    def _y0(item):
        box = item[0]
        if isinstance(box[0], (list, tuple)):
            return box[0][1]
        return box[1]

    raw_sorted = sorted(raw, key=_y0)
    tokens = [item[1].strip() for item in raw_sorted if item[1].strip()]
    return " ".join(tokens)


def extract_bubble_utterances(
    img_path: Path,
    seq_id: str,
    page: int,
    panel_info: Optional[Dict] = None,
    conf_thresh: float = 0.35,
    ocr_fn=None,
    postprocess_fn=None,
) -> List[Dict]:
    """
    Main entry point: run bubble detection → sort RTL → OCR each crop.

    Args:
        img_path:       Path to the manga page image.
        seq_id:         Sequence identifier string.
        page:           0-indexed page number in this sequence.
        panel_info:     Dict with keys 'panels' [[x0,y0,x1,y1],...] and 'size' [w,h].
                        If None, the full page is treated as one panel.
        conf_thresh:    YOLO detection confidence threshold.
        ocr_fn:         Callable(np.array) → [(bbox, text, conf),...].
                        If None, falls back to the project EasyOCR engine.
        postprocess_fn: Optional callable(str) → str to clean each utterance.

    Returns:
        List of dicts (in RTL reading order):
        [
          {
            'page': int,
            'bbox': [x0, y0, x1, y1],
            'panel': int,
            'text': str,
            'conf': float,       # YOLO bubble detection confidence
            'tag': 'dialogue',
          },
          ...
        ]
    """
    img = Image.open(img_path)
    W, H = img.size

    if panel_info:
        panels = panel_info["panels"]
        page_size = tuple(panel_info["size"])
    else:
        panels = [[0, 0, W, H]]
        page_size = (float(W), float(H))

    # Step 1: Detect bubbles
    raw_bubbles = detect_bubbles(img_path, conf_thresh=conf_thresh)
    if not raw_bubbles:
        return []

    # Step 2: Sort in RTL reading order
    sorted_bubbles = sort_bubbles_rtl(raw_bubbles, panels, page_size)

    # Step 3: Lazy-load OCR if not provided
    if ocr_fn is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        ocr_fn = lambda arr: _reader.readtext(arr)

    # Step 4: OCR each crop (with preprocessing: upscale + CLAHE + Otsu)
    results = []
    for bubble in sorted_bubbles:
        x0, y0, x1, y1 = bubble["bbox"]
        # Expand crop by 8px to capture text at bubble edges
        pad = 8
        cx0 = max(0, int(x0) - pad)
        cy0 = max(0, int(y0) - pad)
        cx1 = min(W, int(x1) + pad)
        cy1 = min(H, int(y1) + pad)

        crop = img.crop((cx0, cy0, cx1, cy1))
        # preprocess_crop() is called inside read_bubble_text now
        text = read_bubble_text(crop, ocr_fn)

        if postprocess_fn:
            text = postprocess_fn(text)

        text = text.strip()
        if not text:
            continue

        results.append({
            "page":  page,
            "bbox":  bubble["bbox"],
            "panel": bubble["panel"],
            "text":  text,
            "conf":  bubble["conf"],
            "tag":   "dialogue",
        })

    return results
