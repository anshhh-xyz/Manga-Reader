# ocr/baseline.py

import json
from pathlib import Path

_engine = None
_ENGINE_NAME = None   # track which backend is loaded


def _get_engine():
    """Lazy-load OCR engine. Tries PaddleOCR first, falls back to EasyOCR."""
    global _engine, _ENGINE_NAME

    if _engine is None:
        try:
            from paddleocr import PaddleOCR
            _engine = PaddleOCR(
                lang="en",
                use_textline_orientation=True,
                enable_mkldnn=False,
            )
            _ENGINE_NAME = "paddle"
        except Exception:
            # PaddleOCR not installed — use EasyOCR (always available)
            from ocr.engines import EasyOCREngine
            _engine = EasyOCREngine()
            _ENGINE_NAME = "easyocr"

    return _engine


def _run_engine(image_path: Path) -> list:
    """Dispatch OCR to the loaded engine and return normalised box list."""
    engine = _get_engine()

    if _ENGINE_NAME == "paddle":
        result = engine.predict(str(image_path))
        boxes = []
        for res in result:
            data = res.json
            if callable(data):
                data = data()
            data = data.get("res", data)
            texts  = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            polys  = data.get("rec_polys", data.get("dt_polys", []))
            for pts, text, conf in zip(polys, texts, scores):
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                boxes.append({"box": [min(xs), min(ys), max(xs), max(ys)],
                              "text": str(text), "conf": float(conf)})
        return boxes

    else:  # easyocr
        ocr_boxes = engine.read(str(image_path))
        return [{"box": [b.x0, b.y0, b.x1, b.y1],
                 "text": b.text, "conf": b.conf}
                for b in ocr_boxes]


def ocr_page(image_path, cache_dir=None):
    """
    Run PaddleOCR on one page.

    Returns:
        [
            {
                "box": [x1, y1, x2, y2],
                "text": "...",
                "conf": 0.97
            },
            ...
        ]
    """

    image_path = Path(image_path)

    # -------------------------
    # Precomputed dev cache
    # -------------------------
    dev_cache_path = Path("outputs/ocr_boxes/development.json")
    if dev_cache_path.exists():
        global _DEV_BOXES
        if "_DEV_BOXES" not in globals() or _DEV_BOXES is None:
            _DEV_BOXES = json.loads(dev_cache_path.read_text(encoding="utf-8"))
        seq_name = image_path.parent.name
        if seq_name in _DEV_BOXES:
            imgs = sorted(image_path.parent.glob("*"))
            if image_path in imgs:
                pg_idx = imgs.index(image_path)
                pages = _DEV_BOXES[seq_name].get("pages", [])
                if pg_idx < len(pages):
                    boxes = []
                    for b in pages[pg_idx]:
                        poly = b["poly"]
                        xs = [float(p[0]) for p in poly]
                        ys = [float(p[1]) for p in poly]
                        boxes.append({
                            "box": [min(xs), min(ys), max(xs), max(ys)],
                            "text": str(b.get("text", "")),
                            "conf": float(b.get("conf", 1.0)),
                        })
                    return boxes

    # -------------------------
    # Live OCR (PaddleOCR or EasyOCR fallback)
    # -------------------------
    return _run_engine(image_path)