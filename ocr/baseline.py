# ocr/baseline.py

import json
from pathlib import Path

_engine = None


def _get_engine():
    global _engine

    if _engine is None:
        from paddleocr import PaddleOCR
        _engine = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            enable_mkldnn=False,
        )

    return _engine


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
    # PaddleOCR 3.x inference
    # -------------------------
    result = _get_engine().predict(str(image_path))

    boxes = []

    for res in result:
        data = res.json

        if callable(data):
            data = data()

        # PaddleOCR 3.x stores OCR output under "res"
        data = data.get("res", data)

        texts = data.get("rec_texts", []) #get texts
        scores = data.get("rec_scores", []) #get ocr scores
        polys = data.get("rec_polys", data.get("dt_polys", [])) #get the coordinates

        for pts, text, conf in zip(polys, texts, scores): #club all scores
            xs = [float(p[0]) for p in pts] #chagne to rectangle
            ys = [float(p[1]) for p in pts]

            boxes.append(   #storing
                {
                    "box": [
                        min(xs),
                        min(ys),
                        max(xs),
                        max(ys),
                    ],
                    "text": str(text),
                    "conf": float(conf),
                }
            )

    # -------------------------
    # Save cache
    # -------------------------
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)

        cache.write_text(
            json.dumps(boxes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return boxes


    """Reads tge manga page and find coordinates of the boxes , not making an engine evertime as it is expensive,make one engine and reuse
    convert ocr detected polygons to rectangular boxes create a cache folder to prevent re assessing of an image through , and any future chanfes 
    can be loaded through cache"""