# pipeline/inference.py
"""Full pipeline: image → EasyOCR → story filter → XY-cut RTL order → JSONL.

Run with:
    python -m pipeline.inference --set dev --out outputs/predictions_dev.jsonl

Requires:
    outputs/filtering/filter_model.joblib   (from filtering/train_filter.py)
    outputs/panels_dev.json                 (from detection/run_panels.py)
"""
import argparse
import json
from pathlib import Path

from evaluation.common import OUTPUTS, image_paths, score, split_ids, write_jsonl
from ocr.baseline import ocr_page
from ocr.postprocess import clean_line, join_wrapped
from configs.config import FILTER_THRESHOLD, PAGE_GATE, READING_RTL, PANELS_PATH
from filtering.infer_filter import story_prob
from filtering.common import page_gray
from ordering.reading_order import order_boxes

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Cache folder for EasyOCR results (avoids re-running GPU OCR)
CACHE = OUTPUTS / "ocr_cache" / "easyocr"

# Load panel map once at startup (None → detect on-the-fly)
_panels_path = ROOT / PANELS_PATH if not Path(PANELS_PATH).is_absolute() else Path(PANELS_PATH)
_PANELS: dict | None = None

def _load_panels():
    global _PANELS
    if _PANELS is None:
        if _panels_path.exists():
            _PANELS = json.loads(_panels_path.read_text(encoding="utf-8"))
        else:
            _PANELS = {}   # will fall back to full-page single panel
    return _PANELS


def _get_panels(seq_id: str, page: int):
    """Return {"panels": [...], "size": [w, h]} or None for full-page fallback."""
    key = f"{seq_id}|{page}"
    return _load_panels().get(key)


def predict_sequence(seq_id: str, min_conf: float = 0.3,
                     postprocess: bool = True, join_hyphens: bool = False) -> list:
    """Run the full Phase 4 pipeline for one sequence.

    Returns a list of 3 page-lists, each page-list being:
        [{"speaker": "char1", "text": "..."}, ...]
    """
    img_paths = image_paths(seq_id)
    pages = []

    for pg, img in enumerate(img_paths):
        # ── OCR ───────────────────────────────────────────────────────────────
        raw_boxes = [b for b in ocr_page(img, CACHE) if b["conf"] >= min_conf]

        if not raw_boxes:
            pages.append([])
            continue

        # ── Build DataFrame for filter ─────────────────────────────────────────
        df = pd.DataFrame([{
            "seq_id": seq_id,
            "page":   pg,
            "x0":     min(pt[0] for pt in b["box"]),
            "y0":     min(pt[1] for pt in b["box"]),
            "x1":     max(pt[0] for pt in b["box"]),
            "y1":     max(pt[1] for pt in b["box"]),
            "text":   b["text"],
            "conf":   b["conf"],
        } for b in raw_boxes])

        # ── Story filter ───────────────────────────────────────────────────────
        df["p_story"] = story_prob(df)
        df = df[df["p_story"] >= FILTER_THRESHOLD].reset_index(drop=True)

        if len(df) == 0 or df["p_story"].max() < PAGE_GATE:
            pages.append([])
            continue

        # ── Reading order (XY-cut RTL) ─────────────────────────────────────────
        panel_info = _get_panels(seq_id, pg)
        if panel_info:
            panels = panel_info["panels"]
            size   = panel_info["size"]
        else:
            # fallback: single panel = whole page
            gray   = page_gray(seq_id, pg)
            h, w   = gray.shape
            panels = [(0, 0, w, h)]
            size   = (w, h)

        boxes = df[["x0", "y0", "x1", "y1"]].to_numpy()
        order = order_boxes(boxes, panels, READING_RTL, size)

        # ── Post-process and emit ──────────────────────────────────────────────
        lines = []
        texts = [df.iloc[i]["text"] for i in order]
        if postprocess:
            texts = [clean_line(t) for t in texts]
        if join_hyphens:
            texts = join_wrapped(texts)
        for t in texts:
            t = t.strip()
            if t and score.norm(t):
                lines.append({"speaker": "char1", "text": t})

        pages.append(lines)

    return pages


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run full manga OCR pipeline.")
    ap.add_argument("--set",         default="dev",  choices=["dev", "train", "val", "test"])
    ap.add_argument("--out",         required=True,  help="Output JSONL path")
    ap.add_argument("--min-conf",    type=float,     default=0.3,   help="Min OCR confidence")
    ap.add_argument("--postprocess", action="store_true",           help="Apply OCR post-processing")
    ap.add_argument("--join-hyphens",action="store_true",           help="Join wrapped hyphenated lines")
    args = ap.parse_args()

    ids  = split_ids(args.set)
    rows = {}
    for n, sid in enumerate(ids, 1):
        rows[sid] = predict_sequence(sid, args.min_conf, args.postprocess, args.join_hyphens)
        print(f"[{n}/{len(ids)}] {sid}")

    write_jsonl(rows, args.out)
    print(f"Wrote {args.out}")