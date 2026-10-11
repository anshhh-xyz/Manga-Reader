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


_SPK_MODEL = None
_SPK_CFG = None

def _get_speaker_model():
    global _SPK_MODEL, _SPK_CFG
    if _SPK_MODEL is None:
        model_path = ROOT / "outputs" / "speakers" / "pairwise_xgb.json"
        cfg_path = ROOT / "outputs" / "speakers" / "cluster_cfg.json"
        if model_path.exists() and cfg_path.exists():
            from xgboost import XGBClassifier
            m = XGBClassifier()
            m.load_model(str(model_path))
            _SPK_MODEL = m
            _SPK_CFG = json.loads(cfg_path.read_text(encoding="utf-8"))
    return _SPK_MODEL, _SPK_CFG


def _find_panel(bbox, panels):
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for k, (x0, y0, x1, y1) in enumerate(panels):
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return k
    d = [((cx - (p[0] + p[2]) / 2) ** 2 + (cy - (p[1] + p[3]) / 2) ** 2) for p in panels]
    return int(np.argmin(d)) if len(d) > 0 else -1


def predict_sequence(seq_id: str, min_conf: float = 0.3,
                     postprocess: bool = True, join_hyphens: bool = False,
                     assign_speakers_flag: bool = True) -> list:
    """Run the full manga reading pipeline for one sequence.

    Returns a list of 3 page-lists, each page-list being:
        [{"speaker": "char1", "text": "..."}, ...]
    """
    img_paths = image_paths(seq_id)
    all_seq_boxes = []
    page_sizes = []

    for pg, img in enumerate(img_paths):
        # ── OCR ───────────────────────────────────────────────────────────────
        raw_boxes = [b for b in ocr_page(img, CACHE) if b["conf"] >= min_conf]

        # ── Reading order (XY-cut RTL) setup ──────────────────────────────────
        panel_info = _get_panels(seq_id, pg)
        if panel_info:
            panels = panel_info["panels"]
            size   = panel_info["size"]
        else:
            gray   = page_gray(seq_id, pg)
            h, w   = gray.shape
            panels = [(0, 0, w, h)]
            size   = (w, h)
        page_sizes.append(size)

        if not raw_boxes:
            continue

        # ── Build DataFrame for filter ─────────────────────────────────────────
        rows = []
        for b in raw_boxes:
            box = b["box"]
            if isinstance(box[0], (list, tuple)):
                xs = [float(p[0]) for p in box]; ys = [float(p[1]) for p in box]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                x0, y0, x1, y1 = [float(v) for v in box]
            rows.append({
                "seq_id": seq_id,
                "page":   pg,
                "x0":     x0,
                "y0":     y0,
                "x1":     x1,
                "y1":     y1,
                "text":   b["text"],
                "conf":   b["conf"],
            })
        df = pd.DataFrame(rows)

        # ── Story filter ───────────────────────────────────────────────────────
        df["p_story"] = story_prob(df)
        df = df[df["p_story"] >= FILTER_THRESHOLD].reset_index(drop=True)

        if len(df) == 0 or df["p_story"].max() < PAGE_GATE:
            continue

        # ── Order boxes inside panels ──────────────────────────────────────────
        boxes = df[["x0", "y0", "x1", "y1"]].to_numpy()
        order = order_boxes(boxes, panels, READING_RTL, size)

        # ── Post-process and collect ───────────────────────────────────────────
        for i in order:
            t = df.iloc[i]["text"]
            if postprocess:
                t = clean_line(t)
            t = t.strip()
            if t and score.norm(t):
                b_coords = [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])]
                panel_id = _find_panel(b_coords, panels)
                all_seq_boxes.append({
                    "page": pg,
                    "bbox": b_coords,
                    "text": t,
                    "panel": panel_id,
                    "tag": "dialogue"
                })

    # ── Speaker Diarization ────────────────────────────────────────────────────
    model, cfg = _get_speaker_model() if assign_speakers_flag else (None, None)
    if model is not None and cfg is not None and all_seq_boxes:
        from speakers.character_memory import assign_speakers
        labels = assign_speakers(all_seq_boxes, page_sizes, model, cfg)
    else:
        labels = ["char1"] * len(all_seq_boxes)

    pages = [[], [], []]
    for b, lab in zip(all_seq_boxes, labels):
        pages[b["page"]].append({"speaker": lab, "text": b["text"]})

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