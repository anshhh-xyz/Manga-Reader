"""Use Magi speakers inside pipeline/inference.py.

    from speakers_magi.pipeline_hook import magi_labels
    labels = magi_labels(seq_id, boxes, split="test")      # boxes: your filtered, ordered story boxes

`boxes` need the keys "page" (0-based), "bbox" ([x1,y1,x2,y2], original pixels) and "tag".
Reads the cached Magi result from outputs/magi/<split>/<seq>.json; if it is missing, runs Magi
on the spot (slow the first time because the model loads). Settings come from the file
written by `python -m speakers_magi.grid` (outputs/magi/best_cfg.json).
"""
import json

from . import common
from .core import DEFAULT_CFG, assign_labels


def load_cfg():
    cfg = dict(DEFAULT_CFG)
    p = common.OUT / "best_cfg.json"
    if p.exists():
        with open(p) as f:
            cfg.update(json.load(f))
    return cfg


def magi_labels(seq, boxes, split="test", cfg=None, narr_tag="narration"):
    cache = common.load_cache(seq, split)
    if cache is None:
        from .run_magi import get_model, run_and_cache
        cache = run_and_cache(get_model(), seq, split, do_ocr=False)
    cfg = dict(cfg or load_cfg())
    cfg["narr_tag"] = narr_tag
    labels, _stats, _k = assign_labels(boxes, cache, cfg)
    return labels
