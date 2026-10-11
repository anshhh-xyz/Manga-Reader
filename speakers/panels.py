"""Recursive XY-cut. It looks for white gutters between panels, first horizontal, then vertical, and recurses. 
Each balloon gets the id of the panel containing its centre. Why: It’s independent of whatever format Phase 4 stored, 
and the test pipeline will use the same function. If a page has no clean gutters it falls back to one panel, which matches the old behaviour. 
Panel ids are only used to tell whether two balloons share a panel, so reading direction doesn’t matter here."""

# speakers/panels.py
import numpy as np
from PIL import Image

MIN_GUTTER = 8      # min blank rows/cols (px) to count as a gutter
MIN_SLIVER = 20     # ignore segments thinner than this

def _segments(prof, thr, min_gutter):
    blank = prof <= thr
    segs, start, i, n = [], 0, 0, len(prof)
    while i < n:
        if blank[i]:
            j = i
            while j < n and blank[j]:
                j += 1
            if j - i >= min_gutter and i > 0 and j < n:
                segs.append((start, i)); start = j
            i = j
        else:
            i += 1
    segs.append((start, n))
    return [(a, b) for a, b in segs if b - a >= MIN_SLIVER]

def _xy(mask, x0, y0, x1, y1, depth, out):
    sub = mask[y0:y1, x0:x1]; h, w = sub.shape
    if depth < 6 and h > 60 and w > 60:
        for axis in ("h", "v"):
            prof = sub.sum(axis=1) if axis == "h" else sub.sum(axis=0)
            L = w if axis == "h" else h
            segs = _segments(prof, 0.01 * L, MIN_GUTTER)
            if len(segs) > 1:
                for a, b in segs:
                    if axis == "h": _xy(mask, x0, y0 + a, x1, y0 + b, depth + 1, out)
                    else:           _xy(mask, x0 + a, y0, x0 + b, y1, depth + 1, out)
                return
    out.append((x0, y0, x1, y1))

def page_panels(img):
    mask = np.array(img.convert("L")) < 200            # True = dark ink
    out = []
    _xy(mask, 0, 0, mask.shape[1], mask.shape[0], 0, out)
    return out

def panel_of(panels, bbox):
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for k, (x0, y0, x1, y1) in enumerate(panels):
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return k
    d = [np.hypot(cx - (p[0] + p[2]) / 2, cy - (p[1] + p[3]) / 2) for p in panels]
    return int(np.argmin(d))

def fill_panels(paths, boxes):
    """Set b['panel'] for boxes whose panel is -1. paths = the 3 page image paths."""
    cache = {}
    for b in boxes:
        if b["panel"] != -1:
            continue
        pg = b["page"]
        if pg not in cache:
            cache[pg] = page_panels(Image.open(paths[pg]))
        b["panel"] = panel_of(cache[pg], b["bbox"])