# detection/panel_detector.py
import numpy as np

def _segments(profile, min_gap, thr, min_size):
    low = profile < thr
    n, segs, prev, start = len(low), [], 0, None
    gaps = []
    for i, v in enumerate(low):
        if v and start is None: start = i
        if not v and start is not None:
            if i - start >= min_gap: gaps.append((start, i))
            start = None
    for s, e in gaps:
        segs.append((prev, s)); prev = e
    segs.append((prev, n))
    return [s for s in segs if s[1] - s[0] >= min_size]

def _cut(mask, ox, oy, rtl, min_gap, min_size, thr, depth):
    rows, cols = mask.any(axis=1), mask.any(axis=0)
    if not rows.any():
        return []
    y0, y1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    x0, x1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    mask, ox, oy = mask[y0:y1, x0:x1], ox + x0, oy + y0
    h, w = mask.shape
    leaf = [(ox, oy, ox + w, oy + h)]
    if depth >= 6 or h < 2 * min_size or w < 2 * min_size:
        return leaf
    for axis in (0, 1):  # 0 = horizontal cuts first, then vertical
        prof = mask.mean(axis=1 if axis == 0 else 0)
        segs = _segments(prof, min_gap, thr, min_size)
        if len(segs) > 1:
            if axis == 1 and rtl:
                segs = segs[::-1]
            out = []
            for a, b in segs:
                sub = mask[a:b, :] if axis == 0 else mask[:, a:b]
                sx, sy = (ox, oy + a) if axis == 0 else (ox + a, oy)
                out += _cut(sub, sx, sy, rtl, min_gap, min_size, thr, depth + 1)
            return out
    return leaf

def detect_panels(gray, rtl=True, ink_thr=200, gap_frac=0.006, min_gap_frac=0.008, min_size_frac=0.06):
    h, w = gray.shape
    mask = gray < ink_thr
    min_gap = max(int(min(h, w) * min_gap_frac), 3)
    min_size = int(min(h, w) * min_size_frac)
    panels = _cut(mask, 0, 0, rtl, min_gap, min_size, gap_frac, 0)
    return panels or [(0, 0, w, h)]