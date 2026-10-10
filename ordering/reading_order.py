# ordering/reading_order.py
"""determines the order in which OCR text boxes should be read on a manga page.
It uses panel coordinates (if available) and text-box coordinates. It does not detect text or panels itself."""
import numpy as np

def sort_panels(panels, rtl, row_tol=0.5):
    """Group panels into rows by vertical overlap, then order within rows by direction."""
    ps = sorted(panels, key=lambda p: p[1])
    rows, cur = [], [ps[0]]
    for p in ps[1:]:
        ref = cur[0]
        overlap = min(p[3], ref[3]) - max(p[1], ref[1])
        if overlap > row_tol * min(p[3] - p[1], ref[3] - ref[1]):
            cur.append(p)
        else:
            rows.append(cur); cur = [p]
    rows.append(cur)
    out = []
    for r in rows:
        out += sorted(r, key=lambda p: -p[0] if rtl else p[0])
    return out

def _panel_of(cx, cy, panels):
    best, bd = 0, 1e18
    for i, (x0, y0, x1, y1) in enumerate(panels):
        dx = max(x0 - cx, 0, cx - x1); dy = max(y0 - cy, 0, cy - y1)
        d = dx * dx + dy * dy
        if d < bd: best, bd = i, d
    return best

def order_boxes(boxes, panels, rtl, size, band_frac=0.12):
    """boxes: (n,4) array. Returns positional indices in reading order."""
    boxes = np.asarray(boxes, dtype=float)
    if len(boxes) == 0:
        return []
    panels = sort_panels(panels or [(0, 0, size[0], size[1])], rtl)
    keys = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        pi = _panel_of(cx, cy, panels)
        px0, py0, px1, py1 = panels[pi]
        band = int((cy - py0) / max((py1 - py0) * band_frac, 1))
        keys.append((pi, band, -cx if rtl else cx, i))
    return [k[-1] for k in sorted(keys)]