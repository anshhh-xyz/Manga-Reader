"""Merge word-level OCR boxes into balloon-sized boxes.
Usage: python -m alignment.merge
Reads outputs/ocr_boxes/development_words.json, writes outputs/ocr_boxes/development.json
(the file align.py reads).  Tune the 4 constants and re-run: no OCR needed.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "ocr_boxes" / "development_words.json"
DST = ROOT / "outputs" / "ocr_boxes" / "development.json"

V_GAP = 0.8      # stacked boxes merge if vertical gap < V_GAP * line height
H_OVERLAP = 0.3  # ...and they overlap horizontally by >= this share of the narrower box
H_GAP = 1.0      # same-row boxes merge if horizontal gap < H_GAP * line height
ROW_TOL = 0.5    # reading-row grouping inside a merged box


def bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def should_merge(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    h = min(ay1 - ay0, by1 - by0)
    if h <= 0:
        return False
    v_overlap = min(ay1, by1) - max(ay0, by0)
    h_overlap = min(ax1, bx1) - max(ax0, bx0)
    if v_overlap > 0.5 * h:                       # same row
        return -h_overlap < H_GAP * h
    v_gap = max(ay0, by0) - min(ay1, by1)          # stacked
    w = min(ax1 - ax0, bx1 - bx0)
    return v_gap < V_GAP * h and h_overlap > H_OVERLAP * w


def merge_page(boxes):
    n = len(boxes)
    bb = [bbox(b["poly"]) for b in boxes]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge(bb[i], bb[j]):
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for members in groups.values():
        members.sort(key=lambda i: (bb[i][1] + bb[i][3]) / 2)
        rows, row_cy, row_h = [], None, None
        for i in members:
            cy = (bb[i][1] + bb[i][3]) / 2
            h = bb[i][3] - bb[i][1]
            if rows and abs(cy - row_cy) < ROW_TOL * max(row_h, h):
                rows[-1].append(i)
            else:
                rows.append([i])
                row_cy, row_h = cy, h
        ordered = [i for r in rows for i in sorted(r, key=lambda k: bb[k][0])]
        x0 = min(bb[i][0] for i in members)
        y0 = min(bb[i][1] for i in members)
        x1 = max(bb[i][2] for i in members)
        y1 = max(bb[i][3] for i in members)
        total = sum(max(1, len(boxes[i]["text"])) for i in ordered)
        conf = sum(boxes[i]["conf"] * max(1, len(boxes[i]["text"])) for i in ordered) / total
        merged.append({
            "poly": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "text": " ".join(boxes[i]["text"] for i in ordered).strip(),
            "conf": conf,
        })
    merged.sort(key=lambda b: (b["poly"][0][1], b["poly"][0][0]))
    for k, b in enumerate(merged):
        b["idx"] = k
    return merged


src = json.loads(SRC.read_text(encoding="utf-8"))
out = {sid: {"images": v["images"], "pages": [merge_page(p) for p in v["pages"]]} for sid, v in src.items()}
DST.write_text(json.dumps(out), encoding="utf-8")
counts = [len(p) for v in out.values() for p in v["pages"]]
print(f"saved {DST}: {sum(counts)} boxes, {sum(counts)/len(counts):.1f} per page")