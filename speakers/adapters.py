# speakers/adapter.py
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image

ALIGNED_PATH = Path("outputs/alignment/aligned_boxes.json")
IMG_ROOT = Path("dataset/development/images")
OUT = Path("outputs/speakers"); OUT.mkdir(parents=True, exist_ok=True)

# left = name used in my code, right = name in YOUR aligned file  (EDIT right side)
FIELD = {"seq": "sequence_id", "page": "page", "bbox": "poly", "text": "ocr_text",
         "speaker": "speaker", "tier": "tier", "ambiguous": "ambiguous",
         "tag": "tag", "order": "order_index", "panel": "panel_id"}
TAG_NARR, TAG_NON = "narration", "non-story"                 # EDIT if your tag strings differ

def _bbox(b):
    if b is None: return None
    if len(b) == 4 and not isinstance(b[0], (list, tuple)):
        return [float(v) for v in b]
    xs = [p[0] for p in b]; ys = [p[1] for p in b]          # polygon -> rectangle
    return [min(xs), min(ys), max(xs), max(ys)]

def page_paths(seq):
    # If your data.py already returns the 3 page paths in sequences.json order, call it here instead.
    return sorted((IMG_ROOT / seq).glob("*"))[:3]

_sizes = {}
def page_sizes(seq):
    if seq not in _sizes:
        _sizes[seq] = [Image.open(p).size for p in page_paths(seq)]   # (W, H) per page
    return _sizes[seq]

def load_sequences():
    """-> {seq: [box dicts sorted by (page, gold order)]}. Keeps matched story boxes only."""
    by_seq = defaultdict(list)
    for r in json.load(open(ALIGNED_PATH, encoding="utf-8")):
        b = {k: r.get(v) for k, v in FIELD.items()}
        if b["tier"] not in ("high", "medium") or b["ambiguous"] or b["tag"] == TAG_NON:
            continue
        if b["bbox"] is None or b["order"] is None:
            continue
        b["bbox"] = _bbox(b["bbox"])
        b["panel"] = -1 if b["panel"] is None else int(b["panel"])
        by_seq[b["seq"]].append(b)
    for s in by_seq:
        by_seq[s].sort(key=lambda b: (b["page"], b["order"]))
    from .panels import fill_panels
    for s, bs in by_seq.items():
        fill_panels(page_paths(s), bs)
    return dict(by_seq)