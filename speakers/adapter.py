# speakers/adapter.py
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ALIGNED_PATH = Path("outputs/alignment/aligned_boxes.json")

# logical name -> possible key names in YOUR aligned file (first one found is used)
CANDIDATES = {
    "seq": ["sequence_id", "seq_id", "seq", "sequence"],
    "page": ["page", "page_idx", "page_index", "page_no"],
    "bbox": ["bbox", "box", "coords", "polygon", "poly", "xyxy"],
    "text": ["ocr_text", "text", "box_text", "ocr"],
    "speaker": ["speaker", "gold_speaker", "matched_speaker", "label_speaker"],
    "tier": ["tier", "match_tier"],
    "ambiguous": ["ambiguous", "is_ambiguous", "flag_ambiguous"],
    "tag": ["tag", "tag3", "class", "label", "category"],
    "order": ["order_index", "order_idx", "order", "gold_order", "token_index", "gold_token_index"],
}
REQUIRED = ["seq", "page", "bbox", "text", "speaker", "order"]
BOX_KEYS = CANDIDATES["bbox"]
TAG_NARR, TAG_NON, TAG_DLG = "narration", "non-story", "dialogue"


def _flatten(node, seq=None, page=None, out=None):
    """Finds box records in: [rec..] | {"boxes":[..]} | {seq:[rec..]} | {seq:{page:[rec..]}}."""
    out = [] if out is None else out
    if isinstance(node, list):
        for r in node:
            _flatten(r, seq, page, out)
    elif isinstance(node, dict):
        if any(k in node for k in BOX_KEYS):                      # a box record
            r = dict(node)
            if seq is not None:
                r.setdefault("sequence_id", seq)
            if page is not None:
                r.setdefault("page", page)
            out.append(r)
            return out
        for k in ("sequence_id", "seq_id"):                       # a sequence-level dict
            if isinstance(node.get(k), str):
                seq = node[k]
        if isinstance(node.get("page"), int):                     # a page-level dict
            page = node["page"]
        for k, v in node.items():
            if not isinstance(v, (list, dict)):
                continue
            if seq is None and str(k).startswith("seq"):
                _flatten(v, str(k), page, out)
            elif seq is not None and page is None and re.search(r"\d", str(k)) and not str(k).startswith("seq"):
                _flatten(v, seq, int(re.findall(r"\d+", str(k))[-1]), out)
            else:
                _flatten(v, seq, page, out)
    return out


def _read_records(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit("File not found: %s (edit ALIGNED_PATH at the top of speakers/adapter.py)" % path)
    if path.suffix == ".jsonl":
        data = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    else:
        data = json.load(open(path, encoding="utf-8"))
    rows = _flatten(data)
    if not rows:
        hint = list(data)[:8] if isinstance(data, dict) else type(data).__name__
        raise SystemExit("No box records found in %s. Top level: %s. Paste this line to Claude." % (path, hint))
    return rows


def _mapping(rec):
    m = {k: next((c for c in cands if c in rec), None) for k, cands in CANDIDATES.items()}
    missing = [k for k in REQUIRED if m[k] is None]
    if missing:
        raise SystemExit("Could not find fields %s. Your keys are: %s. Paste this line to Claude."
                         % (missing, sorted(rec)))
    return m


def _bbox(b):
    if len(b) == 4 and not isinstance(b[0], (list, tuple)):
        return [float(v) for v in b]
    xs, ys = [p[0] for p in b], [p[1] for p in b]                 # polygon -> rectangle
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _tag(v):
    s = str(v or "").lower()
    if "narr" in s:
        return TAG_NARR
    if "non" in s or s in ("other", "none", "sfx", "noise"):
        return TAG_NON
    return TAG_DLG


def load_sequences():
    """-> {seq: [box dicts sorted by (page, gold order)]}, matched story boxes only."""
    recs = _read_records(ALIGNED_PATH)
    m = _mapping(recs[0])
    rows = [{k: (r.get(v) if v else None) for k, v in m.items()} for r in recs]
    shift = 1 if min(int(r["page"]) for r in rows) == 1 else 0     # pages stored 1-based -> 0-based
    by_seq = defaultdict(list)
    for b in rows:
        if b["tier"] is not None and b["tier"] not in ("high", "medium"):
            continue
        if b["ambiguous"] or b["order"] is None or b["bbox"] is None:
            continue
        b["tag"] = TAG_NARR if b["speaker"] == "NARRATION" else _tag(b["tag"])
        if b["tag"] == TAG_NON:
            continue
        b["bbox"], b["page"], b["panel"] = _bbox(b["bbox"]), int(b["page"]) - shift, -1
        b["text"] = b["text"] or ""
        by_seq[b["seq"]].append(b)
    for s in by_seq:
        by_seq[s].sort(key=lambda b: (b["page"], b["order"]))
    return dict(by_seq)


if __name__ == "__main__":
    recs = _read_records(ALIGNED_PATH)
    print("file:", ALIGNED_PATH, "| records:", len(recs))
    print("keys in first record:", sorted(recs[0]))
    seqs = load_sequences()
    boxes = [b for v in seqs.values() for b in v]
    print("sequences with boxes:", len(seqs), "| boxes kept:", len(boxes))
    print("pages:", sorted(Counter(b["page"] for b in boxes).items()))
    print("tags:", dict(Counter(b["tag"] for b in boxes)))
    print("speakers per sequence (mean):",
          round(sum(len({b["speaker"] for b in v}) for v in seqs.values()) / max(len(seqs), 1), 2))
    print("example:", boxes[0])