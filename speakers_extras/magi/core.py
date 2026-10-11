"""Core logic: Magi page results + our OCR balloon boxes -> speaker labels.

Pure python (stdlib only) so it can be tested without Magi, a GPU or your dataset.

Inputs
  boxes : list of dicts in reading order, each with
            "page"  (0-based page index),
            "bbox"  ([x1, y1, x2, y2] in the ORIGINAL page pixels),
            "tag"   (story/narration tag; narration boxes get "NARRATION")
  cache : dict written by run_magi.py:
            {"api": str, "sizes": [[W, H], ...], "pages": [<Magi result dict per page>, ...]}
Output
  list of labels, one per box: "NARRATION", "char1", "char2", ...
"""
import math
from collections import Counter

NARR = "\0NARRATION"  # internal sentinel, never leaves this module

DEFAULT_CFG = {
    "unsure": "prev",      # balloons Magi can't link to a character: "prev" (copy previous speaker) | "common"
    "maxk": 0,             # keep only the K most frequent characters per sequence (0 = no cap)
    "scope": "global",     # "global": one Magi character id = one speaker across pages | "page": ids are page-local
    "nearest": True,       # if no text->character link, use the nearest character box
    "match_thr": 0.3,      # min overlap to match our balloon box to a Magi text box
    "near_frac": 0.25,     # nearest-character fallback only within this fraction of the page diagonal
    "narr_tag": "narration",
}


def to_py(x):
    """Recursively turn numpy/torch objects into plain python so the result is JSON-able."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {str(k): to_py(v) for k, v in x.items() if "embed" not in str(k).lower()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        return x.tolist()
    return x


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def match_score(a, b):
    """Overlap that works whether our box is inside theirs or theirs is inside ours."""
    i = _inter(a, b)
    if i <= 0:
        return 0.0
    return max(i / max(_area(a), 1e-9), i / max(_area(b), 1e-9))


def _center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _labels(res, n, page, scope):
    """One identity string per Magi character box on this page.

    Prefers Magi's character_names, then character_cluster_labels. A source is rejected when it
    is degenerate (every box has the same value although there are several boxes), because that
    would silently merge every character into one speaker.
    """
    cands = []
    names = res.get("character_names")
    clus = res.get("character_cluster_labels")
    if names is not None:
        cands.append([str(x) for x in names])
    if clus is not None:
        cands.append(["c%s" % x for x in clus])
    for c in cands:
        if len(c) == n and (len(set(c)) > 1 or n <= 1):
            return ["pg%d:%s" % (page, x) for x in c] if scope == "page" else c
    return ["pg%d:u%d" % (page, i) for i in range(n)]  # no usable identity: page-local, all different


def normalize_pages(cache, cfg):
    pages = []
    sizes = cache.get("sizes") or []
    for i, res in enumerate(cache["pages"]):
        texts = [[float(v) for v in b[:4]] for b in (res.get("texts") or [])]
        chars = [[float(v) for v in b[:4]] for b in (res.get("characters") or [])]
        t2c = {}
        for pair in res.get("text_character_associations") or []:
            t2c.setdefault(int(pair[0]), int(pair[1]))
        w, h = sizes[i] if i < len(sizes) else (1000.0, 1500.0)
        pages.append({
            "texts": texts,
            "chars": chars,
            "labels": _labels(res, len(chars), i, cfg["scope"]),
            "t2c": t2c,
            "diag": math.hypot(w, h),
        })
    return pages


def _raw_label(box, pg, cfg, stats):
    best, best_s = None, 0.0
    for j, t in enumerate(pg["texts"]):
        s = match_score(box["bbox"], t)
        if s > best_s:
            best, best_s = j, s
    if best is not None and best_s >= cfg["match_thr"]:
        stats["matched"] += 1
        c = pg["t2c"].get(best)
        if c is not None and 0 <= c < len(pg["labels"]):
            stats["linked"] += 1
            return pg["labels"][c]
    if cfg["nearest"] and pg["chars"]:
        cx, cy = _center(box["bbox"])
        d, k = min((math.hypot(cx - _center(c)[0], cy - _center(c)[1]), k) for k, c in enumerate(pg["chars"]))
        if d <= cfg["near_frac"] * pg["diag"]:
            stats["nearest"] += 1
            return pg["labels"][k]
    return None


def assign_labels(boxes, cache, cfg=None):
    """Returns (labels, stats, n_speakers)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    pages = normalize_pages(cache, cfg)
    stats = Counter()

    raw = []
    for b in boxes:
        if b.get("tag") == cfg["narr_tag"]:
            raw.append(NARR)
            stats["narration"] += 1
            continue
        stats["story"] += 1
        p = b["page"]
        raw.append(_raw_label(b, pages[p], cfg, stats) if 0 <= p < len(pages) else None)

    real = [r for r in raw if r not in (None, NARR)]
    freq = Counter(real)
    if cfg["maxk"] and len(freq) > cfg["maxk"]:
        keep = {lab for lab, _ in freq.most_common(cfg["maxk"])}
        raw = [r if (r in keep or r in (None, NARR)) else None for r in raw]

    names = {}
    for r in raw:
        if r not in (None, NARR) and r not in names:
            names[r] = "char%d" % (len(names) + 1)
    named = [names.get(r) if r not in (None, NARR) else None for r in raw]

    counts = Counter(n for n in named if n)
    common = counts.most_common(1)[0][0] if counts else "char1"
    first_named = next((n for n in named if n), common)

    out, last = [], None
    for r, n in zip(raw, named):
        if r == NARR:
            out.append("NARRATION")
            continue
        if n is None:
            stats["unsure"] += 1
            n = common if cfg["unsure"] == "common" else (last or first_named)
        out.append(n)
        last = n
    return out, stats, len(set(o for o in out if o != "NARRATION"))  # stats is a Counter: missing key = 0
