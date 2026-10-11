# ordering/pairwise.py
"""Train a pairwise reading-order classifier and evaluate with grouped CV.

Run with:
    python -m ordering.pairwise

Requires:
    outputs/alignment/aligned_boxes.json   (from alignment/merge.py)
    outputs/panels_dev.json                (from detection/run_panels.py)
"""
import json
import itertools
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from scipy.stats import kendalltau

from configs.config import ALIGNED_PATH, PANELS_PATH, SEED
from filtering.dataset import load_aligned
from ordering.reading_order import sort_panels, _panel_of

ROOT = Path(__file__).resolve().parents[1]


def pair_feats(a, b, panels, size):
    w, h = size
    ca = ((a[0] + a[2]) / 2 / w, (a[1] + a[3]) / 2 / h)
    cb = ((b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h)
    pa = _panel_of(ca[0] * w, ca[1] * h, panels)
    pb = _panel_of(cb[0] * w, cb[1] * h, panels)
    return [
        cb[0] - ca[0], cb[1] - ca[1],
        int(pa == pb), pb - pa,
        (b[2] - b[0]) / max(a[2] - a[0], 1),
        (b[3] - b[1]) / max(a[3] - a[1], 1),
    ]


def page_pairs(g, P, rtl):
    panels = sort_panels(P["panels"], rtl)
    B = g[["x0", "y0", "x1", "y1"]].to_numpy(float)
    X, y = [], []
    for i, j in itertools.permutations(range(len(g)), 2):
        X.append(pair_feats(B[i], B[j], panels, P["size"]))
        y.append(int(g.order_idx.iloc[i] < g.order_idx.iloc[j]))
    return X, y


def sort_page(clf, boxes, P, rtl):
    panels = sort_panels(P["panels"], rtl)
    n = len(boxes)
    wins = np.zeros(n)
    for i, j in itertools.permutations(range(n), 2):
        wins[i] += clf.predict_proba([pair_feats(boxes[i], boxes[j], panels, P["size"])])[0, 1]
    return list(np.argsort(-wins))


def main():
    from configs.config import READING_RTL

    aligned_path = Path(ALIGNED_PATH) if Path(ALIGNED_PATH).is_absolute() else ROOT / ALIGNED_PATH
    panels_path  = Path(PANELS_PATH)  if Path(PANELS_PATH).is_absolute()  else ROOT / PANELS_PATH

    if not aligned_path.exists():
        raise FileNotFoundError(f"Aligned boxes not found: {aligned_path}")
    if not panels_path.exists():
        raise FileNotFoundError(
            f"Panel map not found: {panels_path}\n"
            "Run:  python -m detection.run_panels  first."
        )

    df  = load_aligned(str(aligned_path))
    hi  = df[(df.tier == "high") & (~df.ambiguous) & df.order_idx.notna()].copy()
    PAN = json.loads(panels_path.read_text(encoding="utf-8"))

    seqs = hi.seq_id.unique()
    taus = []

    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(seqs):
        X, y = [], []
        for (s, pg), g in hi[hi.seq_id.isin(set(seqs[tr]))].groupby(["seq_id", "page"]):
            key = f"{s}|{pg}"
            if key not in PAN or len(g) < 2:
                continue
            g = g.reset_index(drop=True)
            a, b = page_pairs(g, PAN[key], READING_RTL)
            X += a; y += b

        clf = HistGradientBoostingClassifier(max_depth=3, random_state=SEED).fit(X, y)

        for (s, pg), g in hi[hi.seq_id.isin(set(seqs[te]))].groupby(["seq_id", "page"]):
            key = f"{s}|{pg}"
            if key not in PAN or len(g) < 3:
                continue
            g = g.reset_index(drop=True)
            order = sort_page(clf, g[["x0", "y0", "x1", "y1"]].to_numpy(float), PAN[key], READING_RTL)
            rank = np.empty(len(g))
            rank[order] = np.arange(len(g))
            t = kendalltau(rank, g.order_idx.to_numpy()).correlation
            if not np.isnan(t):
                taus.append(t)

    print(f"Pairwise mean tau (grouped CV): {np.mean(taus):.4f}  pages: {len(taus)}")


if __name__ == "__main__":
    main()