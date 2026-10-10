# ordering/measure_direction.py
"""ordering/measure_direction.py checks which reading-order method works best for your manga dataset: 
normal top-to-bottom ordering or panel-based ordering with right-to-left/left-to-right reading.
Using Kendall Tau"""
import json, numpy as np
from scipy.stats import kendalltau
from configs.config import ALIGNED_PATH, PANELS_PATH
from filtering.dataset import load_aligned
from ordering.reading_order import order_boxes

df = load_aligned(ALIGNED_PATH)
hi = df[(df.tier == "high") & (~df.ambiguous) & df.order_idx.notna()]
P = json.load(open(PANELS_PATH))

def mean_tau(fn):
    taus = []
    for (s, pg), g in hi.groupby(["seq_id", "page"]):
        if len(g) < 3: continue
        order = fn(g, P[f"{s}|{pg}"])
        rank = np.empty(len(g)); rank[order] = np.arange(len(g))
        t = kendalltau(rank, g.order_idx.to_numpy()).correlation
        if not np.isnan(t): taus.append(t)
    return np.mean(taus), len(taus)

cols = ["x0", "y0", "x1", "y1"]
print("naive top-to-bottom:", mean_tau(lambda g, p: list(np.lexsort((g.x0.to_numpy(), g.y0.to_numpy())))))
for rtl in (True, False):
    print(f"XY-cut rtl={rtl}:", mean_tau(lambda g, p: order_boxes(g[cols].to_numpy(), p["panels"], rtl, p["size"])))