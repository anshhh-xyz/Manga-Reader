# speakers/pair_features.py
"""This file creates features for every pair of text boxes to help 
predict whether they belong to the same speaker."""
import numpy as np
from collections import defaultdict

FEATURES = ["same_page", "page_gap", "rank_gap", "gap_parity", "same_panel", "panel_gap", "pp_gap",
            "dx", "dy", "dist", "w_i", "h_i", "w_j", "h_j", "area_ratio",
            "len_i", "len_j", "q_i", "q_j", "ex_i", "ex_j", "cos_sim"]

def _geom(b, size):
    W, H = size
    x1, y1, x2, y2 = b["bbox"]
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)

def seq_pair_features(boxes, emb, sizes):
    n = len(boxes); X, y, pairs = [], [], []
    g = [_geom(b, sizes[b["page"]]) for b in boxes]
    seen = defaultdict(int); pos = []                 # position of each box inside its own panel
    for b in boxes:
        k = (b["page"], b["panel"]); pos.append(seen[k]); seen[k] += 1
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            same_page = a["page"] == b["page"]
            same_panel = int(same_page and a["panel"] >= 0 and a["panel"] == b["panel"])
            dx = g[j][0] - g[i][0] if same_page else np.nan
            dy = g[j][1] - g[i][1] if same_page else np.nan
            dist = float(np.hypot(dx, dy)) if same_page else np.nan
            pgap = abs(a["panel"] - b["panel"]) if same_page and a["panel"] >= 0 and b["panel"] >= 0 else np.nan
            ppg = abs(pos[i] - pos[j]) if same_panel else np.nan
            ta, tb = a["text"] or "", b["text"] or ""
            X.append([int(same_page), abs(a["page"] - b["page"]), j - i, (j - i) % 2, same_panel, pgap, ppg,
                      dx, dy, dist, g[i][2], g[i][3], g[j][2], g[j][3],
                      (g[i][2] * g[i][3]) / max(g[j][2] * g[j][3], 1e-6),
                      len(ta), len(tb), int("?" in ta), int("?" in tb),
                      int("!" in ta), int("!" in tb), float(emb[i] @ emb[j])])
            sa, sb = a.get("speaker"), b.get("speaker")
            ok = sa not in (None, "UNKNOWN") and sb not in (None, "UNKNOWN")
            y.append(int(sa == sb) if ok else -1)
            pairs.append((i, j))
    return np.array(X, dtype=float).reshape(-1, len(FEATURES)), np.array(y), pairs

def prob_matrix(n, pairs, probs):
    P = np.eye(n)
    for (i, j), p in zip(pairs, probs):
        P[i, j] = P[j, i] = p
    return P