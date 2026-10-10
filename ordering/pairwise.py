# ordering/pairwise.py

"""checking via a model"""
import json, itertools, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from scipy.stats import kendalltau
from configs.config import ALIGNED_PATH, PANELS_PATH, SEED
from filtering.dataset import load_aligned
from ordering.reading_order import sort_panels, _panel_of

def pair_feats(a, b, panels, size):
    w, h = size
    ca = ((a[0]+a[2])/2/w, (a[1]+a[3])/2/h); cb = ((b[0]+b[2])/2/w, (b[1]+b[3])/2/h)
    pa, pb = _panel_of(ca[0]*w, ca[1]*h, panels), _panel_of(cb[0]*w, cb[1]*h, panels)
    return [cb[0]-ca[0], cb[1]-ca[1], int(pa == pb), pb - pa,
            (b[2]-b[0])/max(a[2]-a[0], 1), (b[3]-b[1])/max(a[3]-a[1], 1)]

def page_pairs(g, P, rtl):
    p = P
    panels = sort_panels(p["panels"], rtl)
    B = g[["x0","y0","x1","y1"]].to_numpy(float)
    X, y = [], []
    for i, j in itertools.permutations(range(len(g)), 2):
        X.append(pair_feats(B[i], B[j], panels, p["size"]))
        y.append(int(g.order_idx.iloc[i] < g.order_idx.iloc[j]))
    return X, y

def sort_page(clf, boxes, P, rtl):
    panels = sort_panels(P["panels"], rtl)
    n = len(boxes)
    wins = np.zeros(n)
    for i, j in itertools.permutations(range(n), 2):
        wins[i] += clf.predict_proba([pair_feats(boxes[i], boxes[j], panels, P["size"])])[0, 1]
    return list(np.argsort(-wins))

if __name__ == "__main__":
    from configs.config import READING_RTL
    df = load_aligned(ALIGNED_PATH)
    hi = df[(df.tier == "high") & (~df.ambiguous) & df.order_idx.notna()]
    PAN = json.load(open(PANELS_PATH))
    seqs = hi.seq_id.unique(); taus = []
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(seqs):
        X, y = [], []
        for (s, pg), g in hi[hi.seq_id.isin(set(seqs[tr]))].groupby(["seq_id", "page"]):
            if len(g) < 2: continue
            a, b = page_pairs(g, PAN[f"{s}|{pg}"], READING_RTL); X += a; y += b
        clf = HistGradientBoostingClassifier(max_depth=3, random_state=SEED).fit(X, y)
        for (s, pg), g in hi[hi.seq_id.isin(set(seqs[te]))].groupby(["seq_id", "page"]):
            if len(g) < 3: continue
            order = sort_page(clf, g[["x0","y0","x1","y1"]].to_numpy(float), PAN[f"{s}|{pg}"], READING_RTL)
            rank = np.empty(len(g)); rank[order] = np.arange(len(g))
            t = kendalltau(rank, g.order_idx.to_numpy()).correlation
            if not np.isnan(t): taus.append(t)
    print("pairwise mean tau (grouped CV):", np.mean(taus), "pages:", len(taus))