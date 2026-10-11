# speakers/train_pairwise.py
import pickle, numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from .adapters import load_sequences, page_sizes, OUT, TAG_NARR
from .embeddings import get_emb, CACHE
from .pair_features import seq_pair_features, prob_matrix, FEATURES

SEED = 42
def mk():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="logloss", random_state=SEED,
                         tree_method="hist")

def build():
    cache = pickle.load(open(CACHE, "rb"))
    data = {}
    for s, boxes in load_sequences().items():
        story = [b for b in boxes if b["tag"] != TAG_NARR]
        if len(story) < 2:
            continue
        X, y, pairs = seq_pair_features(story, get_emb(story, cache), page_sizes(s))
        data[s] = dict(X=X, y=y, pairs=pairs, n=len(story), gold=[b["speaker"] for b in story])
    return data

def fit(seqs, data, drop=None):
    X = np.vstack([data[s]["X"] for s in seqs]); y = np.concatenate([data[s]["y"] for s in seqs])
    keep = y >= 0
    X, y = X[keep], y[keep]
    if drop is not None: X[:, drop] = np.nan                 # ablation: hide a feature
    m = mk(); m.set_params(scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
    return m.fit(X, y)

def oof(data, drop=None):
    seqs = np.array(sorted(data)); res = {}
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(seqs):
        m = fit(seqs[tr], data, drop)
        for s in seqs[te]:
            X = data[s]["X"].copy()
            if drop is not None: X[:, drop] = np.nan
            res[s] = prob_matrix(data[s]["n"], data[s]["pairs"], m.predict_proba(X)[:, 1])
    return res

def pair_auc(data, res):
    ys, ps = [], []
    for s, d in data.items():
        P = res[s]; k = d["y"] >= 0
        ps += [P[i, j] for (i, j), keep in zip(d["pairs"], k) if keep]; ys += list(d["y"][k])
    return roc_auc_score(ys, ps)

if __name__ == "__main__":
    data = build(); print("sequences:", len(data))
    res = oof(data); print("pairwise AUC (all features):", round(pair_auc(data, res), 4))
    res_noemb = oof(data, drop=FEATURES.index("cos_sim"))
    print("pairwise AUC (no CLIP sim):  ", round(pair_auc(data, res_noemb), 4))
    pickle.dump(dict(data=data, oof=res), open(OUT / "oof.pkl", "wb"))
    final = fit(sorted(data), data); final.save_model(str(OUT / "pairwise_xgb.json"))
    print("saved model + OOF probs")