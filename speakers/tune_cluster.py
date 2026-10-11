# speakers/tune_cluster.py
"""finds the best settings for turning those predictions into character groups.Tuning the clustering process"""
import json, pickle, numpy as np
from sklearn.metrics import adjusted_rand_score
from .adapters import OUT
from .character_memory import cluster

blob = pickle.load(open(OUT / "oof.pkl", "rb")); data, oof = blob["data"], blob["oof"]

def evaluate(t, merge):
    aris, extra = [], []
    for s, d in data.items():
        known = [k for k, g in enumerate(d["gold"]) if g != "UNKNOWN"]
        if not known: continue
        lab = cluster(oof[s], t, merge)
        gold = [d["gold"][k] for k in known]; pred = [lab[k] for k in known]
        aris.append(adjusted_rand_score(gold, pred) if len(set(gold)) > 1 or len(set(pred)) > 1 else 1.0)
        extra.append(len(set(pred)) - len(set(gold)))
    return np.mean(aris), np.mean(extra)

best = None
print("thr  merge   ARI   extra_labels/seq")
for merge in (False, True):
    for t in np.arange(0.30, 0.91, 0.05):
        ari, ex = evaluate(t, merge); print(f"{t:.2f} {merge!s:5}  {ari:.3f}  {ex:+.2f}")
        if best is None or ari > best[0]: best = (ari, float(t), merge)
cfg = {"threshold": round(best[1], 2), "merge_singletons": best[2]}
json.dump(cfg, open(OUT / "cluster_cfg.json", "w")); print("best:", best, "->", cfg)