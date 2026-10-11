# speakers/diagnose.py
import pickle, numpy as np
from collections import Counter
from sklearn.metrics import adjusted_rand_score
from xgboost import XGBClassifier
from .adapters import OUT
from .pair_features import FEATURES
from .character_memory import cluster

blob = pickle.load(open(OUT / "oof.pkl", "rb")); data, oof = blob["data"], blob["oof"]

m = XGBClassifier(); m.load_model(str(OUT / "pairwise_xgb.json"))
print("== feature importance ==")
for n, v in sorted(zip(FEATURES, m.feature_importances_), key=lambda x: -x[1]):
    print(f"{n:11s} {v:.3f}")

X = np.vstack([d["X"] for d in data.values()]); y = np.concatenate([d["y"] for d in data.values()])
sp = X[:, FEATURES.index("same_page")] == 1
print("\n== data checks ==")
print("labelled pairs:", int((y >= 0).sum()), "| same-speaker rate:", round(float(y[y >= 0].mean()), 3))
print("same-page pairs with same_panel=1:", round(float(X[sp, FEATURES.index("same_panel")].mean()), 3),
      "(near 0 means panel ids are missing)")

rows = []
for s, d in data.items():
    known = [k for k, g in enumerate(d["gold"]) if g != "UNKNOWN"]
    if len(known) < 2: continue
    lab = cluster(oof[s], 0.5, True, 4)
    gold = [d["gold"][k] for k in known]; pred = [lab[k] for k in known]
    ari = adjusted_rand_score(gold, pred) if len(set(gold)) > 1 or len(set(pred)) > 1 else 1.0
    rows.append((s, d["n"], len(set(gold)), len(set(pred)), ari))

print("\n== per-sequence ==")
print("mean boxes:", round(np.mean([r[1] for r in rows]), 1),
      "| mean gold speakers:", round(np.mean([r[2] for r in rows]), 2),
      "| mean predicted:", round(np.mean([r[3] for r in rows]), 2),
      "| mean ARI:", round(np.mean([r[4] for r in rows]), 3))
print("gold speaker count distribution:", sorted(Counter(r[2] for r in rows).items()))
print("\nworst 8 by ARI (seq, boxes, gold_k, pred_k, ARI):")
for r in sorted(rows, key=lambda r: r[4])[:8]:
    print(f"  {r[0]}  n={r[1]}  gold={r[2]}  pred={r[3]}  ari={r[4]:.2f}")