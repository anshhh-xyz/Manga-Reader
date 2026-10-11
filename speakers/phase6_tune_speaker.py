# speakers/phase6_tune_speaker.py
"""
Phase 6 — Speaker Model Adaptation (Track A)
=============================================
Goal: Close the speaker gap (0.3418 -> ?) by:
  1. Adding richer text-based features (punctuation, length ratio, keyword flags)
  2. Hyperparameter search over XGBoost (n_estimators, max_depth, learning_rate)
  3. Re-tuning the clustering threshold on the new OOF predictions

Run with:
    python -m speakers.phase6_tune_speaker

Outputs:
    outputs/speakers/phase6_xgb.json          best XGBoost model
    outputs/speakers/phase6_cluster_cfg.json  best clustering config
    outputs/speakers/phase6_results.json      full sweep table
"""

import json
import pickle
import itertools
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from xgboost import XGBClassifier

from .adapters import OUT, TAG_NARR, IMG_ROOT
from .adapter import load_sequences
from .adapters import page_sizes as page_sizes_fn
from .pair_features import prob_matrix, FEATURES
from .panels import fill_panels
from .character_memory import cluster

SEED = 42

# ── Extended feature set ────────────────────────────────────────────────────
EXT_FEATURES = FEATURES + [
    "exclaim_diff",   # |ex_i - ex_j|  (both excited vs one calm)
    "q_diff",         # |q_i - q_j|    (both asking vs one stating)
    "len_ratio",      # min/max text length  (very different lengths → different speakers)
    "narr_flag_i",    # 1 if box_i might be narration (all-caps, short, no punctuation)
    "narr_flag_j",
]


def _narr_flag(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    # Caption-style: all caps, no ellipsis/exclaim, short
    return int(t.isupper() and len(t) < 60 and "..." not in t)


def build_ext():
    """Build feature matrices with extended features."""
    # Load aligned boxes (same as standard training)
    from collections import defaultdict
    import re
    from .adapter import _read_records, _mapping, _bbox, _tag, TAG_NARR, TAG_NON

    recs = _read_records(Path("outputs/alignment/aligned_boxes.json"))
    m = _mapping(recs[0])
    rows = [{k: (r.get(v) if v else None) for k, v in m.items()} for r in recs]
    shift = 1 if min(int(r["page"]) for r in rows if r["page"] is not None) == 1 else 0
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

    # Fill panels using detected panel map
    from .adapters import IMG_ROOT
    for s, bs in by_seq.items():
        paths = sorted((IMG_ROOT / s).glob("*"))[:3]
        if paths:
            fill_panels(paths, bs)

    seqs = dict(by_seq)

    # Build pairwise features (extended)
    def pair_feat_ext(boxes, sizes):
        from collections import defaultdict
        n = len(boxes)
        X, y, pairs = [], [], []

        def geom(b):
            W, H = sizes[b["page"]]
            x1, y1, x2, y2 = b["bbox"]
            return (x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H

        g = [geom(b) for b in boxes]
        seen = defaultdict(int); pos = []
        for b in boxes:
            k = (b["page"], b["panel"]); pos.append(seen[k]); seen[k] += 1

        for i in range(n):
            for j in range(i+1, n):
                a, b = boxes[i], boxes[j]
                same_page = a["page"] == b["page"]
                same_panel = int(same_page and a["panel"] >= 0 and a["panel"] == b["panel"])
                dx = g[j][0] - g[i][0] if same_page else np.nan
                dy = g[j][1] - g[i][1] if same_page else np.nan
                dist = float(np.hypot(dx, dy)) if same_page else np.nan
                pgap = abs(a["panel"] - b["panel"]) if same_page and a["panel"] >= 0 and b["panel"] >= 0 else np.nan
                ppg = abs(pos[i] - pos[j]) if same_panel else np.nan
                ta, tb = a["text"] or "", b["text"] or ""
                # base features (no cos_sim — replaced with nan)
                base = [int(same_page), abs(a["page"]-b["page"]), j-i, (j-i)%2, same_panel, pgap, ppg,
                        dx, dy, dist, g[i][2], g[i][3], g[j][2], g[j][3],
                        (g[i][2]*g[i][3])/max(g[j][2]*g[j][3], 1e-6),
                        len(ta), len(tb), int("?" in ta), int("?" in tb),
                        int("!" in ta), int("!" in tb), np.nan]  # cos_sim=nan
                # extended features
                ext = [
                    abs(int("!" in ta) - int("!" in tb)),
                    abs(int("?" in ta) - int("?" in tb)),
                    min(len(ta), len(tb)) / max(len(ta), len(tb), 1),
                    _narr_flag(ta),
                    _narr_flag(tb),
                ]
                X.append(base + ext)
                sa, sb = a.get("speaker"), b.get("speaker")
                ok = sa not in (None, "UNKNOWN") and sb not in (None, "UNKNOWN")
                y.append(int(sa == sb) if ok else -1)
                pairs.append((i, j))

        return np.array(X, dtype=float).reshape(-1, len(EXT_FEATURES)), np.array(y), pairs

    data = {}
    for s, boxes in seqs.items():
        story = [b for b in boxes if b["tag"] != TAG_NARR]
        if len(story) < 2:
            continue
        sizes = page_sizes_fn(s)
        X, y, pairs = pair_feat_ext(story, sizes)
        data[s] = dict(X=X, y=y, pairs=pairs, n=len(story),
                       gold=[b["speaker"] for b in story])
    print(f"Sequences loaded: {len(data)}")
    return data


def mk_xgb(n_est, depth, lr):
    return XGBClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=SEED, tree_method="hist"
    )


def fit(seqs, data, n_est, depth, lr):
    X = np.vstack([data[s]["X"] for s in seqs])
    y = np.concatenate([data[s]["y"] for s in seqs])
    keep = y >= 0; X, y = X[keep], y[keep]
    m = mk_xgb(n_est, depth, lr)
    m.set_params(scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
    return m.fit(X, y)


def oof_run(data, n_est, depth, lr):
    seqs = np.array(sorted(data)); res = {}
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(seqs):
        m = fit(seqs[tr], data, n_est, depth, lr)
        for s in seqs[te]:
            X = data[s]["X"].copy()
            res[s] = prob_matrix(data[s]["n"], data[s]["pairs"],
                                 m.predict_proba(X)[:, 1])
    return res


def pair_auc(data, res):
    ys, ps = [], []
    for s, d in data.items():
        P = res[s]; k = d["y"] >= 0
        ps += [P[i, j] for (i, j), keep in zip(d["pairs"], k) if keep]
        ys += list(d["y"][k])
    return roc_auc_score(ys, ps) if ys else 0.0


def mean_ari(data, oof, t, merge, max_k):
    aris = []
    for s, d in data.items():
        known = [k for k, g in enumerate(d["gold"]) if g != "UNKNOWN"]
        if not known:
            continue
        lab = cluster(oof[s], t, merge, max_k if max_k else None)
        gold = [d["gold"][k] for k in known]
        pred = [lab[k] for k in known]
        aris.append(adjusted_rand_score(gold, pred)
                    if (len(set(gold)) > 1 or len(set(pred)) > 1) else 1.0)
    return float(np.mean(aris)) if aris else 0.0


# ── Hyperparameter grids ────────────────────────────────────────────────────
XGB_GRID = list(itertools.product(
    [200, 400, 600],       # n_estimators
    [3, 4, 5],             # max_depth
    [0.03, 0.05, 0.08],    # learning_rate
))

CLUSTER_GRID = list(itertools.product(
    np.arange(0.35, 0.75, 0.05),   # threshold
    [False, True],                   # merge_singletons
    [0, 4, 6],                       # max_clusters (0 = unlimited)
))


if __name__ == "__main__":
    data = build_ext()
    all_results = []

    # Phase 1: XGBoost hyperparameter sweep (AUC as proxy)
    print("\n=== XGBoost Hyperparameter Sweep ===")
    print(f"{'n_est':>6} {'depth':>6} {'lr':>6} | {'AUC':>6}")
    best_auc, best_oof, best_xgb_params = -1, None, None

    for n_est, depth, lr in XGB_GRID:
        oof = oof_run(data, n_est, depth, lr)
        auc = pair_auc(data, oof)
        print(f"{n_est:6} {depth:6} {lr:6.3f} | {auc:.4f}")
        all_results.append({"n_est": n_est, "depth": depth, "lr": lr, "auc": round(auc, 4)})
        if auc > best_auc:
            best_auc = auc
            best_oof = oof
            best_xgb_params = (n_est, depth, lr)

    print(f"\nBest XGBoost params: n_est={best_xgb_params[0]}, depth={best_xgb_params[1]}, lr={best_xgb_params[2]}")
    print(f"Best pairwise AUC: {best_auc:.4f}")

    # Phase 2: Cluster threshold sweep on best OOF
    print("\n=== Cluster Threshold Sweep ===")
    print(f"{'t':>5} {'merge':>6} {'maxk':>5} | {'ARI':>6}")
    best_ari, best_cluster_cfg = -1, None

    for t, merge, max_k in CLUSTER_GRID:
        ari = mean_ari(data, best_oof, float(t), merge, max_k)
        print(f"{t:.2f} {str(merge):>6} {max_k:5} | {ari:.4f}")
        all_results.append({"threshold": round(float(t), 2), "merge": merge,
                             "max_k": max_k, "ari": round(ari, 4)})
        if ari > best_ari:
            best_ari = ari
            best_cluster_cfg = {"threshold": round(float(t), 2),
                                 "merge_singletons": bool(merge),
                                 "max_clusters": int(max_k) if max_k else None}

    print(f"\nBest cluster config: {best_cluster_cfg}")
    print(f"Best mean ARI: {best_ari:.4f}")

    # Save best model (trained on all data)
    n_est, depth, lr = best_xgb_params
    final_model = fit(sorted(data), data, n_est, depth, lr)
    final_model.save_model(str(OUT / "phase6_xgb.json"))
    json.dump(best_cluster_cfg, open(OUT / "phase6_cluster_cfg.json", "w"))
    json.dump(all_results, open(OUT / "phase6_results.json", "w"), indent=2)

    print(f"\nSaved:")
    print(f"  {OUT}/phase6_xgb.json")
    print(f"  {OUT}/phase6_cluster_cfg.json")
    print(f"  {OUT}/phase6_results.json")
    print(f"\n--- Phase 6 Track A Complete ---")
    print(f"Run evaluation with:")
    print(f"  python -m speakers.eval_phase6")
