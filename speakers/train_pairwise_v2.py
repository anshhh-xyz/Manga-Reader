# speakers/train_pairwise_v2.py
"""
Pairwise XGBoost Retraining Script — Phase 6 (26 Features + Char Proximity)
=============================================================================
Adds 4 character body proximity features to the existing 22 spatial features:
  - dist_bubble_i_to_nearest_char (normalised)
  - dist_bubble_j_to_nearest_char (normalised)
  - |dist_i - dist_j| proximity difference
  - same_nearest_char (bool)

Saves the new model as: weights/pairwise_xgb_v2.json
To promote to production: copy over weights/pairwise_xgb.json

Run:
    python -m speakers.train_pairwise_v2
    python -m speakers.train_pairwise_v2 --promote   # auto-copies to production
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
OUT_MODEL = WEIGHTS / "pairwise_xgb_v2.json"
PROD_MODEL = WEIGHTS / "pairwise_xgb.json"
PROD_MODEL_BACKUP = WEIGHTS / "pairwise_xgb_v1_backup.json"
SEED = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_sequences() -> Dict[str, Tuple[List[Dict], List[Tuple[float, float]]]]:
    """
    Returns {seq_id: (boxes_with_gold_speakers, page_sizes)}.
    Uses speakers.adapters to load gold aligned boxes with real bboxes and panels.
    """
    from speakers.adapters import load_sequences, page_sizes

    raw_seqs = load_sequences()
    sequences = {}
    for s, boxes in raw_seqs.items():
        # Keep dialogue/story boxes (skip pure narration for speaker attribution)
        story = [b for b in boxes if b.get("tag") != "narration"]
        if len(story) >= 2:
            sizes = [(float(w), float(h)) for w, h in page_sizes(s)]
            sequences[s] = (story, sizes)

    print(f"Loaded {len(sequences)} sequences for training.")
    return sequences


def build_training_data(sequences: Dict) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Build (X, y, seq_ids) for pairwise classification:
      y=1: same speaker, y=0: different speaker.
    Pairs with UNKNOWN speakers are filtered out.
    Includes character proximity features via char_map.
    """
    from speakers.enhanced_spatial_clusterer import extract_pair_features
    from detection.char_detector import detect_chars
    from evaluation.common import image_paths

    all_X, all_y, all_sid = [], [], []

    for sid, (boxes, sizes) in sequences.items():
        # Build char_map for this sequence
        char_map = {}
        try:
            paths = image_paths(sid)
            for pg, p in enumerate(paths):
                char_map[pg] = detect_chars(p)
        except Exception as e:
            print(f"  [WARN] char detection failed for {sid}: {e}")
            char_map = {}

        # Extract features (26 total)
        X, pairs = extract_pair_features(boxes, sizes, char_map=char_map if char_map else None)

        # Build labels (filter out UNKNOWN)
        speakers = [b.get("speaker") for b in boxes]
        valid_X, valid_y = [], []
        for idx, (i, j) in enumerate(pairs):
            sa, sb = speakers[i], speakers[j]
            if sa in (None, "UNKNOWN") or sb in (None, "UNKNOWN"):
                continue
            valid_X.append(X[idx])
            valid_y.append(1 if sa == sb else 0)

        if valid_X:
            all_X.append(np.array(valid_X))
            all_y.append(np.array(valid_y))
            all_sid.extend([sid] * len(valid_y))

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    return X_all, y_all, all_sid


def make_model():
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        tree_method="hist",
        enable_categorical=False,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true",
                    help="After training, copy new model to production path.")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    print("=== Phase 6: Pairwise XGBoost Training (26 features) ===\n")

    sequences = load_all_sequences()
    print("Building pairwise training data (running char detection)...")
    X, y, sids = build_training_data(sequences)

    print(f"  Total valid pairs: {len(y)}")
    print(f"  Same-speaker     : {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  Features         : {X.shape[1]}")
    print()

    # ── Cross-validation AUC Comparison (22 vs 26 features) ──
    seq_ids = np.array(sorted(sequences.keys()))
    sid_to_row = {}
    for k, sid in enumerate(sids):
        sid_to_row.setdefault(sid, []).append(k)

    oof_probs_22 = np.zeros(len(y))
    oof_probs_26 = np.zeros(len(y))

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, te_idx) in enumerate(kf.split(seq_ids)):
        tr_sids = set(seq_ids[tr_idx])
        te_sids = set(seq_ids[te_idx])
        tr_rows = [r for sid in tr_sids for r in sid_to_row.get(sid, [])]
        te_rows = [r for sid in te_sids for r in sid_to_row.get(sid, [])]
        if not tr_rows or not te_rows:
            continue

        ytr = y[tr_rows]
        w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

        # 22-feature baseline
        m22 = make_model()
        m22.set_params(scale_pos_weight=w)
        m22.fit(X[tr_rows, :22], ytr)
        oof_probs_22[te_rows] = m22.predict_proba(X[te_rows, :22])[:, 1]

        # 26-feature with character proximity
        m26 = make_model()
        m26.set_params(scale_pos_weight=w)
        m26.fit(X[tr_rows], ytr)
        oof_probs_26[te_rows] = m26.predict_proba(X[te_rows])[:, 1]

        auc22 = roc_auc_score(y[te_rows], oof_probs_22[te_rows])
        auc26 = roc_auc_score(y[te_rows], oof_probs_26[te_rows])
        print(f"  Fold {fold+1}: 22-feat AUC = {auc22:.4f}  |  26-feat (+char) AUC = {auc26:.4f}")

    valid = oof_probs_26 > 0
    if valid.sum() > 0:
        oof_auc_22 = roc_auc_score(y[valid], oof_probs_22[valid])
        oof_auc_26 = roc_auc_score(y[valid], oof_probs_26[valid])
        print(f"\n  Overall OOF AUC:")
        print(f"    22-feature baseline : {oof_auc_22:.4f}")
        print(f"    26-feature (+char)  : {oof_auc_26:.4f} ({'+' if oof_auc_26 >= oof_auc_22 else ''}{oof_auc_26 - oof_auc_22:+.4f})")

    # ── Final model on all data ──
    print("\nFitting final model on all training data...")
    w = (y == 0).sum() / max((y == 1).sum(), 1)
    final = make_model()
    final.set_params(scale_pos_weight=w)
    final.fit(X, y)
    final.save_model(str(OUT_MODEL))
    print(f"  Saved: {OUT_MODEL}")

    if args.promote:
        if PROD_MODEL.exists():
            shutil.copy(PROD_MODEL, PROD_MODEL_BACKUP)
            print(f"  Backed up old model to: {PROD_MODEL_BACKUP}")
        shutil.copy(OUT_MODEL, PROD_MODEL)
        print(f"  Promoted to production: {PROD_MODEL}")
        print("  NOTE: Clear _MODEL cache by restarting any running server.")

    print("\nDone.")


if __name__ == "__main__":
    main()
