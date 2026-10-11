# filtering/train_filter.py
"""Train an XGBoost story-text filter on the aligned OCR boxes.

Run with:
    python -m filtering.train_filter

Outputs (in outputs/filtering/):
    filter_oof.pkl          — OOF probability DataFrame consumed by tune_threshold
    filter_model.joblib     — Final model bundle consumed by infer_filter
    xgboost_narration_weight_<N>.csv  — per-fold metrics
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, roc_auc_score

from configs.config import (
    SEED,
    ALIGNED_PATH,
    FILTER_MODEL_PATH,
    OOF_PATH,
)
from filtering.dataset import load_aligned, labels, training_mask
from filtering.features import build_features, FEATURES

# ── Configuration ──────────────────────────────────────────────────────────────
N_SPLITS = 5
NARRATION_WEIGHT = 3.0
CLASS_NAMES = ["non-story", "dialogue", "narration"]

ROOT = Path(__file__).resolve().parents[1]


def make_model():
    """Create the XGBoost classifier."""
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",          # GPU acceleration; falls back to CPU automatically
        n_jobs=-1,
        random_state=SEED,
    )


def main():
    aligned_path = ROOT / ALIGNED_PATH if not Path(ALIGNED_PATH).is_absolute() else Path(ALIGNED_PATH)
    oof_base     = ROOT / OOF_PATH     if not Path(OOF_PATH).is_absolute()     else Path(OOF_PATH)
    model_base   = ROOT / FILTER_MODEL_PATH if not Path(FILTER_MODEL_PATH).is_absolute() else Path(FILTER_MODEL_PATH)

    # Create output directories
    oof_base.parent.mkdir(parents=True, exist_ok=True)
    model_base.parent.mkdir(parents=True, exist_ok=True)

    if not aligned_path.exists():
        raise FileNotFoundError(
            f"Aligned boxes not found: {aligned_path}\n"
            "Run the alignment step first (alignment/merge.py)."
        )

    # ── Load data and features ─────────────────────────────────────────────────
    df   = load_aligned(str(aligned_path))
    X    = build_features(df)[FEATURES].copy()
    X    = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y    = labels(df)
    mask = training_mask(df)
    seqs = df["seq_id"].unique()

    print(f"Total boxes:            {len(df)}")
    print(f"Labeled training boxes: {mask.sum()}")
    print(f"Sequences:              {len(seqs)}")
    print(f"Narration weight:       {NARRATION_WEIGHT}")

    if len(seqs) < N_SPLITS:
        raise ValueError(f"Need at least {N_SPLITS} sequences, got {len(seqs)}.")

    if not np.array_equal(np.unique(y[mask]), np.array([0, 1, 2])):
        raise ValueError("Training data must contain all three classes (non-story, dialogue, narration).")

    # ── Cross-validation ───────────────────────────────────────────────────────
    cv  = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(df), 3), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(cv.split(seqs), start=1):
        train_seqs = set(seqs[train_idx])
        test_seqs  = set(seqs[test_idx])

        train_rows = df["seq_id"].isin(train_seqs).to_numpy() & mask
        test_rows  = df["seq_id"].isin(test_seqs).to_numpy()
        test_boxes = np.flatnonzero(test_rows)

        X_train = X.iloc[np.flatnonzero(train_rows)]
        y_train = y[train_rows]
        X_test  = X.iloc[test_boxes]

        if len(np.unique(y_train)) != 3:
            raise ValueError(f"Fold {fold} is missing a class in training data.")

        weights = compute_sample_weight("balanced", y_train)
        weights[y_train == 2] *= NARRATION_WEIGHT

        model = make_model()
        model.fit(X_train, y_train, sample_weight=weights)

        proba = model.predict_proba(X_test)
        for col, class_id in enumerate(model.classes_):
            oof[test_boxes, int(class_id)] = proba[:, col]

        print(f"Fold {fold}: train={train_rows.sum()} labeled rows, held-out={len(test_boxes)} boxes")

    # ── OOF evaluation ─────────────────────────────────────────────────────────
    y_true = y[mask]
    y_pred = oof[mask].argmax(axis=1)

    print("\nXGBoost classification report:")
    print(classification_report(y_true, y_pred, labels=[0,1,2],
                                target_names=CLASS_NAMES, zero_division=0))

    auc = roc_auc_score(y_true > 0, 1 - oof[mask, 0])
    print(f"Story-vs-non-story AUC: {auc:.4f}")

    report = classification_report(y_true, y_pred, labels=[0,1,2],
                                   target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "model": "xgboost",
        "narration_weight": NARRATION_WEIGHT,
        "accuracy":           report["accuracy"],
        "macro_f1":           report["macro avg"]["f1-score"],
        "weighted_f1":        report["weighted avg"]["f1-score"],
        "non_story_f1":       report["non-story"]["f1-score"],
        "dialogue_f1":        report["dialogue"]["f1-score"],
        "narration_f1":       report["narration"]["f1-score"],
        "narration_precision":report["narration"]["precision"],
        "narration_recall":   report["narration"]["recall"],
        "story_auc":          auc,
    }
    print("\nSummary:")
    print(pd.Series(metrics).to_string())

    # ── Save OOF (at the canonical OOF_PATH so tune_threshold finds it) ────────
    oof_df = df.copy()
    oof_df["p_non"] = oof[:, 0]
    oof_df["p_dlg"] = oof[:, 1]
    oof_df["p_nar"] = oof[:, 2]
    oof_df.to_pickle(str(oof_base))       # saved at OOF_PATH exactly
    print(f"\nSaved OOF predictions: {oof_base}")

    # Also save a versioned copy alongside
    versioned_oof = oof_base.with_name(
        f"{oof_base.stem}_xgboost_nar{NARRATION_WEIGHT:g}{oof_base.suffix}"
    )
    oof_df.to_pickle(str(versioned_oof))
    print(f"Saved versioned OOF:   {versioned_oof}")

    metrics_path = oof_base.with_name(f"xgboost_narration_weight_{NARRATION_WEIGHT:g}.csv")
    pd.DataFrame([metrics]).to_csv(str(metrics_path), index=False)
    print(f"Saved metrics:         {metrics_path}")

    # ── Final model on all labeled data ────────────────────────────────────────
    final_weights = compute_sample_weight("balanced", y[mask])
    final_weights[y[mask] == 2] *= NARRATION_WEIGHT

    final_model = make_model()
    final_model.fit(X.loc[mask], y[mask], sample_weight=final_weights)

    joblib.dump({"model": final_model, "features": FEATURES}, str(model_base))
    print(f"Saved final model:     {model_base}")
    print("\nXGBoost training complete.")


if __name__ == "__main__":
    main()
