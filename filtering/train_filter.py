
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


# --------------------------------------------------
# Configuration
# --------------------------------------------------

N_SPLITS = 5
NARRATION_WEIGHT = 5.0

CLASS_NAMES = ["non-story", "dialogue", "narration"]


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
        n_jobs=-1,
        random_state=SEED,
    )


# --------------------------------------------------
# Load data and features
# --------------------------------------------------

df = load_aligned(ALIGNED_PATH)

X = build_features(df)[FEATURES].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

y = labels(df)
mask = training_mask(df)

seqs = df["seq_id"].unique()

print(f"Total boxes: {len(df)}")
print(f"Labeled training boxes: {mask.sum()}")
print(f"Sequences: {len(seqs)}")
print(f"Narration weight multiplier: {NARRATION_WEIGHT}")

if len(seqs) < N_SPLITS:
    raise ValueError(f"Need at least {N_SPLITS} sequences.")

if not np.array_equal(np.unique(y[mask]), np.array([0, 1, 2])):
    raise ValueError("Training data must contain all three classes.")

# Output directories must already exist, or be created manually.
oof_base = Path(OOF_PATH)
model_base = Path(FILTER_MODEL_PATH)

# Split by sequence to avoid putting boxes from one sequence
# into both the training and validation folds.
cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

oof = np.zeros((len(df), 3), dtype=float)


# --------------------------------------------------
# Cross-validation
# --------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv.split(seqs), start=1):

    train_seqs = set(seqs[train_idx])
    test_seqs = set(seqs[test_idx])

    train_rows = (
        df["seq_id"].isin(train_seqs).to_numpy() & mask
    )

    test_rows = df["seq_id"].isin(test_seqs).to_numpy()
    test_boxes = np.flatnonzero(test_rows)

    X_train = X.iloc[np.flatnonzero(train_rows)]
    y_train = y[train_rows]
    X_test = X.iloc[test_boxes]

    if len(np.unique(y_train)) != 3:
        raise ValueError(
            f"Fold {fold} is missing a class in its training data."
        )

    # Balance the classes, then give narration extra weight.
    weights = compute_sample_weight("balanced", y_train)
    weights[y_train == 2] *= NARRATION_WEIGHT

    model = make_model()
    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
    )

    probabilities = model.predict_proba(X_test)

    # Map probability columns to their class IDs.
    for col, class_id in enumerate(model.classes_):
        oof[test_boxes, int(class_id)] = probabilities[:, col]

    print(
        f"Fold {fold}: "
        f"train rows {train_rows.sum()}, "
        f"held-out boxes {len(test_boxes)}"
    )


# --------------------------------------------------
# Evaluate out-of-fold predictions
# --------------------------------------------------

y_true = y[mask]
y_pred = oof[mask].argmax(axis=1)

print("\nXGBoost classification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

auc = roc_auc_score(
    y_true > 0,
    1 - oof[mask, 0],
)

print(f"Story-vs-non-story AUC: {auc:.4f}")

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

metrics = {
    "model": "xgboost",
    "narration_weight": NARRATION_WEIGHT,
    "accuracy": report["accuracy"],
    "macro_f1": report["macro avg"]["f1-score"],
    "weighted_f1": report["weighted avg"]["f1-score"],
    "non_story_f1": report["non-story"]["f1-score"],
    "dialogue_f1": report["dialogue"]["f1-score"],
    "narration_f1": report["narration"]["f1-score"],
    "narration_precision": report["narration"]["precision"],
    "narration_recall": report["narration"]["recall"],
    "story_auc": auc,
}

print("\nSummary:")
print(pd.Series(metrics).to_string())


# --------------------------------------------------
# Save OOF predictions and metrics
# --------------------------------------------------

oof_df = df.copy()
oof_df["p_non"] = oof[:, 0]
oof_df["p_dlg"] = oof[:, 1]
oof_df["p_nar"] = oof[:, 2]

oof_path = oof_base.with_name(
    f"{oof_base.stem}_xgboost_nar{NARRATION_WEIGHT:g}"
    f"{oof_base.suffix}"
)

oof_df.to_pickle(oof_path)

metrics_path = oof_base.with_name(
    f"xgboost_narration_weight_{NARRATION_WEIGHT:g}.csv"
)
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

print(f"\nSaved OOF predictions: {oof_path}")
print(f"Saved metrics: {metrics_path}")


# --------------------------------------------------
# Train final model on all usable labeled data
# --------------------------------------------------

final_weights = compute_sample_weight("balanced", y[mask])
final_weights[y[mask] == 2] *= NARRATION_WEIGHT

final_model = make_model()
final_model.fit(
    X.loc[mask],
    y[mask],
    sample_weight=final_weights,
)

final_path = model_base.with_name(
    f"{model_base.stem}_xgboost_nar{NARRATION_WEIGHT:g}"
    f"{model_base.suffix}"
)

joblib.dump(
    {
        "model": final_model,
        "features": FEATURES,
    },
    final_path,
)

print(f"Saved final model: {final_path}")
print("\nXGBoost training complete.")
