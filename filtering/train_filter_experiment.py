
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier

from configs.config import (
    SEED,
    ALIGNED_PATH,
    FILTER_MODEL_PATH,
    OOF_PATH,
)
from filtering.dataset import load_aligned, labels, training_mask
from filtering.features import build_features, FEATURES


# --------------------------------------------------
# 1. Define the models
# --------------------------------------------------

def make_models():
    return {
        "hist_gradient": HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=300,
            random_state=SEED,
        ),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        ),

        "xgboost": XGBClassifier(
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
        ),
    }


# --------------------------------------------------
# 2. Load data and extract features once
# --------------------------------------------------

df = load_aligned(ALIGNED_PATH)

X = build_features(df)[FEATURES].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

y = labels(df)
mask = training_mask(df)

seqs = df["seq_id"].unique()

print(f"Total boxes: {len(df)}")
print(f"Usable labeled boxes: {mask.sum()}")
print(f"Sequences: {len(seqs)}")
print(f"Features: {len(FEATURES)}")

if len(seqs) < 5:
    raise ValueError("At least 5 sequences are needed for 5-fold CV.")

if len(np.unique(y[mask])) < 3:
    raise ValueError("Training data must contain all three classes.")

# Keep each sequence entirely inside one fold.
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

class_names = ["non-story", "dialogue", "narration"]
results = []

# Convert config values to Paths because they may be strings.
oof_base = Path(OOF_PATH)
model_base = Path(FILTER_MODEL_PATH)


# --------------------------------------------------
# 3. Train and evaluate every model
# --------------------------------------------------

for model_name, model_template in make_models().items():

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    # One set of out-of-fold predictions per model.
    oof = np.zeros((len(df), 3), dtype=float)

    for fold, (train_seq_idx, test_seq_idx) in enumerate(
        cv.split(seqs), start=1
    ):
        train_seqs = set(seqs[train_seq_idx])
        test_seqs = set(seqs[test_seq_idx])

        train_rows = (
            df["seq_id"].isin(train_seqs).to_numpy() & mask
        )

        test_rows = df["seq_id"].isin(test_seqs).to_numpy()
        test_idx = np.flatnonzero(test_rows)

        X_train = X.iloc[np.flatnonzero(train_rows)]
        y_train = y[train_rows]
        X_test = X.iloc[test_idx]

        if len(np.unique(y_train)) < 3:
            raise ValueError(
                f"{model_name}, fold {fold}: "
                "training fold is missing a class."
            )

        # Balance the classes using training rows only.
        weights = compute_sample_weight("balanced", y_train)

        # Create a fresh model for every fold.
        model = model_template.__class__(
            **model_template.get_params()
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=weights,
        )

        probabilities = model.predict_proba(X_test)

        # Place probability columns in the correct class positions.
        for column, class_id in enumerate(model.classes_):
            oof[test_idx, int(class_id)] = probabilities[:, column]

        print(
            f"Fold {fold}: "
            f"train rows {train_rows.sum()}, "
            f"held-out boxes {len(test_idx)}"
        )

    # Evaluate only rows with reliable training labels.
    y_true = y[mask]
    y_pred = oof[mask].argmax(axis=1)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=class_names,
            zero_division=0,
        )
    )

    auc = roc_auc_score(
        y_true > 0,
        1 - oof[mask, 0],
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print(f"Story-vs-non-story AUC: {auc:.4f}")

    results.append({
        "model": model_name,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "non_story_f1": report["non-story"]["f1-score"],
        "dialogue_f1": report["dialogue"]["f1-score"],
        "narration_f1": report["narration"]["f1-score"],
        "narration_recall": report["narration"]["recall"],
        "story_auc": auc,
    })

    # Save this model's out-of-fold predictions.
    oof_df = df.copy()
    oof_df["p_non"] = oof[:, 0]
    oof_df["p_dlg"] = oof[:, 1]
    oof_df["p_nar"] = oof[:, 2]

    oof_path = oof_base.with_name(
        f"{oof_base.stem}_{model_name}{oof_base.suffix}"
    )
    oof_df.to_pickle(oof_path)
    print(f"Saved OOF predictions: {oof_path}")

    # Train the final model on all usable labeled rows.
    final_model = model_template.__class__(
        **model_template.get_params()
    )

    final_model.fit(
        X.loc[mask],
        y[mask],
        sample_weight=compute_sample_weight(
            "balanced", y[mask]
        ),
    )

    final_path = model_base.with_name(
        f"{model_base.stem}_{model_name}{model_base.suffix}"
    )

    joblib.dump(
        {
            "model": final_model,
            "features": FEATURES,
        },
        final_path,
    )

    print(f"Saved final model: {final_path}")


# --------------------------------------------------
# 4. Compare all models in one table
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "macro_f1", ascending=False
)

print("\n" + "=" * 75)
print("MODEL COMPARISON (sorted by macro F1)")
print("=" * 75)
print(results_df.round(4).to_string(index=False))

metrics_path = oof_base.with_name("model_comparison.csv")
results_df.to_csv(metrics_path, index=False)

print(f"\nSaved comparison table: {metrics_path}")
print("\nTraining and comparison complete.")
