# filtering/infer_filter.py
"""Run the trained story filter on a DataFrame of OCR boxes.

Usage:
    from filtering.infer_filter import story_prob
    probs = story_prob(df_boxes)   # returns np.array of P(story) per row
"""
import joblib
import numpy as np
from pathlib import Path

from configs.config import FILTER_MODEL_PATH
from filtering.features import build_features

ROOT = Path(__file__).resolve().parents[1]

_bundle = None


def story_prob(df_boxes):
    """Return P(story) for each row in df_boxes.

    df_boxes must have columns: seq_id, page, x0, y0, x1, y1, text, conf
    Returns a 1-D np.array of floats in [0, 1].
    """
    global _bundle
    if _bundle is None:
        model_path = Path(FILTER_MODEL_PATH)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Filter model not found: {model_path}\n"
                "Run:  python -m filtering.train_filter  first."
            )
        _bundle = joblib.load(str(model_path))

    X = build_features(df_boxes)[_bundle["features"]]
    p = np.zeros((len(df_boxes), 3))
    for col, class_id in enumerate(_bundle["model"].classes_):
        p[:, int(class_id)] = _bundle["model"].predict_proba(X)[:, col]
    return 1 - p[:, 0]   # P(story) = 1 - P(non-story)