# filtering/infer_filter.py
import joblib, numpy as np
from configs.config import FILTER_MODEL_PATH
from filtering.features import build_features

_bundle = None
def story_prob(df_boxes):
    """df_boxes: seq_id,page,x0,y0,x1,y1,text,conf -> np.array of P(story)."""
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(FILTER_MODEL_PATH)
    X = build_features(df_boxes)[_bundle["features"]]
    p = np.zeros((len(df_boxes), 3))
    p[:, _bundle["model"].classes_] = _bundle["model"].predict_proba(X)
    return 1 - p[:, 0]