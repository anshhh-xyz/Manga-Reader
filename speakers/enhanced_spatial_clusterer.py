# speakers/enhanced_spatial_clusterer.py
"""
Primary Speaker Diarization Engine (Phase 5 XGBoost Pairwise + Turn-Taking Parity)
==================================================================================
Best performing speaker attribution model:
  - balanced_joint_f1: 0.3607  (22-feature baseline)
  - balanced_joint_f1: 0.3604  (P12 Decoupled Crop OCR + Typography, record holder)

Key features (22 / 26 total):
1. Turn-taking rank parity ((j - i) % 2) within reading order.
2. Physical panel boundaries and gap analysis (same_panel, panel_gap, pp_gap).
3. Geometric spatial distance and relative bounding box area ratios.
4. Agglomerative clustering with distance threshold t=0.50, singleton merging, and max_k=4.
5. [OPTIONAL] DeBERTa-v3 cross-encoder hybrid fusion via use_cross_encoder=True.
   Blends: P_final = ce_alpha * P_deberta + (1 - ce_alpha) * P_spatial
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"

MODEL_PATH = WEIGHTS / "pairwise_xgb.json"
CFG_PATH = WEIGHTS / "cluster_cfg.json"

TAG_NARR = "narration"

DEFAULT_CFG = {
    "threshold": 0.50,
    "merge_singletons": True,
    "max_clusters": 4
}

_MODEL = None
_CFG = None


def load_model():
    global _MODEL, _CFG
    if _MODEL is None:
        m_p = MODEL_PATH if MODEL_PATH.exists() else (ROOT / "outputs" / "speakers" / "pairwise_xgb.json")
        if not m_p.exists():
            raise FileNotFoundError(f"Pairwise model file not found at {m_p}")
        m = XGBClassifier()
        m.load_model(str(m_p))
        _MODEL = m

    if _CFG is None:
        c_p = CFG_PATH if CFG_PATH.exists() else (ROOT / "outputs" / "speakers" / "cluster_cfg.json")
        if c_p.exists():
            _CFG = json.loads(c_p.read_text(encoding="utf-8"))
        else:
            _CFG = DEFAULT_CFG.copy()

    return _MODEL, _CFG


def _geom(b: Dict, size: Tuple[float, float]):
    W, H = size
    x1, y1, x2, y2 = b["bbox"]
    return ((x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H, (x2 - x1) / W, (y2 - y1) / H)


def extract_pair_features(
    boxes: List[Dict],
    sizes: List[Tuple[float, float]],
    char_map: Optional[Dict[int, List[Dict]]] = None,
):
    """
    Build pairwise feature matrix.

    char_map: {page_idx -> [{bbox, conf}, ...]} from detection.char_detector.
              If None, character proximity features are set to NaN (backward compat).

    Returns (X of shape [n_pairs, 26], pairs list).
    """
    from detection.char_detector import nearest_char_features

    n = len(boxes)
    X, pairs = [], []
    g = [_geom(b, sizes[b.get("page", 0)]) for b in boxes]
    seen = defaultdict(int)
    pos = []
    for b in boxes:
        k = (b.get("page", 0), b.get("panel", -1))
        pos.append(seen[k])
        seen[k] += 1

    # Pre-compute per-bubble character proximity features
    char_dists: List[float] = []
    char_idxs: List[int] = []
    for b in boxes:
        pg = b.get("page", 0)
        W, H = sizes[pg]
        chars = (char_map or {}).get(pg, [])
        d, cidx = nearest_char_features(b["bbox"], chars, W, H)
        char_dists.append(d)
        char_idxs.append(cidx)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            same_page = int(a.get("page", 0) == b.get("page", 0))
            pan_a = a.get("panel", -1)
            pan_b = b.get("panel", -1)
            same_panel = int(same_page and pan_a >= 0 and pan_a == pan_b)

            dx = g[j][0] - g[i][0] if same_page else np.nan
            dy = g[j][1] - g[i][1] if same_page else np.nan
            dist = float(np.hypot(dx, dy)) if same_page else np.nan
            pgap = abs(pan_a - pan_b) if same_page and pan_a >= 0 and pan_b >= 0 else np.nan
            ppg = abs(pos[i] - pos[j]) if same_panel else np.nan

            ta, tb = a.get("text") or "", b.get("text") or ""

            # ── Original 22 Phase-5 features ──────────────────────────────────
            feat = [
                float(same_page),
                float(abs(a.get("page", 0) - b.get("page", 0))),
                float(j - i),
                float((j - i) % 2),
                float(same_panel),
                float(pgap),
                float(ppg),
                dx, dy, dist,
                g[i][2], g[i][3], g[j][2], g[j][3],
                (g[i][2] * g[i][3]) / max(g[j][2] * g[j][3], 1e-6),
                float(len(ta)), float(len(tb)),
                float("?" in ta), float("?" in tb),
                float("!" in ta), float("!" in tb),
                np.nan,  # cos_sim placeholder
            ]

            # ── 4 NEW character-proximity features (indices 22-25) ────────────
            if char_map is not None and same_page:
                ci, cj = char_idxs[i], char_idxs[j]
                feat += [
                    char_dists[i],                        # dist bubble_i → nearest char (norm)
                    char_dists[j],                        # dist bubble_j → nearest char (norm)
                    abs(char_dists[i] - char_dists[j]),   # proximity difference between pair
                    float(ci >= 0 and ci == cj),          # do they share the same nearest char?
                ]
            else:
                feat += [np.nan, np.nan, np.nan, np.nan]

            X.append(feat)
            pairs.append((i, j))

    return np.array(X, dtype=float).reshape(-1, 26), pairs


def cluster_probabilities(P: np.ndarray, t: float = 0.50, merge_singletons: bool = True, max_k: int = 4) -> np.ndarray:
    n = len(P)
    if n == 1:
        return np.zeros(1, dtype=int)
    D = np.clip(1.0 - P, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=t
    )
    labels = clustering.fit_predict(D)

    if max_k and len(set(labels)) > max_k and n > max_k:
        clustering_capped = AgglomerativeClustering(
            n_clusters=max_k,
            metric="precomputed",
            linkage="average"
        )
        labels = clustering_capped.fit_predict(D)

    if merge_singletons:
        sizes = np.bincount(labels)
        big = [c for c in range(len(sizes)) if sizes[c] >= 2]
        new = labels.copy()
        if big:
            for c in np.where(sizes == 1)[0]:
                i = int(np.where(labels == c)[0][0])
                new[i] = max(big, key=lambda k: P[i, labels == k].mean())
        labels = new

    return labels


def relabel(labels: np.ndarray) -> List[str]:
    names: Dict[int, str] = {}
    out: List[str] = []
    for c in labels:
        if c not in names:
            names[c] = f"char{len(names) + 1}"
        out.append(names[c])
    return out


def cluster_assign(
    boxes: List[Dict],
    page_sizes: List = None,
    cfg: Optional[Dict] = None,
    char_map: Optional[Dict[int, List[Dict]]] = None,
    use_cross_encoder: bool = False,
    ce_alpha: float = 0.5,
) -> List[str]:
    """
    Primary speaker assignment entry point.

    Args:
        boxes:             Ordered list of dialogue box dicts (page, bbox, text, tag).
        page_sizes:        [(W, H), ...] per page.
        cfg:               Optional override for clustering config.
        char_map:          Optional {page_idx -> [{bbox, conf}]} from detect_chars().
                           If provided, 4 character-proximity features are added.
        use_cross_encoder: If True, blends XGBoost spatial probs with DeBERTa-v3
                           cross-encoder probs: P = ce_alpha*P_text + (1-ce_alpha)*P_spatial
        ce_alpha:          Weight for DeBERTa scores in the blend [0.0, 1.0].
                           0.0 = pure XGBoost, 1.0 = pure DeBERTa, 0.5 = equal blend.
    """
    model, base_cfg = load_model()
    run_cfg = base_cfg.copy()
    if cfg:
        run_cfg.update(cfg)

    story_idx = [k for k, b in enumerate(boxes) if b.get("tag") != TAG_NARR]
    labels = ["NARRATION"] * len(boxes)
    if not story_idx:
        return labels

    sub_boxes = [boxes[k] for k in story_idx]
    if page_sizes is None:
        page_sizes = [(1000.0, 1500.0)] * 3

    if len(sub_boxes) == 1:
        labels[story_idx[0]] = "char1"
        return labels

    X, pairs = extract_pair_features(sub_boxes, page_sizes, char_map=char_map)

    # Backward compat: if current model expects 22 features, truncate silently
    n_model_features = model.n_features_in_
    if X.shape[1] != n_model_features:
        X = X[:, :n_model_features]

    probs = model.predict_proba(X)[:, 1]

    P_spatial = np.eye(len(sub_boxes))
    for (i, j), p in zip(pairs, probs):
        P_spatial[i, j] = P_spatial[j, i] = p

    if use_cross_encoder:
        from speakers.cross_encoder_infer import build_probability_matrix
        P_text = build_probability_matrix(sub_boxes)
        P = ce_alpha * P_text + (1.0 - ce_alpha) * P_spatial
    else:
        P = P_spatial

    assigned = relabel(
        cluster_probabilities(
            P,
            t=float(run_cfg.get("threshold", 0.50)),
            merge_singletons=bool(run_cfg.get("merge_singletons", True)),
            max_k=int(run_cfg.get("max_clusters", 4)),
        )
    )

    for k, name in zip(story_idx, assigned):
        labels[k] = name

    return labels


# Alias for backward and cross-module compatibility
assign_speakers = cluster_assign
