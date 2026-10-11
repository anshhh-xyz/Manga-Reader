# speakers/hybrid_spatial_tracker.py
"""
Hybrid Spatial Sequential Dialogue Tracker (Option 2 - Refined)
==============================================================
Addresses the four issues identified in the original implementation:
1. Normalized spatial thresholds: Distances computed relative to page (W, H).
   - Stacked: dx / W < 0.12, -0.03 < dy_gap / H < 0.10 (76.2% empirical accuracy).
   - Horizontally split: dx / W > 0.16.
2. Controlled speaker spawning: Eliminates artificial over-splitting across panels.
   - cross_panel_penalty = 0.0
   - new_panel non-last bonus = 0.0
   - new_speaker_threshold = 0.0
   - max_speakers = 4
3. Clean narration: Uses the verified tag ('narration') directly from the filter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# ── SBERT Embeddings (cached) ────────────────────────────────────────────────
_SBERT = None
_SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_sbert():
    global _SBERT
    if _SBERT is None:
        from sentence_transformers import SentenceTransformer
        _SBERT = SentenceTransformer(_SBERT_MODEL_NAME)
    return _SBERT


def _embed(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_sbert()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    return embs.astype(np.float32)


# ── Configuration ────────────────────────────────────────────────────────────
DEFAULT_CFG: Dict = {
    "stack_bonus": 0.65,           # Reward for same speaker on vertically stacked bubbles
    "same_panel_bonus": 0.15,      # Reward for same speaker in same panel
    "alternation_bonus": 0.35,     # Reward for switching speaker when horizontally split
    "cross_panel_penalty": 0.0,    # Cross-panel transitions are 50/50, no penalty
    "alpha": 0.30,                 # Weight of SBERT semantic similarity
    "new_speaker_threshold": 0.0,  # Only spawn when all candidate scores are negative
    "max_speakers": 4,             # Cap on max speakers per sequence
    "window": 3,                   # Rolling embedding window per speaker
    # Normalized geometric thresholds
    "stacked_dx_max": 0.12,        # dx / W < 0.12
    "stacked_dy_min": -0.03,       # dy_gap / H > -0.03
    "stacked_dy_max": 0.10,        # dy_gap / H < 0.10
    "horiz_split_dx_min": 0.16,    # dx / W > 0.16
}

CFG_PATH = Path(__file__).resolve().parents[1] / "weights" / "hybrid_spatial_cfg.json"


def load_cfg(cfg: Optional[Dict] = None) -> Dict:
    base = DEFAULT_CFG.copy()
    if CFG_PATH.exists():
        try:
            base.update(json.loads(CFG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    if cfg:
        base.update(cfg)
    return base


# ── Normalized Geometry ─────────────────────────────────────────────────────
def _spatial_relation(
    b_curr: Dict,
    b_prev: Dict,
    sizes: List[Tuple[float, float]],
    cfg: Dict,
) -> Tuple[bool, bool, bool]:
    """
    Computes spatial relationship normalized by page dimensions.
    Returns: (is_stacked, is_horiz_split, in_same_panel)
    """
    p_curr = b_curr.get("page", 0)
    p_prev = b_prev.get("page", 0)
    pan_curr = b_curr.get("panel", -1)
    pan_prev = b_prev.get("panel", -1)

    in_same_panel = (p_curr == p_prev and pan_curr >= 0 and pan_curr == pan_prev)
    if not in_same_panel:
        return False, False, False

    W, H = sizes[p_curr] if p_curr < len(sizes) else (1000.0, 1500.0)
    bb_curr = b_curr.get("bbox") or [0, 0, 0, 0]
    bb_prev = b_prev.get("bbox") or [0, 0, 0, 0]

    cx_curr = (bb_curr[0] + bb_curr[2]) / 2.0
    cx_prev = (bb_prev[0] + bb_prev[2]) / 2.0

    dx_norm = abs(cx_curr - cx_prev) / max(W, 1.0)
    dy_gap_norm = (bb_curr[1] - bb_prev[3]) / max(H, 1.0)  # top_curr - bottom_prev

    # Step H1: Normalized stacked & horizontal split checks
    is_stacked = (
        dx_norm < cfg["stacked_dx_max"] and
        cfg["stacked_dy_min"] < dy_gap_norm < cfg["stacked_dy_max"]
    )
    is_horiz_split = (dx_norm > cfg["horiz_split_dx_min"])

    return is_stacked, is_horiz_split, in_same_panel


# ── Core Tracker ─────────────────────────────────────────────────────────────
def hybrid_assign(
    boxes: List[Dict],
    page_sizes: List[Tuple[float, float]] = None,
    cfg: Optional[Dict] = None,
    precomputed_embs: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Assigns speaker labels to boxes in reading order.
    """
    cfg = load_cfg(cfg)
    stack_bonus       = float(cfg["stack_bonus"])
    same_panel_bonus  = float(cfg["same_panel_bonus"])
    alternation_bonus = float(cfg["alternation_bonus"])
    cross_panel_pen   = float(cfg["cross_panel_penalty"])
    alpha             = float(cfg["alpha"])
    new_thr           = float(cfg["new_speaker_threshold"])
    max_spk           = int(cfg.get("max_speakers", 4))
    window            = cfg.get("window", 3)

    if page_sizes is None:
        page_sizes = [(1000.0, 1500.0)] * 3

    labels = ["char1"] * len(boxes)

    # Step H2: Use filter's narration tag directly
    story_indices = []
    for idx, b in enumerate(boxes):
        if b.get("tag") == "narration":
            labels[idx] = "NARRATION"
        else:
            story_indices.append(idx)

    if not story_indices:
        return labels

    story_boxes = [boxes[k] for k in story_indices]
    story_texts = [str(b.get("text") or "") for b in story_boxes]

    # Get embeddings
    if precomputed_embs is not None:
        embs = precomputed_embs
    else:
        embs = _embed(story_texts)

    spk_ids: List[str] = []
    spk_embs: Dict[str, List[np.ndarray]] = {}
    spk_mean: Dict[str, np.ndarray] = {}

    story_labels: List[str] = []
    prev_box: Optional[Dict] = None
    last_spk: Optional[str] = None

    for i, (b, emb) in enumerate(zip(story_boxes, embs)):
        if not spk_ids:
            char_id = "char1"
        else:
            is_stacked, is_horiz_split, in_same_panel = (False, False, False)
            if prev_box is not None:
                is_stacked, is_horiz_split, in_same_panel = _spatial_relation(b, prev_box, page_sizes, cfg)

            scores: Dict[str, float] = {}
            for cand in spk_ids:
                score = 0.0

                if in_same_panel:
                    if is_stacked:
                        score += stack_bonus if (cand == last_spk) else -stack_bonus
                    elif is_horiz_split:
                        score += -alternation_bonus if (cand == last_spk) else alternation_bonus
                    else:
                        score += same_panel_bonus if (cand == last_spk) else 0.0
                else:
                    # Step H2: cross-panel penalty is 0.0, non-last bonus is 0.0
                    score += cross_panel_pen if (cand == last_spk) else 0.0

                # Semantic similarity
                mean_emb = spk_mean[cand]
                cos_sim = float(np.dot(emb, mean_emb))
                score += alpha * cos_sim

                scores[cand] = score

            best_cand = max(scores, key=scores.__getitem__)
            best_score = scores[best_cand]

            # Spawning condition
            can_add_new = (max_spk is None or len(spk_ids) < max_spk)
            if is_stacked and last_spk is not None:
                char_id = last_spk
            elif can_add_new and best_score < new_thr:
                char_id = f"char{len(spk_ids) + 1}"
            else:
                char_id = best_cand

        # Register speaker
        if char_id not in spk_ids:
            spk_ids.append(char_id)
            spk_embs[char_id] = []
            spk_mean[char_id] = emb.copy()

        # Update history
        spk_embs[char_id].append(emb)
        hist = spk_embs[char_id]
        if window is not None and len(hist) > window:
            hist = hist[-window:]
            spk_embs[char_id] = hist
        spk_mean[char_id] = np.mean(hist, axis=0)
        norm = np.linalg.norm(spk_mean[char_id])
        if norm > 1e-8:
            spk_mean[char_id] /= norm

        story_labels.append(char_id)
        last_spk = char_id
        prev_box = b

    for idx, lbl in zip(story_indices, story_labels):
        labels[idx] = lbl

    return labels
