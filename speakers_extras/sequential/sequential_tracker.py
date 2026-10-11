# speakers/sequential_tracker.py
"""
Sequential Dialogue Tracker
============================
Assigns speaker labels to manga dialogue boxes by processing them one at a time
in reading order, maintaining a growing history of each character's dialogue.

For each new box:
  1. Embed the text with Sentence-BERT (all-MiniLM-L6-v2).
  2. Score against every known speaker's mean embedding (cosine similarity).
  3. Add a turn-taking prior: in manga, speakers alternate A→B→A within a panel.
  4. If the best score exceeds `new_speaker_threshold`, assign to that speaker.
     Otherwise, create a new character (char1, char2, ...).
  5. Update the speaker's running embedding history.

This replaces the pairwise XGBoost + Agglomerative Clustering approach, which
achieved only ARI=0.13 and balanced_joint_f1=0.314 because it had no access
to semantic dialogue content — only geometry and reading-order position.

Public API
----------
    sequential_assign(boxes, page_sizes, cfg=None) -> list[str]

Config keys (with defaults):
    alpha                  : float = 0.7   # weight of semantic similarity
    beta                   : float = 0.3   # weight of turn-taking prior
    new_speaker_threshold  : float = 0.25  # min score to NOT create new speaker
    max_speakers           : int   = 4     # hard cap (None = unlimited)
    window                 : int   = 5     # rolling history window per speaker (None = all)
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# --------------------------------------------------------------------------- #
# Sentence-BERT lazy loader
# --------------------------------------------------------------------------- #

_SBERT = None
_SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

def _get_sbert():
    """Lazy-load SentenceTransformer once; reuse globally."""
    global _SBERT
    if _SBERT is None:
        from sentence_transformers import SentenceTransformer
        _SBERT = SentenceTransformer(_SBERT_MODEL_NAME)
    return _SBERT


def _embed(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings → L2-normalised embeddings (N, 384).
    Empty/whitespace strings get a zero vector.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_sbert()
    embs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    return embs.astype(np.float32)


# --------------------------------------------------------------------------- #
# Default config
# --------------------------------------------------------------------------- #

DEFAULT_CFG: Dict = {
    "alpha": 0.7,
    "beta": 0.3,
    "new_speaker_threshold": 0.25,
    "max_speakers": 4,
    "window": 5,
}

_WEIGHTS_CFG = Path(__file__).resolve().parents[1] / "weights" / "sequential_cfg.json"


def load_cfg(cfg: Optional[Dict] = None) -> Dict:
    """Merge provided cfg → file cfg → defaults (priority left to right)."""
    base = DEFAULT_CFG.copy()
    if _WEIGHTS_CFG.exists():
        base.update(json.loads(_WEIGHTS_CFG.read_text(encoding="utf-8")))
    if cfg:
        base.update(cfg)
    return base


# --------------------------------------------------------------------------- #
# Turn-taking prior
# --------------------------------------------------------------------------- #

def _turn_prior(candidate_char: str,
                last_spk: Optional[str],
                in_same_panel: bool) -> float:
    """
    Returns a signed scalar bonus/penalty.

    Manga rule: within a panel, speakers strictly alternate (A→B→A→B).
    - Same as last speaker, same panel  → strong penalty  (-0.35)
    - Same as last speaker, diff panel  → slight penalty  (-0.10)
    - Different from last, same panel   → reward          (+0.20)
    - Different from last, diff panel   → small reward    (+0.05)
    - No previous speaker               → neutral          (0.00)
    """
    if last_spk is None:
        return 0.0
    same_spk = (candidate_char == last_spk)
    if same_spk:
        return -0.35 if in_same_panel else -0.10
    else:
        return +0.20 if in_same_panel else +0.05


# --------------------------------------------------------------------------- #
# Main tracker
# --------------------------------------------------------------------------- #

def sequential_assign(
    boxes: List[Dict],
    page_sizes: List,
    cfg: Optional[Dict] = None,
    precomputed_embs: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Assign speaker labels to a sequence of dialogue/narration boxes.

    Parameters
    ----------
    boxes            : list of box dicts. Required keys:
                         text  (str)
                         tag   (str)  — boxes with tag=="narration" → "NARRATION"
                         page  (int)
                         panel (int)  — panel index within page (-1 if unknown)
    page_sizes       : [(W, H), (W, H), (W, H)]  — not used directly but kept for API compat.
    cfg              : optional config override dict.
    precomputed_embs : optional precomputed (N_story, 384) embeddings array to bypass SBERT encoding.

    Returns
    -------
    list of str — same length as boxes. E.g. ["char1", "char2", "char1", "NARRATION", ...]
    """
    cfg = load_cfg(cfg)
    alpha     = float(cfg["alpha"])
    beta      = float(cfg["beta"])
    new_thr   = float(cfg["new_speaker_threshold"])
    max_spk   = cfg.get("max_speakers")        # None = unlimited
    window    = cfg.get("window")              # None = use all history

    TAG_NARR = "narration"

    labels: List[str] = ["NARRATION"] * len(boxes)

    # Separate story (dialogue) boxes from narration
    story_idx = [k for k, b in enumerate(boxes) if b.get("tag", "") != TAG_NARR]
    if not story_idx:
        return labels

    story_boxes = [boxes[k] for k in story_idx]
    story_texts = [str(b.get("text") or "") for b in story_boxes]

    # Batch-encode all story texts upfront (faster than one-by-one)
    if precomputed_embs is not None:
        all_embs = precomputed_embs
    else:
        all_embs = _embed(story_texts)   # (N_story, 384)

    # Per-speaker state
    spk_ids: List[str] = []                          # ordered unique speaker IDs (char1, char2, ...)
    spk_emb_lists: Dict[str, List[np.ndarray]] = {}  # {char_id: [emb, emb, ...]}
    spk_mean: Dict[str, np.ndarray] = {}             # {char_id: mean embedding}

    last_spk:   Optional[str] = None
    last_panel: Optional[int] = None
    last_page:  Optional[int] = None

    story_labels: List[str] = []

    for i, (box, emb) in enumerate(zip(story_boxes, all_embs)):
        cur_page  = box.get("page", 0)
        cur_panel = box.get("panel", -1)
        in_same_panel = (
            last_page == cur_page and
            cur_panel >= 0 and
            last_panel == cur_panel
        )

        if not spk_ids:
            # First dialogue box — always char1
            char_id = "char1"
        else:
            # Score against each known speaker
            scores: Dict[str, float] = {}
            for char_id_cand in spk_ids:
                mean_emb = spk_mean[char_id_cand]
                cos_sim  = float(np.dot(emb, mean_emb))   # both L2-normalised
                prior    = _turn_prior(char_id_cand, last_spk, in_same_panel)
                scores[char_id_cand] = alpha * cos_sim + beta * prior

            best_char  = max(scores, key=scores.__getitem__)
            best_score = scores[best_char]

            # Decide: assign to best_char OR create a new speaker
            can_add_new = (max_spk is None or len(spk_ids) < max_spk)
            if can_add_new and best_score < new_thr:
                char_id = f"char{len(spk_ids) + 1}"
            else:
                char_id = best_char

        # Register new speaker if needed
        if char_id not in spk_ids:
            spk_ids.append(char_id)
            spk_emb_lists[char_id] = []
            spk_mean[char_id] = emb.copy()

        # Update speaker history (rolling window)
        spk_emb_lists[char_id].append(emb)
        hist = spk_emb_lists[char_id]
        if window is not None and len(hist) > window:
            hist = hist[-window:]
            spk_emb_lists[char_id] = hist
        spk_mean[char_id] = np.mean(hist, axis=0)
        # Re-normalise the mean (cosine distance requires unit vectors)
        norm = np.linalg.norm(spk_mean[char_id])
        if norm > 1e-8:
            spk_mean[char_id] /= norm

        story_labels.append(char_id)
        last_spk   = char_id
        last_panel = cur_panel
        last_page  = cur_page

    # Map story labels back into the full box list
    for k, lbl in zip(story_idx, story_labels):
        labels[k] = lbl

    return labels
