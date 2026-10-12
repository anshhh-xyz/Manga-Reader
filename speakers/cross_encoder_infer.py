# speakers/cross_encoder_infer.py
"""
DeBERTa-v3 Cross-Encoder Inference
====================================
Loads the fine-tuned pairwise same-speaker classifier from
  weights/deberta_speaker_cross_encoder/
and scores arbitrary (text_a, text_b) pairs as P(same_speaker).

Usage:
    from speakers.cross_encoder_infer import score_pairs
    probs = score_pairs(boxes)   # boxes: list of {"text", "page", "panel", ...}

Returns a numpy array of shape [n_pairs] with P(same_speaker) for each (i<j) pair,
in the exact order produced by enhanced_spatial_clusterer.extract_pair_features().
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights" / "deberta_speaker_cross_encoder"

_TOKENIZER = None
_MODEL = None
_DEVICE = None

BATCH_SIZE = 32
MAX_LEN = 96


def _load():
    global _TOKENIZER, _MODEL, _DEVICE
    if _MODEL is not None:
        return _TOKENIZER, _MODEL, _DEVICE

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"[cross_encoder_infer] Loading DeBERTa from {WEIGHTS_DIR} ...")
    _TOKENIZER = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
    _MODEL = AutoModelForSequenceClassification.from_pretrained(str(WEIGHTS_DIR))
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    print(f"[cross_encoder_infer] Model loaded on {_DEVICE}")
    return _TOKENIZER, _MODEL, _DEVICE


def _format(box: Dict) -> str:
    pg = box.get("page", 0)
    pan = box.get("panel", -1)
    pan_str = f"[PAN {pan}]" if pan >= 0 else "[PAN ?]"
    text = (box.get("text") or "").strip()
    return f"[P{pg}] {pan_str} {text}"


def score_pairs(
    boxes: List[Dict],
    alpha: float = 1.0,          # weight of DeBERTa output (1-alpha = random fallback)
) -> np.ndarray:
    """
    Score all (i<j) pairs from `boxes`.

    Returns
    -------
    probs : np.ndarray of shape [n_pairs]
        P(same_speaker) for each pair in (i<j) order.
    """
    if not WEIGHTS_DIR.exists():
        raise FileNotFoundError(
            f"DeBERTa weights not found at {WEIGHTS_DIR}. "
            "Run `python -m speakers.train_cross_encoder` first."
        )

    tok, model, device = _load()

    n = len(boxes)
    texts_a, texts_b = [], []
    for i in range(n):
        for j in range(i + 1, n):
            texts_a.append(_format(boxes[i]))
            texts_b.append(_format(boxes[j]))

    if not texts_a:
        return np.array([], dtype=float)

    all_probs: List[float] = []

    for start in range(0, len(texts_a), BATCH_SIZE):
        ta_batch = texts_a[start : start + BATCH_SIZE]
        tb_batch = texts_b[start : start + BATCH_SIZE]

        enc = tok(
            ta_batch,
            tb_batch,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits          # shape [B, 2]
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(same_speaker)

        all_probs.extend(probs.cpu().numpy().tolist())

    return np.array(all_probs, dtype=float)


def build_probability_matrix(
    boxes: List[Dict],
) -> np.ndarray:
    """
    Convenience wrapper: returns full n×n same-speaker probability matrix.
    Diagonal is 1.0 by convention.
    """
    n = len(boxes)
    P = np.eye(n, dtype=float)
    probs = score_pairs(boxes)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            P[i, j] = P[j, i] = probs[k]
            k += 1
    return P
