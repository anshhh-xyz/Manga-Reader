# speakers/sentiment_features.py
"""
Sentiment and Conversational Speech-Act Transition Features
=============================================================
Extracts emotional polarity, dialogue-act transitions, and semantic
relatedness for pairwise speaker diarization.

Features provided per pair (i, j):
  1. sentiment_i: Sentiment polarity score of utterance i in [-1, +1]
  2. sentiment_j: Sentiment polarity score of utterance j in [-1, +1]
  3. sentiment_diff: |sentiment_i - sentiment_j| (emotional divergence)
  4. sentiment_opposing: 1 if opposite emotional valence (positive vs negative), else 0
  5. is_question_i: 1 if utterance i is a question (punctuated or interrogative)
  6. is_question_j: 1 if utterance j is a question
  7. q_to_answer_transition: 1 if i is a question and j is an answer (speech-act alternation)
  8. is_interjection_j: 1 if utterance j begins with an interjection/reaction (reactive turn)
  9. sbert_cosine_sim: SBERT embedding cosine similarity (replaces NaN placeholder)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
import numpy as np

# ── Manga Dialogue Lexicon ────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "good", "great", "cool", "beautiful", "thanks", "thank", "love", "nice",
    "awesome", "happy", "congrats", "best", "proud", "luck", "honest", "trust",
    "rely", "talent", "talented", "fine", "yeah", "yes", "sure", "ok", "okay"
}

NEGATIVE_WORDS = {
    "bad", "embarrassing", "blood", "dead", "afraid", "stupid", "filthy", "sick",
    "terrible", "angry", "hate", "wrong", "no", "against", "cannot", "cant",
    "wont", "none", "nothing", "never", "sorry", "rain", "hurt", "fail", "hard"
}

INTERJECTIONS = {
    "huh", "woah", "whoa", "ehh", "eh", "hm", "hmm", "ah", "wait", "what",
    "hey", "right", "yeah", "no", "gosh", "geez", "see", "man"
}

WH_INTERROGATIVES = {
    "why", "what", "how", "when", "where", "who", "which", "whose",
    "did", "does", "do", "is", "are", "can", "could", "would", "will"
}

_SBERT_MODEL = None


def get_sbert_model():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def analyze_utterance(text: str) -> Dict[str, float]:
    """Extract individual utterance-level tone and speech-act properties."""
    t_clean = (text or "").strip().lower()
    tokens = re.findall(r"\b[a-z']+\b", t_clean)

    if not tokens:
        return {
            "sentiment": 0.0,
            "is_question": float("?" in text),
            "is_interjection": 0.0,
            "is_exclamation": float("!" in text),
        }

    pos_count = sum(1 for w in tokens if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in tokens if w in NEGATIVE_WORDS)
    total_matches = pos_count + neg_count

    if total_matches > 0:
        sentiment = (pos_count - neg_count) / float(total_matches)
    else:
        sentiment = 0.0

    # Question detection: punctuation OR starting interrogative
    has_qmark = "?" in text
    starts_wh = bool(tokens and tokens[0] in WH_INTERROGATIVES)
    is_question = float(has_qmark or (starts_wh and len(tokens) <= 8))

    # Interjection / reaction: first token is a known reaction
    is_interjection = float(bool(tokens and tokens[0] in INTERJECTIONS))

    return {
        "sentiment": sentiment,
        "is_question": is_question,
        "is_interjection": is_interjection,
        "is_exclamation": float("!" in text),
    }


def compute_pairwise_sentiment_features(
    box_i: Dict,
    box_j: Dict,
    emb_i: Optional[np.ndarray] = None,
    emb_j: Optional[np.ndarray] = None,
) -> List[float]:
    """
    Computes pairwise conversational sentiment and speech-act features.
    Returns a list of 8 floats:
      [sentiment_i, sentiment_j, sentiment_diff, sentiment_opposing,
       is_question_i, is_question_j, q_to_answer_transition, is_interjection_j]
    """
    ti = analyze_utterance(box_i.get("text", ""))
    tj = analyze_utterance(box_j.get("text", ""))

    si, sj = ti["sentiment"], tj["sentiment"]
    sentiment_diff = abs(si - sj)
    sentiment_opposing = float((si > 0 and sj < 0) or (si < 0 and sj > 0))

    qi, qj = ti["is_question"], tj["is_question"]
    # Q-to-Answer transition: i asks question, j does not ask
    q_to_answer = float(qi > 0 and qj == 0)

    # Reaction transition: j starts with an interjection
    interj_j = tj["is_interjection"]

    return [
        si,
        sj,
        sentiment_diff,
        sentiment_opposing,
        qi,
        qj,
        q_to_answer,
        interj_j,
    ]


def precompute_sequence_embeddings(texts: List[str]) -> np.ndarray:
    """Precompute normalized SBERT embeddings for all utterances in a sequence."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_sbert_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
