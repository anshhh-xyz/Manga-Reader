# Phase 7: Sequential Dialogue Tracker — Speaker Attribution Redesign

> **Status:** Implementation-ready. This document is a complete handoff guide for any agent continuing this work.  
> **Goal:** Replace the pairwise XGBoost + agglomerative clustering speaker attribution system with a sequential, dialogue-history-aware tracker using Sentence-BERT embeddings.

---

## 1. Why Replace the Current System?

### Current System (Phase 5 XGBoost)
The existing speaker module (`speakers/`) works as follows:
1. For every sequence, build **all pairs (i, j)** of dialogue boxes.
2. XGBoost predicts `P(same_speaker | i, j)` from 22 geometric/text features.
3. Convert `P` to a distance matrix `D = 1 - P`.
4. Run **Agglomerative Clustering** on `D` with a tuned threshold `t` to group boxes.
5. Name clusters `char1`, `char2`, ... by first appearance.

### Why It Fails (Evidence)
| Symptom | Root Cause |
|---|---|
| Pairwise AUC = **0.57** (barely above 0.50 random) | Features are geometry-only; no semantic understanding |
| ARI = **0.13** (1.0 = perfect clustering) | Clustering garbage in → garbage out |
| `balanced_joint_f1` = **0.314** vs ceiling **0.664** | 53% of the possible speaker score is uncaptured |
| Predicts 2.09 speakers/seq vs gold 3.03 | Under-splits: lumps distinct speakers together |

The core problem: the model doesn't know *what* characters say. It only knows *where* boxes are on the page and *how far apart* they are in reading order. In manga, most of the signal for "who is talking" comes from:
- **Dialogue style** (vocabulary, sentence length, tone, punctuation)
- **Turn-taking structure** (A says X → B responds → A replies)
- **Consistency** (a character uses the same words/register throughout)

---

## 2. New Approach: Sequential Dialogue Tracker

### Core Idea
Process dialogue boxes **one at a time, in reading order**. Maintain a running **history** of what each discovered character has said. For each new box, score it against every known character's history using **semantic similarity** — if it matches well, assign it to that character; if it matches nothing, create a new character.

### Algorithm (Greedy Sequential Assignment)
```
speaker_histories = {}       # {char_id: [list of their text lines so far]}
speaker_embeddings = {}      # {char_id: mean SBERT embedding of their lines}

for each dialogue box (in reading order across 3 pages):
    text = clean_text(box["text"])
    emb  = sbert_encode(text)          # 384-dim sentence embedding

    if no speakers yet:
        assign "char1"; update history + embedding
        continue

    # Score against each known speaker
    scores = {}
    for char_id, mean_emb in speaker_embeddings.items():
        cosine_sim  = dot(emb, mean_emb)          # semantic similarity
        turn_prior  = turn_taking_prior(box, char_id, last_speaker)
        scores[char_id] = alpha * cosine_sim + beta * turn_prior

    # Score for "this is a new speaker"
    scores["NEW"] = new_speaker_threshold

    best = argmax(scores)
    if best == "NEW":
        char_id = f"char{len(speaker_histories)+1}"
    else:
        char_id = best

    assign char_id
    update speaker_histories[char_id].append(text)
    update speaker_embeddings[char_id] = mean(all embeddings for char_id)
```

### Turn-Taking Prior
Manga dialogue follows a very strict A→B→A→B pattern within a panel. The prior captures this:
```python
def turn_taking_prior(current_box, candidate_char, last_speaker, last_panel):
    if candidate_char == last_speaker:
        # Same speaker twice in a row: penalise (unlikely in tight exchanges)
        return -0.3 if same_panel(current_box, last_panel) else 0.0
    else:
        # Different speaker after last: reward
        return +0.2 if same_panel(current_box, last_panel) else 0.1
```

### Parameters to Tune (Grid Search on Dev Set)
| Parameter | Meaning | Search Range |
|---|---|---|
| `alpha` | Weight of semantic similarity | [0.3, 0.5, 0.7, 1.0] |
| `beta` | Weight of turn-taking prior | [0.1, 0.3, 0.5] |
| `new_speaker_threshold` | Min score to assign new character | [0.1, 0.2, 0.3, 0.4] |
| `max_speakers` | Hard cap on number of speakers per sequence | [3, 4, 5, None] |
| `window` | How many recent lines per character to include in mean emb | [3, 5, all] |

---

## 3. File Structure

### Files to KEEP (not move)
These are used by the pipeline and must stay in `speakers/`:
```
speakers/
├── __init__.py                  ← keep, update imports
├── adapters.py                  ← keep, loads aligned training data
├── panels.py                    ← keep, fills panel IDs into boxes
├── sequential_tracker.py        ← NEW: the main new module
├── eval_sequential.py           ← NEW: evaluation/scoring script
└── sequential_tracker.md        ← this document
```

### Files to MOVE to `speakers_extras/xgboost_pairwise/`
These implement the old pairwise XGBoost approach. Keep them for reference/ablation:
```
speakers/
├── character_memory.py          → speakers_extras/xgboost_pairwise/
├── pair_features.py             → speakers_extras/xgboost_pairwise/
├── train_pairwise.py            → speakers_extras/xgboost_pairwise/
├── tune_cluster.py              → speakers_extras/xgboost_pairwise/
├── embeddings.py                → speakers_extras/xgboost_pairwise/
├── adapter.py                   → speakers_extras/xgboost_pairwise/
├── diagnose.py                  → speakers_extras/xgboost_pairwise/
├── run_oracle.py                → speakers_extras/xgboost_pairwise/
├── phase6_tune_speaker.py       → speakers_extras/xgboost_pairwise/
└── eval_phase6.py               → speakers_extras/xgboost_pairwise/
```

### Pipeline Connection Files (update, don't move)
```
backend/pipeline_service.py      ← update assign_speakers() call
pipeline/inference.py            ← update _get_speaker_model() and speaker call
generate_submission.py           ← no change needed (uses pipeline_service)
```

---

## 4. Implementation Steps (in order)

### Step 1: Move old files to `speakers_extras/xgboost_pairwise/`
```powershell
mkdir speakers_extras\xgboost_pairwise
Move-Item speakers\character_memory.py speakers_extras\xgboost_pairwise\
Move-Item speakers\pair_features.py    speakers_extras\xgboost_pairwise\
Move-Item speakers\train_pairwise.py   speakers_extras\xgboost_pairwise\
Move-Item speakers\tune_cluster.py     speakers_extras\xgboost_pairwise\
Move-Item speakers\embeddings.py       speakers_extras\xgboost_pairwise\
Move-Item speakers\diagnose.py         speakers_extras\xgboost_pairwise\
Move-Item speakers\run_oracle.py       speakers_extras\xgboost_pairwise\
Move-Item speakers\phase6_tune_speaker.py speakers_extras\xgboost_pairwise\
Move-Item speakers\eval_phase6.py      speakers_extras\xgboost_pairwise\
```

### Step 2: Install sentence-transformers
```bash
pip install sentence-transformers
```
Model to use: `all-MiniLM-L6-v2` — 384-dim, runs fast on CPU, 80MB.  
Store it in HF cache (no explicit download needed, it auto-downloads on first use).

### Step 3: Implement `speakers/sequential_tracker.py`
This is the main new file. See Section 5 for full spec.

### Step 4: Implement `speakers/eval_sequential.py`
Evaluation script that:
1. Loads `outputs/alignment/aligned_boxes.json` (gold labels)
2. Runs `sequential_assign()` on every dev sequence
3. Computes `balanced_joint_f1` using `dataset/score.py`
4. Grid-searches over `alpha`, `beta`, `new_speaker_threshold`, `max_speakers`, `window`
5. Saves best config to `weights/sequential_cfg.json`
6. Prints comparison table vs XGBoost baseline

### Step 5: Connect to `backend/pipeline_service.py`
Change the speaker attribution call in `run_pipeline_on_images()`:
```python
# OLD (remove):
from speakers.character_memory import assign_speakers
labels = assign_speakers(all_seq_boxes, page_sizes, spk_model, spk_cfg)

# NEW:
from speakers.sequential_tracker import sequential_assign
labels = sequential_assign(all_seq_boxes, page_sizes, cfg=seq_cfg)
```
The new function requires NO trained model — just the config JSON.  
Remove `get_speaker_model()` and `_SPEAKER_MODEL` from `pipeline_service.py`.

### Step 6: Connect to `pipeline/inference.py`
Change `predict_sequence()` speaker block:
```python
# OLD (remove):
model, cfg = _get_speaker_model()
labels = assign_speakers(all_seq_boxes, page_sizes, model, cfg)

# NEW:
from speakers.sequential_tracker import sequential_assign
labels = sequential_assign(all_seq_boxes, page_sizes)
```

### Step 7: Run evaluation and update submission
```bash
python -m speakers.eval_sequential          # evaluate on dev set, saves best config
python generate_submission.py               # regenerate submission.jsonl with new speaker logic
python -m evaluation.run_eval --pred outputs/predictions_dev.jsonl --ref dataset/development/labels.jsonl
```

---

## 5. `sequential_tracker.py` Full Spec

### Public API
```python
def sequential_assign(
    boxes: list[dict],          # list of box dicts from pipeline (must have: text, page, panel, tag, bbox)
    page_sizes: list[tuple],    # [(W,H), (W,H), (W,H)]
    cfg: dict = None,           # loaded from weights/sequential_cfg.json (uses defaults if None)
) -> list[str]:
    """
    Assigns speaker labels (char1, char2, ...) to each box in reading order.
    Returns a list of labels of the same length as boxes.
    Boxes tagged 'narration' receive label 'NARRATION'.
    """
```

### Internal Functions
```python
def _get_sbert():
    """Lazy-load SentenceTransformer('all-MiniLM-L6-v2'). Cache globally."""

def _embed(texts: list[str]) -> np.ndarray:
    """Encode a list of strings to embeddings. Returns (N, 384) array."""

def _turn_prior(box, char_id, last_spk, last_panel, same_page) -> float:
    """Return a scalar bonus/penalty based on turn-taking pattern."""

def _new_speaker_score(cfg) -> float:
    """Threshold score for deciding a new character has appeared."""
```

### Default Config (if `weights/sequential_cfg.json` absent)
```json
{
    "alpha": 0.7,
    "beta": 0.3,
    "new_speaker_threshold": 0.25,
    "max_speakers": 4,
    "window": 5
}
```

---

## 6. Expected Performance

Based on the analysis of why the current system fails, the sequential tracker should:

| Metric | XGBoost (current) | Sequential (expected) | Notes |
|---|---|---|---|
| `balanced_joint_f1` | 0.314 | **~0.38–0.45** | Semantic signal replaces geometry |
| ARI (speaker grouping) | 0.13 | **~0.25–0.35** | History-aware > pairwise global |
| Over-splitting | Yes (2.09 pred vs 3.03 gold) | Better (threshold-controlled) | `max_speakers` cap helps |
| Inference time | ~0.5s/seq | ~0.8–1.2s/seq (SBERT) | Still fast enough |
| Needs trained model? | Yes (`pairwise_xgb.json`) | **No** (SBERT is pretrained) | Simpler deployment |

### Where it will still struggle
- 3+ character scenes with similar dialogue style (SBERT can't distinguish)
- Very short single-word lines ("Yes.", "No.", "!") → low embedding signal
- Without visual face detection, misattributions in complex crowd scenes remain

---

## 7. Key Files Reference

| File | Purpose |
|---|---|
| `outputs/alignment/aligned_boxes.json` | Gold-aligned training/eval data (5,692 story boxes, 79 dev sequences) |
| `dataset/development/labels.jsonl` | Gold speaker labels for scoring |
| `dataset/score.py` | Official competition scorer — import as module |
| `weights/sequential_cfg.json` | Output of eval_sequential.py grid search — best hyperparameters |
| `speakers/adapters.py` | Loads aligned_boxes.json into box dict format |
| `speakers/panels.py` | Fills panel IDs into box dicts from XY-cut detection |
| `backend/pipeline_service.py` | Main inference service — update speaker call here |
| `pipeline/inference.py` | Batch inference runner — update speaker call here |
| `generate_submission.py` | Regenerates submission.jsonl — re-run after any change |

---

## 8. Scoring Reference

Competition metric: **`balanced_joint_f1`** (primary for speakers).

```python
# Quick dev-set score check:
from evaluation.common import score, split_ids, write_jsonl
from pipeline.inference import predict_sequence

ids = split_ids("dev")
rows = {sid: predict_sequence(sid) for sid in ids}
write_jsonl(rows, "outputs/predictions_dev.jsonl")
result = score.evaluate("outputs/predictions_dev.jsonl", "dataset/development/labels.jsonl")
print(result)  # → {text_order_score, balanced_joint_f1, ...}
```

Or use `generate_submission.py --set dev --out outputs/predictions_dev.jsonl` which does the same via pipeline_service.
