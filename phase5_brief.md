# Phase 5 — Speaker Diarization: Full Brief

## What Phase 5 Was Supposed to Do

Assign a speaker label (`char1`, `char2`, `NARRATION`, etc.) to every text box so that:
- The **same character keeps the same label** across all 3 pages of a sequence.
- `NARRATION` is used only for caption/narrator text.
- No extra labels are emitted for uncertain balloons (extra labels are penalised by the scorer).

The metric is `balanced_joint_f1` — it weights every gold speaker equally, so missing a minor character hurts just as much as getting the main one wrong. The floor (all lines → `char1`) was **0.2485**.

---

## Attempt 1 — CLIP Visual Embeddings (Initial Approach)

### What We Did
- Cropped the region around each speech balloon from the manga page.
- Embedded each crop using **CLIP** (a vision-language model).
- Used the cosine similarity between embeddings as a "same speaker?" feature.
- Fed this into a pairwise XGBoost classifier that asked: *"Do text box A and text box B belong to the same speaker?"*
- Then used agglomerative clustering on the resulting probability matrix to group boxes.

### Why It Failed
After running diagnostics (`speakers/diagnose.py`), the results were poor:

```
Mean ARI: 0.13  (barely above chance)
Predicted 2.09 speakers/seq vs 3.03 gold
```

Feature importance breakdown:
| Feature | Importance | Problem |
|---|---|---|
| `page_gap` | 0.213 | Model just says "same page = same speaker" — a weak prior |
| `cos_sim` (CLIP) | 0.031 | **Near useless** — B&W manga characters don't have visual identity in crops |
| `same_panel` | 0.000 | **Dead feature** — panel IDs never reached the speaker pipeline |
| `panel_gap` | 0.000 | **Dead feature** — same reason |

**Root cause:** Panel IDs were never populated in the aligned data that the speaker model received. Without panel information, there was no turn-taking signal, and CLIP embeddings on black-and-white manga crops carry no reliable character identity.

**CLIP is fundamentally weak on manga because:**
- Manga characters are black-and-white line drawings.
- Character identity is conveyed by hair shape, clothing patterns, and position — not colour/texture.
- Cropping around a balloon doesn't reliably capture the speaker's face.

---

## Attempt 2 — Fix Panel Detection + Turn-Taking Features

### What We Changed
Instead of relying on visual embeddings, we rewired the feature set to exploit **dialogue structure**:

1. **Fixed panel detection** → `speakers/panels.py` added `fill_panels()` to assign panel IDs to each box using the detected panel map (`outputs/panels_dev.json`).
2. **Added `gap_parity` feature**: In back-and-forth dialogue (`A, B, A, B`), the rank gap between two boxes of the same speaker is always **even**. XGBoost can learn this pattern explicitly.
3. **Added `pp_gap` (panel-position gap)**: Position of each balloon *within* its own panel. Dialogue alternates: box at position 0 and box at position 2 within the same panel are likely the same speaker.
4. **Made CLIP embedding optional** (emb=None): So the model could run at inference without a CLIP encoder.

```python
FEATURES = ["same_page", "page_gap", "rank_gap", "gap_parity", "same_panel", 
            "panel_gap", "pp_gap", "dx", "dy", "dist", "w_i", "h_i", ...]
```

### Result
Massive improvement on panel features, but still limited by the ceiling of what turn-taking alone can do.

---

## Attempt 3 — MAGI v2 (Manga VLM Zero-Shot)

### What MAGI Is
MAGI v2 is a manga-specific vision-language model from Ragav et al. It is designed for character detection, face clustering, and speaker assignment in manga panels. It processes manga panels directly and tries to identify *who* is speaking.

### Why We Tried It
Since CLIP visual features failed, we thought a manga-specific model that understands manga art style might do better at character re-identification.

### Environment Problem
MAGI v2 was built for an older version of HuggingFace Transformers. We had **Transformers 5.x** installed, which broke two things:

1. **`all_tied_weights_keys`** — Transformers 5 requires custom `PreTrainedModel` subclasses to declare which weights are tied. MAGI's `MagiModel` didn't have this.
2. **`AutoBackbone` configuration validation** — Transformers 5 hardened how backbone configs are validated; MAGI's custom `ConvNextBackbone` config injection broke.

**Fixes applied** (`scratch/patch_magi.py`):
```python
# Patch 1: add all_tied_weights_keys
MagiModel.all_tied_weights_keys = []

# Patch 2: inject backbone config before loading
config.backbone_config = ConvNextConfig(...)
```

We got MAGI running on CUDA after patching.

### Grid Search
We evaluated **24 configurations** of MAGI (combining: `context_type` x `neighbour_k` x `narration_mode`):

| Config | balanced_joint_f1 |
|---|---|
| `magi_prev_k4_global_n0` | **0.3312** |
| `magi_common_k4_global_n0` | 0.3289 |
| Floor (all char1) | 0.2485 |

**Best MAGI result: 0.3312** — better than the floor but weaker than our structural XGBoost approach.

---

## Attempt 4 — XGBoost Grid Search with Tuned Clustering (Final)

### What We Did
We ran a full grid search over clustering hyperparameters on top of the pairwise XGBoost model:
- `threshold` (t): 0.4, 0.5, 0.55, 0.6 (agglomerative distance threshold)
- `merge_singletons` (m): 0 or 1
- `max_clusters` (k): 0 (unlimited) or 4

Best configuration found: **`spk2_t0.5_m1_k4`**

| Parameter | Value | Meaning |
|---|---|---|
| `threshold` | 0.5 | Distance threshold for agglomerative clustering |
| `merge_singletons` | True | Merge lone-box clusters into nearest big cluster |
| `max_clusters` | 4 | Hard cap: never predict more than 4 speakers per sequence |

### Final Scores

| Approach | balanced_joint_f1 | text_order_score |
|---|---|---|
| Floor (all char1) | 0.2485 | 0.7477 |
| MAGI v2 (best) | 0.3312 | 0.7477 |
| **XGBoost Turn-Taking (spk2_t0.5_m1_k4)** | **0.3418** | **0.7477** |
| Gold control (oracle speakers) | 0.6639 | 0.7477 |

---

## Why XGBoost Beat MAGI

| Dimension | MAGI v2 | XGBoost Turn-Taking |
|---|---|---|
| Signal | Visual character detection in manga panels | Structural dialogue alternation patterns |
| Robustness | Depends on character visibility in panels | Works even when characters are off-frame |
| Speed | Slow (GPU VLM inference per page) | Fast (CPU XGBoost) |
| Adaptation | None (zero-shot) | Trained on our aligned data (5-fold CV) |
| Key features | Character bounding boxes + re-ID | `gap_parity`, `pp_gap`, `same_panel` |

**Conclusion:** In manga, *where you speak matters more than what you look like.* Turn-taking structure (alternating positions within panels, parity of reading order index) is a stronger signal than visual character identity from a VLM.

---

## What We Moved to `speakers_extras/`

- `speakers_extras/magi/` — MAGI v2 inference + grid evaluation code
- `speakers_extras/clip_embeddings.py` — CLIP embedding code

These were experimental approaches that lost to XGBoost and are kept for reference.

---

## Final State After Phase 5

- **Model saved:** `outputs/speakers/pairwise_xgb.json`
- **Cluster config:** `outputs/speakers/cluster_cfg.json` — `{"threshold": 0.5, "merge_singletons": true, "max_clusters": 4}`
- **Pipeline integrated:** `pipeline/inference.py` calls `assign_speakers()` from `speakers/character_memory.py`
- **Speaker gap remaining:** 0.3418 vs 0.6639 gold = **gap of 0.322** — the biggest remaining bottleneck

---

## What Phase 6 Will Attack

| Gap | Current | Gold/Ceiling | Gap |
|---|---|---|---|
| **Speaker** (`balanced_joint_f1`) | 0.3418 | 0.6639 | **0.3221** biggest |
| **Text** (`text_order_score`) | 0.7477 | 0.7622 (align ceiling) | 0.0145 small |

**Phase 6 runs two parallel tracks:**
1. **Speaker model adaptation** — XGBoost hyperparameter search + added text features on the train split
2. **OCR LoRA trial** — Fine-tune a TrOCR adapter on aligned manga crops to close the small text gap
