# Manga Task — Development Roadmap & Progress Tracker

A system that reads **three consecutive English manga pages**, extracts the story
text in reading order, and labels who spoke each line, built for BYTE Applied ML,
Task 2 (The Manga Task, Advanced). This file tracks the build **phase by phase**.

_Last updated: 2026-10-02_

**Legend:** `[x]` done · `[ ]` to do · ⚠️ known issue · ❓ needs confirmation

Dataset: [BYTE ML Task Data](https://drive.google.com/drive/folders/1UDZ6es3VdDyxd3mEsBQBrrYa7fD2lWYp)

---

## Overview

| Phase | Goal | Status |
|---|---|---|
| 0. Foundation | Repo, dataset, loader, grouped split, results log | Not started |
| 1. Evaluation + first submission | Score anything; floor baselines; plain OCR; first valid test JSONL | Not started |
| 2. Baseline comparison | Other OCRs, zero-shot VLM; pick OCR and decide on a VLM track | Not started |
| 3. Supervision | Align OCR boxes to label lines to create box-level training data | Not started |
| 4. Filter + reading order | Keep story text only, in the right order (first learned part) | Not started |
| 5. Speakers + consistency | Speaker per line, `NARRATION`, same character = same label across 3 pages | Not started |
| 6. Adaptation | Fine-tune/adapt the component causing the most remaining error | Not started |
| 7. Ablations + final run | Per-component contribution, reproducible test inference | Not started |
| 8. Packaging | Code, logs, weights/links, predictions, self-written README | Not started |

**➡️ Current position: start of Phase 0.**

Rule of the roadmap: do NOT build every module at once. Finish a phase, pass its
exam, log the result in `results.md`, commit, then move on. **Always keep a valid
test submission from Phase 1 onward.**

---

## What the task requires (and where it is covered)

| Requirement from the task doc | Covered in |
|---|---|
| Read 3 pages, output story text in reading order with a speaker per line | Phases 1, 4, 5 |
| Same character keeps one label across all three pages; `NARRATION` only for narrator text | Phase 5 |
| Include dialogue, thoughts, narration, spoken screams/grunts, punctuation-only balloons | Phases 3, 4 |
| Exclude SFX, tiny reactions, titles/logos/credits/page numbers/ads/watermarks, intro labels, scanlator notes, signs/clothes/objects, text inside letters/screens/documents | Phases 3, 4 |
| Open-weight models and open-source tools only; no hosted AI inference APIs (training targets or test predictions) | All phases (see rules below) |
| Learn/adapt at least one part of the system with data; unmodified model = baseline only | Phases 4 and 6 |
| Keep all three pages of a sequence together when splitting | Phase 0 |
| Test output generated automatically from images, no manual edits, no use of answers for test images | Phase 7 |
| Cite external data/code and state own contribution; declare LLM help | Phase 8 |
| Submit: training + experiment code and logs, inference code, test JSONL, weights/adapters or links, self-written README | Phase 8 |
| Judged on transcription, reading order, speaker consistency, experimentation, critical thinking, reproducibility, understanding | Whole project (`results.md`, ablations) |

### Hard rules for this project
- No hosted AI inference APIs anywhere, including for pseudo-labels or "quick checks" (this includes Groq, OpenAI, Gemini and similar). Local open-weight models only.
- Never use test images' answers, and never tune on the test set.
- Never hand-edit test predictions.
- Pages are predicted in the order given in `sequences.json`; empty page = `[]`.
- The final README is written by me, in my own words.

---

## Target project layout

```
manga-story-understanding/
├── dataset/                 development/, test/, sequences.json, sample_submission.jsonl, score.py
├── configs/
│   └── config.py            seed, paths, model names
├── data.py                  load a sequence + labels
├── make_split.py            grouped split by sequence
├── alignment/
│   └── align.py             box-to-label matching
├── detection/
│   ├── text_detector.py
│   └── panel_detector.py
├── ocr/
│   ├── baseline.py
│   ├── inference.py
│   └── train_lora.py
├── filtering/
│   └── train_filter.py      story / non-story classifier
├── ordering/
│   └── reading_order.py
├── speakers/
│   ├── speaker_assigner.py
│   ├── embeddings.py
│   └── character_memory.py
├── pipeline/
│   └── inference.py         images in, JSONL out
├── evaluation/
│   ├── run_eval.py          score + append to results.md
│   └── validate_format.py   check JSONL against sample_submission
├── training/
├── outputs/
├── results.md               experiment table
├── notes_data.md            what I learned from reading the data
├── requirements.txt
└── README.md                final, self-written
```

Design rules (from `ARCHITECTURE.md`):
- Each stage is a separate module with a clear input and output, so stages can be swapped and ablated.
- `pipeline/inference.py` is the only place stages are chained.
- Every experiment is run through `evaluation/run_eval.py` and logged.

---

## Phase 0 — Foundation

**Goal:** be able to load any sequence with its labels, with a leak-free split.

- [ ] Repo, `.gitignore` (`.venv/`, `dataset/`, `outputs/`, weights), `git init`
- [ ] `requirements.txt` (start with pillow, numpy, pandas, opencv-python)
- [ ] Download dataset into `dataset/`; confirm layout; run `python dataset/score.py --help`
- [ ] `configs/config.py`: seed and paths
- [ ] `data.py`: load `sequences.json`, return 3 image paths + labels
- [ ] `make_split.py`: grouped by sequence (and by manga series where identifiable); save `dataset/splits.json`
- [ ] Read ~10 sequences next to their labels; write `notes_data.md`
- [ ] Create `results.md`

**Phase 0 exam:** any sequence loads with images and labels; `splits.json` saved; no sequence is split across train/val.

---

## Phase 1 — Evaluation + first submission

**Goal:** score any predictions file with one command, and have a valid (crude) test JSONL.

### 1.1 Evaluation tools
- [ ] `evaluation/run_eval.py`: run `score.py`, append a row to `results.md`
- [ ] `evaluation/validate_format.py`: IDs, 3 page lists, `speaker`/`text` keys, matches `sample_submission.jsonl`
- [ ] Floor 1: all-empty predictions
- [ ] Floor 2: ground-truth text with every speaker = `char1`

### 1.2 Plain OCR baseline
- [ ] `ocr/baseline.py`: PaddleOCR on each page, text + boxes
- [ ] `pipeline/inference.py` skeleton: images → JSONL, single speaker, naive top-to-bottom order
- [ ] Score on dev; produce test JSONL; validate its format
- [ ] Log failure modes: SFX captured, signs captured, missed text, merged/split lines

**Phase 1 exam:** valid test JSONL exists; baseline row in `results.md`. **Milestone M1.**

---

## Phase 2 — Baseline comparison

**Goal:** know which OCR to build on and whether an end-to-end VLM track is realistic. Independent of Phases 3 to 5, so it can run in parallel or be deferred.

- [ ] One or two more OCR options (e.g. Florence-2, a small Qwen-VL) behind the same interface
- [ ] Zero-shot VLM: 3 pages in, JSON out (baseline only)
- [ ] Log all scores; choose OCR; write a yes/no on the VLM track given available GPU

**Phase 2 exam:** chosen OCR and VLM-track decision written in `results.md`.

---

## Phase 3 — Supervision from the labels

**Goal:** the labels have no boxes, so build box-level training data.

- [ ] `alignment/align.py`: fuzzy-match OCR box text to label lines per dev sequence
- [ ] Save per box: page, coordinates, OCR text, matched label line or none, story/non-story tag, speaker, order index
- [ ] Hand-check ~20 samples; record the match rate
- [ ] Log unmatched label lines (OCR misses) and unmatched boxes (non-story examples)

**Phase 3 exam:** aligned dataset saved, reliability documented. ⚠️ If the match rate is poor (say under ~60%), fix OCR or matching before moving on.

---

## Phase 4 — Filter + reading order

**Goal:** keep only story text, in the right order. First learned component.

### 4.1 Story / non-story filter
- [ ] `filtering/train_filter.py`: classifier on box crops + features (position, size, shape, OCR text)
- [ ] Train on train split, evaluate on val split; compare vs unfiltered OCR
- [ ] Plug into the pipeline and rescore

### 4.2 Reading order
- [ ] `detection/panel_detector.py`
- [ ] `ordering/reading_order.py`: panels right-to-left/top-to-bottom, then order inside panels
- [ ] Measure against aligned order indices, by layout type
- [ ] Fallback for pages where the heuristic clearly fails; rescore

**Phase 4 exam:** precision improves without recall collapsing; ordering accuracy reported; `text_order_score` up. **Milestones M2 and M3.**

---

## Phase 5 — Speakers + consistency

**Goal:** a speaker per line, consistent across all three pages.

- [ ] 5a: within-page balloon-to-character association (`speakers/speaker_assigner.py`); Magi is a candidate if it transfers to English
- [ ] 5b: character crops → CLIP/SigLIP embeddings → cluster across 3 pages (`embeddings.py`, `character_memory.py`)
- [ ] `NARRATION` for narrator-style caption text only; thoughts go to the thinking character
- [ ] Oracle analysis: GT text + predicted speakers; GT speakers + predicted text
- [ ] Rescore in the pipeline

**Phase 5 exam:** `balanced_joint_f1` beats the all-`char1` floor. **Milestone M4.**

---

## Phase 6 — Adaptation

**Goal:** a properly adapted model, aimed at the biggest remaining error source.

- [ ] Pick target from error analysis: OCR LoRA on aligned crops (`ocr/train_lora.py`), speaker/character model, or end-to-end QLoRA if Phase 2 said it is realistic
- [ ] Small trial run first (few steps, small subset)
- [ ] Train on train split, validate on val split; save adapters and logs
- [ ] Base vs adapted comparison on the same split

**Phase 6 exam:** comparison logged in `results.md`. **Milestone M5.** If there is no usable GPU, adapt something lighter (filter, embedding head) and say so honestly.

---

## Phase 7 — Ablations + final run

**Goal:** know what each component contributes, then produce the test predictions reproducibly.

- [ ] Remove one component at a time (filter, ordering, speaker model, adaptation); log score change
- [ ] Error analysis on ~20 validation failures; fix only cheap, high-impact issues
- [ ] Freeze configs and model versions
- [ ] One command generates test JSONL; no manual edits; no tuning on test
- [ ] `validate_format.py` passes; re-run from a clean environment
- [ ] Upload adapters/weights; test the download links

**Phase 7 exam:** `predictions.jsonl` reproduces from the saved code alone.

---

## Phase 8 — Packaging

- [ ] Gather code, logs, `results.md`, predictions, weights or links
- [ ] Write the README myself: approach, experiments and comparisons, what worked or failed, what I would try next, own contribution vs external work, with citations
- [ ] Declare any LLM help used
- [ ] Read it once as a stranger would; be able to explain every choice in it

---

## Experiment log (mirrors `results.md`)

| # | Experiment | text_order_score | balanced_joint_f1 | Notes |
|---|---|---|---|---|
| 0 | All empty | | | |
| 1 | GT text, all `char1` | | | |
| 2 | Full-page OCR | | | |
| 3 | Alternative OCRs / zero-shot VLM | | | |
| 4 | + story filter | | | |
| 5 | + reading order | | | |
| 6 | + speakers | | | |
| 7 | + character consistency | | | |
| 8 | Adapted model | | | |

---

## Notes & decisions log

- Pages are labelled in the order listed in `sequences.json`; character names may restart each sequence and need not match the dev annotations.
- `manga-ocr` is trained on Japanese text, so it is not suitable for these English pages.
- No box labels exist, so a YOLO detector cannot be fine-tuned directly; use a pretrained detector and derive box labels by alignment.
- `speaker_accuracy_on_matched` can look high when little text is recovered. Judge by `text_order_score` and `balanced_joint_f1` together.
- Correctly empty pages add nothing, but invented text on empty pages costs points, so the filter matters.
- Cite any external model, code or data (including Magi, PaddleOCR, Florence-2, Qwen, CLIP/SigLIP if used) and check licences.

---

## Immediate next actions

1. Download the dataset; set up the repo and environment.
2. Write `data.py` and `make_split.py`; save `splits.json`.
3. Read ~10 sequences; write `notes_data.md`.
4. Then start Phase 1.
