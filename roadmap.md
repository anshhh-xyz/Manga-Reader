# Manga Task — Development Roadmap & Progress Tracker
 
A system that reads **three consecutive English manga pages**, extracts the story
text in reading order, and labels who spoke each line, built for BYTE Applied ML,
Task 2 (The Manga Task, Advanced). This file tracks the build **phase by phase**.
 
_Last updated: 2026-10-02 (revised after reading `score.py` v2.1)_
 
**Legend:** `[x]` done · `[ ]` to do · ⚠️ known issue · ❓ needs confirmation
 
Dataset: [BYTE ML Task Data](https://drive.google.com/drive/folders/1UDZ6es3VdDyxd3mEsBQBrrYa7fD2lWYp)
 
---
 
## Overview
 
| Phase | Goal | Status |
|---|---|---|
| 0. Foundation | Repo, dataset, loader, grouped split, results log, data notes | Not started |
| 1. Evaluation + first submission | Score anything; floors; plain OCR + cleanup; first valid test JSONL | Not started |
| 2. Baseline comparison | Other OCRs, zero-shot VLM; pick OCR and decide on a VLM track | Not started |
| 3. Supervision | Align OCR boxes to label lines (scorer-based) and compute the alignment ceiling | Not started |
| 4. Filter + reading order | Keep story text only, in the right order (first learned part) | Not started |
| 5. Speakers + consistency | Speaker per line as clustering, `NARRATION`, same character = same label across 3 pages | Not started |
| 6. Adaptation | Fine-tune/adapt the component with the largest remaining gap | Not started |
| 7. Ablations + final run | Per-component contribution, reproducible test inference | Not started |
| 8. Packaging | Code, logs, weights/links, predictions, self-written README | Not started |
 
**➡️ Current position: start of Phase 0.**
 
Rule of the roadmap: do NOT build every module at once. Finish a phase, pass its
exam, log the result in `results.md`, commit, then move on. **Always keep a valid
test submission from Phase 1 onward.**
 
---
 
## What the scorer rewards (read from `score.py`)
 
These facts drive several decisions below. Re-check them if `score.py` changes.
 
- **Line boundaries don't matter.** A page's lines are joined into one string for the text score and speakers are assigned per token. Merging or splitting lines is free unless two speakers end up in one line.
- **Order matters for both metrics.** Token alignment is monotonic per page; swapped blocks only match once.
- **One speaker-name mapping is fitted across all 3 pages**, so cross-page consistency is scored directly.
- **Tokens under 3 characters need an exact match**; longer ones need ≥0.6 similarity. A misread standalone "I" is a guaranteed miss.
- **Empty pages are asymmetric:** gold empty + predict nothing = page excluded; predict anything = page scores 0 and counts.
- **Every gold speaker has equal weight** in `balanced_joint_f1`; every predicted label that doesn't map to a real gold speaker adds a penalty (including a `NARRATION` label when none exists).
- `UNKNOWN` gold speakers are masked for speaker scoring but their text still counts.
- **`balanced_joint_f1` is `None` (left out of the macro) for sequences with no known gold speakers** (e.g. all pages empty, or only `UNKNOWN`). A sequence missing from the predictions file scores 0 on text.
- Case and curly quotes/dashes are normalised by the scorer; don't spend effort on them.
- All-empty predictions score ≈0 by construction (a format check only).
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
- Do not create a new speaker label for an uncertain balloon; extra labels are penalised.
- The final README is written by me, in my own words.
---
 
## Target project layout
 
```
manga-story-understanding/
├── dataset/                 development/, test/, sequences.json, sample_submission.jsonl, score.py
├── configs/
│   └── config.py            seed, paths, model names, reading direction, thresholds
├── data.py                  load a sequence + labels
├── make_split.py            grouped split / 5-fold CV by sequence
├── alignment/
│   ├── align.py             box-to-label matching (scorer-based)
│   └── ceiling.py           alignment-ceiling predictions export
├── detection/
│   ├── text_detector.py
│   └── panel_detector.py
├── ocr/
│   ├── baseline.py
│   ├── inference.py
│   ├── postprocess.py       I/l fixes, contractions, hyphen joins
│   └── train_lora.py
├── filtering/
│   └── train_filter.py      dialogue / narration / non-story classifier
├── ordering/
│   └── reading_order.py
├── speakers/
│   ├── speaker_assigner.py
│   ├── embeddings.py
│   └── character_memory.py
├── pipeline/
│   └── inference.py         images in, JSONL out
├── evaluation/
│   ├── run_eval.py          score + save full report + append to results.md
│   ├── validate_format.py   wraps score.load_jsonl + sample ID check
│   └── triage.py            per-sequence error triage, token precision/recall
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
- Alignment and format validation reuse `score.py` functions rather than re-implementing them.
---
 
## Phase 0 — Foundation
 
**Goal:** be able to load any sequence with its labels, with a leak-free split.
 
- [ ] Repo, `.gitignore` (`.venv/`, `dataset/`, `outputs/`, weights), `git init`
- [ ] `requirements.txt` (start with pillow, numpy, pandas, opencv-python, rapidfuzz optional)
- [ ] Download dataset into `dataset/`; confirm layout; run `python dataset/score.py --help`
- [ ] `configs/config.py`: seed, paths, reading direction (to be measured), thresholds
- [ ] `data.py`: load `sequences.json`, return 3 image paths + labels; flag `UNKNOWN` speakers
- [ ] `make_split.py`: grouped by sequence, **5-fold grouped CV** (small dev set, noisy macro score); group by manga series where identifiable (check filenames/folders first; else cluster first-page CLIP embeddings and split by cluster); save `dataset/splits.json`
- [ ] Read ~10 sequences next to their labels; write `notes_data.md` covering:
  - [ ] fraction of empty gold pages
  - [ ] how often `UNKNOWN` and `NARRATION` appear
  - [ ] speakers per sequence (typical, max)
  - [ ] how line-wrap hyphenation appears in gold text
  - [ ] how SFX, narration and punctuation-only balloons are treated
  - [ ] apparent reading direction
- [ ] Create `results.md`
**Phase 0 exam:** any sequence loads with images and labels; `splits.json` saved; no sequence is split across train/val.
 
---
 
## Phase 1 — Evaluation + first submission
 
**Goal:** score any predictions file with one command, and have a valid (crude) test JSONL.
 
### 1.1 Evaluation tools
- [ ] `evaluation/run_eval.py`: run `score.py`, save the **full report JSON** per experiment, append a row to `results.md`
- [ ] `evaluation/validate_format.py`: call `score.load_jsonl` (don't re-implement rules), then check sequence IDs against `sample_submission.jsonl`
- [ ] `evaluation/triage.py`: per-sequence token precision/recall (matched weight ÷ predicted tokens / gold tokens) and a triage label: predicted ≫ gold → filter; low coverage → OCR or order; high text score but low `balanced_joint_f1` → speakers
- [ ] Floor 1: all-empty predictions (≈0 by construction; format check only)
- [ ] Floor 2: ground-truth text with every speaker = `char1` (the meaningful floor)
### 1.2 Plain OCR baseline
- [ ] `ocr/baseline.py`: PaddleOCR on each page, text + boxes (its built-in detector is the default text detector; look at the boxes on a few dev pages by eye)
- [ ] `ocr/postprocess.py`: fix standalone I/l/`|`, rejoin split contractions (`don ' t`), remove line-wrap hyphens, collapse spacing noise; log the score gain separately
- [ ] `pipeline/inference.py` skeleton: images → JSONL, single speaker, naive top-to-bottom order; drop/clean lines that would fail `load_jsonl` (empty after normalisation, control characters)
- [ ] Score on dev; produce test JSONL; validate its format
- [ ] Log failure modes: SFX captured, signs captured, missed text, merged/split lines
**Phase 1 exam:** valid test JSONL exists; baseline row in `results.md`. **Milestone M1.**
 
---
 
## Phase 2 — Baseline comparison
 
**Goal:** know which OCR to build on and whether an end-to-end VLM track is realistic. Independent of Phases 3 to 5, so it can run in parallel or be deferred.
 
- [ ] One or two more OCR options (e.g. Florence-2, a small Qwen-VL) behind the same interface
- [ ] Zero-shot VLM: 3 pages in, JSON out in the submission format, with the exclusion rules in the prompt; strict JSON parse with fallback to empty pages; run locally, 4-bit quantised if needed (e.g. Qwen2.5-VL 3B/7B). Baseline only.
- [ ] Log all scores; choose OCR
- [ ] Write a yes/no on the VLM track. **Decision rule:** "no" if it does not fit in memory with 3 images, or its zero-shot score is far below the OCR pipeline. Record the VRAM figure.
**Phase 2 exam:** chosen OCR and VLM-track decision written in `results.md`.
 
---
 
## Phase 3 — Supervision from the labels
 
**Goal:** the labels have no boxes, so build box-level training data using the scorer's own matching.
 
### Method (`alignment/align.py`)
`score.py` is importable; use `tokens` and `align_tokens` so "matched" here means "matched" in the score. Per page (labels are already split by page):
 
- [ ] **Candidates:** for every OCR box, score against every gold line *and* every adjacent pair of lines (for boxes straddling two balloons):
  - `w` = sum of weights from `align_tokens(line_tokens, box_tokens)`
  - `precision = w / len(box_tokens)`; `coverage = w / len(line_tokens)`
- [ ] **Assign** each box to its best candidate if precision ≥ 0.6. Tiers: **high** ≥ 0.85, **medium** 0.6–0.85, **none** → non-story negative
- [ ] **Resolve ambiguity** (repeated lines like "Huh?"/"No!", one- or two-token boxes): one-to-one assignment (Hungarian) on precision, tie-broken by consistency with label order and position; if still ambiguous, flag and exclude from training
- [ ] **Order index:** gold token index of the first token a box matched (from the alignment pairs); boxes within a line sort by it. Gives token-level order
- [ ] **Punctuation-only balloons:** accept only if the string is unique on the page; otherwise flag ambiguous and count separately
- [ ] **Speaker:** copied from the matched line; `UNKNOWN` lines stay story text but are excluded from speaker training
- [ ] Save per box: page, coordinates, OCR text, matched line or none, tier, ambiguous flag, 3-class tag (dialogue / narration / non-story), speaker, order index
- [ ] Log unmatched gold lines (OCR misses) and unmatched boxes (non-story examples)
- [ ] Implementation: wrap each OCR box as `{"speaker": "box", "text": ...}` so `score.tokens` accepts it; `align_tokens` returns `(gold_token, box_token, weight)` triples
- [ ] Expect short boxes (tokens under 3 characters need an exact match) to land among the unmatched; count how many
- [ ] Optional fallback for leftover boxes/lines on the same page: pair by position and order into a separate low-confidence tier; never use it for filter training
### Reliability
- [ ] Draw ~20 samples (box + matched label on the image); report match **precision**, not only the match rate
- [ ] **Alignment ceiling** (`alignment/ceiling.py`): predictions from matched boxes only, OCR text, ordered by gold token index, with gold speakers; score with `score.py`. This is the best this OCR can do with perfect filter, order and speakers
### Uses of the tiers
- Filter training: **high** only
- OCR LoRA (Phase 6): high + **medium**, with the **label text** as the target (training only on high would be circular)
**Phase 3 exam:** aligned dataset saved; alignment ceiling logged in `results.md`; match precision documented. ⚠️ If the ceiling or match precision is poor, fix OCR or matching before moving on.
 
---
 
## Phase 4 — Filter + reading order
 
**Goal:** keep only story text, in the right order. First learned component.
 
### 4.1 Story / non-story filter
- [ ] `filtering/train_filter.py`, 3 classes (dialogue / narration / non-story; narration labels come from aligned `speaker == NARRATION`)
- [ ] Features: normalised position, size, aspect ratio, margin distance; text height relative to page median; OCR confidence, text length, all-caps ratio, has-letters flag; **inside-a-white-balloon** (threshold the page, connected components, is the box inside a large mostly-white component)
- [ ] Model 1: gradient boosting on features. Model 2: add CLIP/SigLIP crop embedding; compare as an ablation
- [ ] Grouped CV by sequence, class weights; tune the decision threshold on the **pipeline score**, not accuracy
- [ ] **Page-level gate:** tune an "output nothing for this page" threshold on `text_order_score` (empty-page asymmetry); keep the filter conservative
- [ ] Plug into the pipeline and rescore; compare vs unfiltered OCR
### 4.2 Reading order
- [ ] `detection/panel_detector.py`: recursive XY-cut (horizontal gutters, then vertical, recurse); fallback OpenCV contours; last resort whole page = one panel
- [ ] **Measure reading direction** on aligned data (RTL vs LTR panel order vs true order indices); set in config
- [ ] `ordering/reading_order.py`: panels in direction, then balloons inside each panel top-to-bottom with direction tie-break
- [ ] Learned alternative / fallback: pairwise "does A come before B?" classifier (position differences, same-panel flag, sizes); sort by its predictions
- [ ] Measure pairwise order accuracy / Kendall's tau on matched boxes, by layout type; rescore
**Phase 4 exam:** token precision improves without recall collapsing; ordering accuracy reported; `text_order_score` up. **Milestones M2 and M3.**
 
---
 
## Phase 5 — Speakers + consistency
 
**Goal:** a speaker per line, consistent across all three pages. Treat it as **clustering, not classification** (names are arbitrary per sequence). Intrinsic metric: pairwise same-speaker accuracy or ARI over line pairs; final metric: `balanced_joint_f1`.
 
- [ ] 5a, within a page (`speakers/speaker_assigner.py`): nearest detected character (Magi or a person/face detector); same-panel alternation heuristic; optional learned pairwise "same speaker?" classifier (same panel, distance, alternation, nearest-character match, text cues)
- [ ] 5b, across pages (`embeddings.py`, `character_memory.py`): crop the associated character per line, embed with SigLIP/CLIP, agglomerative clustering with a distance threshold (speaker count unknown); constraints: different characters in one panel cannot merge, lines from one balloon must merge
- [ ] Test Magi's character re-identification against generic CLIP (CLIP is weak on black-and-white manga identity)
- [ ] Tune the clustering threshold directly on `balanced_joint_f1`; do not emit extra labels for uncertain balloons
- [ ] `NARRATION` for narrator-style caption text only (and only when present); thoughts go to the thinking character
- [ ] Oracle analysis: GT text + predicted speakers; GT speakers + predicted text (write each as JSONL and score)
- [ ] Rescore in the pipeline
**Phase 5 exam:** `balanced_joint_f1` beats the all-`char1` floor. **Milestone M4.**
 
---
 
## Phase 6 — Adaptation
 
**Goal:** a properly adapted model, aimed at the biggest remaining error source.
 
- [ ] Pick target by the **largest gap** in the oracle runs and triage: OCR LoRA on aligned crops with label-text targets (`ocr/train_lora.py`), speaker/character model (embedding head), or end-to-end QLoRA if Phase 2 said it is realistic
- [ ] Small trial run first: 10–50 examples; loss falls; can overfit a tiny set; then scale up
- [ ] Train on train folds, validate on val folds; save adapters and logs
- [ ] Base vs adapted comparison on the same split, same pipeline, per-sequence paired differences
**Phase 6 exam:** comparison logged in `results.md`. **Milestone M5.** If there is no usable GPU, adapt something CPU-friendly (filter, pairwise ordering, embedding head) and say so honestly.
 
---
 
## Phase 7 — Ablations + final run
 
**Goal:** know what each component contributes, then produce the test predictions reproducibly.
 
- [ ] Remove one component at a time (postprocess, filter, ordering, speaker model, adaptation), same seed, one change per run; log score change with paired per-sequence differences
- [ ] Error analysis on ~20 validation failures, using `triage.py`: categorise (missed text, extra SFX, wrong order, wrong speaker, split/merged lines), count, and fix only the top one or two
- [ ] Freeze configs and model versions
- [ ] One command generates test JSONL; no manual edits; no tuning on test
- [ ] `validate_format.py` passes; re-run from a clean environment
- [ ] Upload adapters/weights; test the download links
**Phase 7 exam:** `predictions.jsonl` reproduces from the saved code alone.
 
---
 
## Phase 8 — Packaging
 
- [ ] Gather code, logs, `results.md`, predictions, weights or links
- [ ] Write the README myself: approach, experiments and comparisons, what worked or failed, what I would try next, own contribution vs external work (including use of `score.py` as a library), with citations
- [ ] Declare any LLM help used
- [ ] Read it once as a stranger would; be able to explain every choice in it
---
 
## Experiment log (mirrors `results.md`)
 
| # | Experiment | text_order_score | balanced_joint_f1 | Notes |
|---|---|---|---|---|
| 0 | All empty | | | ≈0 by construction; format check |
| 1 | GT text, all `char1` | | | meaningful floor |
| 2 | Full-page OCR | | | |
| 2b | + OCR postprocess | | | I/l, contractions, hyphens |
| 3 | Alternative OCRs / zero-shot VLM | | | |
| 3b | Alignment ceiling | | | matched boxes + gold order + gold speakers |
| 4 | + story filter (+ page gate) | | | |
| 5 | + reading order | | | |
| 6 | + speakers | | | |
| 7 | + character consistency | | | |
| 7b | Oracles: GT text + pred speakers; GT speakers + pred text | | | locates error source |
| 8 | Adapted model | | | |
 
---
 
## Notes & decisions log
 
- Pages are labelled in the order listed in `sequences.json`; character names may restart each sequence and need not match the dev annotations.
- `manga-ocr` is trained on Japanese text, so it is not suitable for these English pages.
- No box labels exist, so a YOLO detector cannot be fine-tuned directly; use a pretrained detector and derive box labels by alignment. Alignment is text matching, not an image problem, so neither YOLO nor OpenCV does it. OpenCV is for panels only.
- Text detector order of preference: PaddleOCR's built-in detector, then comic-text-detector, then Magi (optional). Verify availability and licence first.
- OCR training pairs come from the label text, not the OCR text, but only for boxes OCR already reads well enough to match. This biases the set toward easy crops; say so in the README.
- Alignment uses the scorer's own tokenisation and matching, not a separate fuzzy matcher.
- `speaker_accuracy_on_matched` can look high when little text is recovered. Judge by `text_order_score` and `balanced_joint_f1` together.
- Correctly empty pages add nothing, but invented text on empty pages costs a full page score, so the filter and page-level gate matter.
- The dev set is small; the macro is noisy. Use grouped CV and paired per-sequence differences.
- Cite any external model, code or data (including Magi, PaddleOCR, Florence-2, Qwen, CLIP/SigLIP if used) and check licences.
---
 
## Immediate next actions
 
1. Download the dataset; set up the repo and environment.
2. Write `data.py` and `make_split.py`; save `splits.json`.
3. Read ~10 sequences; write `notes_data.md` (including the empty-page, `UNKNOWN`, `NARRATION` and hyphenation checks).
4. Write `validate_format.py` and `run_eval.py`; run the two floors.
5. Then start Phase 1.