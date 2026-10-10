# Manga Story Understanding — Architecture
 
_Revised after reading `score.py` (v2.1). Sections marked **[scorer]** are driven by how the scorer actually works._
 
## 1. Vision
 
A modular system for BYTE Applied ML Task 2 that reads **three consecutive English
manga pages** and returns the story text in reading order, with a consistent
speaker label for every line.
 
It is not "run OCR on a page". Its core loop is:
 
```text
3 manga pages (one sequence)
        ↓
Find text regions
        ↓
Read the text (OCR) + clean it (post-processing)
        ↓
Keep only story text (drop SFX, signs, credits, documents, ...)
        ↓
Put it in reading order
        ↓
Decide who speaks each line
        ↓
Keep character labels consistent across all 3 pages
        ↓
One JSON object per sequence (JSONL)
```
 
The architecture stays modular so that the OCR model, the filter, the ordering
logic and the speaker model can each be replaced, measured and ablated
independently.
 
---
 
## 2. Core Design Principles
 
### Task-first
The task doc defines success: transcription, reading order and speaker
consistency, plus experimentation, critical thinking and reproducibility. Every
component is justified by a measurable improvement in `results.md`.
 
### Open-weight and local only
Only open-weight models and open-source tools. No hosted AI inference APIs for
training targets or test predictions.
 
### Adapt something with data
At least one learned part must be trained or adapted on the development data. An
unmodified model call is a baseline, not the final answer.
 
### Supervision from the labels
The labels contain text and speaker only, with no boxes. The pipeline derives
box-level supervision by aligning OCR output to the label lines, **using the
scorer's own token matching** (section 5, `alignment/align.py`).
 
### Measure every stage
Each stage is evaluated alone (against aligned data) and in the full pipeline
(through `score.py`). Oracle experiments locate where errors come from.
 
### Reproducible
Fixed seeds, pinned versions, one command from images to test JSONL, and no
manual edits to predictions.
 
### Honest attribution
External models, code and data are cited, and the README states what was my own
contribution and what LLM help was used.
 
---
 
## 3. Repository Structure
 
```text
manga-story-understanding/
│
├── dataset/                 development/, test/, sequences.json,
│                            sample_submission.jsonl, score.py, splits.json
├── configs/
│   └── config.py
│
├── data.py
├── make_split.py
│
├── alignment/
│   ├── align.py             box-to-label matching (scorer-based)
│   └── ceiling.py           alignment-ceiling predictions export
│
├── detection/
│   ├── text_detector.py
│   └── panel_detector.py
│
├── ocr/
│   ├── baseline.py
│   ├── inference.py
│   ├── postprocess.py       rule-based cleanup (I/l, contractions, hyphens)
│   └── train_lora.py
│
├── filtering/
│   └── train_filter.py
│
├── ordering/
│   └── reading_order.py
│
├── speakers/
│   ├── speaker_assigner.py
│   ├── embeddings.py
│   └── character_memory.py
│
├── pipeline/
│   └── inference.py
│
├── evaluation/
│   ├── run_eval.py          score + save full report JSON + append to results.md
│   ├── validate_format.py   wraps score.load_jsonl + sequence-ID check
│   └── triage.py            per-sequence error triage, token precision/recall
│
├── training/
├── outputs/
├── results.md
└── README.md
```
 
---
 
## 4. Data and Sequences
 
### `data.py`
Loads one sequence: the three image paths from `sequences.json`, plus labels if the
sequence is a development one. Labels with speaker `UNKNOWN` are kept as story text
but flagged (the scorer masks them for speaker scoring).
 
### `make_split.py`
Grouped train/validation split, preferably **5-fold grouped CV** because the
development set is small and the macro score is noisy.
 
Rules:
- all three pages of a sequence always stay together
- sequences from the same manga series stay together where identifiable (check
  filenames/folders first; otherwise cluster first-page CLIP embeddings and split by cluster)
- the test sequences are never used for any decision
- the split is saved to disk and reused by every experiment
---
 
## 5. Pipeline Stages
 
### `detection/text_detector.py`
Finds text regions (boxes) on a page using a pretrained detector. No detector
fine-tuning is planned unless alignment produces enough box labels.
 
Choice, in order of preference (verify availability and licence before relying on any):
1. **PaddleOCR's built-in detector**: comes with the OCR, so it is the baseline.
2. **comic-text-detector**: built for comics/manga; try it if PaddleOCR splits or misses balloons.
3. **Magi**: optional; also gives panels and characters, but was built for Japanese manga.
Not used: **YOLO**, because training it needs box labels and the dataset has none
(alignment produces box labels *after* detection, so it cannot be the first step).
**OpenCV** is used for panels (`panel_detector.py`), not for text boxes or alignment.
 
The detector decides what alignment ever sees, so check its boxes by eye on a few
dev pages before building on it.
 
Output: list of boxes per page.
 
### `detection/panel_detector.py`
Finds panel boxes for reading order.
 
1. Recursive XY-cut: split on horizontal gutters (near-all-white rows), then
   vertical gutters inside each strip, and recurse.
2. If that fails (borderless or overlapping panels): OpenCV contours.
3. Last resort: whole page = one panel.
### `ocr/`
Reads text from regions or full pages.
 
- `baseline.py`: plain OCR, no adaptation
- `inference.py`: the chosen OCR behind a common interface
- `postprocess.py`: rule-based cleanup, **[scorer]** because tokens shorter than 3
  characters get no fuzzy credit, so a misread "I" (as `l` or `|`) is a guaranteed miss.
  Rules: fix I/l/|, rejoin split contractions (`don ' t` → `don't`), remove line-wrap
  hyphens, collapse OCR spacing noise. Case and curly quotes are already normalised
  by the scorer and need no handling.
- `train_lora.py`: LoRA/PEFT adaptation on aligned crop-to-text pairs (if chosen in Phase 6)
OCR must handle English. `manga-ocr` is trained on Japanese and is not suitable.
 
### `alignment/align.py` — the method
 
**Goal:** per page, decide which OCR boxes are story text, which label line each
belongs to, and in what order. Done only on development data.
 
**Why not generic fuzzy matching:** the scorer's `align_tokens` / `token_match`
define what counts as a match (token-level, exact below 3 characters, ≥0.6
similarity above). Aligning with the same function means "matched" in my
supervision equals "matched" in the score. `score.py` is imported as a library.
 
```text
per page:
  gold lines   -> tokens(rows)            (norm + tokenise, with speaker)
  OCR boxes    -> tokens(text)
 
  1. Candidates: for every box b, score against every gold line l AND every
     adjacent pair (l, l+1) concatenated (for boxes straddling two balloons).
        w         = sum of weights from align_tokens(line_tokens, box_tokens)
        precision = w / len(box_tokens)      # how much of the box is explained
        coverage  = w / len(line_tokens)     # how much of the line the box covers
  2. Assign each box to its best candidate if precision >= 0.6.
        tier high   : precision >= 0.85
        tier medium : 0.6 <= precision < 0.85
        else        : unmatched -> non-story negative
  3. Ambiguity: identical or near-identical lines ("Huh?", "No!") and one- or
     two-token boxes match several lines. Resolve with a one-to-one assignment
     (Hungarian) over precision, tie-broken by position consistency with the
     label order (top-to-bottom, reading direction measured in Phase 0). If still
     ambiguous, mark ambiguous and exclude from training.
  4. Order index: gold token index of the first token the box matched
     (from the alignment pairs). Boxes in the same line sort by that index. This
     gives token-level order, not just line-level.
  5. Punctuation-only balloons ("...", "?!"): punctuation tokens match exactly, so
     accept only if the string is unique on the page; otherwise ambiguous.
  6. Speaker: copied from the matched line. UNKNOWN lines stay story text but are
     excluded from speaker training.
```
 
Implementation notes (checked against `score.py`):
- `tokens(rows)` expects rows shaped `{"speaker": ..., "text": ...}`. For OCR boxes,
  wrap each as `{"speaker": "box", "text": ocr_text}`.
- `align_tokens` returns `(gold_token, box_token, weight)` triples; `w` is the sum of
  the weights, and the gold token index gives the order index.
- Cost is O(tokens_gold x tokens_box) per call, which is fine at box-by-line scale.
Known limits of this alignment:
- **Short tokens:** tokens under 3 characters must match exactly, so short boxes
  ("I", "NO") with any OCR error will not align. Expect them among the unmatched.
- **Selection bias:** only boxes OCR already reads reasonably get matched, so the
  hardest crops are underrepresented in OCR training data. Optional fallback: pair
  leftover boxes with leftover gold lines on the same page by position and order,
  flagged as a separate low-confidence tier and kept out of the filter's training set.
Per box output: page, coordinates, OCR text, matched line (or none), tier,
ambiguous flag, story / non-story, 3-class tag (dialogue / narration / non-story),
speaker, order index.
 
Also logged: unmatched gold lines (OCR misses) and unmatched boxes (non-story examples).
 
**Reliability checks**
- Draw ~20 samples (box + matched label) and report match **precision**, not just match rate.
- **Alignment ceiling** (`alignment/ceiling.py`): build predictions from matched
  boxes only, with OCR text, ordered by gold token index, with gold speakers, and
  score them with `score.py`. This is the best this OCR can do with perfect
  filtering, order and speakers, and is the Phase 3 exam.
Training use of tiers: the filter trains on **high** only. OCR LoRA also uses
**medium**, with the **label text** as the target (so the model learns from its
own near-misses; training on high only would be circular).
 
### `filtering/train_filter.py`
Classifies each box as dialogue, narration or non-story (3 classes; narration
labels come free from `speaker == NARRATION` in aligned data).
 
Targets the exclusion list in the task: SFX and captions, tiny reactions, titles,
logos, credits, page numbers, ads, watermarks, intro labels, scanlator notes,
writing on signs/clothes/objects, and text inside letters, phone screens or other
documents.
 
Features:
- normalised position, size, aspect ratio, distance from page margin
- text height relative to the page median (SFX and titles are much larger)
- OCR confidence, text length, all-caps ratio, contains-letters flag
- **inside a white balloon**: threshold the page, take connected components, check
  whether the box lies in a large, mostly white component (strong story signal)
- crop appearance (CLIP/SigLIP embedding), compared against the feature-only model
Models: gradient boosting first, then add crop embeddings and compare (an ablation).
 
Training: grouped CV by sequence, class weights, and the decision threshold tuned on
the **pipeline score**, not accuracy.
 
**[scorer]** Page-level gate: if a gold page is empty and nothing is predicted, it
is excluded from the average; if anything is predicted, it scores 0 and counts. So
the filter is conservative, and a page-level "output nothing" threshold is tuned on
`text_order_score`.
 
### `ordering/reading_order.py`
Turns story boxes into reading order. **[scorer]** The token alignment is monotonic
per page, so swapped blocks lose credit on both text and speaker metrics; order is
as important as OCR quality.
 
```text
panels in reading direction
        ↓
balloon order inside each panel (top-to-bottom, tie-break by direction)
        ↓
fallback / correction when the layout says otherwise
```
 
- **Direction is measured, not assumed:** compare RTL and LTR panel order against
  aligned order indices on the dev data; make it a config option.
- **Learned alternative:** pairwise classifier "does box A come before box B?" on
  position differences, same-panel flag and sizes; sort by its predictions. Also the
  fallback where the heuristic fails.
- Metric: pairwise order accuracy / Kendall's tau on matched boxes, by layout type.
### `speakers/`
Names are arbitrary per sequence, so speaker assignment is **clustering, not
classification**. Intrinsic metric: pairwise same-speaker accuracy (or ARI) over
line pairs; final metric: `balanced_joint_f1`.
 
- `speaker_assigner.py` (within a page): nearest detected character (Magi or a
  person/face detector), same-panel alternation heuristic, optionally a learned
  pairwise "same speaker?" classifier (same panel, distance, alternation, text cues)
- `embeddings.py`: crop embeddings (CLIP/SigLIP); generic CLIP is weak on
  black-and-white manga identity, so test Magi's re-identification too
- `character_memory.py`: per-sequence registry; agglomerative clustering with a
  distance threshold (speaker count unknown), with constraints: different characters
  in one panel cannot merge, lines from one balloon must merge
```text
Sequence
├── char1
├── char2
└── NARRATION   (narrator text only; thoughts belong to the thinking character)
```
 
**[scorer]** Rules that follow from the metric:
- one name mapping is fitted across all 3 pages, so cross-page consistency is scored directly
- every gold speaker has equal weight; a two-word character counts as much as the lead
- every predicted label that does not map to a real gold speaker adds 1 to the
  denominator, including a `NARRATION` label when the sequence has none
- so: do not create new labels for uncertain balloons; tune the clustering threshold
  on `balanced_joint_f1` directly
### `pipeline/inference.py`
The only place the stages are chained.
 
```text
images → detect → OCR → postprocess → filter → order → speakers → JSONL
```
 
It reads images only, never labels, and writes the final JSONL automatically.
It drops or cleans any line that would fail `score.load_jsonl` (empty after
normalisation, control characters).
 
---
 
## 6. Output Format
 
One JSON object per sequence, one per line.
 
```json
{"sequence_id":"seq_001","pages":[
  [{"speaker":"char1","text":"Where are you going?"},{"speaker":"char2","text":"Home."}],
  [{"speaker":"char1","text":"Wait for me!"}],
  []
]}
```
 
- the three lists are the first, second and third images of the sequence
- lines inside a list are in reading order
- empty list = no included text on that page
- `speaker` is a consistent label for that sequence, or `NARRATION`
- `text` keeps stutters, repeats and meaningful punctuation, without printed line wrapping
- keys are exactly `sequence_id` and `pages`; each line has exactly `speaker` and `text`
---
 
## 7. Adaptation Strategy
 
The chosen target depends on baseline and error analysis, not on preference. The
target is picked from the **largest gap in the oracle runs** (section 8).
 
| Option | What is adapted | Training data | Risk |
|---|---|---|---|
| A. Story filter | classifier on box features/crops | aligned boxes (high tier) | low; cheap; needs good alignment |
| B. OCR LoRA | OCR/VLM-OCR on crops | aligned crop-to-label-text pairs (high + medium) | medium; needs GPU |
| C. Speaker/character model | embedding head or association model | aligned speaker labels | medium; weak labels |
| D. End-to-end VLM QLoRA | 3 pages → JSON | 80 sequences | high; small data, heavy GPU |
 
Plan: Option A is built in Phase 4, so the adaptation requirement is met early.
Phase 6 then adapts whichever of B, C or D addresses the largest remaining error,
after a small trial run (10-50 examples; loss falls; can overfit a tiny set).
Every adapted model is compared against its base on the same validation split.
With no GPU, adapt something CPU-friendly (filter, pairwise ordering, embedding head) and say so.
 
VLM-track rule: if a zero-shot 3-page VLM does not fit in memory, or scores far
below the OCR pipeline, the answer is "no" for end-to-end training. Record the VRAM number.
 
Candidate models (verify licences and English support before relying on them):
PaddleOCR, Florence-2, Qwen2.5-VL (small sizes), CLIP/SigLIP, Magi.
 
---
 
## 8. Evaluation Architecture
 
```text
predictions.jsonl
        ↓
evaluation/validate_format.py   (score.load_jsonl + sequence IDs match sample_submission.jsonl)
        ↓
dataset/score.py                (text_order_score, balanced_joint_f1, diagnostics)
        ↓
evaluation/run_eval.py          (save full report JSON, append row to results.md)
        ↓
evaluation/triage.py            (per-sequence triage, token precision/recall)
```
 
Experiments:
- floors: all-empty (≈0 by construction, so it only checks the format); ground-truth
  text with every speaker = `char1` (the meaningful floor)
- alignment ceiling: matched OCR boxes + gold order + gold speakers
- pipeline experiments: OCR → +postprocess → +filter → +ordering → +speakers → +consistency → adapted
- oracle experiments: ground-truth text with predicted speakers; ground-truth speakers with predicted text
- ablations: remove one component at a time, same seed, one change per run
Derived measures from the report:
- token recall = `matched_token_coverage × gold_tokens / gold_tokens`; precision =
  matched weight / `predicted_tokens`
- triage per sequence: predicted tokens ≫ gold → filter; low coverage → OCR or order;
  good text score but low `balanced_joint_f1` → speakers
Rules for reading the scores:
- the two main metrics are `text_order_score` and `balanced_joint_f1`
- `speaker_accuracy_on_matched` alone can mislead when little text is recovered
- macro averages over sequences on a small set are noisy: use grouped CV and compare
  experiments by **per-sequence paired differences**
- the scorer guides decisions but is not the whole selection rubric
---
 
## 9. Compliance Map
 
| Task rule | How the architecture satisfies it |
|---|---|
| Open-weight, open-source only | all models run locally; no hosted AI APIs anywhere, including for pseudo-labels |
| Actual learning/adaptation | story filter (Phase 4) plus an adapted component (Phase 6), with base-vs-adapted comparison |
| Keep sequences together | grouped split in `make_split.py` |
| Test output automatic, no manual edits, no test answers | single `pipeline/inference.py` run, images only |
| Cite external work | citations and own-contribution statement in README; licences checked (incl. use of `score.py` as a library) |
| Required submission items | code and logs, inference code, test JSONL, weights or links, self-written README |
| Pipeline choice is free | modular stages; any stage may be replaced |
 
---
 
## 10. LLM-Readable Architecture Contract
 
Treat these as constraints whenever another LLM or coding assistant works on this project:
 
1. `pipeline/inference.py` is the only place stages are chained.
2. Every stage has a clear input and output and can be swapped independently.
3. No hosted AI inference API is ever called, for any purpose.
4. Test labels do not exist and test images are never used for tuning.
5. Test predictions are never edited by hand.
6. All three pages of a sequence always stay in the same split.
7. Character labels are per sequence; `NARRATION` is for narrator text only, and never emitted when the sequence has no narrator text.
8. Empty pages return `[]`; nothing is invented to fill them.
9. Every experiment goes through `evaluation/run_eval.py` and is logged in `results.md`.
10. Seeds, model versions and paths come from `configs/config.py`.
11. External code, models and data are listed with citations and licences.
12. The final README is written by the student, not generated.
13. Alignment uses the scorer's own tokenisation and matching (`score.tokens`, `score.align_tokens`), not a separate fuzzy matcher.
14. Format validation uses `score.load_jsonl`; do not re-implement the rules.
15. Do not create a new speaker label for an uncertain balloon; extra labels are penalised.
16. Thresholds (filter, page-level gate, clustering) are tuned on pipeline score under grouped CV, never on the test set.
---
 
## 11. Growing the System
 
Do NOT build everything at once. Start with:
 
```text
data.py → make_split.py → evaluation tools → plain OCR → valid JSONL
```
 
Then add, in order: alternative baselines, alignment (+ceiling), story filter,
reading order, speakers, character consistency, adaptation, ablations, packaging.
 
The goal is to grow the pipeline from a working baseline, keeping a valid
submission at every stage.
 d5