# Manga Story Understanding — Architecture

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
Read the text (OCR)
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
box-level supervision by aligning OCR output to the label lines.

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
│   └── align.py
│
├── detection/
│   ├── text_detector.py
│   └── panel_detector.py
│
├── ocr/
│   ├── baseline.py
│   ├── inference.py
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
│   ├── run_eval.py
│   └── validate_format.py
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
sequence is a development one.

### `make_split.py`
Grouped train/validation split (or k-fold) by sequence.

Rules:
- all three pages of a sequence always stay together
- sequences from the same manga series stay together where identifiable
- the test sequences are never used for any decision
- the split is saved to disk and reused by every experiment

---

## 5. Pipeline Stages

### `detection/text_detector.py`
Finds text regions (boxes) on a page using a pretrained detector. No detector
fine-tuning is planned unless alignment produces enough box labels.

Output: list of boxes per page.

### `detection/panel_detector.py`
Finds panel boxes for reading order. OpenCV contours first; a pretrained detector
if that fails.

### `ocr/`
Reads text from regions or full pages.

- `baseline.py`: plain OCR, no adaptation
- `inference.py`: the chosen OCR behind a common interface
- `train_lora.py`: LoRA/PEFT adaptation on aligned crop-to-text pairs (if chosen in Phase 6)

OCR must handle English. `manga-ocr` is trained on Japanese and is not suitable.

### `alignment/align.py`
Matches each OCR box to a label line by fuzzy text matching.

```text
OCR boxes + label lines
        ↓
fuzzy match
        ↓
aligned boxes: story / non-story, speaker, order index, crop-to-text pair
```

This produces the training data for the filter, the ordering evaluation, the
speaker model and OCR adaptation.

### `filtering/train_filter.py`
Classifies each box as story text or not.

Targets the exclusion list in the task: SFX and captions, tiny reactions, titles,
logos, credits, page numbers, ads, watermarks, intro labels, scanlator notes,
writing on signs/clothes/objects, and text inside letters, phone screens or other
documents.

Features: box position, size, shape, OCR text, crop appearance, balloon vs caption
style. This is also the main source of the `NARRATION` signal.

### `ordering/reading_order.py`
Turns story boxes into reading order.

```text
panels right-to-left, top-to-bottom
        ↓
order of balloons inside each panel
        ↓
fallback / correction when the layout says otherwise
```

Panel layout and balloon connections take priority over a blind right-to-left
assumption.

### `speakers/`
- `speaker_assigner.py`: links each balloon to a character (tail direction, proximity, or an existing model such as Magi if it transfers)
- `embeddings.py`: crop embeddings (CLIP/SigLIP)
- `character_memory.py`: per-sequence registry that clusters characters across the three pages and hands out labels

```text
Sequence
├── char1
├── char2
└── NARRATION   (narrator text only; thoughts belong to the thinking character)
```

Labels only need to be consistent within a sequence, not match the dev annotations.

### `pipeline/inference.py`
The only place the stages are chained.

```text
images → detect → OCR → filter → order → speakers → JSONL
```

It reads images only, never labels, and writes the final JSONL automatically.

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

---

## 7. Adaptation Strategy

The chosen target depends on baseline and error analysis, not on preference.

| Option | What is adapted | Training data | Risk |
|---|---|---|---|
| A. Story filter | classifier on box features/crops | aligned boxes | low; cheap; needs good alignment |
| B. OCR LoRA | OCR/VLM-OCR on crops | aligned crop-to-text pairs | medium; needs GPU |
| C. Speaker/character model | embedding head or association model | aligned speaker labels | medium; weak labels |
| D. End-to-end VLM QLoRA | 3 pages → JSON | 80 sequences | high; small data, heavy GPU |

Plan: Option A is built in Phase 4, so the adaptation requirement is met early.
Phase 6 then adapts whichever of B, C or D addresses the largest remaining error,
after a small trial run. Every adapted model is compared against its base on the
same validation split.

Candidate models (verify licences and English support before relying on them):
PaddleOCR, Florence-2, Qwen2.5-VL (small sizes), CLIP/SigLIP, Magi.

---

## 8. Evaluation Architecture

```text
predictions.jsonl
        ↓
evaluation/validate_format.py   (format check against sample_submission.jsonl)
        ↓
dataset/score.py                (text_order_score, balanced_joint_f1, diagnostics)
        ↓
evaluation/run_eval.py          (append row to results.md)
```

Experiments:
- floors: all-empty; ground-truth text with every speaker = `char1`
- pipeline experiments: OCR → +filter → +ordering → +speakers → +consistency → adapted
- oracle experiments: ground-truth text with predicted speakers; ground-truth speakers with predicted text
- ablations: remove one component at a time

Rules for reading the scores:
- the two main metrics are `text_order_score` and `balanced_joint_f1`
- `speaker_accuracy_on_matched` alone can mislead when little text is recovered
- the scorer guides decisions but is not the whole selection rubric

---

## 9. Compliance Map

| Task rule | How the architecture satisfies it |
|---|---|
| Open-weight, open-source only | all models run locally; no hosted AI APIs anywhere, including for pseudo-labels |
| Actual learning/adaptation | story filter (Phase 4) plus an adapted component (Phase 6), with base-vs-adapted comparison |
| Keep sequences together | grouped split in `make_split.py` |
| Test output automatic, no manual edits, no test answers | single `pipeline/inference.py` run, images only |
| Cite external work | citations and own-contribution statement in README; licences checked |
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
7. Character labels are per sequence; `NARRATION` is for narrator text only.
8. Empty pages return `[]`; nothing is invented to fill them.
9. Every experiment goes through `evaluation/run_eval.py` and is logged in `results.md`.
10. Seeds, model versions and paths come from `configs/config.py`.
11. External code, models and data are listed with citations and licences.
12. The final README is written by the student, not generated.

---

## 11. Growing the System

Do NOT build everything at once. Start with:

```text
data.py → make_split.py → evaluation tools → plain OCR → valid JSONL
```

Then add, in order: alternative baselines, alignment, story filter, reading order,
speakers, character consistency, adaptation, ablations, packaging.

The goal is to grow the pipeline from a working baseline, keeping a valid
submission at every stage.
