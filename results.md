# Results

| experiment | split | text_order | balanced_joint_f1 | spk_acc_matched | joint_f1 | coverage | notes |
|---|---|---|---|---|---|---|---|
| 0 all-empty | val | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Format check passes; all-empty baseline produces 0 |
| 1 GT text, all char1 | val | 1.0000 | 0.2563 | 0.5661 | 0.5661 | 1.0000 | text ceiling; speaker floor |
| P2 paddle naive | val | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | crashed (oneDNN NotImplementedError), fix pending |
| P2 easyocr naive | val | 0.5352 | 0.1679 | 0.5994 | 0.3594 | 0.5796 | 2.1s/seq, bagP 0.767, bagR 0.820, ordR 0.611 |

---

## Benchmark Analysis & Triage Summary

### 1. Summary of Benchmark Scores (Validation Set)

| Experiment | `text_order_score` | `balanced_joint_f1` | `spk_acc_matched` | `coverage` | What it means |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0 all-empty** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | Format check passes; all-empty baseline produces 0. |
| **1 GT text, all `char1`** | `1.0000` | `0.2563` | `0.5661` | `1.0000` | **Upper ceiling for text** (1.00) and **speaker floor** (0.256). |
| **P2 paddle naive** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | Empty file due to Paddle Windows/Python 3.13 oneDNN crash. |
| **P2 easyocr naive** | **`0.5352`** | **`0.1679`** | **`0.5994`** | **`0.5796`** | **Solid first working baseline!** Recovers 58% of tokens directly. |

---

### 2. Triage Diagnostic Breakdown (80 Sequences)

Running `evaluation/triage.py` on `outputs/preds_easyocr_development.jsonl`:

| Diagnostic Metric | Score | Key Takeaway |
| :--- | :---: | :--- |
| **Mean Bag Recall** | **82.0%** | **EasyOCR catches 82% of all words!** Text detection & recognition is strong. |
| **Mean Bag Precision** | **76.7%** | Most text detected is legitimate story text (minor non-story noise). |
| **Mean Ordered Recall** | **61.1%** | **~21% drop** between words found and words in the right order. |

#### Triage Category Counts:
- **`ORDER`: 54 sequences (67.5% of dataset)** — The major bottleneck. EasyOCR reads text correctly, but naive top-to-bottom order breaks manga right-to-left panel flow. Fixing reading order will boost text score from ~0.53 towards ~0.75+.
- **`ok / speakers`: 21 sequences** — Text & order are reasonable, but all lines are labeled `char1`.
- **`FILTER (too much predicted)`: 3 sequences** — Non-story noise / SFX captured.
- **`OCR (missed text)`: only 1 sequence** — Pure OCR misses are extremely rare with EasyOCR.
- **`INVENTED-TEXT`: 1 sequence** (and 7 pages with text predicted on empty gold pages, which score 0).

---

## Next Steps

### Step A: Fix PaddleOCR
- **Why**: A valid Paddle row is needed to compare against EasyOCR and apply Step 0 decision rules.
- **Action**: Add `enable_mkldnn=False` in `ocr/engines.py` under `PaddleEngine.__init__` in the 3.x branch to bypass the oneDNN Windows crash.

### Step B: Clear broken Paddle output & re-run
- Clear `outputs/preds_paddle_development.jsonl` and cache `outputs/cache/paddle`.
- Test on 3 sequences, then run the full development set and run `run_eval.py` + `triage.py`.

### Step C: Engine Comparison
- Compare PaddleOCR vs EasyOCR on speed, bag recall, order recall, and accuracy.

### Step D: VLM Status
- **Local Qwen**: Verified as text-only models (`Qwen2.5-0.5B-Instruct` and `Qwen2.5-3B-Instruct` in HF cache), not Vision models (no Qwen2-VL).
- **Florence-2**: Already fully cached in HF hub (`microsoft/Florence-2-large`), optional for baseline comparison.
