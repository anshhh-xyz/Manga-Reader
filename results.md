# Results

| experiment | split | text_order | balanced_joint_f1 | spk_acc_matched | joint_f1 | coverage | notes |
|---|---|---|---|---|---|---|---|
| 0 all-empty | val | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Format check passes; all-empty baseline produces 0 |
| 1 GT text, all char1 | val | 1.0000 | 0.2563 | 0.5661 | 0.5661 | 1.0000 | text ceiling; speaker floor |
| P2 paddle naive | val | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | crashed (oneDNN NotImplementedError), fix pending |
| P2 easyocr naive | val | 0.5352 | 0.1679 | 0.5994 | 0.3594 | 0.5796 | 2.1s/seq, bagP 0.767, bagR 0.820, ordR 0.611 |
| P3 alignment ceiling (high+med) | dev | 0.7622 | 0.6742 | 0.9985 | 0.7409 | 0.6454 | matched OCR boxes + gold order + gold speakers |

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
| P3 alignment ceiling (high+med) | dev | 0.7622 | 0.6742 | 0.9985 | 0.7409 | 0.6454 |  |





OCR boxes: 7243
tiers: {'none': 1551, 'high': 4955, 'medium': 737} {'none': '21.4%', 'high': '68.4%', 'medium': '10.2%'}
tags: {'non-story': 1551, 'dialogue': 5333, 'narration': 359}
ambiguous boxes: 740
gold lines: 1753, with no box: 177 (10.1%)
short boxes (<=2 tokens): 6009, of which unmatched: 1391
gold lines: 1753, with no box: 177 (10.1%)
short boxes (<=2 tokens): 6009, of which unmatched: 1391
unmatched lines by speaker type: Counter({'character': 169, 'UNKNOWN': 6, 'NARRATION': 2})
sample unmatched gold lines:
short boxes (<=2 tokens): 6009, of which unmatched: 1391
unmatched lines by speaker type: Counter({'character': 169, 'UNKNOWN': 6, 'NARRATION': 2})
sample unmatched gold lines:
   seq_8f7b98bae85cd5e5 p 1 | haah
unmatched lines by speaker type: Counter({'character': 169, 'UNKNOWN': 6, 'NARRATION': 2})  
ocr results

new results after merging 
2233 boxes, 356 gold lines with no box -> D:\Projects\BYTE\Manga Task Reader AML-2\outputs\alignment
OCR boxes: 2233
tiers: {'none': 613, 'high': 1170, 'medium': 450} {'none': '27.5%', 'high': '52.4%', 'medium': '20.2%'}
tags: {'non-story': 613, 'dialogue': 1540, 'narration': 80}
ambiguous boxes: 143
gold lines: 1753, with no box: 356 (20.3%)
short boxes (<=2 tokens): 784, of which unmatched: 347
unmatched lines by speaker type: Counter({'character': 331, 'NARRATION': 15, 'UNKNOWN': 10})


initial scores with histgradientboost on classier 
          precision    recall  f1-score   support

   non-story       0.82      0.75      0.78       608
    dialogue       0.80      0.88      0.84       925
   narration       0.33      0.12      0.18        58

    accuracy                           0.80      1591
   macro avg       0.65      0.58      0.60      1591
weighted avg       0.79      0.80      0.79      1591

story-vs-non-story AUC: 0.8944624404347594

MODEL COMPARISON (sorted by macro F1)
===========================================================================
        model  accuracy  macro_f1  weighted_f1  non_story_f1  dialogue_f1  narration_f1  narration_recall  story_auc
      xgboost    0.8052    0.6027       0.7960        0.7831       0.8430        0.1818            0.1207     0.9053
hist_gradient    0.8014    0.5988       0.7927        0.7790       0.8402        0.1772            0.1207     0.8945
random_forest    0.7932    0.5979       0.7866        0.7770       0.8305        0.1860            0.1379     0.9052
  extra_trees    0.8152    0.5592       0.8002        0.7933       0.8530        0.0312            0.0172     0.9089


changed the numerical value of narration to 3 on xgboost
from 3 classification,making a threshold cretiera by 1-p(non-story)

heurestic reading order model based on kendall tau vs use a model for reading order
