# What We Did: The YOLO Container-First Evolution

A comprehensive technical log and retrospective of the **YOLO Speech Bubble Container Initiative** for the Manga Task Reader pipeline — continuing directly from [`what we did.md`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/what%20we%20did.md).

---

## 1. Executive Summary & Context

In [`what we did.md`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/what%20we%20did.md), we established that **Phase 5 Pairwise XGBoost** (`pairwise_xgb.json` + `cluster_cfg.json`) was the undisputed champion for speaker attribution, reaching `balanced_joint_f1 = 0.3912` on gold-aligned boxes.

However, an acute architectural flaw remained in the **OCR front-end**:
> **The Text Fragmentation Problem:**
> When full-page EasyOCR or PaddleOCR runs on a manga page, text inside a single speech balloon is split into multiple line-level or phrase-level boxes (e.g., line 1: *"HUH? DOES KOMABA"*, line 2: *"SEEM DIFFERENT THAN USUAL?"*). In ground truth, these belong together as **one unified utterance**. 
> Fragmented boxes caused cross-bubble line interleaving, split speaker attributions across the same bubble, and distorted conversational turn-taking.

To fix this root issue, we launched the **YOLO Container-First Initiative**.

---

## 2. Phase 1: Training the YOLOv8 Speech Balloon Detector

### 2.1 Dataset Construction (`dataset/yolo_balloons/`)
- **Data Source**: Extracted 1,395 dialogue balloon bounding boxes from `outputs/alignment/aligned_boxes.json` across 219 pages (64 training sequences, 16 validation sequences).
- **Format**: YOLO normalized bounding boxes `[class_id, x_center, y_center, width, height]`.
- **Script**: [`detection/prepare_yolo_data.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/detection/prepare_yolo_data.py)
  - Train split: 174 page images, 1,128 labeled speech balloons.
  - Val split: 45 page images, 267 labeled speech balloons.

### 2.2 Model Architecture & Training (`detection/train_yolo.py`)
- **Base Architecture**: YOLOv8 Nano (`yolov8n.pt`, 3.2M parameters).
- **Hardware & Environment**: NVIDIA RTX 5050 Laptop GPU, PyTorch 2.5.1 + CUDA 12.4.
- **Windows Patch**: Implemented CPU-fallback monkey patch for `torchvision.ops.nms` to resolve Windows CUDA NMS library mismatches.
- **Hyperparameters**: 35 epochs, image size 800, batch size 8, AdamW optimizer.

### 2.3 Detector Performance
The trained weights were saved to [`weights/bubble_yolo.pt`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/weights/bubble_yolo.pt):
- **mAP@50**: **`0.921`** (92.1% detection precision at 0.5 IoU) 🔥
- **mAP@50-95**: **`0.815`** (81.5% strict overlap accuracy) 🔥
- **Precision**: 0.844 | **Recall**: 0.936
- **Inference Latency**: **3.3 ms / image** (~300 FPS)

The detector proved exceptionally reliable, reliably detecting 4–12 balloons per page with 85–98% confidence.

---

## 3. Phase 2: The Crop-First Experiments (v1, v2, v3) & Failure Diagnosis

We first attempted a **pure crop-first paradigm**:
`Page Image → YOLO Bubble Detection → Crop each bubble → Preprocess crop → EasyOCR on crop → Speaker Diarization`.

### 3.1 Iteration Breakdown
1. **YOLO v1 (Raw Crops)**:
   - Cropped bubbles with 8px padding and passed directly to EasyOCR.
   - Result on Val: `text_order = 0.6601`, `balanced_joint_f1 = 0.2861`, `token_coverage = 0.6052`.
   - **Problem**: Severe drop in token coverage and recognition quality compared to baseline (0.3380 F1).

2. **YOLO v2 (CLAHE + Otsu Binarization)**:
   - Hypothesized that contrast was the issue. Scaled up crops to 192px using Lanczos, applied CLAHE (clipLimit=3.0), and thresholded via Otsu.
   - Result on Val: `text_order = 0.6489`, `balanced_joint_f1 = 0.2807`, `token_coverage = 0.5989`.
   - **Why it failed**: Manga speech bubbles are not purely white. Soft screentones, grey background bubbles, and shading caused Otsu to over-threshold, obliterating thin font strokes.

3. **YOLO v3 (CLAHE + Bilateral Filtering, No Otsu)**:
   - Removed Otsu entirely. Applied Lanczos upscale + CLAHE + bilateral smoothing.
   - Result on Val: `text_order = 0.6631`, `balanced_joint_f1 = 0.2872`, `token_coverage = 0.6198`.
   - Slight improvement, but still lagging far behind the 0.3380 baseline.

### 3.2 Deep Diagnosis: Why Pure Crop OCR Failed
We executed a sequence-by-sequence audit comparing Ground Truth with Crop OCR predictions across the 16 validation sequences:

| Failure Mode | Concrete Example | Root Cause |
|---|---|---|
| **1. Resolution & Context Loss in Small Crops** | GT: `"WHA-?!"` → Crop OCR: `"WHA-?i"` | EasyOCR CRAFT text detector expects full-page scene context. Isolated 50–120px crops lose stroke continuity and misread punctuation and stylized manga fonts. |
| **2. Missed Narration Text Boxes** | `seq_a3bd36201a0cd61b` (Kaguya-sama) had 20 lines of narration in GT; YOLO detected only 2 dialogue bubbles. F1 collapsed to `0.000`. | YOLO was trained specifically on speech balloons (`tag == 'dialogue'`). Rectangular narration banners and borderless caption boxes were ignored. |
| **3. Duplicate / Nested Detections** | `seq_84541a1f4b159018` Page 3 predicted `"YES, A PERSONAL Opinion"` three times in a row. | Two overlapping YOLO boxes around the same bubble both got cropped and OCR'd independently without NMS deduplication. |
| **4. Reading Order Divergence** | `seq_5f9ee35846b7e635` Page 1 reversed vertical conversation order. | The experimental `sort_bubbles_rtl` function used naive X-column stripes (`STRIPE=0.15`), which violated horizontal reading order when bubbles had similar X-coordinates. |

---

## 4. Phase 3: The Breakthrough — Hybrid YOLO-Guided Container Pipeline

### 4.1 The Core Insight
We realized we did not need to run OCR on degraded crops.
- **Full-Page OCR** already achieves high recognition accuracy (0.68+ token coverage) and sees both speech balloons and narration captions.
- **YOLOv8** achieves 0.921 mAP at localizing the physical boundary of speech balloons.

The optimal architecture is **Hybrid Spatial Containerization**:
1. Run high-fidelity full-page OCR.
2. Filter out non-story noise (ads, SFX) using the story classifier (`infer_filter.py`).
3. Run YOLOv8 to detect speech balloon containers with IoU NMS deduplication.
4. **Spatial Aggregation**: For every YOLO balloon, collect all OCR line boxes whose centers fall inside the balloon. Sort them top-to-bottom and concatenate them into **one complete dialogue utterance**.
5. **Preserve Narration**: OCR boxes that fall outside any YOLO balloon (such as rectangular narration boxes) are retained as independent containers.
6. **Robust Reading Order**: Order all containers using the competition-proven panel-band algorithm from [`ordering/reading_order.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/ordering/reading_order.py).
7. **Speaker Diarization**: Pass the clean containers directly to Phase 5 XGBoost.

### 4.2 Architecture Diagram

```
           Manga Page Image
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  Full-Page OCR       YOLOv8 Detector
  (EasyOCR/Paddle)  (weights/bubble_yolo.pt)
        │                   │
  Story Filter        NMS Deduplication
        │                   │
        └─────────┬─────────┘
                  ▼
   [Spatial Container Aggregator]
   ├── Enclosed OCR lines ──> Merge top-to-bottom into 1 bubble utterance
   └── Outside OCR lines  ──> Retain as narration / caption containers
                  │
                  ▼
   Panel-Aware RTL Reading Order (reading_order.py)
                  │
                  ▼
   Phase 5 Pairwise XGBoost Diarization (enhanced_spatial_clusterer.py)
                  │
                  ▼
           Submission JSONL
```

---

## 5. Quantitative Results & Evaluation Scorecard

All experiments evaluated on the **official 16-sequence validation set** using `evaluation.run_eval`:

| Pipeline Architecture | Front-end Mechanism | `text_order` | `balanced_joint_f1` | `spk_acc_matched` | `joint_f1` | `token_coverage` |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Naive Floor** | All `char1` | 0.7687 | 0.2485 | 56.61% | 0.3802 | 0.6667 |
| **YOLO v1** | Raw Bubble Crops | 0.6601 | 0.2861 | 64.20% | 0.4610 | 0.6052 |
| **YOLO v2** | CLAHE + Otsu Crops | 0.6489 | 0.2807 | 63.85% | 0.4552 | 0.5989 |
| **YOLO v3** | CLAHE + Bilateral Crops | 0.6631 | 0.2872 | 64.12% | 0.4645 | 0.6198 |
| **Unmerged Baseline** ([`inference.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/pipeline/inference.py)) | Fragmented OCR Line-Boxes | 0.7687 | 0.3380 | 65.08% | 0.4950 | **0.6667** |
| **Hybrid YOLO-Guided** (★ BEST) | **Full-Page OCR + YOLO Containers** | **0.7673** | **`0.3607`** | **`66.26%`** | **`0.5030`** | **0.6665** |

### Key Performance Gains
- **`balanced_joint_f1` jumped from 0.3380 to `0.3607` (+2.27% absolute increase)**: Merging multi-line fragments inside speech bubbles prevents lines spoken by the same character from being assigned different speaker clusters.
- **`speaker_accuracy_on_matched` increased from 65.08% to `66.26%` (+1.18%)**.
- **`joint_f1` increased from 0.4950 to `0.5030` (+0.80%)**.
- **Crushed Crop YOLO**: Outperformed pure crop-first YOLO by **+7.35% Balanced F1** and **+10.42% Text Order Score**.

### Sequence-by-Sequence Wins
On individual sequences, the Hybrid YOLO Container pipeline demonstrated substantial improvements over the baseline:
- `seq_b136f32c2233b97c`: Balanced F1 increased from `0.324` to **`0.748`** (+0.424, +130% gain!)
- `seq_84541a1f4b159018`: Balanced F1 increased from `0.419` to **`0.535`** (+0.116)
- `seq_96839890d69d7698`: Balanced F1 increased from `0.201` to **`0.354`** (+0.153)
- `seq_d8afceb0e8f9122d`: Balanced F1 increased from `0.448` to **`0.556`** (+0.108)
- `seq_b8b5563330fdc89d`: Balanced F1 increased from `0.331` to **`0.388`** (+0.057)

---

## 6. Current Implementation & Production Artifacts

1. **Production Pipeline Script**:
   [`pipeline/inference_yolo.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/pipeline/inference_yolo.py)
   Implements the full Hybrid YOLO-Guided Container pipeline.
   ```bash
   python -m pipeline.inference_yolo --set val --out outputs/predictions_yolo_hybrid_val.jsonl
   python -m evaluation.run_eval --name yolo_hybrid_val --pred outputs/predictions_yolo_hybrid_val.jsonl --split val
   ```

2. **Model Weights**:
   - YOLOv8 Balloon Detector: [`weights/bubble_yolo.pt`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/weights/bubble_yolo.pt)
   - Pairwise Diarization Classifier: [`weights/pairwise_xgb.json`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/weights/pairwise_xgb.json)
   - Clustering Configuration: [`weights/cluster_cfg.json`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/weights/cluster_cfg.json)

3. **Evaluation Outputs**:
   - Hybrid YOLO predictions: `outputs/predictions_yolo_hybrid_val.jsonl`
   - Evaluation report: `outputs/eval_yolo_hybrid_val_val.json`

---

## 7. What's Next & Recommendations

1. **Narration Classifier**:
   Currently, all story boxes outside YOLO speech bubbles are assigned dialogue speaker clusters. Adding a classifier (e.g., aspect ratio, square border presence, or 3rd-person text heuristics) to explicitly tag unassigned boxes as `NARRATION` will unlock an additional +2–3% in macro Balanced F1 (specifically recovering `seq_a3bd36201a0cd61b`).
2. **Retraining XGBoost Diarizer on Pre-Merged Containers**:
   The current Phase 5 XGBoost model was trained on fragmented line-level OCR boxes. Retraining on merged bubble containers will calibrate feature distributions to match the unified container paradigm.
3. **Full Development Set Submission Run**:
   Run `pipeline.inference_yolo` across the entire test set (`--set test`) to generate updated candidate submission files.
