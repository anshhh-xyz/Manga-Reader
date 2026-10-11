# YOLO Bubble Detector Approach & Implementation Roadmap

## 1. Executive Summary & Objective

In our current pipeline, OCR runs full-page text detection first, producing isolated, word-level or line-level bounding boxes (`"IT LOOKS"`, `"LIKE THAT"`, `"IS BAD:"`). Although geometric heuristics (`merge_balloons`) help fuse stacked lines post-hoc, text-first pipelines suffer from two persistent failure modes:
1. **Cross-Bubble Interleaving**: Lines belonging to adjacent bubbles in the same panel can get interleaved during XY-cut sorting if their vertical coordinates overlap.
2. **Fragmented Dialogue Utterances**: Individual dialogue lines are separated rather than grouped into single, coherent speech acts per speaker.

The **YOLO Bubble Detector Approach** flips this paradigm from **Text-First** to **Container-First**:
$$\text{Raw Manga Page Image} \xrightarrow{\textbf{YOLO Bubble Detector}} \text{Speech Balloon BBoxes} \xrightarrow{\textbf{RTL Reading Order}} \text{Ordered Bubbles} \xrightarrow{\textbf{OCR on Balloon Crops}} \text{Complete Sentences} \xrightarrow{\textbf{XGBoost Diarization}} \text{Speaker Labels}$$

This document specifies the exact architecture, data preparation pipeline, training procedure, and integration steps so that any engineer or agent can execute or build upon this system.

---

## 2. Problems It Solves

| Challenge | Text-First Approach (Current) | YOLO Container-First Approach |
| :--- | :--- | :--- |
| **Bubble Boundary Integrity** | Relies on post-processing distance heuristics (`v_gap`, `h_overlap`) to guess if lines share a bubble. | YOLO directly detects the speech balloon boundary as a unified container. |
| **Reading Order Precision** | Slices text into arbitrary horizontal bands; adjacent bubbles frequently interleave lines. | Bubbles are ordered in Manga Right-to-Left (RTL) sequence *before* text recognition runs. |
| **Speaker Attribution Granularity** | Diarization clusterer must handle 20–50 fragmented line entries per sequence. | Diarization clusters clean speech acts (10–20 bubbles per sequence), mirroring ground-truth transcripts. |
| **OCR Efficiency & Focus** | OCR scans entire pages including backgrounds, screentones, and non-story noise. | OCR is focused specifically on cropped speech bubbles. |

---

## 3. Architecture Blueprint

```
                      +-----------------------------+
                      |   Raw Manga Page Image      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   YOLOv8 Bubble Detector    |
                      |   (Trained on Aligned BBoxes|
                      +--------------+--------------+
                                     |
                                     | [x0, y0, x1, y1] per balloon
                                     v
                      +-----------------------------+
                      |   Panel Assignment &        |
                      |   Manga RTL Bubble Ordering |
                      +--------------+--------------+
                                     |
                                     | Sorted Bubble Crops
                                     v
                      +-----------------------------+
                      |   Targeted Bubble OCR       |
                      |   (EasyOCR / PaddleOCR      |
                      |    Paragraph Mode on Crop)  |
                      +--------------+--------------+
                                     |
                                     | Complete Dialogue Sentences
                                     v
                      +-----------------------------+
                      |   Phase 5 XGBoost Diarizer  |
                      |   (Rank Parity + Panel Gaps |
                      |    + Agglomerative Cluster) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Final Submission JSONL    |
                      +-----------------------------+
```

---

## 4. Dataset Preparation Strategy

### Source Data
Our alignment pipeline created [`outputs/alignment/aligned_boxes.json`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/outputs/alignment/aligned_boxes.json), which contains verified high- and medium-confidence text bounding boxes matched to gold dialogue lines across all 80 development sequences (over 1,380 aligned boxes).

### Bounding Box Aggregation into Balloons
In `aligned_boxes.json`, annotations that share the same `(sequence_id, page, order_index)` or belong to the same gold transcript line form the ground-truth speech bubbles.
- Polygon to rectangle conversion:
  $$x_0 = \min(x), \quad y_0 = \min(y), \quad x_1 = \max(x), \quad y_1 = \max(y)$$
- With padding ($\sim 4\text{--}8\text{px}$) to capture the outer speech bubble border.

### YOLO Annotation Format
For an image with width $W$ and height $H$, each detected balloon is converted to YOLO `.txt` format:
$$\text{class\_id} \quad x_{\text{center\_norm}} \quad y_{\text{center\_norm}} \quad w_{\text{norm}} \quad h_{\text{norm}}$$
where:
$$x_{\text{center\_norm}} = \frac{x_0 + x_1}{2 \cdot W}, \quad y_{\text{center\_norm}} = \frac{y_0 + y_1}{2 \cdot H}, \quad w_{\text{norm}} = \frac{x_1 - x_0}{W}, \quad h_{\text{norm}} = \frac{y_1 - y_0}{H}$$
- Single class: `0: balloon`

### Dataset Directory Structure
```
dataset/yolo_balloons/
├── data.yaml
├── images/
│   ├── train/  (Images from dataset/splits.json train split)
│   └── val/    (Images from dataset/splits.json val split)
└── labels/
    ├── train/  (.txt label files corresponding to train images)
    └── val/    (.txt label files corresponding to val images)
```

---

## 5. Model Selection & Training Recipe

### Base Architecture
- **Model**: `yolov8s.pt` or `yolov8n.pt` (Ultralytics).
- **Target Hardware**: NVIDIA GeForce RTX 5050 Laptop GPU (CUDA enabled).
- **Image Resolution**: `imgsz=1024` (manga pages are typically $\sim 800 \times 1200$, so 1024 maintains small text detail without downsampling artifacts).

### Training Hyperparameters
```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
results = model.train(
    data="dataset/yolo_balloons/data.yaml",
    epochs=40,
    imgsz=1024,
    batch=8,
    device=0,          # CUDA GPU
    optimizer="AdamW",
    lr0=0.001,
    augment=True,      # Scale, flip (horizontal only, not vertical)
    mosaic=0.5,        # Moderate mosaic to prevent splitting manga panels
    save_period=10,
    project="runs/bubble_yolo",
    name="yolov8s_manga_balloons"
)
```

---

## 6. Inference Pipeline Integration

Once trained (`weights/bubble_yolo.pt`):

```python
def extract_balloons_yolo(img_path, model, conf_thresh=0.35):
    """
    Returns list of balloon dicts:
    [{'bbox': [x0, y0, x1, y1], 'conf': float, 'crop': PIL.Image}, ...]
    """
    results = model.predict(img_path, conf=conf_thresh, imgsz=1024)[0]
    boxes = []
    w, h = results.orig_shape[1], results.orig_shape[0]
    for b in results.boxes:
        xyxy = b.xyxy[0].tolist()
        boxes.append({
            "bbox": [round(v, 1) for v in xyxy],
            "conf": float(b.conf[0])
        })
    return boxes
```

Inside each detected balloon crop:
```python
def read_balloon_text(crop_img, ocr_engine):
    """
    Runs OCR on the isolated crop.
    Sorts internal lines top-to-bottom and joins as a single utterance.
    """
    raw_lines = ocr_engine.readtext(np.array(crop_img))
    raw_lines.sort(key=lambda item: item[0][0][1])  # sort by y0
    text = " ".join(item[1] for item in raw_lines).strip()
    return clean_line(text)
```

---

## 7. Step-by-Step Implementation Roadmap

1. **Step 1: Install Dependencies**
   - Install `ultralytics` (`pip install ultralytics`).
2. **Step 2: Generate YOLO Dataset (`detection/prepare_yolo_data.py`)**
   - Parse `outputs/alignment/aligned_boxes.json` and group text boxes by gold sequence/page/order.
   - Generate normalized bounding boxes for each balloon.
   - Write `data.yaml`, `images/`, and `labels/` matching `dataset/splits.json`.
3. **Step 3: Train YOLO Balloon Detector (`detection/train_yolo.py`)**
   - Run training on RTX 5050 GPU for 30–40 epochs.
   - Save best model checkpoint to `weights/bubble_yolo.pt`.
4. **Step 4: Benchmark Bubble Detection Accuracy**
   - Evaluate Mean Average Precision ($mAP_{50}$ and $mAP_{50-95}$) against validation split.
5. **Step 5: Pipeline Integration & Comparison**
   - Hook `extract_balloons_yolo` into `backend/pipeline_service.py` and `pipeline/inference.py`.
   - Run `evaluation/run_eval.py` to compare competition `text_order_score` and `balanced_joint_f1` against the baseline.
6. **Step 6: Update Submissions**
   - Regenerate `submission.jsonl` and `submission.json`.
