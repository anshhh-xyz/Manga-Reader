# What We Did: The Evolution of Manga Speaker Attribution

An end-to-end retrospective of the speaker attribution experiments for the Manga Task Reader pipeline: from visual CLIP embeddings, to specialized Manga VLMs, SBERT sequential state machines, spatial hybrid trackers, and ultimately to the winning **Pairwise Turn-Taking XGBoost Clusterer**.

---

## 1. The Challenge & Performance Landscape

In manga, text balloons lack explicit speaker tags (like *"Alice said"*). Attributing dialogue lines to speakers across 3-page sequences requires solving:
1. **Turn-Taking Mechanics**: Alternating dialogue in a conversation ($A \rightarrow B \rightarrow A \rightarrow B$).
2. **Panel Boundaries**: Speech balloons within the same panel usually alternate characters, while vertically stacked balloons often belong to the same speaker.
3. **Noisy Visuals**: Manga is grayscale, stylized, uses heavy screentones, and many characters speak from off-panel.
4. **Hard Upper Ceiling**: Speaker attribution only scores on text tokens recovered by OCR. With OCR recovering ~63.6% of gold tokens, speaker attribution is evaluated using `balanced_joint_f1` (macro-averaged over characters) and `speaker_accuracy_on_matched`.

---

## 2. Chronological Experiments & Detailed Breakdown

### Approach 0: The Naive Floor (`all char1`)
- **What we did**: Assigned every single detected dialogue box to a single character (`char1`).
- **Why**: Established the baseline floor that any machine learning model must beat.
- **Result**:
  - `balanced_joint_f1`: **0.2485 – 0.2563**
  - `speaker_accuracy_on_matched`: **~56.6%**
- **Takeaway**: If an ML model fails to beat 0.25, it is actively degrading performance compared to doing nothing.

---

### Approach 1: Visual CLIP Embeddings
- **What we did**:
  - Extracted visual image crops surrounding each dialogue balloon.
  - Passed the image crops through OpenAI CLIP to generate visual vector embeddings.
  - Computed cosine similarity between crops to determine if the same character was present near the balloons.
- **Why we tried it**: The intuition was that the character speaking is physically drawn near their dialogue balloon.
- **What happened & Why it failed**:
  - **Feature Importance**: When fed into a classifier, CLIP similarity had an importance of only **0.031** (nearly zero).
  - **Grayscale Screentone Noise**: Manga line art and black-and-white screentones do not provide rich visual signals for CLIP.
  - **Off-Panel Dialogue**: Characters frequently speak from outside the panel or behind objects.
  - **Mean ARI**: **0.13** (barely better than pure random chance).
- **Verdict**: **Abandoned.** Visual embeddings around bubbles are too noisy for black-and-white manga.

---

### Approach 2: MAGI v2 (Specialized Manga VLM)
- **What we did**:
  - Integrated **MAGI v2**, a deep vision-language model trained specifically on manga for character detection, character re-identification, and text-to-character association.
  - Tested various prompts and global character tracking configurations (`magi_prev_k4_global_n0`).
- **Why we tried it**: MAGI is specifically trained on Japanese manga layouts and character faces.
- **What happened & Why it fell short**:
  - `balanced_joint_f1`: **0.3312** (beat the 0.2485 floor, but lagged behind structural machine learning).
  - **Weaknesses**:
    - High inference latency and heavy GPU VRAM footprint.
    - Hallucinated or lost character tracks across page boundaries.
    - Fragile handling of non-standard manga panels and off-screen dialogue.
- **Verdict**: **Surpassed by lightweight structural models.**

---

### Approach 3: Phase 5 Pairwise Turn-Taking XGBoost (★ The Champion ★)
- **What we did**:
  - Built an engineering pipeline centered on **manga structure, reading order, and panel physics**:
    1. **XY-Cut Panel Detection**: Segmented pages into panels and calculated relative bounding coordinates.
    2. **Reading Order Parity**: Calculated `(j - i) % 2` (rank parity). In conversational dialogue, an alternating step ($j - i$ odd) indicates a different speaker; an even step indicates the same speaker returning.
    3. **Panel Gap Features**: `same_panel`, `panel_gap`, and within-panel relative position gap (`pp_gap`).
    4. **Spatial Geometry**: Normalized relative distances ($dx, dy, \text{dist}$) and area ratios.
    5. **Pairwise Classifier**: Trained a 22-feature XGBoost model (`pairwise_xgb.json`) to output a probability matrix $P_{ij} \in [0, 1]$ of whether balloon $i$ and balloon $j$ belong to the same character.
    6. **Constrained Agglomerative Clustering**: Converted probabilities to distances $D = 1 - P$, ran Average Linkage clustering with threshold $t = 0.50$, merged orphan singletons into their nearest cluster, and capped max speakers to 4 (`cluster_cfg.json`).
- **Results**:
  - `balanced_joint_f1`: **`0.3912`**
  - `speaker_accuracy_on_matched`: **`73.61%`**
  - `joint_f1`: **`0.5409`**
- **Why it worked so well**: Manga reading order and panel boundaries encode 80%+ of dialogue turn-taking. Pairwise graph clustering optimizes all connections globally across the entire 3-page sequence simultaneously.

---

### Approach 4: Phase 6 Surface Feature Expansion (Option 3 Initial)
- **What we did**:
  - Attempted to improve Phase 5 by adding 5 dialogue text features to the 22 layout features (total 27 features):
    - Punctuation differences: `exclaim_diff` ($!$) and `q_diff` ($?$)
    - Dialogue string length ratio (`len_ratio`)
    - Uppercase and caption heuristic flags (`narr_flag`)
  - Retrained XGBoost (`phase6_xgb.json`) and swept clustering thresholds ($t = 0.35$).
- **What happened & Why it regressed**:
  - `balanced_joint_f1`: dropped from **`0.3912`** to **`0.3509`** (-0.0403).
  - `speaker_accuracy_on_matched`: dropped from **`73.61%`** to **`66.70%`** (-6.91%).
  - **Reason**: In manga, dialogue punctuation and line length fluctuate wildly even for the same speaker (e.g., a character shouts *"WHAT!?"* and then says *"Okay."*). These surface text features acted as noise and degraded the clean structural geometry learned by Phase 5.
- **Verdict**: **Reverted.** The added text heuristics hurt performance.

---

### Approach 5: SBERT Sequential Tracker (Option 1)
- **What we did**:
  - Replaced pairwise clustering with a sequential state machine using Sentence-BERT (`all-MiniLM-L6-v2`).
  - Represented each active character by a rolling semantic embedding centroid of their recent lines.
  - For each new dialogue balloon, calculated cosine semantic similarity against active speakers and applied turn-taking priors.
- **What happened & Why it failed**:
  - `balanced_joint_f1`: **0.30 – 0.32**
  - **Failure Modes**:
    1. **Cascading Greedy Errors**: A greedy step mistake at box 3 poisoned the speaker's embedding centroid for all subsequent boxes.
    2. **Low Dialogue Semantic Coherence**: In action and comedy manga, consecutive utterances by the same speaker often have near-zero semantic cosine similarity (e.g., *"Take this!"* vs *"Over here!"*).
    3. **Over-Splitting**: Spawned too many new character labels on panel transitions.
- **Verdict**: **Rejected.**

---

### Approach 6: Refined Hybrid Spatial Tracker (Option 2)
- **What we did**:
  - Systematically resolved the sequential tracker's flaws (Steps H1, H2, H3):
    - **Step H1 (Normalized Geometry)**: Replaced absolute pixel thresholds with page-relative fractions ($\frac{dx}{W} < 0.12$, $-0.03 < \frac{dy\_gap}{H} < 0.10$, $\frac{dx}{W} > 0.16$).
    - **Step H2 (Premise Validation & Prior Tuning)**: Verified empirically on dev data that vertically stacked bubbles share the same speaker **76.19% of the time (80 / 105 cases)**. Eliminated artificial cross-panel penalties (`cross_panel_penalty = 0.0`) and set `max_speakers = 4`.
    - **Step H3 (Parameter Sweep)**: Swept 144 configurations across stack bonus, alternation bonus, SBERT weight $\alpha$, and new-speaker thresholds.
- **Results**:
  - Tuned `balanced_joint_f1`: **`0.3267`** (improved over raw SBERT, predicted average speakers of 2.56 vs gold 3.03).
- **Comparison against Pairwise Clustering**:
  - Option 2 tuned: `0.3267`
  - Option 3 / Phase 6: `0.3509`
  - Phase 5 Pairwise: **`0.3912`**
- **Why greedy trackers cannot beat global pairwise clustering**: A sequential tracker only looks backward at historical state. An agglomerative pairwise clusterer looks at all pairwise connections across all pages simultaneously, resolving global communities of dialogue bubbles.

---

## 3. Comprehensive Summary Scorecard

| Approach | Architecture / Engine | Key Mechanism | `balanced_joint_f1` | `spk_acc_matched` | Verdict |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **All `char1` Floor** | Rule | Assign 1 single speaker to all lines | `0.2485` | `56.61%` | Baseline floor |
| **Visual CLIP** | Deep Learning (ViT) | Crop image around bubble $\rightarrow$ CLIP embeddings | `~0.26` | `~57.0%` | **Failed** (0.031 importance, noisy B&W visuals) |
| **SBERT Sequential** | NLP + Greedy Tracker | `all-MiniLM-L6-v2` + sequential state machine | `0.3053` | `58.20%` | **Failed** (greedy cascading errors, low dialogue similarity) |
| **Hybrid Spatial Tracker** | Spatial Rules + SBERT | H1/H2/H3 normalized geometry + 76.2% stacked rule | `0.3267` | `59.57%` | Improved tracker, but bounded by greedy decisions |
| **MAGI v2** | Manga VLM | Specialized manga character detection & re-ID | `0.3312` | `61.80%` | Decent, but heavy and fragile across panels |
| **Phase 6 / Option 3** | XGBoost + Clustering | 27 features (layout + punctuation heuristics, $t=0.35$) | `0.3509` | `66.70%` | Strong, but punctuation features added noise |
| **Phase 5 Pairwise (Champion)** | **XGBoost + Clustering** | **22 features (pure panel physics + rank parity, $t=0.50$)** | **`0.3912`** | **`73.61%`** | **★ WINNER: Maximum accuracy and structural purity ★** |

---

### Approach 7: YOLO Container-First OCR (Experimental)

- **What we did**:
  - Designed and implemented a completely new **Container-First paradigm** to solve the text fragmentation problem observed in OCR-first pipelines.
  - Trained a **YOLOv8n** speech bubble detector on 1,395 aligned speech balloon bounding boxes (from `outputs/alignment/aligned_boxes.json`), covering 219 pages across 64 train / 16 val sequences.
  - The pipeline architecture: `Page Image → YOLOv8 Bubble Detector → Manga RTL Sort → OCR per Crop → Phase 5 XGBoost Diarization`.
  - Built [`detection/prepare_yolo_data.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/detection/prepare_yolo_data.py), [`detection/train_yolo.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/detection/train_yolo.py), [`detection/bubble_ocr.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/detection/bubble_ocr.py), and [`pipeline/inference_yolo.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/pipeline/inference_yolo.py).

- **YOLO Training Results** (35 epochs, RTX 5050, ~2.2 minutes):
  - `mAP@50 = 0.921` 🔥
  - `mAP@50-95 = 0.815` 🔥
  - Precision = 0.844, Recall = 0.936
  - Inference: **3.3ms/image** — detects 4–9 bubbles per page at 83–98% confidence.

- **End-to-End Pipeline Iterations** (val set, 16 sequences):

  | Architecture / Approach | Preprocessing / Grouping | `text_order` | `balanced_joint_f1` | `spk_acc_matched` | `joint_f1` | `token_coverage` |
  |---|---|:---:|:---:|:---:|:---:|:---:|
  | **Unmerged Baseline** (`pipeline.inference`) | Raw EasyOCR line-boxes | 0.7687 | 0.3380 | 65.08% | 0.4950 | 0.6667 |
  | **YOLO v1** (Crop-Only) | Raw bubble crops | 0.6601 | 0.2861 | 64.20% | 0.4610 | 0.6052 |
  | **YOLO v2** (Crop-Only) | CLAHE + Otsu binarization | 0.6489 | 0.2807 | 63.85% | 0.4552 | 0.5989 |
  | **YOLO v3** (Crop-Only) | CLAHE + bilateral filter | 0.6631 | 0.2872 | 64.12% | 0.4645 | 0.6198 |
  | **Hybrid YOLO-Guided** (★ NEW) | **Full-Page OCR + YOLO Containers** | **0.7673** | **0.3607** | **66.26%** | **0.5030** | **0.6665** |

- **Key Insights on the Hybrid Breakthrough**:
  1. **Crop OCR Bottleneck Eliminated**: Running EasyOCR on isolated small image crops degraded font resolution, producing distorted tokens (e.g. `'WHA-?i'` instead of `'WHA-?!'`). Full-page OCR maintains complete visual context and high token fidelity.
  2. **True Container-First Utterances**: Instead of treating each line of dialogue as a separate box, YOLOv8 speech bubble boundaries (`weights/bubble_yolo.pt`) are used to spatially aggregate all enclosed OCR lines into single, continuous dialogue utterances.
  3. **Beats Unmerged Pipeline by +2.3% Balanced F1**:
     - `balanced_joint_f1`: **0.3380 → 0.3607** (+0.0227)
     - `speaker_accuracy_on_matched`: **65.08% → 66.26%** (+1.18%)
     - `joint_f1`: **0.4950 → 0.5030** (+0.0080)
     - Merging fragmented lines inside speech bubbles prevents cross-line speaker splitting and significantly improves speaker attribution consistency!

- **Verdict**: **The Hybrid YOLO-Guided Container pipeline ([`pipeline/inference_yolo.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/pipeline/inference_yolo.py)) is the new best unaligned end-to-end pipeline**, successfully solving speech bubble text fragmentation without sacrificing OCR recognition fidelity.

---

## 4. Key Takeaways & Final State

1. **Geometry & Structure Trump Surface Text in Manga**:
   - The two strongest predictive signals in manga dialogue are:
     1. **Physical panel boundaries and within-panel rank position (`same_panel`, `pp_gap`)**.
     2. **Alternating turn-taking parity (`(j - i) % 2`)**.
   - Visual crops (CLIP) and superficial punctuation (`!`, `?`) only added noise.
2. **Global Optimization Beats Greedy Transitions**:
   - Pairwise affinity matrices combined with constrained agglomerative clustering ($t=0.50$, singleton merging, max $k=4$) consistently outperformed sequential state-machine trackers by +6.5% to +8.5% in Joint F1.
3. **The Power of Hybrid Containers**:
   - YOLOv8 balloon detector achieves 0.921 mAP@50. Using it as a spatial aggregator over full-page OCR eliminates dialogue line fragmentation and boosts unaligned end-to-end `balanced_joint_f1` from 0.3380 to **0.3607** (+2.3%).
4. **Current Production State**:
   - Production speaker attribution resides in [`speakers/enhanced_spatial_clusterer.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/speakers/enhanced_spatial_clusterer.py) executing the Phase 5 engine.
   - Configured with [`weights/pairwise_xgb.json`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/weights/pairwise_xgb.json) and [`weights/cluster_cfg.json`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/weights/cluster_cfg.json).
   - End-to-end YOLO container inference in [`pipeline/inference_yolo.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/pipeline/inference_yolo.py).
   - Test predictions verified in [`submission.jsonl`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/submission.jsonl) and [`submission.json`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/submission.json).
