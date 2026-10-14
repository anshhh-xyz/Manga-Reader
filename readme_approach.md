# Manga Task 

BYTE Applied ML, Task 2

## What the architecture was supposed to be (the initial plan)
Thought to begin with ocr so the architecture was:
`raw manga page --> OCR engine to get text + word boxes --> box annotations / merge words into bubble boxes --> text cleaning / filter out ads and SFX --> sort lines by reading order --> some model like histboost or xgb or clustering for speaker attribution --> then predictions into JSONL`

Basically: detect where the text is, read it, throw away junk ads, order it right-to-left like manga, figure out who said what, and output valid format.

## Setup
- read the task doc, went to gpt and claude then turned it into ROADMAP.md + ARCHITECTURE.md (modular pipeline: detect, OCR, filter, order, speakers, JSONL)
- Phase 0: loaded the 80 dev sequences, split 64 train / 16 val grouped by sequence (all 3 pages together), 15 test
- Phase 1: built the eval tools (run_eval, validate_format), made the two floors (all-empty, gold text with one speaker, to check if our text part is good but speaker is bad), first plain OCR baseline + valid JSONL

## Oracles (sanity checks)
Claude asked to make oracles, learned what they are, seemed helpful so went with it.
So made three oracles to find the actual ceilings:
- **Floor 1 (all empty)**: gave 0.0 across everything. Just to verify format check passes.
- **Floor 2 (Gold text, all labeled `char1`)**: gave text_order `1.0000`, balanced F1 `0.2563`. (meaning even if my OCR is 100% perfect, and if I can't do speaker attribution and just predict one person for everything, the bottom score is ~0.256)
- **Alignment Oracle / Ceiling**: took actual EasyOCR boxes that matched gold text, and gave them gold reading order + gold speaker labels. Result: text_order `0.7622`, balanced F1 `0.6742`, coverage `0.6454`.
  - this told that with standard EasyOCR, no matter how better a speaker classifier I build, the ceiling was ~0.674 F1. If the text isn't in the box or is cut off, speaker matching fails anyway. So text recovery was just as important as speaker clustering.

## OCR choice
- thought to start with a few OCRs: PaddleOCR, EasyOCR, Florence-2, plus a zero-shot Qwen2.5-VL
- PaddleOCR crashed on my Windows / Python 3.13 / GPU setup (oneDNN error), fixed it once but dropped it as was trouble setting up and then got to know that it works on cpu or something, thought all ocrs work similarly somewhere so removed both paddle and florence
- Florence-2 cut to save time
- EasyOCR was the one that worked: 80/80 sequences in ~1.5s each, text_order_score 0.535 naive
- built triage.py: bag recall 0.82, precision 0.767, ordered recall only 0.611, 54/80 flagged as ORDER problems
- final: EasyOCR as the only OCR

## Alignment (Phase 3)
- labels have no boxes, so I had to make them
- EasyOCR gives word-level boxes, so asked ai to write merge.py to group them into balloon boxes (2077 boxes, ~8.7 per page)
- aligned boxes to label lines using the scorer's own token matching (score.py), tiers: high 1052 / medium 417 / none 608
- alignment ceiling (matched boxes + gold order + gold speakers): text_order 0.7622, balanced_joint_f1 0.6742, coverage 0.6454
- this is the best EasyOCR can do with a perfect filter, order and speakers

## Filter + reading order (Phase 4)
- **Story vs non-story filter (XGBoost)**:
  - Trained an XGBoost classifier on box geometry, text length, caps ratio, confidence, and white-balloon background to output `p_story` to remove sfx and non-needed text picked up by OCR.
  - Swept thresholds in `tune_threshold.py` to maximize `text_order_score` and locked threshold at **`0.30`** (`p_story >= 0.30` kept, below dropped as junk/SFX). But then OCR detected some random stuff on empty pages which was asked to delete, but then some pages which were almost empty had quiet text like "Look..." and this was getting removed as well which dropped the scores further, so removed the whole-page gating thing completely (kept `PAGE_GATE = 0.0`).
  - text_order_score has Longest Common Subsequence (LCS) for all the words extracted and sees how many appear in the exact chronological order of the gold reference text.
- **Panel detection & reading order**:
  - Manga is read Right-to-Left (RTL) panel-by-panel. Naive top-to-bottom reading order scrambled 54/80 sequences in our initial triage.
  - Detected panel borders by finding white gutter margins between panels.
  - Grouped panels into horizontal rows by vertical overlap, then sorted panels in each row Right-to-Left (`-x`).
  - Assigned each dialogue box to its enclosing panel, divided panels into vertical reading bands, and sorted text inside each panel top-to-bottom and RTL (`-cx`).
  - Claude said about Kendall tau rank correlation: this gave RTL panel-flow hit **`tau = 0.891`** vs naive top-to-bottom's **`tau = 0.755`** but LCS text order score was better so went with it.

---

## Speakers (Phase 5)

Treated speakers as clustering, not classification (names are per sequence: char1, char2, etc).
### 1. Visual CLIP Embeddings (`openai/clip-vit-base-patch32`) — Failed
- Thought: crop the artwork around each speech balloon (masking out the text with white) and compute CLIP visual embeddings to see which character face is near the bubble.
- Result: pairwise AUC actually *dropped* from **0.5735 to 0.5715**, feature importance was a miserable **0.031**. Manga is black-and-white screentone line art, and bubbles often sit in empty white gutters. CLIP had zero clue. Dropped.

### 2. MAGI v2 (Specialized Manga VLM) — Failed
- Thought: use MAGI off-the-shelf since it was trained specifically for manga character detection and speaker re-ID.
- Result: got `balanced_joint_f1 = 0.3312`. Better than floor (0.2485), but super heavy GPU VRAM, slow latency, had PyTorch/vision environment conflicts on Windows, and lost character tracks across page borders. Dropped.

### 3. Pairwise Turn-Taking XGBoost (Best)
- Antigravity Agents suggested manga reading order and panel boundaries encode 80%+ of conversational turn-taking
- **How it works**: for every pair of dialogue bubbles (i, j) across the 3 pages, extract features — same panel? rank gap even or odd? physical distance? XGBoost outputs `P(same_speaker)` for that pair. Convert all those to a distance matrix (`1 - P`), then Agglomerative Clustering cuts at threshold `t` to form groups = character labels. XGBoost doesn't predict speakers directly, it scores every pair, clustering finds the actual groups globally — that's why it beats greedy trackers that decide bubble by bubble.
- Engineered 22 pure layout & structural features:
  - `(j - i) % 2` (turn-taking rank parity — consecutive dialogue steps alternate characters)
  - physical panel gaps (`same_panel`, `panel_gap`, `pp_gap`)
  - normalized distances ($\Delta x$, $\Delta y$, reading distance)
- Trained pairwise XGBoost (`pairwise_xgb.json`) to output $P(\text{same\_speaker}) \in [0, 1]$.
- Fed into Agglomerative Clustering (average linkage, $t=0.50$, max 4 clusters).
- **Result**: F1 jumped to **`0.3912` on dev** and **`73.61%` speaker accuracy**! This proved structural panel physics completely destroy visual embeddings in manga.

### 4. Phase 6 Surface Text Feature Expansion — Failed Ablation
- Thought: maybe dialogue text features (punctuation `!`, `?`, uppercase, sentence length ratio) will help tell speakers apart (e.g. angry character vs calm character). Added 5 text features (27 features total).
- Result: F1 dropped from **0.3912 to 0.3509 dev** (-0.0403)! Punctuation in manga fluctuates wildly even for the same character. It just added noise. Reverted back to the clean 22 features.

### 5. SBERT Sequential Markov Tracker (Option 1) — Failed
- **How it worked**: instead of pairwise scoring, built a sequential state machine. For each new bubble, compute SBERT embedding, compare against rolling "centroid" embeddings of known characters, assign to the closest one or create a new character if too far.
- **Why it failed**:
  - one wrong assignment shifts the centroid, which causes the next bubble to also be wrong,
  - Manga dialogue has near-zero semantic similarity

### 6. Refined Hybrid Spatial Tracker (Option 2) — Decent but Bounded
- **How it worked**: instead of SBERT, used normalized geometry rules to assign speakers step-by-step:(ALL CODE WRITTEN BY AI,lol)
  - H1: if next bubble is directly below the previous one (small dx, small dy gap) → same speaker
  - H2: if it's offset horizontally by a significant amount → speaker switch
  - H3: empirically verified on dev data that vertically stacked bubbles share the same speaker **76.19% of the time (80/105 cases)**, so tuned the stacking bonus accordingly. Swept 144 parameter combinations.
- Result: reached **0.3267 F1** — better than raw SBERT, but still well below Pairwise XGBoost (0.3912).
---

## Architectural Change(i suggested but ai wrote improvisions)

- **The problem with our old pipeline**: `merge.py` was a naive heuristic. It guessed bubble boundaries using coordinate distance thresholds.
  - If a sentence had a slightly large line gap, `merge.py` split it into two separate broken utterances and if two separate speech bubbles were close to each other, `merge.py` merged them into one monster box.
- **Training the YOLO detector**:
  - Used the 1,395 aligned bounding boxes made by ocr from `outputs/alignment/aligned_boxes.json` (high + medium confidence tiers) as ground truth speech bubble labels.
  - Trained `yolov8s.pt` for 35 epochs 
  - **Results**: `mAP@50 = 0.921`, `mAP@50-95 = 0.815`, Precision = 0.844, Recall = 0.936. Inference: **3.3ms/image**, detects 4–9 bubbles per page at 83–98% confidence.
- **First attempt — Crop-Only YOLO (Failed)**:
  - First tried running OCR directly on each cropped bubble instead of full-page OCR.
  - Tried 3 preprocessing variations(DOEN BY AI)
    - v1: raw bubble crops → text_order `0.6601`, F1 `0.2861`
    - v2: CLAHE + Otsu binarization → text_order `0.6489`, F1 `0.2807` (worse!)
    - v3: CLAHE + bilateral filter → text_order `0.6631`, F1 `0.2872`
  - **issue**: EasyOCR CRAFT text detector needs full-page scene context. Isolated 50–120px crops lose stroke continuity and misread punctuation + stylized manga fonts. Also YOLO missed narration boxes entirely (trained only on dialogue bubbles), and overlapping YOLO boxes without NMS deduplication caused duplicated text.(AI TOLD ME ABOUT THIS)
- **Then came up with Hybrid Container Architecture**:
  - Run full-page OCR (high fidelity, sees everything including narration).
  - Run YOLO for bubble boundaries only (as spatial containers, not for OCR).
  - For every YOLO bubble, collect all OCR lines whose centers fall inside it, sort top-to-bottom, concatenate into one clean utterance.
  - OCR lines outside any bubble (narration boxes, captions) are kept as independent containers.
  `Page --> YOLOv8 (weights/bubble_yolo.pt) detects actual speech bubble containers --> full-page OCR lines inside each bubble are grouped together --> sorted top-to-bottom inside the bubble to make one clean sentence --> panel-aware RTL ordering --> Phase 5 XGBoost speaker clustering`
- **Result**:
  - Sentences were finally complete utterances instead of chopped fragments.
  - Speaker accuracy jumped from 58.2% to **`66.26%`**, Joint F1 hit **`0.5030`**, and val balanced F1 reached **`0.3607`**.

---

## More Things Tried for spekaer assignment

1. **Qwen2.5-VL-3B Multimodal Diarizer**:
   - Thought: pass manga image crops to local Qwen VLM and prompt it to diarize dialogue.
   - Result: collapsed to **`0.2019` F1**  The VLM hallucinated speaker labels, scrambled IDs across pages, and broke the Hungarian matching metric.
2. **Qwen2.5-0.5B small LM**:
   - Text-only diarizer prompt on a small 0.5B model. Also tanked to **`0.3043` F1**. Failed JSON formatting or echoed the prompt on 14/16 sequences, causing fallback to XGBoost.
3. **The NARRATION Trap Bug**:
   - Thought: tag unassigned boxes as `NARRATION` instead of a character label.
   - Found out from digging into `dataset/score.py` that `NARRATION` is a **fixed label** — it's never re-mapped by the Hungarian matching algorithm. So if any dialogue line gets tagged `NARRATION` incorrectly, it can't match a gold character and **adds an 
4. **26-Feature XGBoost with Character Body Detection**:
   - Built a YOLO character body detector (`detection/char_detector.py`) to measure bubble-to-character proximity.
   - Result: F1 dropped from 0.3607 to **`0.3487`**. Manga characters stand in weird poses, overlap panels, or speak from off-screen. The detections were too noisy and poisoned the feature set.
5. **Sentiment Features (27-feat XGBoost)**:
   - Thought speaker changes align with sentiment contrast (angry vs calm). Added VADER sentiment scores.
   - Result: tiny +0.0057 F1 gain, but added messy dependencies and wasn't clean, so not promoted.
6. **TrOCR LoRA Fine-tuning (Phase 6 Track B)**:
   - Thought: fine-tune `microsoft/trocr-base-printed` with LoRA (`r=8, alpha=16`) on 1,381 high + medium confidence aligned manga dialogue crops.
   - Training loss dropped from 5.0009 → 1.0869 across 3 epochs. Crop evaluation: **50.62% exact match** and **58.40% character-level accuracy**. Domain adaptation to manga typography worked.
   - But with baseline OCR already at `text_order = 0.7477` vs ceiling `0.7622` (gap of only **0.0145**), OCR accuracy wasn't the bottleneck. Dropped.

---

## Another Architectural Change 

- **The "Horizontal Bleed" Bug**:
  diagnosed our worst-performing sequence (`seq_952f154fb1505883`, which lost 38.5% of its words).
  When two speech bubbles sit side-by-side horizontally in a panel, EasyOCR full-page would read straight across both bubbles in one line!
  Character A's words and Character B's words got interleaved into one monster line:
  `"Bet IS She'Sthe GoiNg BeSt OUT To Be OF ALL OF WKAGE; US HErE:"`
  This corrupted the text into unmatchable word-salad and merged two different speakers' lines into one.
- **The Fix**:
  1. **Bleed Decoupling**: If an OCR box is abnormally wide compared to the YOLO bubble ($w_{\text{ocr}} > 1.35 \times w_{\text{bubble}}$), we discard it. Instead, we crop that specific speech bubble directly from the high-res image and run a zoomed, isolated EasyOCR pass on just that bubble crop.
  2. **Zoomed Crop Recovery**: If a bubble has 0 OCR lines, run zoomed crop OCR at 256px resolution.
  3. **Manga Typography Normalizer**: Manga text often hyphenates words across line breaks (`"EvERy- BODY"` $\rightarrow$ `"EVERYBODY"`, `"VICE - PRESIDENT"` $\rightarrow$ `"VICE-PRESIDENT"`) or misreads comic fonts (`CLLB` $\rightarrow$ `CLUB`, `YOLR` $\rightarrow$ `YOUR`). Wrote clean regex rules to stitch hyphenated words back together.
- **Result**:
  - Broke the **0.80 text order barrier** for the first time in project history: **`0.8066`**!
  - Token coverage jumped from 66.65% to **`72.88%`** (+6.23% gain).
  - Joint F1 rose to **`0.5069`**, balanced F1 at **`0.3604`**.

---

## DeBERTa-v3 Pairwise Cross-Encoder (`used encoders.md` / P13 — Tested & Archived)

- Thought: can a deep language model learn conversational back-and-forth better than tabular XGBoost
- Formatted 12,368 sequence dialogue pairs from our dataset and fine-tuned `microsoft/deberta-v3-base` to predict $P(\text{same\_speaker})$.
- Tested pure DeBERTa ($\alpha=1.0$): collapsed to **`0.2378` F1**. Without spatial/panel coordinates, DeBERTa thinks everyone is the same speaker because in manga everyone in the same panel talks about the exact same topic.
- Hybrid blend with XGBoost ($\alpha=0.3$): got **`0.3379` F1**, still below pure spatial XGBoost (0.3604).
- Conclusion: spatial layout + turn-taking priors are strictly superior to text semantics in manga.

---

## Finally 

- Looked at our Agglomerative Clustering setup in `speakers/enhanced_spatial_clusterer.py`.
- We had been using hardcoded defaults since the start:
  `{"threshold": 0.50, "merge_singletons": true, "max_clusters": 4}`
  **thoguht to tune those paramters**
- Ran a systematic sweep on the actual production decoupled boxes.
  `merge_singletons: true` was secretly wrecking our score the whole time
  In manga, characters often say a single short line ("Huh?!", "Behind you!", or random crowd chatter).
  Because `merge_singletons` was set to `True`, the clusterer was forcefully taking every single 1-line speaker and merging them into `char1` or `char2`! This polluted the main character clusters and destroyed speaker purity.
- **The new optimal config**:
  - `threshold: 0.48` (slightly tighter than 0.50, prevents false merges)
  - `merge_singletons: false` (lets singleton speakers stay independent)
  - `max_clusters: 6` (allows complex scenes with 5–6 speakers without clipping)
- **Result**:
  - Balanced Joint F1 jumped from 0.3604 to **`0.3755`** (NEW ALL-TIME PROJECT RECORD!)
  - Speaker accuracy on matched tokens: **`64.22%`**
  - Text order score: **`0.8066`**
  - Token coverage: **`72.88%`**
  - Joint F1: **`0.5077`**
  - Checked on 20 dev sequences to make sure it didn't overfit: dev F1 also went up from `0.2626` to `0.2718`.

---

## Final Scoreboard

| Pipeline | text_order | balanced_joint_f1 | spk_acc | joint_f1 | coverage | Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| Floor (all-empty) | 0.0000 | 0.0000 | 0.00% | 0.0000 | 0.00% | baseline |
| Floor (GT text, all char1) | 1.0000 | 0.2563 | 56.61% | 0.5661 | 100.0% | speaker floor |
| P2 EasyOCR naive | 0.5352 | 0.1679 | 59.94% | 0.3594 | 57.96% | first working run |
| P3 Alignment Ceiling | 0.7622 | 0.6742 | 99.85% | 0.7409 | 64.54% | oracle ceiling |
| P5 CLIP Embeddings | 0.7477 | ~0.2600 | ~57.0% | ~0.3700 | 63.61% | AUC dropped to 0.5715 |
| P5 MAGI v2 VLM | 0.7477 | 0.3312 | 61.80% | 0.4400 | 63.61% | heavy, lost track across pages |
| P5 SBERT Sequential | 0.7477 | 0.3053 | 58.20% | 0.4100 | 63.61% | greedy cascading errors |
| P5 Hybrid Spatial Tracker | 0.7477 | 0.3267 | 59.57% | 0.4380 | 63.61% | 76.2% stacked rule |
| P5 Pairwise XGBoost (Dev Champion) | 0.7477 | 0.3912 (dev) | 73.61% | 0.5409 | 63.61% | pure panel physics & parity |
| P6 Surface Text Expansion (27-feat) | 0.7477 | 0.3509 (dev) | 66.70% | 0.4883 | 63.61% | punctuation added noise |
| P10 Hybrid YOLO baseline | 0.7673 | 0.3607 (val) | 66.26% | 0.5030 | 66.65% | 1st big container jump |
| P11 Character body proximity | 0.7931 | 0.3487 (val) | 64.37% | 0.4884 | 70.55% | noisy body detections |
| P12 Decoupled Crop + Typography | 0.8066 | 0.3604 (val) | 64.07% | 0.5069 | 72.88% | broke 0.80 text order |
| P13 DeBERTa Cross-Encoder blend | 0.7748 | 0.3379 (val) | 62.62% | 0.4887 | 68.59% | topic overlap killed text LM |
| **P14 Optimal Clustering (Champion)** | **0.8066** | **0.3755 (val)** | **64.22%** | **0.5077** | **72.88%** | **Current best production** |
| *Oracle Upper Bound* | *0.8066* | *0.6639* | *100.0%* | *0.8120* | *72.88%* | gold speakers |

---

## Where I landed (Current Production State)
- **Detection**: YOLOv8 speech bubble detector (`weights/bubble_yolo.pt`) finds containers.
- **OCR**: EasyOCR full-page, with **Decoupled Zoomed Crop Recovery** for bleed/multi-bubble columns and empty bubbles.
- **Postprocessing**: Manga typography normalizer (hyphen rejoining, OCR character noise fix).
- **Ordering**: Panel-aware Right-to-Left (RTL) reading order sort.
- **Speakers**: Phase 5 Pairwise XGBoost (22 spatial/turn-taking features) + Agglomerative Clustering ($t=0.48$, `merge_singletons: false`, `max_clusters: 6`).
- **Configs**: `weights/cluster_cfg.json` active in production.
- **Submissions**: `submission.jsonl` and `submission.json` regenerated on test set and verified.

---

## What would i do later 
got bored of this project but if really had to i'd do 

- **Speaker detection**: XGBoost pairwise model is good but trained on a tiny dataset. if had access to a large manga dataset with speaker labels, would finetune a proper neural network (graph neural net or a transformer) on speaker diarization — adding richer patterns than 22 handcrafted features can.
- **Text coverage / OCR**: to push token coverage above 72.88% — finetune an OCR on manga. something like `manga-ocr` or TrOCR finetuned on manga crops. 
