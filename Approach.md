Started by discussing it with ChatGPT to understand clearly about whats been asked and what to do
Saw the given dataset structure and samples
Tried to read score.py,someweher successful,gave it to chat gpt to brief it out and explain technicality 
asked it to make a sample roadmap and an architecutre file which would have techstack and workflow info
went to claude to review everthing and ask for updates

*What they suggested-*
they gave a good head start about datachecking,training methods (k-folds and experimentations), alignment system,ocr training and recommendation,filtering system etc

Ran into problem with alignment system-
tried using self made algo,open cv manga comic and faster rcnn 
now training yolo for detecting bubbles and boxes
IDEA IS-
train yolo for bubbles and boxes 
then use ocr to get text from thosee and then a filtering system that helps get the spoken text only and then character assignment

Why not YOLO for alignment? YOLO needs box labels to train, and getting box labels is exactly the problem. Alignment doesn't need a trained detector. A pretrained detector and OCR find the boxes, and text matching links each one to a label line. After alignment you'll have pseudo-box labels, so a detector could be fine-tuned later if it turns out to be the weak point. That is optional.

Why not OpenCV? Alignment is a text-matching problem, so image processing doesn't help with it. I'm not aware of a specific "manga edition" of OpenCV. Plain OpenCV is useful for panel detection (contours). For text boxes, use a pretrained detector such as PaddleOCR's or comic-text-detector (check this one is still available).

Turned score.py into a module to use its functions
Using score.py as a module and then importing it funtions and testing methods to improve our work

need to fix ocr stuff to scoremax / using regex based algo for that


OCR model choosing--easy ocr,paddle ocr and florence,not using paddle as it works on cpu not gpu
removed florence too,had some isseus while running,using just easy ocr, and easyocr performs well 


OCR is skipping small boxes from every page , mainly those which are one word per box 
merging ballon texts for now,received better results
updateing the texts span in merged box so to get better results
still missing on boxes,there are 83 lon unatched boxes ,
EasyOCR’s paragraph=True mode is dropping or scrambling text before alignment ever sees it. The diagnostic below checks that directly.
Taking boxes annotations for speakers and converting area arounf thsoe boxes to embeddings using CLIP
GOT BAD RESUTLS MADE A DIAGNOSTIC.PY TO EVALUATE- ITS RESULTS WERE-
The speaker model beats the floor, so the Phase 5 exam is technically met. But it recovers only about 20% of the available gap (0.085 of 0.416). Mean ARI of 0.13 says it is barely better than chance at grouping lines by speaker. The diagnose output explains why:

Panel features are dead. same_panel and panel_gap have importance 0.000, and the same-panel rate is exactly 0.0. Panel ids never reached the speaker data, so the strongest cue, turn-taking within a panel, is missing. This is the main fix.
page_gap dominates at 0.213. The model is mostly saying “balloons on the same page are probably the same person”. That is a weak prior, and with no panel information it has nothing better.
It under-splits. It predicts 2.09 speakers per sequence against 3.03 gold. The scorer weights every gold speaker equally, so missing minor speakers hurts.
CLIP similarity is nearly useless at 0.031 importance, as expected for black-and-white manga.

The fix is a self-contained panel finder, plus two turn-taking features, gap_parity and pp_gap. In dialogue, A, B, A, B means an even rank gap usually means the same speaker, and a tree model finds that easier if you give it explicitly.


Replacing pair_features with gpa parity and pp gap

### Phase 5 Speaker Diarization Comparison & Decision

| Approach | Configuration | balanced_joint_f1 | text_order_score | Verdict / Notes |
|---|---|---|---|---|
| **Floor** | All char1 | 0.2485 | 0.7477 | Baseline floor |
| **Magi v2** | magi_prev_k4_global_n0 | **0.3312** | 0.7477 | Manga VLM (visual character detection & re-ID) |
| **XGBoost + XY-cut Panels** | spk2_t0.5_m1_k4 | **0.3418** | 0.7477 | **Selected Approach** (Turn-taking parity + panel gaps) |
| **Gold Control** | Oracle gold speakers | 0.6639 | 0.7477 | Theoretical upper bound with perfect speaker labels |

**Decision:**
Proceeding with **XGBoost Turn-Taking (spk2_t0.5_m1_k4)** as the primary speaker engine.
- Outperformed zero-shot MAGI (0.3418 vs 0.3312) and far exceeded the baseline floor (0.2485).
- Relies on dialogue structure within panels rather than noisy visual embeddings.
- Lightweight, deterministic, and fast for final submission inference.

---

### Phase 6 Adaptation Experiments (Dual-Track)

#### Track A: Speaker Adaptation Sweep
- **Exploration:** Extended pairwise XGBoost with dialogue punctuation differences (`exclaim_diff`, `q_diff`), character length ratio (`len_ratio`), and caption/narration heuristic flags (`narr_flag`).
- **Results:** 27 hyperparameter combinations swept. The extended features achieved OOF AUC of 0.5797, but on dev sequence evaluation scored `balanced_joint_f1 = 0.3509` vs baseline `0.3912` (delta: -0.0403).
- **Finding:** Shallow surface punctuation added variance and noise compared to pure spatial reading-order parity and physical panel bounding gaps.
- **Verdict:** Retained Phase 5 baseline (`pairwise_xgb.json` with `cluster_cfg.json`, threshold 0.50, merge singletons, max clusters 4).

#### Track B: OCR Adaptation (TrOCR LoRA)
- **Exploration:** Fine-tuned `microsoft/trocr-base-printed` with LoRA (`r=8, alpha=16`) on 1,381 high- and medium-confidence aligned manga dialogue crops on GPU.
- **Results:**
  - Training loss dropped from 5.0009 -> 1.0869 across 3 epochs.
  - Crop evaluation: **50.62% exact match** (699/1,381) and **58.40% character-level accuracy**.
- **Finding:** Domain adaptation to manga typography succeeded, but with baseline OCR already delivering 0.7477 `text_order_score` against the 0.7622 ceiling (a gap of only 0.0145), OCR accuracy is not the primary bottleneck in the system.

---

### Final End-to-End Pipeline Evaluation (Unassisted)

Executed [`pipeline/inference.py`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/pipeline/inference.py) over all 71 development sequences with no ground-truth boxes:
$$\text{Raw Page Images} \xrightarrow{\text{PaddleOCR}} \text{XY-Cut Panels} \xrightarrow{\text{Story Filter}} \text{RTL Reading Order} \xrightarrow{\text{XGBoost Turn-Taking}} \text{Speaker Assignment}$$

**Benchmark Scores:**
- **`text_order_score`**: **0.6874**
- **`balanced_joint_f1`**: **0.3141**
- **`speaker_accuracy_on_matched`**: **0.6741** (67.4% accuracy across matched dialogue lines)
- **`joint_f1`**: **0.4625**
- **`matched_token_coverage`**: **0.5966**

