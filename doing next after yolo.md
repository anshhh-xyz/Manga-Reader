# Doing Next After YOLO: The Qwen2.5 Conversational VLM Leap

A complete technical specification, architectural blueprint, and execution roadmap for replacing pure spatial tree clustering with **Conversational Multimodal LLM Diarization (Qwen2.5-VL)** to break through the 0.36 F1 ceiling.

---

## 1. The Core Problem & Motivation

In [`what we did yolo.md`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/what%20we%20did%20yolo.md), the Hybrid YOLO-Guided Container pipeline reached:
- `balanced_joint_f1`: **`0.3607`** (up from 0.3380 baseline).
- `speaker_accuracy_on_matched`: **`66.26%`**.
- `text_order_score`: **`0.7673`**.

However, benchmark oracle tests in `outputs/eval_control_gold_dev.json` revealed:
- **Oracle Speaker Upper Bound**: **`0.6639` Balanced Joint F1** with the *exact same* OCR text.

```
Current Hybrid Pipeline:   0.3607 F1
Oracle Speaker Bound:      0.6639 F1
Available Ceiling Gap:    +0.3032 F1  (Score can be nearly DOUBLED)
```

### Why Phase 5 XGBoost is Mathematically Capped
1. **The Binary Parity Trap**: XGBoost relies on `(j - i) % 2` (turn parity). In conversations with 3+ characters (e.g. Protagonist, Rival, Teacher), modulo 2 cannot differentiate Speaker 1 from Speaker 3. It **under-splits** (predicts 2.09 characters vs gold 3.03). Missing a 3rd character produces `0.000 F1` for that speaker, which severely degrades the macro-averaged score.
2. **Zero Cross-Page Character Persona**: Between Page 1 and Page 3, pairwise coordinates have `dist = 0` and `same_panel = 0`. Spatial trees are essentially guessing blind across page boundaries.
3. **No Conversational Semantics**: A tree classifier cannot understand that *"Why are you quitting?"* is answered by a *different* person saying *"We lost the farm"*, while *"Thank you for the meal"* is an interjecting elder.

---

## 2. The Solution: Qwen2.5-VL Conversational Diarizer

Instead of treating speaker attribution as a pairwise geometric distance problem, we treat it as a **Conversational Multimodal Dialogue Diarization problem**.

### 2.1 The Model
- **Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Location**: Already downloaded and cached locally at:
  `C:\Users\ASUS-PC\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct`
- **Hardware Footprint**: 3 Billion parameters. Under `bfloat16` precision, it occupies **~6.0 GB VRAM**, fitting comfortably inside the **8.52 GB VRAM** of the NVIDIA RTX 5050 Laptop GPU.
- **Weights Preservation**: Symlinked/referenced under [`weights/qwen_diarizer/`](file:///d:/Projects/BYTE/Manga%20Task%20Reader%20AML-2/weights/).

### 2.2 Input Representation
For each 3-page sequence, we provide:
1. **The 3 Page Images**: Downsampled / resized for efficient visual tokenization.
2. **The Ordered Utterance Transcript**:
   Extracted by the Hybrid YOLO pipeline, numbered `0` to `N - 1`:
   ```text
   [Page 1 | Panel 1] Box 0: "Why are you dropping out?"
   [Page 1 | Panel 1] Box 1: "We lost the farm. I have to work to save money."
   [Page 1 | Panel 2] Box 2: "Are you crazy?! What about your dream?!"
   [Page 2 | Panel 1] Box 3: "TRAINS. BANKS. HER FAMILY'S ASSETS TOTAL TWO HUNDRED TRILLION YEN."
   [Page 2 | Panel 1] Box 4: "Thank you for the meal, kids."
   ```

### 2.3 System Prompt & Task Specification
The model is instructed to act as a **Manga Dialogue & Speaker Attribution Engine**:
- Identify distinct speaking entities across the 3 pages: `char1`, `char2`, `char3`, `char4`.
- Identify narrative captions, scene descriptions, and non-spoken voiceovers as `NARRATION`.
- Understand turn-taking pragmatics:
  - Questions and their answers belong to different speakers.
  - Successive stacked lines in the same panel with matching tone belong to the same speaker.
  - Character personas remain consistent across pages (e.g., student tone vs adult tone).
- Output strict JSON mapping:
  ```json
  {
    "speakers": [
      {"id": 0, "speaker": "char1", "reason": "asking question"},
      {"id": 1, "speaker": "char2", "reason": "answering"},
      {"id": 2, "speaker": "char1", "reason": "emotional response"},
      {"id": 3, "speaker": "NARRATION", "reason": "expository caption"},
      {"id": 4, "speaker": "char3", "reason": "elderly parent"}
    ]
  }
  ```

### 2.4 Robust Fallback Architecture
If the LLM output times out, exceeds token limits, or encounters JSON decode errors:
The pipeline **automatically falls back** to the proven Phase 5 Pairwise XGBoost model (`cluster_assign`), guaranteeing 100% submission validity and zero catastrophic drops.

---

## 3. Directory Hygiene & Refactoring Plan

To ensure clean pair programming and maintainability for any agent:
- **Core Production Code**:
  - `pipeline/inference_yolo.py`: Primary end-to-end pipeline.
  - `detection/bubble_ocr.py`: YOLO balloon detection & spatial grouping.
  - `speakers/qwen_diarizer.py`: The new Qwen2.5-VL conversational diarizer.
  - `speakers/enhanced_spatial_clusterer.py`: Phase 5 XGBoost (backup/fallback).
- **Secondary / Archive Code**:
  - Move scratch exploration scripts (`scratch/`) and superseded test scripts to `speakers_extras/` and `detection/extras/`.

---

## 4. Step-by-Step Execution Roadmap

| Step | Action | Deliverable | Status |
|:---:|---|---|:---:|
| **1** | Write architecture documentation | `doing next after yolo.md` | **Done** |
| **2** | Workspace hygiene & archiving | Move auxiliary test scripts to `extras/` | In Progress |
| **3** | Build Qwen2.5-VL Diarizer | `speakers/qwen_diarizer.py` | Next |
| **4** | Benchmark on Validation Set | Evaluate on 16 val sequences (`evaluation.run_eval`) | Next |
| **5** | Integrate into Main Pipeline | Wire into `pipeline/inference_yolo.py` with XGBoost fallback | Next |
| **6** | Regenerate Submissions | Update `submission.jsonl` and `submission.json` | Next |

---

## 5. Expected Performance Impact

- **Baseline (Phase 5 XGBoost)**: `0.3607` Balanced F1
- **Target with Qwen2.5-VL**: **`0.45 – 0.55+` Balanced F1**
  - Recovers `seq_a3bd36201a0cd61b` (Kaguya-sama narration: `0.000` $\rightarrow$ `~0.70`).
  - Correctly splits 3rd and 4th characters in multi-person scenes (e.g. `seq_81b1782990ab7950`, `seq_c780ac6d1dbd1242`).
  - Resolves cross-page character identity consistency.
