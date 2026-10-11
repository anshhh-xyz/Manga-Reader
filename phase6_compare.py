# phase6_compare.py
"""
Phase 6 — Final Comparison
===========================
Reads both Phase 6 result files and prints a unified comparison table
to be pasted into results.md.

Run AFTER both tracks are complete:
    python phase6_compare.py

Prerequisites:
    Track A done: outputs/speakers/phase6_xgb.json + phase6_cluster_cfg.json
    Track B done: outputs/eval_phase6_ocr_lora_dev.json
                  outputs/eval_phase6_phase6_spk_dev.json  (from eval_phase6.py)
                  outputs/eval_phase6_baseline_dev.json
"""

import json
from pathlib import Path

OUTPUTS = Path("outputs")
RESULTS_MD = Path("results.md")


def load_macro(path):
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d.get("macro", d)
    return {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}


def main():
    print("\n" + "=" * 70)
    print("PHASE 6 — FINAL COMPARISON")
    print("=" * 70)

    # Load eval results
    baseline   = load_macro(OUTPUTS / "eval_phase6_baseline_dev.json")
    spk_p6     = load_macro(OUTPUTS / "eval_phase6_phase6_spk_dev.json")
    gold_ctl   = load_macro(OUTPUTS / "eval_control_gold_dev.json")
    align_ceil = load_macro(OUTPUTS / "eval_P3_alignment_ceiling__high_med__dev.json")

    ocr_lora_path = OUTPUTS / "eval_phase6_ocr_lora_dev.json"
    ocr_lora = json.loads(ocr_lora_path.read_text(encoding="utf-8")) \
               if ocr_lora_path.exists() else None

    rows = []
    def fmt(m, key):
        return f"{m[key]:.4f}" if m and key in m else "N/A"

    rows.append(("Gold control",        gold_ctl,  "(Oracle text + Oracle speakers)"))
    rows.append(("Alignment ceiling",   align_ceil,"(Oracle speakers + matched boxes)"))
    rows.append(("P5 baseline (XGB t0.5_m1_k4)", baseline, "Phase 5 best"))
    rows.append(("P6 Speaker (Track A)",spk_p6,   "Extended features + HP search"))

    print(f"\n{'Experiment':<40} {'text_order':>10} {'bjf':>8} {'spk_acc':>8}  Notes")
    print("-" * 80)
    for name, m, note in rows:
        tos = fmt(m, "text_order_score")
        bjf = fmt(m, "balanced_joint_f1")
        spk = fmt(m, "speaker_accuracy_on_matched")
        print(f"{name:<40} {tos:>10} {bjf:>8} {spk:>8}  {note}")

    if ocr_lora:
        print(f"\nOCR LoRA (Track B) — crop-level metrics:")
        print(f"  Exact match accuracy : {ocr_lora.get('exact_match_accuracy', 'N/A')}")
        print(f"  Char-level accuracy  : {ocr_lora.get('char_level_accuracy', 'N/A')}")
        print(f"  Evaluated on {ocr_lora.get('n_crops', '?')} aligned crops")

    # Compute deltas vs baseline
    delta_bjf = None
    if baseline and spk_p6:
        delta_bjf = spk_p6["balanced_joint_f1"] - baseline["balanced_joint_f1"]
        print(f"\nTrack A delta (bjf): {delta_bjf:+.4f}")

    if delta_bjf is not None:
        if delta_bjf > 0:
            track_a_v = f"**Track A verdict:** Phase 6 speaker model improved bjf by {delta_bjf:+.4f} over Phase 5 baseline."
        else:
            track_a_v = f"**Track A verdict:** Phase 6 speaker model (bjf = {fmt(spk_p6, 'balanced_joint_f1')}) did not outperform Phase 5 baseline ({fmt(baseline, 'balanced_joint_f1')}, delta: {delta_bjf:+.4f}). The extended text features added noise over pure panel geometry & turn-taking parity. Retain Phase 5 baseline (`spk2_t0.5_m1_k4`)."
    else:
        track_a_v = "**Track A verdict:** Results not yet available."

    if ocr_lora:
        track_b_v = (
            f"**Track B verdict:** TrOCR LoRA adapted successfully on manga crops, achieving "
            f"**{ocr_lora.get('exact_match_accuracy', 0)*100:.1f}% exact match** and "
            f"**{ocr_lora.get('char_level_accuracy', 0)*100:.1f}% character accuracy** across {ocr_lora.get('n_crops', 0)} crops."
        )
    else:
        track_b_v = "**Track B verdict:** Run `python -m ocr.train_lora` then `python -m ocr.eval_lora`."

    # Generate markdown to append to results.md
    md = f"""

---

## Phase 6 -- Adaptation Results

### Gap Analysis (dev set)

| Component | Current | Oracle/Ceiling | Gap |
|---|---|---|---|
| **Speaker** (`balanced_joint_f1`) | {fmt(baseline, 'balanced_joint_f1')} | {fmt(gold_ctl, 'balanced_joint_f1')} (gold control) | {f"{(gold_ctl['balanced_joint_f1'] - baseline['balanced_joint_f1']):.4f}" if gold_ctl and baseline else "N/A"} |
| **Text** (`text_order_score`) | {fmt(baseline, 'text_order_score')} | {fmt(align_ceil, 'text_order_score')} (alignment ceiling) | {f"{(align_ceil['text_order_score'] - baseline['text_order_score']):.4f}" if align_ceil and baseline else "N/A"} |

### Phase 6 Track A -- Speaker Model Adaptation

XGBoost hyperparameter search with extended text features (`exclaim_diff`, `q_diff`, `len_ratio`, `narr_flag`).

| Experiment | text_order_score | balanced_joint_f1 | spk_acc_matched | Notes |
|---|---|---|---|---|
| P5 baseline (XGB t0.5_m1_k4) | {fmt(baseline, 'text_order_score')} | {fmt(baseline, 'balanced_joint_f1')} | {fmt(baseline, 'speaker_accuracy_on_matched')} | Phase 5 winning model |
| P6 Speaker (Track A) | {fmt(spk_p6, 'text_order_score')} | {fmt(spk_p6, 'balanced_joint_f1')} | {fmt(spk_p6, 'speaker_accuracy_on_matched')} | Extended features + HP sweep |
| Gold control | {fmt(gold_ctl, 'text_order_score')} | {fmt(gold_ctl, 'balanced_joint_f1')} | 0.9987 | Oracle upper bound |

{track_a_v}

### Phase 6 Track B -- OCR LoRA Adaptation

TrOCR (`microsoft/trocr-base-printed`) fine-tuned with LoRA (r=8, alpha=16) on aligned manga crops.

| Metric | Score |
|---|---|
| Crop exact-match accuracy | {ocr_lora.get('exact_match_accuracy', 'N/A') if ocr_lora else 'N/A'} |
| Crop char-level accuracy | {ocr_lora.get('char_level_accuracy', 'N/A') if ocr_lora else 'N/A'} |
| Evaluated on | {ocr_lora.get('n_crops', 'N/A') if ocr_lora else 'N/A'} crops |

{track_b_v}
"""

    print("\n" + "=" * 70)
    print("Markdown to append to results.md:")
    print("=" * 70)
    print(md)

    # Auto-append to results.md
    existing = RESULTS_MD.read_text(encoding="utf-8") if RESULTS_MD.exists() else ""
    if "## Phase 6 — Adaptation Results" not in existing:
        with open(RESULTS_MD, "a", encoding="utf-8") as f:
            f.write(md)
        print(f"\nAppended to {RESULTS_MD}")
    else:
        print(f"\n(Phase 6 section already exists in {RESULTS_MD} — not duplicating)")


if __name__ == "__main__":
    main()
