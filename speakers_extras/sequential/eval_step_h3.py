"""
speakers/eval_step_h3.py  –  Step H3 quick sweep for hybrid_spatial_tracker
Runs base config then a 162-combo sweep. Streams one line per combo.
Usage:  python -m speakers.eval_step_h3
"""
import itertools, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.common import score as scorer, split_ids, write_jsonl
from speakers.adapters import load_sequences, page_sizes as get_page_sizes
from speakers.hybrid_spatial_tracker import hybrid_assign, DEFAULT_CFG, _embed, CFG_PATH
from sklearn.metrics import adjusted_rand_score
import numpy as np

DEV_LABELS = ROOT / "dataset" / "development" / "labels.jsonl"
BASELINE_XGB = 0.3509  # Option 3 / Phase 6

# ---------------------------------------------------------------------------
def stacked_accuracy(sequences):
    c = tot = 0
    cfg = DEFAULT_CFG
    for sid, boxes in sequences.items():
        sizes = get_page_sizes(sid)
        story = [b for b in boxes if b.get("tag") != "narration"]
        for i in range(1, len(story)):
            b1, b2 = story[i-1], story[i]
            if b1["page"] != b2["page"]: continue
            if b1.get("panel", -1) < 0 or b1.get("panel") != b2.get("panel"): continue
            s1, s2 = b1.get("speaker"), b2.get("speaker")
            if s1 in (None, "UNKNOWN", "NARRATION") or s2 in (None, "UNKNOWN", "NARRATION"): continue
            W, H = sizes[b1["page"]]
            cx1 = (b1["bbox"][0] + b1["bbox"][2]) / 2
            cx2 = (b2["bbox"][0] + b2["bbox"][2]) / 2
            dx = abs(cx1 - cx2) / W
            dg = (b2["bbox"][1] - b1["bbox"][3]) / H
            if dx < cfg["stacked_dx_max"] and cfg["stacked_dy_min"] < dg < cfg["stacked_dy_max"]:
                tot += 1
                if s1 == s2: c += 1
    return (c / tot * 100 if tot else 0.0, c, tot)


def run_eval(sequences, references, all_dev, seq_embs, cfg):
    preds = {}
    pred_counts, gold_counts, aris = [], [], []
    for sid, boxes in sequences.items():
        sizes = get_page_sizes(sid)
        stripped = [{k: v for k, v in b.items() if k != "speaker"} for b in boxes]
        labels = hybrid_assign(stripped, sizes, cfg=cfg, precomputed_embs=seq_embs.get(sid))

        pages = [[], [], []]
        for b, lbl in zip(boxes, labels):
            pg = b.get("page", 0)
            if 0 <= pg < 3:
                pages[pg].append({"speaker": lbl, "text": b.get("text", "")})
        preds[sid] = pages

        g_spks = [b["speaker"] for b in boxes
                  if b.get("speaker") not in (None, "UNKNOWN", "NARRATION")
                  and b.get("tag") != "narration"]
        p_spks = [lbl for b, lbl in zip(boxes, labels) if b.get("tag") != "narration"]
        if g_spks: gold_counts.append(len(set(g_spks)))
        if p_spks: pred_counts.append(len(set(p_spks)))

        g_l = [b["speaker"] for b in boxes if b.get("speaker") not in (None,"UNKNOWN","NARRATION") and b.get("tag") != "narration"]
        p_l = [lbl for b, lbl in zip(boxes, labels) if b.get("speaker") not in (None,"UNKNOWN","NARRATION") and b.get("tag") != "narration"]
        if len(g_l) > 1 and (len(set(g_l)) > 1 or len(set(p_l)) > 1):
            aris.append(adjusted_rand_score(g_l, p_l))

    for sid in all_dev:
        if sid not in preds: preds[sid] = [[], [], []]

    report = scorer.evaluate(references, preds)
    m = report["macro"]
    return {
        "bjf":      m.get("balanced_joint_f1",        0.0) or 0.0,
        "spk_acc":  m.get("speaker_accuracy_on_matched", 0.0) or 0.0,
        "jf1":      m.get("joint_f1",                 0.0) or 0.0,
        "pred_spk": float(np.mean(pred_counts)) if pred_counts else 0.0,
        "gold_spk": float(np.mean(gold_counts)) if gold_counts else 3.03,
        "ari":      float(np.mean(aris)) if aris else 0.0,
    }


def main():
    print("=" * 65)
    print("  STEP H3: HYBRID SPATIAL TRACKER – SWEEP")
    print("=" * 65)

    seqs = load_sequences()
    refs = scorer.load_jsonl(DEV_LABELS, reference=True)
    all_dev = split_ids("dev")
    print(f"Sequences: {len(seqs)}")

    acc, c, tot = stacked_accuracy(seqs)
    print(f"\nStacked-rule empirical accuracy: {acc:.2f}% ({c}/{tot})")

    print("\nPrecomputing SBERT embeddings ...")
    seq_embs = {}
    for sid, boxes in seqs.items():
        texts = [str(b.get("text") or "") for b in boxes if b.get("tag") != "narration"]
        seq_embs[sid] = _embed(texts)
    print("Done.\n")

    # --- Base evaluation ---
    r0 = run_eval(seqs, refs, all_dev, seq_embs, DEFAULT_CFG)
    print(f"Base (H2 defaults)")
    print(f"  bjf={r0['bjf']:.4f}  spk_acc={r0['spk_acc']:.4f}  jf1={r0['jf1']:.4f}"
          f"  ari={r0['ari']:.4f}  pred_spk={r0['pred_spk']:.2f}  gold_spk={r0['gold_spk']:.2f}")
    print(f"  vs XGB baseline: {r0['bjf'] - BASELINE_XGB:+.4f}\n")

    # --- Sweep ---
    grid = {
        "stack_bonus":           [0.40, 0.65, 0.90],
        "same_panel_bonus":      [0.00, 0.20],
        "alternation_bonus":     [0.20, 0.40],
        "alpha":                 [0.10, 0.30],
        "new_speaker_threshold": [-0.05, 0.00, 0.05],
        "max_speakers":          [3, 4],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"Sweep: {len(combos)} combinations\n")
    print(f"  {'#':>4}  {'bjf':>7}  {'spk_acc':>7}  {'ari':>6}  {'pred':>5}  note")

    best_bjf = r0["bjf"]
    best_cfg = DEFAULT_CFG.copy()
    best_r   = r0

    for n, combo in enumerate(combos, 1):
        cfg = DEFAULT_CFG.copy()
        cfg.update(dict(zip(keys, combo)))
        r = run_eval(seqs, refs, all_dev, seq_embs, cfg)
        note = ""
        if r["bjf"] > best_bjf:
            best_bjf = r["bjf"]
            best_cfg = cfg.copy()
            best_r   = r
            note = "<-- NEW BEST"
        if note or n % 50 == 0:
            print(f"  {n:4d}  {r['bjf']:.4f}  {r['spk_acc']:.4f}  {r['ari']:.4f}"
                  f"  {r['pred_spk']:.2f}  {note}")
        sys.stdout.flush()

    print("\n" + "=" * 65)
    print("BEST RESULT")
    print(f"  bjf={best_r['bjf']:.4f}  spk_acc={best_r['spk_acc']:.4f}"
          f"  jf1={best_r['jf1']:.4f}  ari={best_r['ari']:.4f}")
    print(f"  pred_spk={best_r['pred_spk']:.2f}  gold_spk={best_r['gold_spk']:.2f}")
    print(f"  vs XGB baseline: {best_r['bjf'] - BASELINE_XGB:+.4f}")
    print(f"\nBest config:\n{json.dumps(best_cfg, indent=2)}")

    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
    print(f"\nSaved -> {CFG_PATH}")


if __name__ == "__main__":
    main()
