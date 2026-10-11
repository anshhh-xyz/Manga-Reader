# speakers/eval_comparison.py
"""
Head-to-head evaluation: Option 2 vs Option 3
===========================================
Compares:
  - Option 2: Hybrid Spatial Sequential Tracker (hybrid_spatial_tracker.py)
  - Option 3: Enhanced Spatial Layout Clusterer (enhanced_spatial_clusterer.py)
against the official competition metric on the development set.
"""

import json
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from evaluation.common import score as scorer, split_ids, write_jsonl
from speakers.adapters import load_sequences, page_sizes as get_page_sizes
from speakers.hybrid_spatial_tracker import hybrid_assign
from speakers.enhanced_spatial_clusterer import cluster_assign

DEV_LABELS = ROOT / "dataset" / "development" / "labels.jsonl"


def eval_method(method_fn, method_name, sequences, references, all_dev):
    predictions = {}
    aris = []

    for seq_id, boxes in sequences.items():
        sizes = get_page_sizes(seq_id)
        stripped = [{k: v for k, v in b.items() if k != "speaker"} for b in boxes]
        labels = method_fn(stripped, sizes)

        # Build prediction format
        pages = [[], [], []]
        for b, lbl in zip(boxes, labels):
            pg = b.get("page", 0)
            if 0 <= pg < 3:
                pages[pg].append({"speaker": lbl, "text": b.get("text", "")})
        predictions[seq_id] = pages

        # Compute ARI on non-narration boxes
        gold_spks = [b.get("speaker") for b in boxes]
        valid = [(g, p) for g, p in zip(gold_spks, labels) if g not in (None, "NARRATION", "UNKNOWN")]
        if len(valid) > 1:
            g_l, p_l = zip(*valid)
            if len(set(g_l)) > 1 or len(set(p_l)) > 1:
                aris.append(adjusted_rand_score(g_l, p_l))

    for sid in all_dev:
        if sid not in predictions:
            predictions[sid] = [[], [], []]

    out_file = ROOT / "outputs" / f"preds_eval_{method_name}.jsonl"
    write_jsonl(predictions, str(out_file))

    report = scorer.evaluate(references, predictions)
    macro = report["macro"]
    mean_ari = float(np.mean(aris)) if aris else 0.0

    return {
        "name": method_name,
        "balanced_joint_f1": macro.get("balanced_joint_f1", 0.0) or 0.0,
        "spk_acc_matched": macro.get("speaker_accuracy_on_matched", 0.0) or 0.0,
        "joint_f1": macro.get("joint_f1", 0.0) or 0.0,
        "text_order_score": macro.get("text_order_score", 0.0) or 0.0,
        "coverage": macro.get("matched_token_coverage", 0.0) or 0.0,
        "mean_ari": mean_ari,
    }


def main():
    print("\n" + "=" * 65)
    print("      EVALUATING OPTION 2 VS OPTION 3 ON DEVELOPMENT SET")
    print("=" * 65)

    sequences = load_sequences()
    references = scorer.load_jsonl(DEV_LABELS, reference=True)
    all_dev = split_ids("dev")
    print(f"Loaded {len(sequences)} sequences from dev alignment.\n")

    print("[1/2] Running Option 2: Hybrid Spatial Sequential Tracker...")
    res_opt2 = eval_method(hybrid_assign, "option2_hybrid_spatial", sequences, references, all_dev)

    print("[2/2] Running Option 3: Enhanced Spatial Layout Clusterer...")
    res_opt3 = eval_method(cluster_assign, "option3_spatial_clusterer", sequences, references, all_dev)

    print("\n" + "=" * 78)
    print(f" {'Metric':<25} | {'Option 2 (Hybrid Tracker)':<23} | {'Option 3 (XGB Clusterer)':<23}")
    print("-" * 78)
    print(f" {'balanced_joint_f1':<25} | {res_opt2['balanced_joint_f1']:<23.4f} | {res_opt3['balanced_joint_f1']:<23.4f}")
    print(f" {'speaker_acc_matched':<25} | {res_opt2['spk_acc_matched']:<23.4f} | {res_opt3['spk_acc_matched']:<23.4f}")
    print(f" {'joint_f1':<25} | {res_opt2['joint_f1']:<23.4f} | {res_opt3['joint_f1']:<23.4f}")
    print(f" {'mean_ARI (clustering)':<25} | {res_opt2['mean_ari']:<23.4f} | {res_opt3['mean_ari']:<23.4f}")
    print(f" {'text_order_score':<25} | {res_opt2['text_order_score']:<23.4f} | {res_opt3['text_order_score']:<23.4f}")
    print("=" * 78)

    winner = "Option 2" if res_opt2["balanced_joint_f1"] >= res_opt3["balanced_joint_f1"] else "Option 3"
    print(f"\n WINNER: {winner} (based on competition balanced_joint_f1)\n")


if __name__ == "__main__":
    main()
