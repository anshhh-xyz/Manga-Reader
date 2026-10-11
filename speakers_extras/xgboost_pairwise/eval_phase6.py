# speakers/eval_phase6.py
"""
Evaluate the Phase 6 speaker model against the Phase 5 baseline.

Loads the gold-aligned dev sequences, runs both models (baseline + phase6),
computes balanced_joint_f1, and prints a comparison table.

Run with:
    python -m speakers.eval_phase6
"""

import json
import pickle
import numpy as np
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

from evaluation.common import OUTPUTS
from .adapters import OUT, load_sequences, page_sizes as page_sizes_fn, TAG_NARR
from .pair_features import seq_pair_features, prob_matrix
from .character_memory import labels_from_P
from .phase6_tune_speaker import _narr_flag, EXT_FEATURES

BASELINE_MODEL_PATH  = OUT / "pairwise_xgb.json"
BASELINE_CFG_PATH    = OUT / "cluster_cfg.json"
PHASE6_MODEL_PATH    = OUT / "phase6_xgb.json"
PHASE6_CFG_PATH      = OUT / "phase6_cluster_cfg.json"


def pair_feat_ext(boxes, sizes):
    from collections import defaultdict
    n = len(boxes)
    X, y, pairs = [], [], []

    def geom(b):
        W, H = sizes[b["page"]]
        x1, y1, x2, y2 = b["bbox"]
        return (x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H

    g = [geom(b) for b in boxes]
    seen = defaultdict(int); pos = []
    for b in boxes:
        k = (b["page"], b["panel"]); pos.append(seen[k]); seen[k] += 1

    for i in range(n):
        for j in range(i+1, n):
            a, b = boxes[i], boxes[j]
            same_page = a["page"] == b["page"]
            same_panel = int(same_page and a["panel"] >= 0 and a["panel"] == b["panel"])
            dx = g[j][0] - g[i][0] if same_page else np.nan
            dy = g[j][1] - g[i][1] if same_page else np.nan
            dist = float(np.hypot(dx, dy)) if same_page else np.nan
            pgap = abs(a["panel"] - b["panel"]) if same_page and a["panel"] >= 0 and b["panel"] >= 0 else np.nan
            ppg = abs(pos[i] - pos[j]) if same_panel else np.nan
            ta, tb = a["text"] or "", b["text"] or ""
            base = [int(same_page), abs(a["page"]-b["page"]), j-i, (j-i)%2, same_panel, pgap, ppg,
                    dx, dy, dist, g[i][2], g[i][3], g[j][2], g[j][3],
                    (g[i][2]*g[i][3])/max(g[j][2]*g[j][3], 1e-6),
                    len(ta), len(tb), int("?" in ta), int("?" in tb),
                    int("!" in ta), int("!" in tb), np.nan]
            ext = [
                abs(int("!" in ta) - int("!" in tb)),
                abs(int("?" in ta) - int("?" in tb)),
                min(len(ta), len(tb)) / max(len(ta), len(tb), 1),
                _narr_flag(ta),
                _narr_flag(tb),
            ]
            X.append(base + ext)
            sa, sb = a.get("speaker"), b.get("speaker")
            ok = sa not in (None, "UNKNOWN") and sb not in (None, "UNKNOWN")
            y.append(int(sa == sb) if ok else -1)
            pairs.append((i, j))

    return np.array(X, dtype=float).reshape(-1, len(EXT_FEATURES)), np.array(y), pairs


def load_model(model_path, cfg_path):
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(str(model_path))
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    return m, cfg


def score_model(model_path, cfg_path, label="model"):
    """Score a speaker model by injecting predicted speakers into the dev sequences."""
    if not Path(model_path).exists():
        print(f"  {label}: model not found at {model_path}")
        return None

    model, cfg = load_model(model_path, cfg_path)
    seqs_data = load_sequences()
    all_ids = [json.loads(line)["sequence_id"] for line in open("dataset/development/labels.jsonl", encoding="utf-8")]

    results = {}
    for sid in all_ids:
        boxes = seqs_data.get(sid, [])
        story_idx = [k for k, b in enumerate(boxes) if b.get("tag") != TAG_NARR]
        labels = ["NARRATION"] * len(boxes)

        if len(story_idx) >= 2:
            sizes = page_sizes_fn(sid)
            sub = [boxes[k] for k in story_idx]
            if getattr(model, "n_features_in_", 22) == 27:
                X, _, pairs = pair_feat_ext(sub, sizes)
            else:
                X, _, pairs = seq_pair_features(sub, None, sizes)

            if pairs and len(X) > 0:
                probs = model.predict_proba(X)[:, 1]
                P = prob_matrix(len(sub), pairs, probs)
            else:
                P = np.ones((len(sub), len(sub)))

            sub_labels = labels_from_P(sub, P, cfg)
            for k, nm in zip(story_idx, sub_labels):
                labels[k] = nm
        elif len(story_idx) == 1:
            labels[story_idx[0]] = "char1"

        pages = [[], [], []]
        for b, lab in zip(boxes, labels):
            pg = b["page"]
            if pg < 3:
                pages[pg].append({"speaker": lab, "text": b.get("text", "")})
        results[sid] = pages

    # Write to temp JSONL and score
    tmp = OUTPUTS / f"_phase6_{label}.jsonl"
    with open(tmp, "w", encoding="utf-8") as f:
        for sid in all_ids:
            f.write(json.dumps({"sequence_id": sid, "pages": results[sid]}, ensure_ascii=False) + "\n")

    # Use the evaluation scorer
    eval_name = f"phase6_{label}"
    eval_out = OUTPUTS / f"eval_{eval_name}_dev.json"
    proc = subprocess.run(
        [sys.executable, "-m", "evaluation.run_eval",
         "--name", eval_name, "--pred", str(tmp), "--split", "dev"],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-1000:])
        return None

    ev = json.loads(eval_out.read_text(encoding="utf-8"))
    m = ev["macro"]
    return {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6 — Speaker Model Evaluation")
    print("=" * 60)

    print("\n[1] Baseline (Phase 5 XGBoost t0.5_m1_k4):")
    baseline = score_model(BASELINE_MODEL_PATH, BASELINE_CFG_PATH, label="baseline")
    if baseline:
        print(f"    text_order_score:    {baseline.get('text_order_score', 'N/A')}")
        print(f"    balanced_joint_f1:   {baseline.get('balanced_joint_f1', 'N/A')}")
        print(f"    spk_acc_matched:     {baseline.get('speaker_accuracy_on_matched', 'N/A')}")

    print("\n[2] Phase 6 (extended features + tuned hyperparams):")
    phase6 = score_model(PHASE6_MODEL_PATH, PHASE6_CFG_PATH, label="phase6_spk")
    if phase6:
        print(f"    text_order_score:    {phase6.get('text_order_score', 'N/A')}")
        print(f"    balanced_joint_f1:   {phase6.get('balanced_joint_f1', 'N/A')}")
        print(f"    spk_acc_matched:     {phase6.get('speaker_accuracy_on_matched', 'N/A')}")

    if baseline and phase6:
        bjf_delta = round(phase6["balanced_joint_f1"] - baseline["balanced_joint_f1"], 4)
        print(f"\n==> balanced_joint_f1 delta: {bjf_delta:+.4f}")
        if bjf_delta > 0:
            print("    Phase 6 speaker model IMPROVED over baseline.")
        else:
            print("    Phase 6 speaker model did NOT improve; keep baseline.")

    print("\nDone.")
