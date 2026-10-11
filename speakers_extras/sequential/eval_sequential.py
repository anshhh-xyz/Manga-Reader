# speakers/eval_sequential.py
"""
Evaluate the Sequential Dialogue Tracker on the dev set.

Usage:
    python -m speakers.eval_sequential               # grid search + save best config
    python -m speakers.eval_sequential --quick       # single run with default/saved config

Saves best hyperparameters to: weights/sequential_cfg.json
Prints a comparison table vs the XGBoost baseline (balanced_joint_f1 = 0.3141).
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ── Competition scorer ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(ROOT))
from evaluation.common import score as scorer, split_ids, write_jsonl
from speakers.sequential_tracker import sequential_assign, DEFAULT_CFG, _embed, load_cfg
from speakers.adapters import load_sequences, page_sizes as get_page_sizes

# ── Paths ─────────────────────────────────────────────────────────────────────
DEV_LABELS  = ROOT / "dataset" / "development" / "labels.jsonl"
OUT_PRED    = ROOT / "outputs" / "predictions_seq_dev.jsonl"
WEIGHTS_CFG = ROOT / "weights" / "sequential_cfg.json"
BASELINE_BJF = 0.3141  # Phase 5 XGBoost end-to-end balanced_joint_f1


def precompute_data():
    """Load sequences, precompute embeddings, and load gold references."""
    sequences = load_sequences()
    print(f"[eval_sequential] Loaded {len(sequences)} sequences from aligned boxes.")
    print("[eval_sequential] Precomputing SBERT embeddings...")
    seq_embs = {}
    for seq_id, boxes in sequences.items():
        story_boxes = [b for b in boxes if b.get("tag", "") != "narration"]
        texts = [str(b.get("text") or "") for b in story_boxes]
        seq_embs[seq_id] = _embed(texts)

    references = scorer.load_jsonl(DEV_LABELS, reference=True)
    all_dev = split_ids("dev")
    return sequences, seq_embs, references, all_dev


def run_on_dev(cfg: dict, sequences: dict, seq_embs: dict, references: dict, all_dev: list) -> dict:
    """
    Run sequential_assign on every dev sequence.
    Returns scorer macro dict.
    """
    predictions = {}
    for seq_id, boxes in sequences.items():
        sizes = get_page_sizes(seq_id)
        stripped = [
            {k: v for k, v in b.items() if k != "speaker"}
            for b in boxes
        ]
        labels = sequential_assign(stripped, sizes, cfg=cfg, precomputed_embs=seq_embs.get(seq_id))
        pages = [[], [], []]
        for b, lbl in zip(boxes, labels):
            pg = b.get("page", 0)
            if 0 <= pg < 3:
                pages[pg].append({"speaker": lbl, "text": b.get("text", "")})
        predictions[seq_id] = pages

    for sid in all_dev:
        if sid not in predictions:
            predictions[sid] = [[], [], []]

    report = scorer.evaluate(references, predictions)
    return report["macro"]


def grid_search(sequences, seq_embs, references, all_dev) -> dict:
    """Grid search over key hyperparameters. Returns best cfg dict."""
    param_grid = {
        "alpha":                 [0.5, 0.7, 1.0],
        "beta":                  [0.1, 0.3, 0.5],
        "new_speaker_threshold": [0.15, 0.20, 0.25, 0.30, 0.35],
        "max_speakers":          [3, 4, 5],
        "window":                [3, 5, None],
    }

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    print(f"\n[eval_sequential] Grid search: {total} combinations\n")
    print(f"  {'#':>4}  {'alpha':>5}  {'beta':>5}  {'thr':>5}  {'maxk':>5}  {'win':>5}  {'bjf':>7}  {'text_ord':>9}")
    print("  " + "-" * 62)

    best_bjf = -1.0
    best_cfg = DEFAULT_CFG.copy()

    for n, combo in enumerate(combos, 1):
        cfg = dict(zip(keys, combo))
        try:
            macro = run_on_dev(cfg, sequences, seq_embs, references, all_dev)
        except Exception as e:
            print(f"  [{n:4}/{total}] ERROR: {e}")
            continue

        bjf      = macro.get("balanced_joint_f1", 0.0) or 0.0
        text_ord = macro.get("text_order_score", 0.0) or 0.0
        win_str  = str(cfg["window"]) if cfg["window"] else "all"

        if bjf > best_bjf:
            best_bjf = bjf
            best_cfg = cfg.copy()
            print(f" *[{n:4}/{total}] {cfg['alpha']:5.1f}  {cfg['beta']:5.1f}  {cfg['new_speaker_threshold']:5.2f}  "
                  f"{cfg['max_speakers']:5}  {win_str:>5}  {bjf:7.4f}  {text_ord:9.4f} (NEW BEST)")
        elif n % 50 == 0 or n == total:
            print(f"  [{n:4}/{total}] {cfg['alpha']:5.1f}  {cfg['beta']:5.1f}  {cfg['new_speaker_threshold']:5.2f}  "
                  f"{cfg['max_speakers']:5}  {win_str:>5}  {bjf:7.4f}  {text_ord:9.4f}")

    print(f"\n[eval_sequential] Best balanced_joint_f1 = {best_bjf:.4f}  (XGBoost baseline = {BASELINE_BJF})")
    delta = best_bjf - BASELINE_BJF
    sign  = "+" if delta >= 0 else ""
    print(f"[eval_sequential] Delta vs baseline       = {sign}{delta:.4f}")
    return best_cfg


def single_run(cfg: dict, sequences, seq_embs, references, all_dev, save_pred=True):
    """Run once with given config and print results."""
    print(f"\n[eval_sequential] Running with config: {cfg}")

    predictions = {}
    for seq_id, boxes in sequences.items():
        sizes = get_page_sizes(seq_id)
        stripped = [
            {k: v for k, v in b.items() if k != "speaker"}
            for b in boxes
        ]
        labels = sequential_assign(stripped, sizes, cfg=cfg, precomputed_embs=seq_embs.get(seq_id))
        pages = [[], [], []]
        for b, lbl in zip(boxes, labels):
            pg = b.get("page", 0)
            if 0 <= pg < 3:
                pages[pg].append({"speaker": lbl, "text": b.get("text", "")})
        predictions[seq_id] = pages

    for sid in all_dev:
        if sid not in predictions:
            predictions[sid] = [[], [], []]

    if save_pred:
        write_jsonl(predictions, str(OUT_PRED))

    report = scorer.evaluate(references, predictions)
    macro = report["macro"]

    bjf = macro.get('balanced_joint_f1', 0) or 0.0
    print(f"\n  text_order_score   : {macro.get('text_order_score', 0):.4f}")
    print(f"  balanced_joint_f1  : {bjf:.4f}  (XGBoost baseline = {BASELINE_BJF})")
    print(f"  spk_acc_matched    : {macro.get('speaker_accuracy_on_matched', 0):.4f}")
    print(f"  joint_f1           : {macro.get('joint_f1', 0):.4f}")
    print(f"  coverage           : {macro.get('matched_token_coverage', 0):.4f}")
    delta = bjf - BASELINE_BJF
    sign  = "+" if delta >= 0 else ""
    print(f"\n  Improvement vs XGBoost baseline: {sign}{delta:.4f}")
    return macro


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate Sequential Dialogue Tracker on dev set.")
    ap.add_argument("--quick", action="store_true",
                    help="Single run with saved/default config (skip grid search).")
    args = ap.parse_args()

    sequences, seq_embs, references, all_dev = precompute_data()

    if args.quick:
        cfg = load_cfg()
        single_run(cfg, sequences, seq_embs, references, all_dev, save_pred=True)
    else:
        best_cfg = grid_search(sequences, seq_embs, references, all_dev)
        WEIGHTS_CFG.parent.mkdir(parents=True, exist_ok=True)
        WEIGHTS_CFG.write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
        print(f"\n[eval_sequential] Saved best config -> {WEIGHTS_CFG}")
        print("\n[eval_sequential] Final evaluation with best config:")
        single_run(best_cfg, sequences, seq_embs, references, all_dev, save_pred=True)
        print(f"\n[eval_sequential] Predictions written -> {OUT_PRED}")
        print("[eval_sequential] Ready to re-run submission: python generate_submission.py")
