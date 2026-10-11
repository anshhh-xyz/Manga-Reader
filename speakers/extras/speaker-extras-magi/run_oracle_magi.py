"""Experiment 7b with Magi speakers: your aligned boxes + gold order + Magi-predicted speakers.

    python -m speakers_magi.run_oracle_magi --score magi_default
    python -m speakers_magi.run_oracle_magi --unsure common --maxk 5 --scope page --score magi_x

Writes outputs/speakers/oracle_gt_order_magi_speakers.jsonl in the same format as before.
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from . import common
from .core import DEFAULT_CFG, assign_labels

OUT_FILE = Path("outputs/speakers/oracle_gt_order_magi_speakers.jsonl")


def load_inputs():
    from speakers.adapter import TAG_NARR, load_sequences  # your existing aligned-box loader

    seqs = load_sequences()
    caches = {s: common.load_cache(s, "dev") for s in common.seq_ids("dev")}
    return seqs, caches, TAG_NARR


def build(cfg, seqs, caches, out_path=OUT_FILE, verbose=True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tot, missing = Counter(), []
    n_spk = []
    with open(out_path, "w", encoding="utf-8") as f:
        for s in common.seq_ids("dev"):
            boxes = seqs.get(s, [])
            cache = caches.get(s)
            if cache is not None and boxes:
                labels, st, k = assign_labels(boxes, cache, cfg)
                tot.update(st)
                n_spk.append(k)
            else:
                if boxes and cache is None:
                    missing.append(s)
                labels = ["NARRATION" if b["tag"] == cfg["narr_tag"] else "char1" for b in boxes]
            pages = [[], [], []]
            for b, lab in zip(boxes, labels):
                pages[b["page"]].append({"speaker": lab, "text": b["text"]})
            f.write(json.dumps({"sequence_id": s, "pages": pages}, ensure_ascii=False) + "\n")
    if verbose:
        n = max(tot["story"], 1)
        print("wrote %s" % out_path)
        print("story balloons %d | matched a Magi text box %.0f%% | linked to a character %.0f%% | "
              "nearest-character fallback %.0f%% | unsure %.0f%% | mean speakers/seq %.2f"
              % (tot["story"], 100 * tot["matched"] / n, 100 * tot["linked"] / n,
                 100 * tot["nearest"] / n, 100 * tot["unsure"] / n,
                 sum(n_spk) / max(len(n_spk), 1)))
        if missing:
            print("WARNING: no Magi cache for %d sequences (labelled char1): %s ..." % (len(missing), missing[:3]))
    return dict(tot), missing


def score(name, pred):
    """Run your evaluation.run_eval and pull out the two numbers."""
    r = subprocess.run([sys.executable, "-m", "evaluation.run_eval", "--name", name, "--pred", str(pred),
                        "--split", "dev"], capture_output=True, text=True)
    txt = r.stdout + r.stderr
    num = r"([0-9.eE+-]+)"
    b = re.search(r'"balanced_joint_f1":\s*' + num, txt)
    t = re.search(r'"text_order_score":\s*' + num, txt)
    if not b:
        print("could not read the score; run_eval said:\n" + txt[-600:])
    return (float(b.group(1)) if b else None), (float(t.group(1)) if t else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsure", choices=["prev", "common"], default=DEFAULT_CFG["unsure"])
    ap.add_argument("--maxk", type=int, default=DEFAULT_CFG["maxk"])
    ap.add_argument("--scope", choices=["global", "page"], default=DEFAULT_CFG["scope"])
    ap.add_argument("--nearest", type=int, default=int(DEFAULT_CFG["nearest"]))
    ap.add_argument("--thr", type=float, default=DEFAULT_CFG["match_thr"])
    ap.add_argument("--score", metavar="NAME", default=None, help="also score it with evaluation.run_eval")
    a = ap.parse_args()

    seqs, caches, tag_narr = load_inputs()
    cfg = {"unsure": a.unsure, "maxk": a.maxk, "scope": a.scope, "nearest": bool(a.nearest),
           "match_thr": a.thr, "narr_tag": tag_narr}
    print("cfg:", cfg)
    build(cfg, seqs, caches)
    if a.score:
        b, t = score(a.score, OUT_FILE)
        print("balanced_joint_f1 = %s | text_order_score = %s" % (b, t))
        print("(floor all-char1 0.2485 | your best layout model 0.3418 | gold-speaker control 0.6639)")


if __name__ == "__main__":
    main()
