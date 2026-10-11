"""One command: try every setting against balanced_joint_f1 and keep the best.

    python -m speakers_magi.grid

Saves the winner to outputs/magi/best_cfg.json (read by pipeline_hook.py) and its predictions to
outputs/magi/best_oracle.jsonl. Tuned on the dev set only; never on test.
"""
import itertools
import json
import shutil

from . import common
from .core import DEFAULT_CFG
from .run_oracle_magi import OUT_FILE, build, load_inputs, score


def main():
    seqs, caches, tag_narr = load_inputs()
    have = sum(1 for c in caches.values() if c is not None)
    print("Magi cache present for %d / %d dev sequences" % (have, len(caches)))
    if have == 0:
        raise SystemExit("no cache found: run `python -m speakers_magi.run_magi --split dev` first")

    rows = []
    for unsure, maxk, scope, nearest in itertools.product(["prev", "common"], [0, 4, 6], ["global", "page"], [False, True]):
        cfg = {**DEFAULT_CFG, "unsure": unsure, "maxk": maxk, "scope": scope, "nearest": nearest, "narr_tag": tag_narr}
        build(cfg, seqs, caches, verbose=False)
        name = "magi_%s_k%d_%s_n%d" % (unsure, maxk, scope, int(nearest))
        b, t = score(name, OUT_FILE)
        rows.append((b if b is not None else -1.0, t, cfg))
        print("%-28s balanced_joint_f1=%s  text_order_score=%s" % (name, b, t))
        if b is not None and b >= max(r[0] for r in rows):
            shutil.copy(OUT_FILE, common.OUT / "best_oracle.jsonl")

    rows.sort(key=lambda r: -r[0])
    print("\nTop 3:")
    for b, t, cfg in rows[:3]:
        print("  %.4f  %s" % (b, {k: v for k, v in cfg.items() if k != "narr_tag"}))
    best = rows[0][2]
    common.OUT.mkdir(parents=True, exist_ok=True)
    with open(common.OUT / "best_cfg.json", "w") as f:
        json.dump({k: v for k, v in best.items() if k != "narr_tag"}, f, indent=2)
    print("\nsaved outputs/magi/best_cfg.json")
    print("reference: floor 0.2485 | your best layout model 0.3418 | gold-speaker control 0.6639")


if __name__ == "__main__":
    main()
