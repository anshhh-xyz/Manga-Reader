# ordering/rescore.py
"""Compare naive vs XY-cut ordering using final evaluation scores.

Run with:
    python -m ordering.rescore

Requires:
    outputs/filtering/filter_oof.pkl    (from filtering/train_filter.py)
    outputs/panels_dev.json             (from detection/run_panels.py)
"""
import json
import pandas as pd
from pathlib import Path

from configs.config import (
    OOF_PATH, PANELS_PATH, READING_RTL,
    FILTER_THRESHOLD, PAGE_GATE,
)
from filtering.tune_threshold import build_predictions, write_jsonl, dev_seq_ids, naive_order
from evaluation.common import REFERENCES, score
from ordering.reading_order import order_boxes

ROOT = Path(__file__).resolve().parents[1]


def main():
    oof_path    = Path(OOF_PATH)    if Path(OOF_PATH).is_absolute()    else ROOT / OOF_PATH
    panels_path = Path(PANELS_PATH) if Path(PANELS_PATH).is_absolute() else ROOT / PANELS_PATH
    tmp_path    = ROOT / "outputs" / "_tmp.jsonl"

    if not oof_path.exists():
        raise FileNotFoundError(
            f"OOF file not found: {oof_path}\n"
            "Run:  python -m filtering.train_filter  first."
        )
    if not panels_path.exists():
        raise FileNotFoundError(
            f"Panel map not found: {panels_path}\n"
            "Run:  python -m detection.run_panels  first."
        )

    df  = pd.read_pickle(str(oof_path))
    P   = json.loads(panels_path.read_text(encoding="utf-8"))
    ids = dev_seq_ids()

    # Load references once — score.load_jsonl returns {seq_id: pages}
    refs = score.load_jsonl(str(REFERENCES), reference=True)

    cols = ["x0", "y0", "x1", "y1"]

    def xycut_order(s, pg, g):
        key = f"{s}|{pg}"
        p   = P.get(key, {"panels": None, "size": [1, 1]})
        return order_boxes(g[cols].to_numpy(), p["panels"], READING_RTL, p["size"])

    for name, fn in [("naive", naive_order), ("xy-cut", xycut_order)]:
        preds_list = build_predictions(df, FILTER_THRESHOLD, PAGE_GATE, fn, ids)
        write_jsonl(preds_list, tmp_path)

        preds  = score.load_jsonl(str(tmp_path))
        report = score.evaluate(refs, preds)
        print(f"{name:10s}  {report['macro']}")


if __name__ == "__main__":
    main()