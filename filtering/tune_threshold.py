# filtering/tune_threshold.py
import json, itertools
from pathlib import Path
import numpy as np
import pandas as pd

from configs.config import OOF_PATH
from ocr.postprocess import clean_line
from evaluation.common import score  # score module loaded from dataset/score.py

ROOT = Path(__file__).resolve().parents[1]
TMP_PATH = ROOT / "outputs" / "_tmp.jsonl"
REF_PATH = ROOT / "dataset" / "development" / "labels.jsonl"


def dev_seq_ids(path=None):
    path = path or REF_PATH
    return [json.loads(l)["sequence_id"] for l in open(path, encoding="utf-8")]


def naive_order(seq, pg, g):
    """Top-to-bottom, left-to-right. g must be reset_index(drop=True) before call."""
    return list(np.lexsort((g.x0.to_numpy(), g.y0.to_numpy())))


def build_predictions(df, thr, gate, order_fn, seq_ids, speaker="char1"):
    groups = {}
    for (s, pg), g in df.groupby(["seq_id", "page"]):
        groups[(s, pg)] = g.reset_index(drop=True)

    out = []
    for s in seq_ids:
        pages = []
        for pg in range(3):
            g = groups.get((s, pg))
            if g is None:
                pages.append([])
                continue
            ps = 1 - g.p_non
            g = g[ps >= thr].reset_index(drop=True)
            if len(g) == 0 or (1 - g.p_non).max() < gate:
                pages.append([])
                continue
            lines = []
            for i in order_fn(s, pg, g):
                t = clean_line(g.iloc[i].text)
                if t.strip():
                    lines.append({"speaker": speaker, "text": t})
            pages.append(lines)
        out.append({"sequence_id": s, "pages": pages})
    return out


def write_jsonl(preds, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    oof_path = Path(OOF_PATH)
    if not oof_path.is_absolute():
        oof_path = ROOT / oof_path

    if not oof_path.exists():
        raise FileNotFoundError(
            f"OOF file not found: {oof_path}\n"
            "Run:  python -m filtering.train_filter  first to generate it."
        )

    df = pd.read_pickle(oof_path)
    ids = dev_seq_ids()

    # Load references once — score.load_jsonl returns dict {seq_id: pages}
    refs = score.load_jsonl(str(REF_PATH), reference=True)

    rows = []
    for thr, gate in itertools.product(
        [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.0, 0.6, 0.8, 0.9],
    ):
        if gate and gate < thr:
            continue

        preds_list = build_predictions(df, thr, gate, naive_order, ids)
        write_jsonl(preds_list, TMP_PATH)

        # score.load_jsonl returns dict {seq_id: pages}
        preds = score.load_jsonl(str(TMP_PATH))

        # score.evaluate returns {"version":..., "macro": {...}, "sequences": {...}}
        report = score.evaluate(refs, preds)
        macro = report["macro"]

        row = (thr, gate, macro["text_order_score"], macro["balanced_joint_f1"])
        rows.append(row)
        print(row)

    res = pd.DataFrame(rows, columns=["thr", "gate", "text_order", "joint_f1"])
    print("\nTop 10 by text_order_score:")
    print(res.sort_values("text_order", ascending=False).head(10))