# filtering/tune_threshold.py
import json, itertools, numpy as np, pandas as pd
from configs.config import OOF_PATH
from ocr.postprocess import clean_line          # ADAPT: your Phase 1 cleanup fn
from evaluation.common import score      # ADAPT: fn(pred_path)->dict with
                                                # "text_order_score","balanced_joint_f1"

def dev_seq_ids(path="dataset/development/labels.jsonl"):
    return [json.loads(l)["sequence_id"] for l in open(path, encoding="utf-8")]

def naive_order(seq, pg, g):
    return list(np.lexsort((g.x0.to_numpy(), g.y0.to_numpy())))

def build_predictions(df, thr, gate, order_fn, seq_ids, speaker="char1"):
    groups = {k: g for k, g in df.groupby(["seq_id", "page"])}
    out = []
    for s in seq_ids:
        pages = []
        for pg in range(3):
            g = groups.get((s, pg))
            if g is None:
                pages.append([]); continue
            ps = 1 - g.p_non
            g = g[ps >= thr]
            if len(g) == 0 or (1 - g.p_non).max() < gate:
                pages.append([]); continue
            lines = []
            for i in order_fn(s, pg, g):
                t = clean_line(g.iloc[i].text)
                if t.strip():
                    lines.append({"speaker": speaker, "text": t})
            pages.append(lines)
        out.append({"sequence_id": s, "pages": pages})
    return out

def write_jsonl(preds, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    df = pd.read_pickle(OOF_PATH)
    ids = dev_seq_ids()
    rows = []
    for thr, gate in itertools.product([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                       [0.0, 0.6, 0.8, 0.9]):
        if gate and gate < thr:
            continue
        write_jsonl(build_predictions(df, thr, gate, naive_order, ids), "outputs/_tmp.jsonl")
        preds = score.load_jsonl("outputs/_tmp.jsonl")
        refs = score.load_jsonl(
            "dataset/development/labels.jsonl",
            reference=True,
        )
        r = score.evaluate(refs, preds)
        rows.append((thr, gate, r["text_order_score"], r["balanced_joint_f1"]))
        print(rows[-1])
    res = pd.DataFrame(rows, columns=["thr", "gate", "text_order", "joint_f1"])
    print(res.sort_values("text_order", ascending=False).head(10))