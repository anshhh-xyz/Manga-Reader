# ordering/rescore.py
"""compares two reading-order methods to see which gives better final evaluation scores."""
import json, pandas as pd
from configs.config import OOF_PATH, PANELS_PATH, READING_RTL, FILTER_THRESHOLD, PAGE_GATE
from filtering.tune_threshold import build_predictions, write_jsonl, dev_seq_ids, naive_order
from evaluation.common import REFERENCES, score
from ordering.reading_order import order_boxes

df = pd.read_pickle(OOF_PATH)
P = json.load(open(PANELS_PATH))
ids = dev_seq_ids()
cols = ["x0", "y0", "x1", "y1"]

def xycut_order(s, pg, g):
    p = P[f"{s}|{pg}"]
    return order_boxes(g[cols].to_numpy(), p["panels"], READING_RTL, p["size"])

for name, fn in [("naive", naive_order), ("xy-cut", xycut_order)]:
    write_jsonl(build_predictions(df, FILTER_THRESHOLD, PAGE_GATE, fn, ids), "outputs/_tmp.jsonl")
    refs = score.load_jsonl(REFERENCES, reference=True)
    preds = score.load_jsonl("outputs/_tmp.jsonl")
    print(name, score.evaluate(refs, preds)["macro"])