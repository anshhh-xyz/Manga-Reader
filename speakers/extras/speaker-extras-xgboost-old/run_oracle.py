# speakers/run_oracle.py
import argparse, json, pickle
from .adapters import load_sequences, OUT, TAG_NARR
from .character_memory import labels_from_P

ap = argparse.ArgumentParser()
ap.add_argument("--t", type=float, default=None)
ap.add_argument("--merge", type=int, default=None)
ap.add_argument("--maxk", type=int, default=0)
ap.add_argument("--gold", action="store_true", help="control: use gold speakers on the same boxes")
ap.add_argument("--single", action="store_true", help="floor: every story box = char1")
a = ap.parse_args()

cfg = json.load(open(OUT / "cluster_cfg.json"))
if a.t is not None: cfg["threshold"] = a.t
if a.merge is not None: cfg["merge_singletons"] = bool(a.merge)
cfg["max_clusters"] = a.maxk or None
print("cfg:", cfg, "| gold control:", a.gold)

oof = pickle.load(open(OUT / "oof.pkl", "rb"))["oof"]
seqs = load_sequences()
all_ids = [json.loads(l)["sequence_id"] for l in open("dataset/development/labels.jsonl", encoding="utf-8")]
out = OUT / "oracle_gt_order_pred_speakers.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for s in all_ids:                                  # every dev sequence, even ones with no boxes
        boxes = seqs.get(s, [])
        story_n = sum(b["tag"] != TAG_NARR for b in boxes)
        if a.gold:
            labels = [b["speaker"] or "UNKNOWN" for b in boxes]
        elif a.single:
            labels = ["NARRATION" if b["tag"] == TAG_NARR else "char1" for b in boxes]
        elif story_n >= 2 and s in oof:
            labels = labels_from_P(boxes, oof[s], cfg)
        else:
            labels = ["NARRATION" if b["tag"] == TAG_NARR else "char1" for b in boxes]
        pages = [[], [], []]
        for b, lab in zip(boxes, labels):
            pages[b["page"]].append({"speaker": lab, "text": b["text"]})
        f.write(json.dumps({"sequence_id": s, "pages": pages}, ensure_ascii=False) + "\n")
print("wrote", out)