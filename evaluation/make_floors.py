# evaluation/make_floors.py
import argparse

from evaluation.common import OUTPUTS, REFERENCES, score, split_ids, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", choices=["dev", "train", "val"])
args = ap.parse_args()

ids = split_ids(args.split)
refs = score.load_jsonl(REFERENCES, reference=True)

empty = {i: [[], [], []] for i in ids}
gt_char1 = {i: [[{"speaker": "char1", "text": r["text"]} for r in page] for page in refs[i]]
            for i in ids}

write_jsonl(empty, OUTPUTS / f"floor_empty_{args.split}.jsonl")
write_jsonl(gt_char1, OUTPUTS / f"floor_gt_char1_{args.split}.jsonl")
print("wrote floors to outputs/")


"""tells us how metrics in score.py change with no output, different outputs, and different character assignments.
no output+no char and correct output+same chars"""