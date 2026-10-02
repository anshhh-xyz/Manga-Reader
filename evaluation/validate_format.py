# evaluation/validate_format.py
import argparse
import sys

from evaluation.common import score, split_ids


def validate(pred_path, expected_ids):
    preds = score.load_jsonl(pred_path)          # raises ValueError on any format problem
    expected = set(expected_ids)
    missing = sorted(expected - preds.keys())
    extra = sorted(preds.keys() - expected)
    if missing or extra:
        raise ValueError(f"missing {len(missing)} ids {missing[:3]}, extra {len(extra)} ids {extra[:3]}")
    return len(preds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--set", default="test", choices=["test", "dev", "train", "val"])
    args = ap.parse_args()
    try:
        n = validate(args.pred, split_ids(args.set))
    except ValueError as e:
        print(f"INVALID: {e}")
        sys.exit(1)
    print(f"VALID: {n} sequences, format OK for '{args.set}'")



"""validate_format.py makes sure your prediction JSONL is structurally correct and contains 
exactly the sequences it is supposed to contain — before you try to score it."""