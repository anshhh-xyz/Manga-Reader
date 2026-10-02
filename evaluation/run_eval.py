# evaluation/run_eval.py
import argparse
import json

from evaluation.common import OUTPUTS, REFERENCES, RESULTS, score, split_ids

COLS = ["text_order_score", "balanced_joint_f1", "speaker_accuracy_on_matched",
        "joint_f1", "matched_token_coverage"]


def fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def append_row(name, split, macro, notes):
    if not RESULTS.exists():
        RESULTS.write_text(
            "# Results\n\n| experiment | split | text_order | balanced_joint_f1 | spk_acc_matched "
            "| joint_f1 | coverage | notes |\n|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8")
    row = f"| {name} | {split} | " + " | ".join(fmt(macro[c]) for c in COLS) + f" | {notes} |\n"
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="val", choices=["dev", "train", "val"])
    ap.add_argument("--notes", default="")
    ap.add_argument("--worst", type=int, default=5)
    args = ap.parse_args()

    ids = split_ids(args.split)
    all_refs = score.load_jsonl(REFERENCES, reference=True)
    refs = {i: all_refs[i] for i in ids}
    preds = {k: v for k, v in score.load_jsonl(args.pred).items() if k in refs}

    report = score.evaluate(refs, preds)
    OUTPUTS.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in args.name)
    (OUTPUTS / f"eval_{safe}_{args.split}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    append_row(args.name, args.split, report["macro"], args.notes)
    print(json.dumps(report["macro"], indent=2))
    if report["missing_sequences"]:
        print(f"WARNING: {len(report['missing_sequences'])} sequences missing from predictions (scored 0)")

    worst = sorted(report["sequences"].items(), key=lambda kv: kv[1]["text_order_score"])[:args.worst]
    print(f"\nWorst {args.worst} by text_order_score:")
    for sid, r in worst:
        print(f"  {sid}  text={r['text_order_score']:.3f}  pages={r['page_text_scores']}")


"""It takes your predictions → compares them with the reference labels → calculates scores → 
saves the results → shows the worst-performing sequences."""