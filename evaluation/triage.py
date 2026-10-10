# evaluation/triage.py
import argparse, json, re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "dataset" / "development" / "labels.jsonl"


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["sequence_id"]] = r["pages"]
    return out


def toks(pages):
    t = []
    for pg in pages:
        for ln in pg:
            t += re.findall(r"[a-z0-9']+", ln["text"].lower())
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    gold, pred = load(GOLD), load(args.pred)
    rows, labels = [], Counter()
    bad_empty = missed_pages = 0

    for sid, gp in gold.items():
        pp = pred.get(sid)
        if pp is None:
            continue
        for g, p in zip(gp, pp):
            if not g and p:
                bad_empty += 1          # invented text on an empty gold page (scores 0)
            if g and not p:
                missed_pages += 1
        g, p = toks(gp), toks(pp)
        if not g:
            lab = "INVENTED-TEXT (gold all empty)" if p else "ok (both empty)"
            labels[lab] += 1
            continue
        bag = sum((Counter(g) & Counter(p)).values())
        brec, bprec = bag / len(g), (bag / len(p) if p else 0.0)
        sm = SequenceMatcher(None, g, p, autojunk=False)
        orec = sum(m.size for m in sm.get_matching_blocks()) / len(g)
        if len(p) > 1.5 * len(g):
            lab = "FILTER (too much predicted)"
        elif brec < 0.6:
            lab = "OCR (missed text)"
        elif brec - orec > 0.15:
            lab = "ORDER"
        else:
            lab = "ok / speakers"
        labels[lab] += 1
        rows.append((orec, sid, len(g), len(p), brec, bprec, lab))

    rows.sort()
    print(f"{'seq':24}{'gold':>6}{'pred':>6}{'bagR':>7}{'bagP':>7}{'ordR':>7}  triage")
    for orec, sid, ng, np_, brec, bprec, lab in rows:
        print(f"{sid:24}{ng:>6}{np_:>6}{brec:>7.2f}{bprec:>7.2f}{orec:>7.2f}  {lab}")

    n = max(len(rows), 1)
    print("\nmean bag recall   :", round(sum(r[4] for r in rows) / n, 3))
    print("mean bag precision:", round(sum(r[5] for r in rows) / n, 3))
    print("mean ordered recall:", round(sum(r[0] for r in rows) / n, 3))
    print("pages with text predicted on EMPTY gold page:", bad_empty)
    print("pages with nothing predicted but gold has text:", missed_pages)
    print("missing from predictions:", len(set(gold) - set(pred)))
    print("\ntriage counts:", dict(labels))


if __name__ == "__main__":
    main()