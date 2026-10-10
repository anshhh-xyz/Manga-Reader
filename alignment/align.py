"""matches the OCR-detected text boxes to the ground-truth label lines using the matching logic from score.py..
Usage:  python -m alignment.align development
Outputs: outputs/alignment/aligned_boxes.json, outputs/alignment/unmatched_lines.json

Why use the scorer’s matcher: “matched” in my supervision then means exactly “matched” in your final score. For example, tokens under 3 characters need an exact match, and longer ones need ≥0.6 similarity.

How it works, per box:

Score the box against every gold line, and every adjacent pair of lines (a box can straddle two balloons).
precision = w / box_tokens is how much of the box is explained.
coverage = w / line_tokens is how much of the line the box covers.
A pair only wins if it beats the best single line by more than 0.10 precision. Otherwise the single line wins, because a full-line box also scores 1.0 against any pair containing it.
Tier: high ≥ 0.85, medium 0.60–0.85, none < 0.60 (a non-story negative).
Ambiguity: if several different lines tie (repeated “HUH?”/”NO!”), the box is flagged ambiguous. A page-level Hungarian assignment (linear_sum_assignment) then resolves it one-to-one by precision plus position. Ambiguous boxes stay excluded from training even after resolving. Resolving only matters for the ceiling.
Order index: the gold token position of the first matched token. This gives token-level order.
"""
import json
import sys
from pathlib import Path

from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset"))
import score as S  # noqa: E402  (the scorer, used as a library)

OUT = ROOT / "outputs" / "alignment"
HIGH, MED = 0.85, 0.60
PAIR_MARGIN = 0.10   # a 2-line candidate must beat the best single line by this
TIE_EPS = 0.02       # singles within this precision of the best are "tied"


# ---------------------------- ADAPTERS ----------------------------
# Only these three functions touch score.py. If Step 0 showed different shapes, edit here.
def toks(text, speaker="x"):
    """text -> list of score.py tokens."""
    return list(S.tokens([{"speaker": speaker, "text": text}]))


def align(gold_toks, box_toks):
    """-> list of (gold_position, weight). gold_position is the index into gold_toks."""
    out, last = [], -1
    for g, _b, w in S.align_tokens(gold_toks, box_toks):
        if isinstance(g, int):
            gp = g
        else:  # token objects: find the next occurrence after the previous match
            gp = next((i for i in range(last + 1, len(gold_toks)) if gold_toks[i] == g), last + 1)
        last = gp
        out.append((gp, float(w)))
    return out
# ------------------------------------------------------------------


def score_candidate(G, offsets, line_ids, box_toks):
    gold = [t for i in line_ids for t in G[i]]
    pairs = align(gold, box_toks)
    if not pairs:
        return {"lines": line_ids, "precision": 0.0, "coverage": 0.0, "first_pos": None}
    w = sum(p[1] for p in pairs)
    first = min(p[0] for p in pairs)
    return {
        "lines": line_ids,
        "precision": w / len(box_toks),
        "coverage": w / max(1, len(gold)),
        "first_pos": offsets[line_ids[0]] + first,  # global token index on the page
    }


def align_page(boxes, gold_lines):
    L = len(gold_lines)
    G = [toks(l["text"], l["speaker"]) for l in gold_lines]
    offsets, run = [], 0
    for g in G:
        offsets.append(run)
        run += len(g)

    recs, single_prec = [], []
    for b in boxes:
        bt = toks(b["text"])
        rec = {"box": b, "n_tokens": len(bt), "lines": None, "precision": 0.0,
               "coverage": 0.0, "first_pos": None, "ambiguous": False}
        sp = [0.0] * L
        if bt and L:
            singles = [score_candidate(G, offsets, (i,), bt) for i in range(L)]
            sp = [s["precision"] for s in singles]
            best = max(singles, key=lambda s: (s["precision"], s["coverage"]))
            if L > 1:
                pairs = [score_candidate(G, offsets, (i, i + 1), bt) for i in range(L - 1)]
                bp = max(pairs, key=lambda s: (s["precision"], s["coverage"]))
                if bp["precision"] > best["precision"] + PAIR_MARGIN:
                    best = bp
            rec.update(lines=best["lines"], precision=best["precision"],
                       coverage=best["coverage"], first_pos=best["first_pos"])
            if len(best["lines"]) == 1 and best["precision"] >= MED:
                tied = [i for i in range(L) if sp[i] >= best["precision"] - TIE_EPS]
                rec["ambiguous"] = len(tied) > 1
                rec["tied"] = tied
        single_prec.append(sp)
        recs.append(rec)

    # --- resolve ambiguous boxes one-to-one (Hungarian) ---
    amb = [k for k, r in enumerate(recs) if r["ambiguous"]]
    if amb:
        cols = sorted({i for k in amb for i in recs[k]["tied"]})
        n = max(1, len(recs))
        cost = []
        for k in amb:
            row = []
            for i in cols:
                if i in recs[k]["tied"]:
                    pos_pen = abs(k / n - i / max(1, L - 1))  # box rank vs line rank
                    row.append((1 - single_prec[k][i]) + 0.5 * pos_pen)
                else:
                    row.append(10.0)
            cost.append(row)
        ri, ci = linear_sum_assignment(cost)
        for r_, c_ in zip(ri, ci):
            k, line = amb[r_], cols[c_]
            if cost[r_][c_] < 10.0:
                recs[k]["lines"] = (line,)
                recs[k]["first_pos"] = offsets[line]

    out = []
    for r in recs:
        p = r["precision"]
        tier = "high" if p >= HIGH else "medium" if p >= MED else "none"
        if r["lines"] is None or tier == "none":
            tier, r["lines"], speaker, tag = "none", None, None, "non-story"
        else:
            speaker = gold_lines[r["lines"][0]]["speaker"]
            tag = "narration" if speaker == "NARRATION" else "dialogue"
        out.append({
            "idx": r["box"]["idx"], "poly": r["box"]["poly"], "ocr_text": r["box"]["text"],
            "conf": r["box"]["conf"], "n_tokens": r["n_tokens"],
            "lines": list(r["lines"]) if r["lines"] else None,
            "precision": round(p, 4), "coverage": round(r["coverage"], 4),
            "tier": tier, "ambiguous": bool(r["ambiguous"]), "tag": tag,
            "speaker": speaker, "speaker_train": speaker not in (None, "UNKNOWN"),
            "order_index": r["first_pos"] if tier != "none" else None,
        })
    return out


def main(split="development"):
    cache = json.loads((ROOT / "outputs" / "ocr_boxes" / f"{split}.json").read_text(encoding="utf-8"))
    labels = {}
    for line in (ROOT / "dataset" / split / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            labels[d["sequence_id"]] = d["pages"]

    all_boxes, unmatched = [], []
    for sid, labs in labels.items():
        for p in range(3):
            res = align_page(cache[sid]["pages"][p], labs[p])
            covered = set()
            for r in res:
                r.update(sequence_id=sid, page=p)
                if r["lines"]:
                    covered.update(r["lines"])
            all_boxes.extend(res)
            for i, ln in enumerate(labs[p]):
                if i not in covered:
                    unmatched.append({"sequence_id": sid, "page": p, "line": i,
                                      "speaker": ln["speaker"], "text": ln["text"]})
        print("aligned", sid)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aligned_boxes.json").write_text(json.dumps(all_boxes), encoding="utf-8")
    (OUT / "unmatched_lines.json").write_text(json.dumps(unmatched, indent=1), encoding="utf-8")
    print(f"{len(all_boxes)} boxes, {len(unmatched)} gold lines with no box -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "development")