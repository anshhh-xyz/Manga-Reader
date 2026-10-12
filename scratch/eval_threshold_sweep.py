"""
Clustering Threshold Sweep
==========================
Sweeps agglomerative clustering distance threshold t ∈ [0.35, 0.65]
on the 16 val sequences using the full production pipeline.

Goal: find the t that maximises balanced_joint_f1 without any retraining.

See: 'using threshold sweep.md' for full documentation.

Usage:
    python scratch/eval_threshold_sweep.py
"""
import sys
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(r"d:\Projects\BYTE\Manga Task Reader AML-2")
sys.path.insert(0, str(ROOT))

from evaluation.common import image_paths, score as scorer, split_ids, write_jsonl
from configs.config import FILTER_THRESHOLD, PAGE_GATE, READING_RTL, PANELS_PATH
from ocr.baseline import ocr_page
from ocr.postprocess import clean_line
from filtering.infer_filter import story_prob
from filtering.common import page_gray
from ordering.reading_order import order_boxes
from detection.bubble_ocr import detect_bubbles
from speakers.enhanced_spatial_clusterer import (
    load_model, extract_pair_features, cluster_probabilities, relabel
)

_panels_path = ROOT / PANELS_PATH if not Path(PANELS_PATH).is_absolute() else Path(PANELS_PATH)
_PANELS = None
TAG_NARR = "narration"


def get_panels(seq_id: str, page: int):
    global _PANELS
    if _PANELS is None:
        _PANELS = json.loads(_panels_path.read_text(encoding="utf-8")) if _panels_path.exists() else {}
    return _PANELS.get(f"{seq_id}|{page}")


def find_panel(bbox, panels):
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for k, (x0, y0, x1, y1) in enumerate(panels):
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return k
    dists = [((cx - (p[0] + p[2]) / 2) ** 2 + (cy - (p[1] + p[3]) / 2) ** 2) for p in panels]
    return int(np.argmin(dists)) if dists else -1


def box_in_bubble(ocr_box, bubble_box, margin=6.0):
    ox0, oy0, ox1, oy1 = ocr_box
    ocx, ocy = (ox0 + ox1) / 2.0, (oy0 + oy1) / 2.0
    bx0, by0, bx1, by1 = bubble_box
    return (bx0 - margin) <= ocx <= (bx1 + margin) and (by0 - margin) <= ocy <= (by1 + margin)


def is_sfx_or_junk(text: str) -> bool:
    t = text.strip()
    if re.match(r"^sfx\b", t, re.I):
        return True
    letters = re.sub(r"[^a-zA-Z]", "", t)
    valid_short = {"I", "NO", "OH", "AH", "OK", "HE", "WE", "ME", "MY", "GO", "DO", "SO",
                   "AM", "IS", "IT", "UP", "ON", "AT", "IN", "TO", "BY", "IF", "HI", "?", "!"}
    return len(letters) <= 2 and letters.upper() not in valid_short and t not in valid_short


def merge_unassigned(unassigned_rows: list, v_thresh=35.0):
    if not unassigned_rows:
        return []
    unassigned_rows.sort(key=lambda r: (r["panel"], r["y0"], r["x0"]))
    groups, cur = [], [unassigned_rows[0]]
    for r in unassigned_rows[1:]:
        p = cur[-1]
        if (r["panel"] == p["panel"] and -10 <= r["y0"] - p["y1"] <= v_thresh
                and (min(r["x1"], p["x1"]) - max(r["x0"], p["x0"]) > -20
                     or abs((r["x0"] + r["x1"]) / 2 - (p["x0"] + p["x1"]) / 2) < 80)):
            cur.append(r)
        else:
            groups.append(cur); cur = [r]
    groups.append(cur)
    merged = []
    for g in groups:
        txt = " ".join(i["text"] for i in g if i["text"])
        if not txt or is_sfx_or_junk(txt):
            continue
        merged.append({
            "page": g[0]["page"],
            "bbox": [min(i["bbox"][0] for i in g), min(i["bbox"][1] for i in g),
                     max(i["bbox"][2] for i in g), max(i["bbox"][3] for i in g)],
            "text": txt, "panel": g[0]["panel"], "tag": "dialogue"
        })
    return merged


# ── Pre-extract all val sequence boxes (OCR+YOLO) once so the sweep is fast ──

def extract_boxes_for_sequence(seq_id: str):
    """Returns (all_seq_boxes, page_sizes) for one sequence."""
    img_paths = image_paths(seq_id)
    all_boxes, page_sizes = [], []

    for pg, img in enumerate(img_paths):
        panel_info = get_panels(seq_id, pg)
        gray = page_gray(seq_id, pg)
        h, w = gray.shape
        size = (float(w), float(h))
        panels = panel_info["panels"] if panel_info else (
            __import__("detection.panel_detector", fromlist=["detect_panels"])
            .detect_panels(gray, rtl=True)
        )
        page_sizes.append(size)

        raw_boxes = [b for b in ocr_page(img) if b.get("conf", 1.0) >= 0.3]
        if not raw_boxes:
            continue

        rows = []
        for b in raw_boxes:
            box = b["box"]
            if isinstance(box[0], (list, tuple)):
                xs = [float(p[0]) for p in box]; ys = [float(p[1]) for p in box]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                x0, y0, x1, y1 = map(float, box)
            rows.append({"seq_id": seq_id, "page": pg,
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "text": b["text"], "conf": b["conf"]})

        df = pd.DataFrame(rows)
        df["p_story"] = story_prob(df)
        df = df[df["p_story"] >= FILTER_THRESHOLD].reset_index(drop=True)
        if len(df) == 0 or df["p_story"].max() < PAGE_GATE:
            continue

        raw_bubbles = detect_bubbles(img, conf_thresh=0.35)
        valid_bubbles = []
        for b in sorted(raw_bubbles, key=lambda x: -x["conf"]):
            bx = b["bbox"]
            keep = True
            for vb in valid_bubbles:
                vbx = vb["bbox"]
                ix0, iy0 = max(bx[0], vbx[0]), max(bx[1], vbx[1])
                ix1, iy1 = min(bx[2], vbx[2]), min(bx[3], vbx[3])
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                    ab = (bx[2] - bx[0]) * (bx[3] - bx[1])
                    av = (vbx[2] - vbx[0]) * (vbx[3] - vbx[1])
                    iou = inter / (ab + av - inter)
                    if iou > 0.45 or inter / min(ab, av) > 0.8:
                        keep = False; break
            if keep:
                valid_bubbles.append(b)

        bubble_members = {k: [] for k in range(len(valid_bubbles))}
        unassigned_items = []

        for idx, row in df.iterrows():
            obox = [row.x0, row.y0, row.x1, row.y1]
            assigned = False
            for bk, binfo in enumerate(valid_bubbles):
                if box_in_bubble(obox, binfo["bbox"]):
                    bubble_members[bk].append(idx); assigned = True; break
            if not assigned:
                t = clean_line(row.text).strip()
                if not is_sfx_or_junk(t):
                    pid = find_panel(obox, panels)
                    unassigned_items.append({
                        "page": pg, "bbox": obox,
                        "x0": row.x0, "y0": row.y0, "x1": row.x1, "y1": row.y1,
                        "text": t, "panel": pid,
                    })

        containers = []
        for bk, ocr_idxs in bubble_members.items():
            if not ocr_idxs:
                continue
            ocr_idxs.sort(key=lambda i: (df.iloc[i].y0, df.iloc[i].x0))
            texts = [clean_line(df.iloc[i].text).strip() for i in ocr_idxs]
            texts = [t for t in texts if not is_sfx_or_junk(t)]
            merged_text = " ".join(t for t in texts if t)
            if not merged_text or not scorer.norm(merged_text):
                continue
            bbox = valid_bubbles[bk]["bbox"]
            pid = find_panel(bbox, panels)
            containers.append({
                "page": pg, "bbox": [float(v) for v in bbox],
                "text": merged_text, "panel": pid, "tag": "dialogue"
            })

        containers.extend(merge_unassigned(unassigned_items))
        if not containers:
            continue

        c_boxes = np.array([c["bbox"] for c in containers])
        c_order = order_boxes(c_boxes, panels, READING_RTL, size)
        for ord_idx in c_order:
            all_boxes.append(containers[ord_idx])

    return all_boxes, page_sizes


def assign_with_threshold(boxes, page_sizes, t: float, max_k: int = 4):
    """Run clustering at a specific threshold t, return speaker labels."""
    model, _ = load_model()

    story_idx = [k for k, b in enumerate(boxes) if b.get("tag") != TAG_NARR]
    labels = ["NARRATION"] * len(boxes)
    if not story_idx:
        return labels

    sub_boxes = [boxes[k] for k in story_idx]
    if len(sub_boxes) == 1:
        labels[story_idx[0]] = "char1"
        return labels

    X, pairs = extract_pair_features(sub_boxes, page_sizes)
    n_feat = model.n_features_in_
    if X.shape[1] != n_feat:
        X = X[:, :n_feat]

    probs = model.predict_proba(X)[:, 1]
    P = np.eye(len(sub_boxes))
    for (i, j), p in zip(pairs, probs):
        P[i, j] = P[j, i] = p

    assigned = relabel(cluster_probabilities(P, t=t, merge_singletons=True, max_k=max_k))
    for k, name in zip(story_idx, assigned):
        labels[k] = name
    return labels


if __name__ == "__main__":
    val_ids = split_ids("val")
    ref_all = scorer.load_jsonl(str(ROOT / "dataset" / "development" / "labels.jsonl"))
    ref = {k: v for k, v in ref_all.items() if k in set(val_ids)}

    print(f"Pre-extracting boxes for {len(val_ids)} val sequences (once)...")
    cache = {}
    for i, sid in enumerate(val_ids, 1):
        cache[sid] = extract_boxes_for_sequence(sid)
        print(f"  [{i}/{len(val_ids)}] {sid} — {len(cache[sid][0])} boxes")

    # Sweep thresholds
    THRESHOLDS = [round(0.35 + 0.02 * i, 2) for i in range(16)]  # 0.35 to 0.65

    print(f"\nThreshold Sweep (t = {THRESHOLDS[0]:.2f} → {THRESHOLDS[-1]:.2f})")
    print("=" * 72)
    print(f"{'t':>5}  {'text_order':>10}  {'bal_f1':>8}  {'spk_acc':>8}  {'joint_f1':>9}  {'coverage':>9}")
    print("-" * 72)

    best_f1 = 0.0
    best_t = 0.50
    best_report = None
    results_rows = []

    for t in THRESHOLDS:
        preds = {}
        for sid in val_ids:
            boxes, page_sizes = cache[sid]
            if boxes:
                labels = assign_with_threshold(boxes, page_sizes, t=t)
                pages = [[], [], []]
                for b, lab in zip(boxes, labels):
                    pages[b["page"]].append({"speaker": lab, "text": b["text"]})
            else:
                pages = [[], [], []]
            preds[sid] = pages

        out_p = ROOT / "outputs" / f"preds_val_t{t:.2f}.jsonl"
        write_jsonl(preds, str(out_p))
        hyp = scorer.load_jsonl(str(out_p))
        hyp = {k: v for k, v in hyp.items() if k in set(val_ids)}
        report = scorer.evaluate(ref, hyp)["macro"]

        marker = " ◄" if report["balanced_joint_f1"] > best_f1 else ""
        print(
            f"{t:>5.2f}  "
            f"{report['text_order_score']:>10.4f}  "
            f"{report['balanced_joint_f1']:>8.4f}  "
            f"{report['speaker_accuracy_on_matched']:>8.4f}  "
            f"{report['joint_f1']:>9.4f}  "
            f"{report['matched_token_coverage']:>9.4f}{marker}"
        )

        if report["balanced_joint_f1"] > best_f1:
            best_f1 = report["balanced_joint_f1"]
            best_t = t
            best_report = report

        results_rows.append({"t": t, **report})

    print("=" * 72)
    print(f"\nBest threshold: t = {best_t:.2f}  →  balanced_joint_f1 = {best_f1:.4f}")
    print(f"Baseline (t=0.50):               balanced_joint_f1 = 0.3604")
    delta = best_f1 - 0.3604
    print(f"Delta vs P12 champion:           {delta:+.4f}")

    if best_report:
        print("\nFull metrics at best t:")
        for k in ["text_order_score", "balanced_joint_f1", "speaker_accuracy_on_matched",
                  "joint_f1", "matched_token_coverage"]:
            print(f"  {k:<35}: {best_report[k]:.4f}")

    # Save best t to cluster_cfg if it's an improvement
    cfg_path = ROOT / "weights" / "cluster_cfg.json"
    cfg = json.loads(cfg_path.read_text())
    old_t = cfg.get("threshold", 0.50)

    if best_t != old_t and best_f1 > 0.3604:
        cfg["threshold"] = best_t
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"\n✓ Updated cluster_cfg.json: threshold {old_t} → {best_t}")
    else:
        print(f"\n— cluster_cfg.json unchanged (t={old_t} retained)")
