"""
Sweep agglomerative clustering threshold t on the EXACT production pipeline boxes.
Caches production boxes once (takes ~18s), then sweeps t in seconds.
"""
import sys
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.common import score as scorer, split_ids, write_jsonl
from speakers.enhanced_spatial_clusterer import cluster_assign
import pipeline.inference_yolo as p_yolo

# Monkey-patch or hook into predict_sequence_hybrid_yolo to capture all_seq_boxes
original_cluster_assign = p_yolo.cluster_assign

boxes_cache = {}

def hook_cluster_assign(boxes, page_sizes, **kwargs):
    # Just return char1 so we don't do work during extraction
    return ["char1"] * len(boxes)

p_yolo.cluster_assign = hook_cluster_assign

val_ids = split_ids("val")
ref_all = scorer.load_jsonl(str(ROOT / "dataset" / "development" / "labels.jsonl"))
ref = {k: v for k, v in ref_all.items() if k in set(val_ids)}

print(f"Extracting production boxes for {len(val_ids)} val sequences...")

# Let's write a function to get (boxes, page_sizes) for each sequence
# We can copy the exact logic from predict_sequence_hybrid_yolo up to line 282
from pipeline.inference_yolo import (
    image_paths, page_gray, _get_panels, detect_panels, ocr_page,
    story_prob, FILTER_THRESHOLD, PAGE_GATE, detect_bubbles, _box_in_bubble,
    clean_line, _find_panel, _classify_unassigned, preprocess_crop, _get_crop_reader,
    order_boxes, READING_RTL
)
from PIL import Image
import pandas as pd

def extract_prod_boxes(seq_id: str, yolo_conf=0.35, min_ocr_conf=0.25, iou_nms=0.45, postprocess=True, crop_recovery=True):
    img_paths = image_paths(seq_id)
    all_seq_boxes = []
    page_sizes = []

    for pg, img in enumerate(img_paths):
        panel_info = _get_panels(seq_id, pg)
        gray = page_gray(seq_id, pg)
        h, w = gray.shape
        size = (float(w), float(h))
        if panel_info:
            panels = panel_info["panels"]
        else:
            panels = detect_panels(gray, rtl=True)
        page_sizes.append(size)

        raw_boxes = [b for b in ocr_page(img) if b.get("conf", 1.0) >= min_ocr_conf]
        if not raw_boxes:
            continue

        rows = []
        for b in raw_boxes:
            box = b["box"]
            if isinstance(box[0], (list, tuple)):
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                x0, y0, x1, y1 = [float(v) for v in box]
            rows.append({
                "seq_id": seq_id, "page": pg,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "text": b["text"], "conf": b["conf"],
            })
        df = pd.DataFrame(rows)
        df["p_story"] = story_prob(df)
        df = df[df["p_story"] >= FILTER_THRESHOLD].reset_index(drop=True)
        if len(df) == 0 or df["p_story"].max() < PAGE_GATE:
            continue

        raw_bubbles = detect_bubbles(img, conf_thresh=yolo_conf)
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
                    area_b = (bx[2] - bx[0]) * (bx[3] - bx[1])
                    area_vb = (vbx[2] - vbx[0]) * (vbx[3] - vbx[1])
                    iou = inter / (area_b + area_vb - inter)
                    if iou > iou_nms or (inter / min(area_b, area_vb)) > 0.8:
                        keep = False
                        break
            if keep:
                valid_bubbles.append(b)

        bubble_members = {k: [] for k in range(len(valid_bubbles))}
        unassigned_indices = []

        for idx, row in df.iterrows():
            obox = [row.x0, row.y0, row.x1, row.y1]
            assigned = False
            for bk, binfo in enumerate(valid_bubbles):
                if _box_in_bubble(obox, binfo["bbox"]):
                    bubble_members[bk].append(idx)
                    assigned = True
                    break
            if not assigned:
                unassigned_indices.append(idx)

        containers = []
        im_pil = None

        for bk, ocr_idxs in bubble_members.items():
            bbox = valid_bubbles[bk]["bbox"]
            bw = bbox[2] - bbox[0]
            panel_id = _find_panel(bbox, panels)
            merged_text = ""

            is_cross_bleed = False
            if ocr_idxs and crop_recovery:
                for oi in ocr_idxs:
                    ow = df.iloc[oi].x1 - df.iloc[oi].x0
                    if ow > 1.35 * bw or (ow > 180 and bw < 140):
                        is_cross_bleed = True
                        break

            if ocr_idxs and not is_cross_bleed:
                ocr_idxs.sort(key=lambda i: (df.iloc[i].y0, df.iloc[i].x0))
                texts = [
                    (clean_line(df.iloc[i].text) if postprocess else df.iloc[i].text).strip()
                    for i in ocr_idxs
                ]
                merged_text = " ".join(t for t in texts if t)
            elif crop_recovery and (is_cross_bleed or not ocr_idxs) and valid_bubbles[bk]["conf"] >= 0.40:
                if im_pil is None:
                    im_pil = Image.open(img)
                bx0 = max(0, int(bbox[0] - 4))
                by0 = max(0, int(bbox[1] - 4))
                bx1 = min(int(w), int(bbox[2] + 4))
                by1 = min(int(h), int(bbox[3] + 4))
                if bx1 > bx0 + 8 and by1 > by0 + 8:
                    crop = im_pil.crop((bx0, by0, bx1, by1))
                    prep = preprocess_crop(crop, target_side=256)
                    reader = _get_crop_reader()
                    res = reader.reader.readtext(prep)
                    words = [
                        (clean_line(r[1]) if postprocess else r[1]).strip()
                        for r in res
                        if clean_line(r[1]).strip()
                    ]
                    words = [w for w in words if any(c.isalnum() for c in w)]
                    merged_text = " ".join(words)
                    if sum(c.isalpha() for c in merged_text) < 3:
                        merged_text = ""

            if not merged_text or not scorer.norm(merged_text):
                continue

            containers.append({
                "page": pg,
                "bbox": [float(v) for v in bbox],
                "text": merged_text,
                "panel": panel_id,
                "tag": "dialogue",
            })

        for u_idx in unassigned_indices:
            row = df.iloc[u_idx]
            t = (clean_line(row.text) if postprocess else row.text).strip()
            if not t or not scorer.norm(t):
                continue
            bbox = [float(row.x0), float(row.y0), float(row.x1), float(row.y1)]
            panel_id = _find_panel(bbox, panels)
            tag = _classify_unassigned(bbox, size)
            containers.append({
                "page": pg,
                "bbox": bbox,
                "text": t,
                "panel": panel_id,
                "tag": tag,
            })

        if not containers:
            continue

        c_boxes = np.array([c["bbox"] for c in containers])
        c_order = order_boxes(c_boxes, panels, READING_RTL, size)
        for ord_idx in c_order:
            all_seq_boxes.append(containers[ord_idx])

    return all_seq_boxes, page_sizes


cache = {}
for i, sid in enumerate(val_ids, 1):
    cache[sid] = extract_prod_boxes(sid)
    print(f"[{i}/{len(val_ids)}] {sid} ({len(cache[sid][0])} boxes)")

print("\n--- Sweeping thresholds t on PRODUCTION boxes ---")
print(f"{'t':>6} | {'text_ord':>8} | {'bal_f1':>8} | {'spk_acc':>8} | {'joint_f1':>8} | {'coverage':>8}")
print("-" * 65)

# Sweep fine-grained
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]

best_f1 = -1
best_t = None
best_rep = None

for t in THRESHOLDS:
    preds = {}
    cfg = {"threshold": t, "merge_singletons": True, "max_clusters": 4}
    for sid in val_ids:
        boxes, page_sizes = cache[sid]
        if boxes:
            labels = cluster_assign(boxes, page_sizes, cfg=cfg)
            pages = [[], [], []]
            for b, lab in zip(boxes, labels):
                pages[b["page"]].append({"speaker": lab, "text": b["text"]})
        else:
            pages = [[], [], []]
        preds[sid] = pages

    hyp = {k: v for k, v in preds.items() if k in set(val_ids)}
    rep = scorer.evaluate(ref, hyp)["macro"]

    is_best = rep["balanced_joint_f1"] > best_f1
    if is_best:
        best_f1 = rep["balanced_joint_f1"]
        best_t = t
        best_rep = rep

    mark = " ◄ BEST" if is_best else ""
    print(f"{t:6.2f} | {rep['text_order_score']:8.4f} | {rep['balanced_joint_f1']:8.4f} | {rep['speaker_accuracy_on_matched']:8.4f} | {rep['joint_f1']:8.4f} | {rep['matched_token_coverage']:8.4f}{mark}")

print("=" * 65)
print(f"Optimal threshold for production: t = {best_t:.2f} with bal_f1 = {best_f1:.4f}")
