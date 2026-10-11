# backend/pipeline_service.py
"""
Manga Task Reader — Backend Inference Service  [v2: YOLO Hybrid Pipeline]
=========================================================================
Delegates to the winning Hybrid YOLO-Guided Container Pipeline:
1. YOLOv8 bubble detection  (weights/bubble_yolo.pt)
2. Full-page OCR + story filter  (weights/filter_model.joblib)
3. Container aggregation + smart NARRATION tagging
4. Panel-aware RTL reading order
5. Phase 5 Pairwise XGBoost speaker attribution  (weights/pairwise_xgb.json)
6. Clean monochrome visualization generation (annotated panels + dialogue)
"""

import base64
import io
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ── Core pipeline (best-performing combination) ──
from pipeline.inference_yolo import predict_sequence_hybrid_yolo, _get_panels
from detection.panel_detector import detect_panels
from ocr.baseline import ocr_page

ROOT = Path(__file__).resolve().parents[1]





def render_annotated_page(img: Image.Image, panels: list, boxes: list, page_idx: int) -> str:
    """Renders a clean black & white manga overlay showing panels and labeled text bubbles."""
    annotated = img.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    # 1. Draw panels: Crisp dashed/thin gray borders with subtle panel markers
    for idx, (px0, py0, px1, py1) in enumerate(panels):
        # Draw clean panel border
        draw.rectangle([px0, py0, px1, py1], outline="#71717a", width=2)
        # Panel indicator tag
        tag_w, tag_h = 36, 18
        draw.rectangle([px0, py0, px0 + tag_w, py0 + tag_h], fill="#18181b")
        draw.text((px0 + 4, py0 + 2), f"P{idx+1}", fill="#ffffff")

    # 2. Draw speech bubbles & speakers
    for b in boxes:
        x0, y0, x1, y1 = [int(v) for v in b["bbox"]]
        speaker = b.get("speaker", "char1")
        order_num = b.get("order_num", 1)

        # Bubble bounding box (high contrast black border with white backdrop)
        draw.rectangle([x0, y0, x1, y1], outline="#09090b", width=3)

        # Character Badge on top-left of box
        badge_text = f"#{order_num} {speaker}"
        badge_w = max(len(badge_text) * 8 + 10, 50)
        badge_h = 20
        badge_y0 = max(0, y0 - badge_h)
        badge_y1 = badge_y0 + badge_h

        # Solid ink-black badge with crisp white text
        draw.rectangle([x0, badge_y0, x0 + badge_w, badge_y1], fill="#09090b", outline="#ffffff", width=1)
        draw.text((x0 + 4, badge_y0 + 3), badge_text, fill="#ffffff")

    # Encode to base64 PNG
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def run_pipeline_on_images(image_paths: List[Path], seq_name: str = "seq_custom") -> Dict[str, Any]:
    """
    Run the full 3-page YOLO hybrid pipeline on 3 image files.
    Saves images to a temp location so the pipeline can OCR them,
    then runs predict_sequence_hybrid_yolo() for the best result.
    """
    import tempfile, shutil
    if len(image_paths) != 3:
        raise ValueError(f"Exactly 3 images are required; received {len(image_paths)}.")

    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    page_sizes = [(img.width, img.height) for img in pil_images]

    # ── Detect panels for visualization ──
    page_panels_list = []
    for img in pil_images:
        gray = np.array(img.convert("L"))
        panel_info = None  # backend uses on-the-fly panel detection
        panels = detect_panels(gray, rtl=True)
        page_panels_list.append(panels)

    # ── Run the YOLO Hybrid Pipeline ──
    # predict_sequence_hybrid_yolo takes a seq_id and looks up dataset paths.
    # For custom images we run the core logic directly.
    from filtering.common import page_gray as _page_gray
    from pipeline.inference_yolo import (
        _classify_unassigned, _box_in_bubble, _find_panel,
        ocr_page as _ocr_page, story_prob, clean_line,
        detect_bubbles, order_boxes, FILTER_THRESHOLD, PAGE_GATE, READING_RTL,
    )
    import pandas as pd
    from speakers import cluster_assign

    all_seq_boxes = []

    for pg, (img, path) in enumerate(zip(pil_images, image_paths)):
        gray = np.array(img.convert("L"))
        panels = page_panels_list[pg]
        W, H = page_sizes[pg]
        size = (float(W), float(H))

        raw_boxes = [b for b in _ocr_page(path) if b.get("conf", 1.0) >= 0.30]
        if not raw_boxes:
            continue

        rows = []
        for b in raw_boxes:
            box = b["box"]
            if isinstance(box[0], (list, tuple)):
                xs = [float(p[0]) for p in box]; ys = [float(p[1]) for p in box]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                x0, y0, x1, y1 = [float(v) for v in box]
            rows.append({"seq_id": seq_name, "page": pg,
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "text": b["text"], "conf": b["conf"]})
        df = pd.DataFrame(rows)
        df["p_story"] = story_prob(df)
        df = df[df["p_story"] >= FILTER_THRESHOLD].reset_index(drop=True)
        if len(df) == 0 or df["p_story"].max() < PAGE_GATE:
            continue

        # YOLO bubble detection + NMS
        raw_bubbles = detect_bubbles(path, conf_thresh=0.35)
        valid_bubbles = []
        for b in sorted(raw_bubbles, key=lambda x: -x["conf"]):
            bx = b["bbox"]; keep = True
            for vb in valid_bubbles:
                vbx = vb["bbox"]
                ix0, iy0 = max(bx[0], vbx[0]), max(bx[1], vbx[1])
                ix1, iy1 = min(bx[2], vbx[2]), min(bx[3], vbx[3])
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1-ix0)*(iy1-iy0)
                    iou = inter / ((bx[2]-bx[0])*(bx[3]-bx[1]) + (vbx[2]-vbx[0])*(vbx[3]-vbx[1]) - inter)
                    if iou > 0.45 or (inter / min((bx[2]-bx[0])*(bx[3]-bx[1]), (vbx[2]-vbx[0])*(vbx[3]-vbx[1]))) > 0.8:
                        keep = False; break
            if keep:
                valid_bubbles.append(b)

        bubble_members = {k: [] for k in range(len(valid_bubbles))}
        unassigned = []
        for idx, row in df.iterrows():
            obox = [row.x0, row.y0, row.x1, row.y1]
            assigned = False
            for bk, binfo in enumerate(valid_bubbles):
                if _box_in_bubble(obox, binfo["bbox"]):
                    bubble_members[bk].append(idx); assigned = True; break
            if not assigned:
                unassigned.append(idx)

        containers = []
        for bk, ocr_idxs in bubble_members.items():
            if not ocr_idxs: continue
            ocr_idxs.sort(key=lambda i: (df.iloc[i].y0, df.iloc[i].x0))
            texts = [clean_line(df.iloc[i].text).strip() for i in ocr_idxs]
            merged = " ".join(t for t in texts if t)
            if not merged: continue
            bbox = valid_bubbles[bk]["bbox"]
            containers.append({"page": pg, "bbox": [float(v) for v in bbox],
                               "text": merged, "panel": _find_panel(bbox, panels), "tag": "dialogue"})

        for u_idx in unassigned:
            row = df.iloc[u_idx]
            t = clean_line(row.text).strip()
            if not t: continue
            bbox = [float(row.x0), float(row.y0), float(row.x1), float(row.y1)]
            tag = _classify_unassigned(bbox, size)
            containers.append({"page": pg, "bbox": bbox, "text": t,
                               "panel": _find_panel(bbox, panels), "tag": tag})

        if not containers: continue
        c_boxes = np.array([c["bbox"] for c in containers])
        for ord_idx in order_boxes(c_boxes, panels, READING_RTL, size):
            all_seq_boxes.append(containers[ord_idx])

    # ── Speaker Attribution ──
    labels = cluster_assign(all_seq_boxes, page_sizes) if all_seq_boxes else []
    for b, lab in zip(all_seq_boxes, labels):
        b["speaker"] = lab

    # 6. Group into pages & chronological script
    pages_output = [[], [], []]
    script_output = []
    global_order = 1

    for b in all_seq_boxes:
        pg = b["page"]
        b["order_num"] = global_order
        pages_output[pg].append({
            "order": global_order,
            "speaker": b["speaker"],
            "text": b["text"],
            "bbox": [round(v, 1) for v in b["bbox"]],
            "panel": b["panel"] + 1,
        })
        script_output.append({
            "order": global_order,
            "page": pg + 1,
            "panel": b["panel"] + 1,
            "speaker": b["speaker"],
            "text": b["text"]
        })
        global_order += 1

    # 7. Generate Annotated Visual Overlays
    annotated_images = []
    for pg in range(3):
        pg_boxes = [b for b in all_seq_boxes if b["page"] == pg]
        annotated_b64 = render_annotated_page(pil_images[pg], page_panels_list[pg], pg_boxes, pg)
        annotated_images.append(annotated_b64)

    # 8. Compute Sequence Statistics
    speakers_detected = sorted(list({b["speaker"] for b in all_seq_boxes}))
    total_panels = sum(len(p) for p in page_panels_list)
    total_words = sum(len(b["text"].split()) for b in all_seq_boxes)

    return {
        "sequence_id": seq_name,
        "pages": pages_output,
        "script": script_output,
        "annotated_images": annotated_images,
        "stats": {
            "total_dialogue_bubbles": len(all_seq_boxes),
            "total_panels": total_panels,
            "total_words": total_words,
            "characters_detected": speakers_detected,
            "character_count": len(speakers_detected)
        }
    }
