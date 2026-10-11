# backend/pipeline_service.py
"""
Manga Task Reader — Backend Inference Service
=============================================
Encapsulates the end-to-end reading pipeline using models in weights/:
1. Panel detection (XY-cut)
2. OCR extraction (PaddleOCR / precomputed cache)
3. Story filtering (weights/filter_model.joblib)
4. RTL Reading order inside panels
5. Character clustering (weights/pairwise_xgb.json + cluster_cfg.json)
6. Clean monochrome visualization generation (annotated panels + dialogue)
"""

import base64
import io
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from xgboost import XGBClassifier

from detection.panel_detector import detect_panels
from ordering.reading_order import order_boxes
from ocr.baseline import ocr_page
from ocr.postprocess import clean_line
from filtering.features import balloon_feats, text_feats, FEATURES
from speakers import cluster_assign

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"

# Global model caches
_FILTER_BUNDLE = None


def get_filter_bundle():
    global _FILTER_BUNDLE
    if _FILTER_BUNDLE is None:
        p = WEIGHTS / "filter_model.joblib"
        if not p.exists():
            p = ROOT / "outputs" / "filtering" / "filter_model.joblib"
        _FILTER_BUNDLE = joblib.load(str(p))
    return _FILTER_BUNDLE


def build_page_features(df_page: pd.DataFrame, gray: np.ndarray) -> pd.DataFrame:
    h, w = gray.shape
    b = df_page[["x0", "y0", "x1", "y1"]].to_numpy(dtype=float)
    bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    med_h = max(float(np.median(bh)), 1.0) if len(bh) else 1.0
    med_a = max(float(np.median(bw * bh)), 1.0) if len(bw) else 1.0
    bal = balloon_feats(gray, b)
    rows = []
    for i, (idx, r) in enumerate(df_page.iterrows()):
        x0, y0, x1, y1 = b[i]
        geom = [
            (x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
            bw[i] / w, bh[i] / h, bw[i] / max(bh[i], 1.0),
            min(x0, y0, w - x1, h - y1) / min(w, h),
            bh[i] / med_h,
            bw[i] * bh[i] / med_a,
            len(df_page)
        ]
        rows.append([idx] + geom + text_feats(r["text"], r["conf"]) + bal[i])
    X = pd.DataFrame(rows, columns=["_idx"] + FEATURES).set_index("_idx")
    return X.loc[df_page.index]


def filter_story_boxes(df_boxes: pd.DataFrame, gray: np.ndarray, threshold: float = 0.3) -> pd.DataFrame:
    if len(df_boxes) == 0:
        return df_boxes
    bundle = get_filter_bundle()
    X = build_page_features(df_boxes, gray)[bundle["features"]]
    p = np.zeros((len(df_boxes), 3))
    for col, class_id in enumerate(bundle["model"].classes_):
        p[:, int(class_id)] = bundle["model"].predict_proba(X)[:, col]
    p_story = 1 - p[:, 0]
    df_boxes = df_boxes.copy()
    df_boxes["p_story"] = p_story
    return df_boxes[df_boxes["p_story"] >= threshold].reset_index(drop=True)


def find_panel_index(bbox: List[float], panels: List[Tuple[int, int, int, int]]) -> int:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    for k, (x0, y0, x1, y1) in enumerate(panels):
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return k
    dists = [((cx - (p[0] + p[2]) / 2) ** 2 + (cy - (p[1] + p[3]) / 2) ** 2) for p in panels]
    return int(np.argmin(dists)) if dists else 0


def should_merge_lines(a: List[float], b: List[float], v_gap_mult: float = 2.2, h_overlap_mult: float = 0.15) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ha, hb = ay1 - ay0, by1 - by0
    h_line = min(ha, hb)
    h_max = max(ha, hb)
    if h_line <= 0:
        return False

    v_overlap = min(ay1, by1) - max(ay0, by0)
    h_overlap = min(ax1, bx1) - max(ax0, bx0)
    wa, wb = ax1 - ax0, bx1 - bx0
    w_min = min(wa, wb)
    w_max = max(wa, wb)

    # Same line horizontal continuation
    if v_overlap > 0.4 * h_line:
        h_gap = max(ax0, bx0) - min(ax1, bx1)
        return h_gap < 1.5 * h_line

    # Vertically stacked lines in same bubble
    v_gap = max(ay0, by0) - min(ay1, by1)
    if v_gap < v_gap_mult * h_max or v_gap < 45.0:
        if h_overlap > h_overlap_mult * w_min:
            return True
        ca_x = (ax0 + ax1) / 2
        cb_x = (bx0 + bx1) / 2
        if abs(ca_x - cb_x) < 0.65 * w_max:
            return True

    return False


def merge_balloons(df_boxes: pd.DataFrame, panels: list) -> pd.DataFrame:
    n = len(df_boxes)
    if n <= 1:
        return df_boxes

    panel_ids = [find_panel_index([r.x0, r.y0, r.x1, r.y1], panels) for _, r in df_boxes.iterrows()]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if panel_ids[i] == panel_ids[j]:
                box_i = [df_boxes.iloc[i].x0, df_boxes.iloc[i].y0, df_boxes.iloc[i].x1, df_boxes.iloc[i].y1]
                box_j = [df_boxes.iloc[j].x0, df_boxes.iloc[j].y0, df_boxes.iloc[j].x1, df_boxes.iloc[j].y1]
                if should_merge_lines(box_i, box_j):
                    parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged_rows = []
    for members in groups.values():
        members.sort(key=lambda idx: (df_boxes.iloc[idx].y0, df_boxes.iloc[idx].x0))
        cleaned_tokens = [clean_line(str(df_boxes.iloc[idx].text)).strip() for idx in members]
        joined_text = " ".join(t for t in cleaned_tokens if t).strip()
        if not joined_text:
            continue

        x0 = min(df_boxes.iloc[idx].x0 for idx in members)
        y0 = min(df_boxes.iloc[idx].y0 for idx in members)
        x1 = max(df_boxes.iloc[idx].x1 for idx in members)
        y1 = max(df_boxes.iloc[idx].y1 for idx in members)
        conf = float(np.mean([df_boxes.iloc[idx].conf for idx in members]))
        p_id = panel_ids[members[0]]

        merged_rows.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "text": joined_text,
            "conf": conf,
            "panel": p_id,
        })

    return pd.DataFrame(merged_rows) if merged_rows else df_boxes


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
    """Runs the full 3-page sequence reader pipeline on 3 image files."""
    if len(image_paths) != 3:
        raise ValueError(f"Exactly 3 images are required; received {len(image_paths)}.")

    all_seq_boxes = []
    page_sizes = []
    page_panels_list = []
    pil_images = []

    for pg, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        pil_images.append(img)
        w, h = img.size
        page_sizes.append((w, h))

        # 1. Panel detection (on-the-fly XY-cut)
        gray = np.array(img.convert("L"))
        panels = detect_panels(gray, rtl=True)
        page_panels_list.append(panels)

        # 2. OCR detection
        raw_ocr = ocr_page(path)
        if not raw_ocr:
            continue

        # 3. Prepare DataFrame for story filter
        rows = []
        for b in raw_ocr:
            box = b["box"]
            if isinstance(box[0], (list, tuple)):
                xs = [float(p[0]) for p in box]; ys = [float(p[1]) for p in box]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                x0, y0, x1, y1 = [float(v) for v in box]
            rows.append({
                "seq_id": seq_name,
                "page": pg,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "text": b.get("text", ""),
                "conf": b.get("conf", 1.0),
            })

        df = pd.DataFrame(rows)
        # Filter story dialogue (pass gray for balloon features)
        df_story = filter_story_boxes(df, gray, threshold=0.3)
        if len(df_story) == 0:
            continue

        # Merge fragmented lines within the same speech balloon
        df_story = merge_balloons(df_story, panels)

        # 4. RTL XY-cut Reading Order inside panels
        boxes_np = df_story[["x0", "y0", "x1", "y1"]].to_numpy()
        order_indices = order_boxes(boxes_np, panels, rtl=True, size=(w, h))

        for rank, i in enumerate(order_indices, start=1):
            t = clean_line(str(df_story.iloc[i]["text"])).strip()
            if t:
                b_coords = [
                    float(boxes_np[i][0]), float(boxes_np[i][1]),
                    float(boxes_np[i][2]), float(boxes_np[i][3])
                ]
                panel_id = find_panel_index(b_coords, panels)
                all_seq_boxes.append({
                    "page": pg,
                    "bbox": b_coords,
                    "text": t,
                    "panel": panel_id,
                    "tag": "dialogue",
                    "order_in_page": rank
                })

    # 5. Speaker Attribution (Enhanced Spatial Layout Clusterer)
    if all_seq_boxes:
        labels = cluster_assign(all_seq_boxes, page_sizes)
    else:
        labels = []

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
