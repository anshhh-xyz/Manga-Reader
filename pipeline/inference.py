from ocr.postprocess import clean_line, join_wrapped   
import argparse
import unicodedata
from evaluation.common import OUTPUTS, image_paths, score, split_ids, write_jsonl
from ocr.baseline import ocr_page

CACHE = OUTPUTS / "ocr_cache" / "paddle"
from configs.config import FILTER_THRESHOLD, PAGE_GATE, READING_RTL
from filtering.infer_filter import story_prob
from ordering.reading_order import order_boxes
from detection.panel_detector import detect_panels

# df_boxes: one row per OCR box for ONE sequence page, columns as above
df_boxes["p_story"] = story_prob(df_boxes)
for (seq, pg), g in df_boxes.groupby(["seq_id", "page"]):
    g = g[g.p_story >= FILTER_THRESHOLD]
    if len(g) == 0 or g.p_story.max() < PAGE_GATE:
        page_lines = []
    else:
        gray = page_gray(seq, pg); h, w = gray.shape
        panels = detect_panels(gray, rtl=False)
        order = order_boxes(g[["x0","y0","x1","y1"]].to_numpy(), panels, READING_RTL, (w, h))
        page_lines = [{"speaker": "char1", "text": clean_text(g.iloc[i].text)} for i in order]

def clean(text):
    text = "".join(c for c in text if not (unicodedata.category(c) == "Cc" and not c.isspace()))
    return text.strip()


def naive_order(boxes):
    # Phase 1 only: top-to-bottom, then left-to-right. Real reading order comes in Phase 4.
    return sorted(boxes, key=lambda b: (b["box"][1], b["box"][0]))

def predict_sequence(seq_id, min_conf, pp=False, join=False):
    pages = []
    for img in image_paths(seq_id):
        boxes = [b for b in ocr_page(img, CACHE) if b["conf"] >= min_conf]
        texts = [clean(b["text"]) for b in naive_order(boxes)]
        if pp:
            texts = [clean_line(t) for t in texts]
        if join:
            texts = join_wrapped(texts)
        pages.append([{"speaker": "char1", "text": t} for t in texts if score.norm(t)])
    return pages


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["dev", "train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--join-hyphens", action="store_true")
    args = ap.parse_args()

    ids = split_ids(args.set)
    rows = {}
    for n, sid in enumerate(ids, 1):
        rows[sid] = predict_sequence(sid, args.min_conf, args.postprocess, args.join_hyphens)
        print(f"[{n}/{len(ids)}] {sid}")
    write_jsonl(rows, args.out)
    print(f"wrote {args.out}")






"""Gets manga pages for each sequence.
Runs PaddleOCR to detect and read text.
Removes low-confidence OCR results using --min-conf.
Sorts text top-to-bottom, then left-to-right.
Optionally cleans OCR mistakes with --postprocess.
Optionally joins hyphenated lines with --join-hyphens.
Temporarily labels all text as char1 because speaker detection comes later.
Saves everything as JSONL for evaluation with score.py.

So basically:

Images → OCR → filter → clean → order → JSONL → scoring."""