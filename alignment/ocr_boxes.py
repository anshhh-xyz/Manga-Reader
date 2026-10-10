"""Run EasyOCR on every page of a split and cache boxes to outputs/ocr_boxes/<split>.json
Usage:  python -m alignment.ocr_boxes development
"""
import json
import sys
from pathlib import Path

import easyocr

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset"
OUT = ROOT / "outputs" / "ocr_boxes"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def _entry(seqs, seq_id):
    """Find this sequence's entry in sequences.json, whatever the container shape."""
    if isinstance(seqs, dict):
        if seq_id in seqs:
            return seqs[seq_id]
        for v in seqs.values():
            if isinstance(v, list):
                seqs = v
                break
    if isinstance(seqs, list):
        for e in seqs:
            if isinstance(e, dict) and e.get("sequence_id") == seq_id:
                return e
    return None


def find_pages(split, seq_id, seqs):
    """Return the 3 image paths in the order given by sequences.json (fallback: sorted filenames)."""
    folder = DS / split / "images" / seq_id
    e = _entry(seqs, seq_id)
    names = None
    if isinstance(e, list):
        names = e
    elif isinstance(e, dict):
        for k in ("pages", "images", "image_paths", "files"):
            if isinstance(e.get(k), list):
                names = e[k]
                break
    if names:
        paths = []
        for n in names:
            p = Path(str(n))
            cands = [p, folder / p.name, DS / split / p]
            paths.append(next((c for c in cands if c.is_file()), None))
        if all(paths):
            return paths
    return sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT)


def main(split):
    seqs = json.loads((DS / "sequences.json").read_text(encoding="utf-8"))
    seq_ids = sorted(p.name for p in (DS / split / "images").iterdir() if p.is_dir())
    reader = easyocr.Reader(["en"], gpu=True)  # falls back to CPU if no CUDA

    result = {}
    for n, sid in enumerate(seq_ids, 1):
        pages = find_pages(split, sid, seqs)
        assert len(pages) == 3, f"{sid}: expected 3 pages, got {len(pages)}"
        page_boxes = []
        for pth in pages:
            raw = reader.readtext(str(pth),  x_ths=0.4, y_ths=0.3)
            boxes = []
            for item in raw:
                poly, text = item[0], item[1]
                conf = float(item[2]) if len(item) > 2 else 1.0  # paragraph mode has no confidence
                pts = [[int(x), int(y)] for x, y in poly]
                boxes.append({"poly": pts, "text": text.strip(), "conf": conf})
            # naive reading order: top, then left
            boxes.sort(key=lambda b: (min(p[1] for p in b["poly"]), min(p[0] for p in b["poly"])))
            for i, b in enumerate(boxes):
                b["idx"] = i
            page_boxes.append(boxes)
        result[sid] = {"images": [str(p) for p in pages], "pages": page_boxes}
        print(f"[{n}/{len(seq_ids)}] {sid}: {[len(b) for b in page_boxes]} boxes")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{split}.json").write_text(json.dumps(result), encoding="utf-8")
    print("saved", OUT / f"{split}.json")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "development")