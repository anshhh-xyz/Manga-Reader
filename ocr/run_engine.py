# ocr/run_engine.py
import argparse, json, time
from pathlib import Path
from configs.data import all_ids, get_sequence
from ocr.engines import ENGINES

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

try:
    from ocr.postprocess import clean_line
except Exception:
    import re
    def clean_line(t):
        return re.sub(r"\s+", " ", t).strip()


def page_paths(split, seq_id):
    # Adapter: if your data.py already returns the 3 ordered page paths, call that instead.
    d = ROOT / "dataset" / split / "images" / seq_id
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
    assert len(files) == 3, f"{seq_id}: expected 3 pages, got {len(files)}"
    return files


def naive_page_lines(boxes):
    """Top-to-bottom by box centre, then left-to-right. Single speaker."""
    boxes = sorted(boxes, key=lambda b: ((b.y0 + b.y1) / 2, b.x0))
    lines = []
    for b in boxes:
        t = clean_line(b.text)
        if t:
            lines.append({"speaker": "char1", "text": t})
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--split", default="development", choices=["development", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all sequences")
    args = ap.parse_args()

    engine = ENGINES[args.engine]()
    img_root = ROOT / "dataset" / args.split / "images"
    seq_ids = sorted(p.name for p in img_root.iterdir() if p.is_dir())
    if args.limit:
        seq_ids = seq_ids[:args.limit]

    cache = ROOT / "outputs" / "cache" / args.engine / args.split
    cache.mkdir(parents=True, exist_ok=True)
    out_path = ROOT / "outputs" / f"preds_{args.engine}_{args.split}.jsonl"
    times = []

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sid in enumerate(seq_ids, 1):
            cf = cache / f"{sid}.json"
            if cf.exists():
                pages = json.loads(cf.read_text(encoding="utf-8"))
            else:
                t0 = time.time()
                sequence = get_sequence(sid)
                pages = [
                    naive_page_lines(engine.read(str(p)))
                    for p in sequence["image_paths"]
                ]
                times.append(time.time() - t0)
                cf.write_text(json.dumps(pages), encoding="utf-8")
            f.write(json.dumps({"sequence_id": sid, "pages": pages}, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(seq_ids)}] {sid}")

    if times:
        print(f"avg {sum(times)/len(times):.1f}s per sequence (uncached ones)")
    print("wrote", out_path)


if __name__ == "__main__":
    main()