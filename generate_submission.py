#!/usr/bin/env python3
"""
generate_submission.py
======================
Generates the final competition submission file (submission.jsonl) by running
the end-to-end manga reading pipeline on the test set.

The script uses the same pipeline as the FastAPI backend (backend/pipeline_service.py):
  Panel Detection -> OCR -> Story Filter -> RTL Reading Order -> Speaker Attribution

Output format per line:
  {"sequence_id": "seq_XXXX", "pages": [[{speaker, text}, ...], [...], [...]]}

Usage:
    python generate_submission.py
    python generate_submission.py --set test --out submission.jsonl
    python generate_submission.py --set dev  --out outputs/predictions_dev.jsonl

Re-run this script any time you change the backend, OCR, or filtering code.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Parse args
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Generate competition submission JSONL.")
parser.add_argument("--set",     default="test",   choices=["test", "dev", "val", "train"],
                    help="Which split to run on (default: test).")
parser.add_argument("--out",     default="submission.jsonl",
                    help="Output JSONL path (default: submission.jsonl in project root).")
parser.add_argument("--conf",    type=float, default=0.3,
                    help="Minimum OCR confidence threshold (default: 0.3).")
parser.add_argument("--verbose", action="store_true",
                    help="Print stack traces on errors.")
args = parser.parse_args()

# --------------------------------------------------------------------------- #
# Locate image sequences
# --------------------------------------------------------------------------- #
SPLITS_JSON = ROOT / "dataset" / "splits.json"
IMG_DIRS = {
    "test":  ROOT / "dataset" / "test"  / "images",
    "dev":   ROOT / "dataset" / "development" / "images",
    "train": ROOT / "dataset" / "development" / "images",
    "val":   ROOT / "dataset" / "development" / "images",
}

seq_ids = []
img_root = IMG_DIRS[args.set]

if SPLITS_JSON.exists():
    splits = json.loads(SPLITS_JSON.read_text(encoding="utf-8"))
    seq_ids = splits.get(args.set, [])

if not seq_ids:
    # Fallback: auto-discover from image directory
    seq_ids = sorted(d.name for d in img_root.iterdir() if d.is_dir())

print(f"[generate_submission] Running on {len(seq_ids)} sequences from '{args.set}' split.")
print(f"[generate_submission] Images root : {img_root}")
print(f"[generate_submission] Output      : {args.out}\n")

# --------------------------------------------------------------------------- #
# Load pipeline (lazy — models load on first call)
# --------------------------------------------------------------------------- #
from backend.pipeline_service import run_pipeline_on_images

# --------------------------------------------------------------------------- #
# Run pipeline on each sequence
# --------------------------------------------------------------------------- #
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

results = []
errors = []

for n, seq_id in enumerate(seq_ids, 1):
    seq_dir = img_root / seq_id
    if not seq_dir.exists():
        print(f"  [{n:3}/{len(seq_ids)}] SKIP {seq_id} — directory not found")
        results.append({"sequence_id": seq_id, "pages": [[], [], []]})
        continue

    # Gather exactly 3 images (PNG preferred, fallback to JPEG)
    img_paths = sorted(seq_dir.glob("*.png"))
    if len(img_paths) < 3:
        img_paths = sorted(seq_dir.glob("*.jpg")) + sorted(seq_dir.glob("*.jpeg"))
    img_paths = sorted(img_paths)[:3]

    if len(img_paths) != 3:
        print(f"  [{n:3}/{len(seq_ids)}] WARN {seq_id} — found {len(img_paths)} images (expected 3)")
        results.append({"sequence_id": seq_id, "pages": [[], [], []]})
        errors.append(seq_id)
        continue

    try:
        pipeline_out = run_pipeline_on_images(img_paths, seq_name=seq_id)

        # Strip visual-only fields; keep only speaker + text per box
        clean_pages = []
        for page_boxes in pipeline_out["pages"]:
            clean_pages.append([
                {"speaker": b["speaker"], "text": b["text"]}
                for b in page_boxes
            ])

        results.append({"sequence_id": seq_id, "pages": clean_pages})

        n_boxes = sum(len(p) for p in clean_pages)
        spkrs = pipeline_out["stats"]["characters_detected"]
        print(f"  [{n:3}/{len(seq_ids)}] OK   {seq_id} — {n_boxes} boxes | chars: {spkrs}")

    except Exception as exc:
        import traceback
        print(f"  [{n:3}/{len(seq_ids)}] ERR  {seq_id} — {exc}")
        if args.verbose:
            traceback.print_exc()
        results.append({"sequence_id": seq_id, "pages": [[], [], []]})
        errors.append(seq_id)

# --------------------------------------------------------------------------- #
# Write JSONL
# --------------------------------------------------------------------------- #
with open(out_path, "w", encoding="utf-8") as fout:
    for r in results:
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")

total_boxes = sum(len(b) for r in results for p in r["pages"] for b in [p])

print(f"\n[generate_submission] Done.")
print(f"  Sequences processed  : {len(results)}")
print(f"  Sequences with errors: {len(errors)}")
print(f"  Total dialogue boxes : {total_boxes}")
print(f"  Output written to    : {out_path.resolve()}")

if errors:
    print(f"\n  Failed sequences ({len(errors)}):")
    for e in errors:
        print(f"    - {e}")
