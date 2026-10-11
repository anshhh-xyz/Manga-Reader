# detection/run_panels.py
"""Detect panels for every (sequence, page) in the aligned dataset.

Run with:
    python -m detection.run_panels

Output:  outputs/panels_dev.json
"""
import json
from pathlib import Path

from configs.config import ALIGNED_PATH, PANELS_PATH
from filtering.dataset import load_aligned
from filtering.common import page_gray
from detection.panel_detector import detect_panels

ROOT = Path(__file__).resolve().parents[1]


def main():
    aligned_path = Path(ALIGNED_PATH) if Path(ALIGNED_PATH).is_absolute() else ROOT / ALIGNED_PATH
    panels_path  = Path(PANELS_PATH)  if Path(PANELS_PATH).is_absolute()  else ROOT / PANELS_PATH

    if not aligned_path.exists():
        raise FileNotFoundError(
            f"Aligned boxes not found: {aligned_path}\n"
            "Run the alignment step (alignment/merge.py) first."
        )

    panels_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_aligned(str(aligned_path))
    out = {}

    for s, pg in df[["seq_id", "page"]].drop_duplicates().itertuples(index=False):
        gray = page_gray(s, pg)
        h, w = gray.shape
        # rtl=False keeps panel list direction-neutral; reading direction is
        # applied separately in ordering/reading_order.py
        panels = detect_panels(gray, rtl=False)
        out[f"{s}|{pg}"] = {
            "size":   [w, h],
            "panels": [list(map(int, p)) for p in panels],
        }

    with panels_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    n1 = sum(len(v["panels"]) == 1 for v in out.values())
    print(f"Processed {len(out)} pages — {n1} detected as single panel")
    print(f"Saved panel map to: {panels_path}")


if __name__ == "__main__":
    main()