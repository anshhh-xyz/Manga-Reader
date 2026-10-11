# ordering/measure_directions.py
"""Compare naive vs XY-cut reading order using Kendall Tau on gold annotations.

Run with:
    python -m ordering.measure_directions

Requires:
    outputs/alignment/aligned_boxes.json   (from alignment/merge.py)
    outputs/panels_dev.json                (from detection/run_panels.py)
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau

from configs.config import ALIGNED_PATH, PANELS_PATH
from filtering.dataset import load_aligned
from ordering.reading_order import order_boxes

ROOT = Path(__file__).resolve().parents[1]


def mean_tau(df_hi, panels, fn):
    taus = []
    for (s, pg), g in df_hi.groupby(["seq_id", "page"]):
        key = f"{s}|{pg}"
        if key not in panels or len(g) < 3:
            continue
        g = g.reset_index(drop=True)
        order = fn(g, panels[key])
        rank = np.empty(len(g))
        rank[order] = np.arange(len(g))
        t = kendalltau(rank, g.order_idx.to_numpy()).correlation
        if not np.isnan(t):
            taus.append(t)
    return (float(np.mean(taus)), len(taus)) if taus else (float("nan"), 0)


def main():
    aligned_path = Path(ALIGNED_PATH) if Path(ALIGNED_PATH).is_absolute() else ROOT / ALIGNED_PATH
    panels_path  = Path(PANELS_PATH)  if Path(PANELS_PATH).is_absolute()  else ROOT / PANELS_PATH

    if not aligned_path.exists():
        raise FileNotFoundError(f"Aligned boxes not found: {aligned_path}")
    if not panels_path.exists():
        raise FileNotFoundError(
            f"Panel map not found: {panels_path}\n"
            "Run:  python -m detection.run_panels  first."
        )

    df = load_aligned(str(aligned_path))
    hi = df[(df.tier == "high") & (~df.ambiguous) & df.order_idx.notna()].copy()
    panels = json.loads(panels_path.read_text(encoding="utf-8"))

    cols = ["x0", "y0", "x1", "y1"]

    def naive(g, p):
        return list(np.lexsort((g.x0.to_numpy(), g.y0.to_numpy())))

    tau, n = mean_tau(hi, panels, naive)
    print(f"naive top-to-bottom:   tau={tau:.4f}  pages={n}")

    for rtl in (True, False):
        def xycut(g, p, _rtl=rtl):
            return order_boxes(g[cols].to_numpy(), p["panels"], _rtl, p["size"])
        tau, n = mean_tau(hi, panels, xycut)
        print(f"XY-cut rtl={rtl}:        tau={tau:.4f}  pages={n}")


if __name__ == "__main__":
    main()