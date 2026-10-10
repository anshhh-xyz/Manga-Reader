# detection/run_panels.py
import json
from configs.config import ALIGNED_PATH, PANELS_PATH
from filtering.dataset import load_aligned
from filtering.common import page_gray
from detection.panel_detector import detect_panels

df = load_aligned(ALIGNED_PATH)
out = {}
for (s, pg) in df[["seq_id", "page"]].drop_duplicates().itertuples(index=False):
    gray = page_gray(s, pg)
    h, w = gray.shape
    # cache with rtl=False so we get a direction-neutral panel list; Step 7 orders them
    out[f"{s}|{pg}"] = {"size": [w, h], "panels": [list(map(int, p)) for p in detect_panels(gray, rtl=False)]}
json.dump(out, open(PANELS_PATH, "w"))
n1 = sum(len(v["panels"]) == 1 for v in out.values())
print(f"{len(out)} pages, {n1} detected as a single panel")