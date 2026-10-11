"""Paths and small helpers shared by the scripts. Edit the constants here if your layout differs."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("outputs/magi")
DEV_IMG = Path("dataset/development/images")
TEST_IMG = Path("dataset/test/images")
LABELS = Path("dataset/development/labels.jsonl")
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def page_paths(seq, split):
    """The 3 page images of a sequence, sorted by file name.

    EDIT HERE if your data.py returns the pages in sequences.json order and that differs from
    sorted file names. The page order must match the `page` index of your aligned boxes.
    """
    root = DEV_IMG if split == "dev" else TEST_IMG
    return sorted(p for p in (root / seq).iterdir() if p.suffix.lower() in IMG_EXT)[:3]


def seq_ids(split):
    if split == "dev":
        with open(LABELS, encoding="utf-8") as f:
            return [json.loads(line)["sequence_id"] for line in f if line.strip()]
    return sorted(p.name for p in TEST_IMG.iterdir() if p.is_dir())


def cache_path(seq, split):
    return OUT / split / ("%s.json" % seq)


def load_cache(seq, split):
    p = cache_path(seq, split)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_page(path):
    """Magi's model card reads pages as grayscale converted back to RGB."""
    return np.array(Image.open(path).convert("L").convert("RGB"))
