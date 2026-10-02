import json
import random
from collections import defaultdict

from configs.config import (SEED, VAL_FRACTION, SPLITS_JSON, REF_TRAIN, REF_VAL)
from configs import data


def series_key(seq_id, image_paths):
    """Group sequences from the same manga series.
    ADJUST: return something shared by the same series (folder name, filename
    prefix, ...) once you find it while reading the data in step 7.
    Until then every sequence is its own group."""
    return seq_id


def make_split():
    groups = defaultdict(list)
    for seq_id in data.dev_ids():
        seq = data.get_sequence(seq_id)
        groups[series_key(seq_id, seq["image_paths"])].append(seq_id)

    keys = sorted(groups)                 # sort first so the shuffle is reproducible
    random.Random(SEED).shuffle(keys)

    total = sum(len(v) for v in groups.values())
    target_val = round(total * VAL_FRACTION)
    train, val = [], []
    for k in keys:
        (val if len(val) < target_val else train).extend(groups[k])
    return sorted(train), sorted(val)


def assert_no_leak(train, val):
    assert not set(train) & set(val), "sequence appears in both splits"
    train_imgs = {p for i in train for p in data.get_sequence(i)["image_paths"]}
    val_imgs = {p for i in val for p in data.get_sequence(i)["image_paths"]}
    assert not train_imgs & val_imgs, "an image appears in both splits"


def write_refs(ids, path):
    rows = data.load_reference_rows()
    with open(path, "w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    train, val = make_split()
    assert_no_leak(train, val)
    SPLITS_JSON.write_text(
        json.dumps({"seed": SEED, "train": train, "val": val}, indent=2), encoding="utf-8"
    )
    write_refs(train, REF_TRAIN)
    write_refs(val, REF_VAL)
    print(f"train: {len(train)} sequences | val: {len(val)} sequences")