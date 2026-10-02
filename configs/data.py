import json
from pathlib import Path

from configs.config import DATASET, SEQUENCES_JSON, DEV_REFERENCES


def _load_sequence_index():
    """Return {sequence_id: [3 absolute image Paths]} in the listed page order."""
    raw = json.loads(SEQUENCES_JSON.read_text(encoding="utf-8"))
    index = {}

    # ADJUST: written for two common shapes; edit to match what you saw in step 3.
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = ((r["sequence_id"], r) for r in raw)

    for seq_id, value in items:
        if isinstance(value, dict):
            value = value.get("pages") or value.get("images")
        paths = [Path(p) if Path(p).is_absolute() else DATASET / p for p in value]
        if len(paths) != 3:
            raise ValueError(f"{seq_id}: expected 3 pages, got {len(paths)}")
        index[seq_id] = paths
    return index


def load_reference_rows(path=DEV_REFERENCES):
    """Return {sequence_id: full jsonl row} for a reference file."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["sequence_id"]] = row
    return rows


_INDEX = _load_sequence_index()
_REFS = load_reference_rows()


def all_ids():
    return list(_INDEX)


def dev_ids():
    return [i for i in _INDEX if i in _REFS]


def test_ids():
    return [i for i in _INDEX if i not in _REFS]


def get_sequence(seq_id):
    """Dict with sequence_id, image_paths (3, in order), labels (3 lists or None)."""
    row = _REFS.get(seq_id)
    return {
        "sequence_id": seq_id,
        "image_paths": _INDEX[seq_id],
        "labels": row["pages"] if row else None,
    }