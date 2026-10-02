# evaluation/common.py
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results.md"

REFERENCES = DATASET / "development" / "labels.jsonl"
SAMPLE_SUB = DATASET / "sample_submission.jsonl"
SPLITS = DATASET / "splits.json"

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def _load_score():
    spec = importlib.util.spec_from_file_location("score", DATASET / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = _load_score()  # scorer


def split_ids(name):
    """name: 'dev' (all 80), 'train', 'val', or 'test'."""
    if name == "dev":
        return list(score.load_jsonl(REFERENCES, reference=True))
    if name == "test":
        lines = SAMPLE_SUB.read_text(encoding="utf-8").splitlines()
        return [json.loads(l)["sequence_id"] for l in lines if l.strip()]
    return list(json.loads(SPLITS.read_text(encoding="utf-8"))[name])


def image_paths(seq_id):
    """Three page images in order. Swap for your data.py function if you have one."""
    for sub in ("development", "test"):
        folder = DATASET / sub / "images" / seq_id
        if folder.exists():
            paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
            assert len(paths) == 3, f"{seq_id}: expected 3 pages, found {len(paths)}"
            return paths
    raise FileNotFoundError(f"No image folder for {seq_id}")


def write_jsonl(rows, path):
    """rows: {sequence_id: [page1, page2, page3]}"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for seq_id, pages in rows.items():
            f.write(json.dumps({"sequence_id": seq_id, "pages": pages}, ensure_ascii=False) + "\n")


"""score = _load_score() → loads score.py so we can use its functions.
split_ids(name) → tells us which sequence IDs belong to dev/train/val/test.
image_paths(seq_id) → finds the 3 page images belonging to a sequence.
write_jsonl(rows, path) → saves our data/predictions into the JSONL format expected by the project."""