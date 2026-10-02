
import json, itertools
from pathlib import Path

root = Path("dataset")

seq = json.loads((root / "sequences.json").read_text(encoding="utf-8"))
print("sequences.json top-level type:", type(seq).__name__, "| length:", len(seq))
if isinstance(seq, dict):
    for k, v in itertools.islice(seq.items(), 2):
        print(repr(k), "->", json.dumps(v)[:300])
else:
    for item in seq[:2]:
        print(json.dumps(item)[:300])

# find the reference file(s)
for p in root.rglob("*.jsonl"):
    n = sum(1 for _ in p.open(encoding="utf-8"))
    print(p, "lines:", n)