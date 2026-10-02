import json
from configs import data
from configs.config import SPLITS_JSON

splits = json.loads(SPLITS_JSON.read_text(encoding="utf-8"))
train, val = set(splits["train"]), set(splits["val"])

assert not train & val, "leak: sequence in both splits"
assert train | val == set(data.dev_ids()), "splits don't cover all dev sequences"

for sid in list(train | val) + data.test_ids():
    s = data.get_sequence(sid)
    assert len(s["image_paths"]) == 3
    assert all(p.exists() for p in s["image_paths"]), f"missing image in {sid}"
    if sid in train | val:
        assert len(s["labels"]) == 3

assert all(data.get_sequence(i)["labels"] is None for i in data.test_ids())
print("Phase 0 OK:", len(train), "train,", len(val), "val,", len(data.test_ids()), "test")