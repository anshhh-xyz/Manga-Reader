"""Checks the label-assignment logic with fake Magi output. Needs no GPU, no model, no dataset.

    python -m speakers_magi.selftest
"""
import json

from .core import assign_labels, to_py


def box(page, x1, y1, x2, y2, tag="dialogue"):
    return {"page": page, "bbox": [x1, y1, x2, y2], "tag": tag}


def make_cache(**page_overrides):
    # 3 pages 1000x1500. Page 0: two texts, two characters. Page 1: one text, one character. Page 2: empty.
    p0 = {"texts": [[100, 100, 300, 200], [600, 100, 800, 200]],
          "characters": [[80, 300, 280, 700], [580, 300, 780, 700]],
          "character_names": ["A", "B"],
          "text_character_associations": [[0, 0], [1, 1]]}
    p1 = {"texts": [[100, 100, 300, 200]],
          "characters": [[80, 300, 280, 700]],
          "character_names": ["B"],            # character B again on page 1
          "text_character_associations": [[0, 0]]}
    p2 = {"texts": [], "characters": [], "text_character_associations": []}
    pages = [p0, p1, p2]
    for k, v in page_overrides.items():
        pages[int(k[1])] = v
    return {"api": "fake", "sizes": [[1000, 1500]] * 3, "pages": pages}


def test_basic_and_cross_page():
    boxes = [box(0, 110, 110, 290, 190), box(0, 610, 110, 790, 190), box(1, 110, 110, 290, 190)]
    labels, st, k = assign_labels(boxes, make_cache())
    assert labels == ["char1", "char2", "char2"], labels      # B on page 1 = same speaker as B on page 0
    assert k == 2 and st["linked"] == 3 and st["unsure"] == 0


def test_scope_page_splits_ids():
    boxes = [box(0, 110, 110, 290, 190), box(0, 610, 110, 790, 190), box(1, 110, 110, 290, 190)]
    labels, _, k = assign_labels(boxes, make_cache(), {"scope": "page"})
    assert labels == ["char1", "char2", "char3"] and k == 3, labels


def test_narration_and_unmatched():
    boxes = [box(0, 110, 110, 290, 190), box(2, 0, 0, 50, 50, tag="narration"), box(2, 400, 1200, 500, 1300)]
    labels, st, _ = assign_labels(boxes, make_cache())
    assert labels[1] == "NARRATION"
    assert labels[2] == "char1" and st["unsure"] == 1          # page 2 has nothing -> copies previous speaker


def test_nearest_fallback_and_off():
    c = make_cache()
    c["pages"][0]["text_character_associations"] = []           # no links at all
    boxes = [box(0, 90, 100, 270, 200), box(0, 590, 100, 780, 200)]
    labels, st, _ = assign_labels(boxes, c, {"nearest": True})
    assert labels == ["char1", "char2"] and st["nearest"] == 2, (labels, st)
    labels, st, _ = assign_labels(boxes, c, {"nearest": False})
    assert labels == ["char1", "char1"] and st["unsure"] == 2, (labels, st)


def test_degenerate_names_fall_back_to_clusters():
    c = make_cache()
    c["pages"][0]["character_names"] = ["unknown", "unknown"]   # would merge everyone
    c["pages"][0]["character_cluster_labels"] = [0, 1]
    boxes = [box(0, 110, 110, 290, 190), box(0, 610, 110, 790, 190)]
    labels, _, k = assign_labels(boxes, c)
    assert labels == ["char1", "char2"] and k == 2


def test_maxk_cap():
    c = make_cache()
    boxes = [box(0, 110, 110, 290, 190), box(0, 610, 110, 790, 190), box(1, 110, 110, 290, 190)]
    labels, _, k = assign_labels(boxes, c, {"maxk": 1})
    assert k == 1 and set(labels) == {"char1"}, labels          # B is most frequent -> everyone becomes char1


def test_unsure_common_policy():
    c = make_cache()
    boxes = [box(0, 110, 110, 290, 190), box(0, 610, 110, 790, 190), box(1, 110, 110, 290, 190),
             box(2, 400, 1200, 500, 1300)]
    labels, _, _ = assign_labels(boxes, c, {"unsure": "common"})
    assert labels[3] == "char2", labels                         # most common speaker (B = char2)


def test_to_py_numpy():
    try:
        import numpy as np
    except ImportError:
        return
    out = to_py({"a": np.array([[1, 2]]), "b": np.float32(1.5), "embeddings": np.zeros(3), "c": (np.int64(3), 4)})
    json.dumps(out)
    assert "embeddings" not in out and out["a"] == [[1, 2]] and out["c"] == [3, 4]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("\nall %d checks passed" % len(tests))


if __name__ == "__main__":
    main()
