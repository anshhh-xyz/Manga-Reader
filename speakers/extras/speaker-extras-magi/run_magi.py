"""Run Magi on every sequence and cache the raw results as JSON (resume-safe).

    python -m speakers_magi.run_magi --split dev
    python -m speakers_magi.run_magi --split test
    python -m speakers_magi.run_magi --split dev --limit 3      # quick trial
"""
import argparse
import json
import time

from . import common
from .core import to_py

MODEL_ID = "ragavsachdeva/magiv2"
_MODEL = {}


def get_model(device=None):
    if "m" not in _MODEL:
        import torch
        from transformers import AutoModel

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("loading %s on %s (first run downloads the weights) ..." % (MODEL_ID, device))
        m = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        _MODEL["m"] = m.to(device).eval()
    return _MODEL["m"]


def call_magi(model, pages, do_ocr=False):
    """Try the model-card API first, then a simpler detection API. Returns (api_name, per-page list)."""
    import torch

    errors = []
    with torch.no_grad():
        # 1) chapter-wide call from the model card, with an empty character bank
        try:
            res = model.do_chapter_wide_prediction(
                pages, {"images": [], "names": []}, use_tqdm=False, do_ocr=do_ocr)
            return "do_chapter_wide_prediction", res
        except Exception as e:  # noqa: BLE001 - we want to try the next API on any failure
            errors.append("do_chapter_wide_prediction: %r" % (e,))
        # 2) plain detections + associations over the whole chapter
        try:
            res = model.predict_detections_and_associations(pages)
            return "predict_detections_and_associations", res
        except Exception as e:  # noqa: BLE001
            errors.append("predict_detections_and_associations: %r" % (e,))
    raise RuntimeError("Both Magi calls failed:\n  " + "\n  ".join(errors))


def run_and_cache(model, seq, split, do_ocr=False):
    paths = common.page_paths(seq, split)
    if len(paths) != 3:
        raise RuntimeError("%s has %d page images, expected 3" % (seq, len(paths)))
    pages = [common.read_page(p) for p in paths]
    api, res = call_magi(model, pages, do_ocr=do_ocr)
    res = to_py(res)
    if not isinstance(res, list) or len(res) != len(pages):
        raise RuntimeError("unexpected Magi result type %s (len %s) for %s"
                           % (type(res).__name__, getattr(res, "__len__", lambda: "?")(), seq))
    cache = {"api": api, "sizes": [[p.shape[1], p.shape[0]] for p in pages], "pages": res}
    out = common.cache_path(seq, split)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=0, help="only the first N sequences")
    ap.add_argument("--force", action="store_true", help="re-run even if cached")
    ap.add_argument("--ocr", action="store_true", help="also run Magi's own OCR (not needed; slower)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    a = ap.parse_args()

    ids = common.seq_ids(a.split)
    if a.limit:
        ids = ids[:a.limit]
    model = get_model(a.device)
    failed, t0 = [], time.time()
    for n, seq in enumerate(ids, 1):
        if common.cache_path(seq, a.split).exists() and not a.force:
            print("[%d/%d] %s cached" % (n, len(ids), seq))
            continue
        try:
            c = run_and_cache(model, seq, a.split, do_ocr=a.ocr)
            n_t = sum(len(p.get("texts", [])) for p in c["pages"])
            n_c = sum(len(p.get("characters", [])) for p in c["pages"])
            print("[%d/%d] %s ok  api=%s texts=%d characters=%d  (%.0fs elapsed)"
                  % (n, len(ids), seq, c["api"], n_t, n_c, time.time() - t0))
        except Exception as e:  # noqa: BLE001
            failed.append(seq)
            print("[%d/%d] %s FAILED: %s" % (n, len(ids), seq, e))
    print("\ndone. failed: %d %s" % (len(failed), failed[:10]))
    if failed:
        print("re-run the same command to retry only the failed ones.")


if __name__ == "__main__":
    main()
