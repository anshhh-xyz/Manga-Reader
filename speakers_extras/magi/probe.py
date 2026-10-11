"""First real check: load Magi, run ONE sequence, print what comes back and draw it on the pages.

    python -m speakers_magi.probe
    python -m speakers_magi.probe --seq seq_xxxxxxxxxxxxxxxx

Look at outputs/magi/probe_*.png: red boxes = Magi text boxes (numbered), blue = characters
(with identity), green lines = text -> character links. If the boxes sit on the balloons and
characters, the coordinate system is right and everything else will work.
"""
import argparse
import json

from PIL import Image, ImageDraw

from . import common
from .core import to_py
from .run_magi import call_magi, get_model


def describe(v, depth=0):
    if isinstance(v, dict):
        return "dict(%s)" % ", ".join(sorted(v))
    if isinstance(v, list):
        return "list[%d]%s" % (len(v), (" of " + type(v[0]).__name__) if v else "")
    return type(v).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--seq", default=None)
    a = ap.parse_args()

    seq = a.seq or common.seq_ids(a.split)[0]
    model = get_model()
    print("\npublic callables on the model:")
    print("  " + ", ".join(m for m in dir(model) if not m.startswith("_") and callable(getattr(model, m, None))
                           and any(k in m for k in ("predict", "do_", "visual", "cluster", "ocr", "chapter"))))

    paths = common.page_paths(seq, a.split)
    pages = [common.read_page(p) for p in paths]
    api, res = call_magi(model, pages, do_ocr=False)
    res = to_py(res)
    print("\nsequence %s | API that worked: %s | result: %s" % (seq, api, describe(res)))
    for i, r in enumerate(res):
        print("\npage %d keys:" % i)
        for k, v in r.items():
            print("   %-32s %s" % (k, describe(v)))
        for k in ("text_character_associations", "character_names", "character_cluster_labels", "texts"):
            if k in r:
                print("   sample %s: %s" % (k, json.dumps(r[k][:4])))

    common.OUT.mkdir(parents=True, exist_ok=True)
    for i, (p, r) in enumerate(zip(paths, res)):
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        names = r.get("character_names") or r.get("character_cluster_labels") or []
        for j, b in enumerate(r.get("texts", [])):
            d.rectangle(b[:4], outline="red", width=4)
            d.text((b[0] + 4, b[1] + 4), "T%d" % j, fill="red")
        for j, b in enumerate(r.get("characters", [])):
            d.rectangle(b[:4], outline="blue", width=4)
            lab = names[j] if j < len(names) else j
            d.text((b[0] + 4, b[1] + 4), "C%d:%s" % (j, lab), fill="blue")
        for t, c in r.get("text_character_associations", []):
            tb, cb = r["texts"][int(t)], r["characters"][int(c)]
            d.line([((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)],
                   fill="green", width=4)
        out = common.OUT / ("probe_%s_p%d.png" % (seq, i))
        im.save(out)
        print("wrote", out)
    with open(common.OUT / ("probe_%s.json" % seq), "w", encoding="utf-8") as f:
        json.dump({"api": api, "pages": res}, f)
    print("\nPaste this whole output to me if anything looks off.")


if __name__ == "__main__":
    main()
