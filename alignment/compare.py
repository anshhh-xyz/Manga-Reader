"""For gold lines that have no box in the merged run, how many of their words appear
anywhere on that page in (a) paragraph-mode OCR, (b) word-level OCR?
Usage: python -m alignment.compare
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OCR = ROOT / "outputs" / "ocr_boxes"
para = json.loads((OCR / "development.json").read_text(encoding="utf-8"))
words = json.loads((OCR / "development_words.json").read_text(encoding="utf-8"))
unm = json.loads((ROOT / "outputs/alignment/unmatched_lines.json").read_text(encoding="utf-8"))


def W(t):
    return re.findall(r"[A-Z0-9']+", t.upper())


def page_words(cache, sid, p):
    return set(w for b in cache[sid]["pages"][p] for w in W(b["text"]))


fp, fw, n = 0.0, 0.0, 0
examples = []
for u in unm:
    g = W(u["text"])
    if len(g) < 3:
        continue
    a = sum(w in page_words(para, u["sequence_id"], u["page"]) for w in g) / len(g)
    b = sum(w in page_words(words, u["sequence_id"], u["page"]) for w in g) / len(g)
    fp, fw, n = fp + a, fw + b, n + 1
    if len(examples) < 6 and b - a > 0.5:
        examples.append((u, a, b))

print(f"unmatched lines with >=3 words: {n}")
print(f"avg share of gold words present on the page: paragraph-mode {fp/n:.2f} | word-level {fw/n:.2f}")
for u, a, b in examples:
    print(f"\n{u['sequence_id']} p{u['page']}  GOLD: {u['text']}   (para {a:.2f}, words {b:.2f})")