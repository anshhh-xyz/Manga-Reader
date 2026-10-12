# ocr/postprocess.py
import re

# lone l / | / ¦ (not part of a word) -> I.  Also catches  l'M, l'LL, l'D
_LONE_I = re.compile(r"(?<![\w'’])[l|¦](?!\w)")
# |T, |F, |S ... (bar glued to a capital) -> IT, IF, IS
_BAR_PREFIX = re.compile(r"[|¦](?=[A-Z])")
# "don ' t"  "DON' T"  "ASH ' S" -> rejoin, only before real contraction suffixes
_CONTRACTION = re.compile(r"(?<=\w)\s*['’]\s*(?=(?:t|s|m|d|ll|re|ve)\b)", re.I)
_DOTS = re.compile(r"\.\s*\.\s*\.(?:\s*\.)*")
_SPACES = re.compile(r"\s+")

COMMON_OCR_REPLACEMENTS = [
    (re.compile(r"\bcllb\b", re.I), "CLUB"),
    (re.compile(r"\byolr\b", re.I), "YOUR"),
    (re.compile(r"\bcolnting\b", re.I), "COUNTING"),
    (re.compile(r"\bcontinle\b", re.I), "CONTINUE"),
    (re.compile(r"\beqlestrian\b", re.I), "EQUESTRIAN"),
    (re.compile(r"\bminlte\b", re.I), "MINUTE"),
    (re.compile(r"\bllck\b", re.I), "LUCK"),
    (re.compile(r"\bgless\b", re.I), "GUESS"),
    (re.compile(r"\bblt\b", re.I), "BUT"),
    (re.compile(r"\bthls\b", re.I), "THIS"),
    (re.compile(r"\byol\b", re.I), "YOU"),
    (re.compile(r"\bwhaaati\b", re.I), "WHAAAT!"),
    (re.compile(r"\bhachikeni\b", re.I), "HACHIKEN!"),
    (re.compile(r"\btodayi\b", re.I), "TODAY!"),
    (re.compile(r"\bproldi\b", re.I), "PROUD!"),
]


def clean_line(text):
    if not text:
        return ""
    text = _BAR_PREFIX.sub("I", text)
    text = _LONE_I.sub("I", text)
    text = _CONTRACTION.sub("'", text)
    text = _DOTS.sub("...", text)          # ". . ." -> "..."
    # Rejoin hyphenated line breaks: "EvERy- BODY" -> "EVERYBODY"
    text = re.sub(r"(\b[A-Za-z]{2,})-\s+([A-Za-z]{2,}\b)", r"\1\2", text)
    # Rejoin spaced compounds: "VICE - PRESIDENT" -> "VICE-PRESIDENT"
    text = re.sub(r"(\b[A-Za-z]+)\s+-\s+([A-Za-z]+\b)", r"\1-\2", text)
    text = re.sub(r"\s*_\s*$", ".", text)
    text = re.sub(r"\s*;\s*$", ".", text)
    for pat, repl in COMMON_OCR_REPLACEMENTS:
        text = pat.sub(repl, text)
    return _SPACES.sub(" ", text).strip()


def join_wrapped(lines):
    """Optional: glue a line ending in 'letter-' to the next line (printed line-wrap hyphens).
    Risk: real stutters like 'W-' / 'WHAT' become 'WWHAT'. Keep only if the score improves."""
    out = []
    for t in lines:
        if out and re.search(r"[A-Za-z]-$", out[-1]) and t[:1].isalpha():
            out[-1] = out[-1][:-1] + t
        else:
            out.append(t)
    return out


if __name__ == "__main__":
    assert clean_line("l'M FINE") == "I'M FINE"
    assert clean_line("DON ' T GO") == "DON'T GO"
    assert clean_line("WAIT . . .") == "WAIT..."
    assert clean_line("|T WORKS") == "IT WORKS"
    assert clean_line("HE'S ALL RIGHT!") == "HE'S ALL RIGHT!"
    print("postprocess OK")