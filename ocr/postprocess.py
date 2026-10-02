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


def clean_line(text):
    text = _BAR_PREFIX.sub("I", text)
    text = _LONE_I.sub("I", text)
    text = _CONTRACTION.sub("'", text)
    text = _DOTS.sub("...", text)          # ". . ." -> "..."
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