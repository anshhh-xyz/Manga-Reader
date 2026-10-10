# filtering/features.py


"""
Creates numerical features for each OCR text box to train the story-text filter.

Functions:
- balloon_feats(): Extracts features from white connected regions, such as possible speech balloons.
- text_feats(): Extracts text features like character count, uppercase ratio, digits, and OCR confidence.
- build_features(): Combines geometry, text, and balloon features into a DataFrame for model training and prediction.

Feature groups:
- GEOM: Box position, size, margins, and page-level statistics.
- TEXT: Text length, word count, capitalization, digits, and OCR confidence.
- BALLOON: White-region size, border contact, box overlap, and whiteness.
"""


import numpy as np, pandas as pd, cv2
from filtering.common import page_gray

BALLOON = ["comp_area_frac","comp_touches_border","box_in_comp_frac","white_frac","has_comp"]
GEOM    = ["cx","cy","bw","bh","aspect","margin","rel_height","rel_area","n_boxes"]
TEXT    = ["n_chars","n_words","caps_ratio","has_letters","no_letters","digit_ratio","conf"]
FEATURES = GEOM + TEXT + BALLOON

def balloon_feats(gray, boxes):
    h, w = gray.shape
    white = (gray > 200).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    out = []
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(x1), w), min(int(y1), h)
        if x1 <= x0 or y1 <= y0:
            out.append([0, 0, 0, 0, 0]); continue
        reg = labels[y0:y1, x0:x1]
        vals = reg[reg > 0]
        if vals.size == 0:
            out.append([0, 0, 0, 0, 0]); continue
        cid = np.bincount(vals).argmax()
        cl, ct, cw, ch, area = stats[cid]
        touches = int(cl == 0 or ct == 0 or cl + cw >= w or ct + ch >= h)
        box_area = (x1 - x0) * (y1 - y0)
        out.append([area / (h * w), touches, box_area / max(cw * ch, 1),
                    vals.size / box_area, 1])
    return out

def text_feats(t, conf):
    letters = [c for c in t if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / max(len(letters), 1)
    return [len(t), len(t.split()), caps, int(len(letters) > 0),
            int(len(letters) == 0), sum(c.isdigit() for c in t) / max(len(t), 1), conf]

def build_features(df):
    """df needs seq_id,page,x0..y1,text,conf. No labels used, so it works at test time."""
    rows = []
    for (seq, pg), g in df.groupby(["seq_id", "page"], sort=False):
        gray = page_gray(seq, pg)
        h, w = gray.shape
        b = g[["x0", "y0", "x1", "y1"]].to_numpy(dtype=float)
        bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        med_h = max(np.median(bh), 1.0)
        med_a = max(np.median(bw * bh), 1.0)
        bal = balloon_feats(gray, b)
        for i, (idx, r) in enumerate(g.iterrows()):
            x0, y0, x1, y1 = b[i]
            geom = [(x0+x1)/2/w, (y0+y1)/2/h, bw[i]/w, bh[i]/h, bw[i]/max(bh[i], 1),
                    min(x0, y0, w-x1, h-y1)/min(w, h), bh[i]/med_h,
                    bw[i]*bh[i]/med_a, len(g)]
            rows.append([idx] + geom + text_feats(r["text"], r["conf"]) + bal[i])
    X = pd.DataFrame(rows, columns=["_idx"] + FEATURES).set_index("_idx")
    return X.loc[df.index]