# speakers/embeddings.py
import pickle, numpy as np, torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor
from .adapters import load_sequences, page_paths, OUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NAME = "openai/clip-vit-base-patch32"          # open weights; cite it in the README
CACHE = OUT / "emb_cache.pkl"

def crop_around(img, bbox, scale=2.5, min_frac=0.2):
    W, H = img.size
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(scale * max(x2 - x1, y2 - y1), min_frac * min(W, H))
    box = [max(0, cx - side/2), max(0, cy - side/2), min(W, cx + side/2), min(H, cy + side/2)]
    im = img.convert("RGB").copy()
    ImageDraw.Draw(im).rectangle([x1, y1, x2, y2], fill="white")   # hide the balloon text
    return im.crop([int(v) for v in box])

def key(b): return f'{b["seq"]}|{b["page"]}|{round(b["bbox"][0])},{round(b["bbox"][1])}'

@torch.no_grad()
def build_cache():
    seqs = load_sequences()
    model = CLIPModel.from_pretrained(NAME).to(DEV).eval()
    proc = CLIPProcessor.from_pretrained(NAME)
    cache = {}
    for s, boxes in seqs.items():
        imgs = [Image.open(p) for p in page_paths(s)]
        crops = [crop_around(imgs[b["page"]], b["bbox"]) for b in boxes]
        for i in range(0, len(crops), 32):
            inp = proc(images=crops[i:i+32], return_tensors="pt")
            pixel_values = inp["pixel_values"].to(DEV)
            out = model.get_image_features(pixel_values=pixel_values)
            e = out.pooler_output if hasattr(out, "pooler_output") else out
            e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
            for b, v in zip(boxes[i:i+32], e):
                cache[key(b)] = v
    pickle.dump(cache, open(CACHE, "wb"))
    print("embedded", len(cache), "boxes ->", CACHE)

def get_emb(boxes, cache):
    return np.stack([cache[key(b)] for b in boxes])

if __name__ == "__main__":
    build_cache()