"""loads a specific manga page and converts it to grayscale so Phase 4's filtering logic can analyze the image."""

import cv2
from configs.data import get_sequence  # ADAPT: your Phase 0 loader

def page_gray(seq_id: str, page: int):
    seq = get_sequence(seq_id)
    path = seq["image_paths"][page]          # ADAPT key name if different
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"cannot read {path}"
    return img