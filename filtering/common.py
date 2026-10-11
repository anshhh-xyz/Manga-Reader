"""filtering/common.py — page image loader used by features.py"""
import cv2
from evaluation.common import image_paths   # uses the real dataset path resolver


def page_gray(seq_id: str, page: int):
    """Load one manga page (0-indexed) and return it as a uint8 grayscale array."""
    paths = image_paths(seq_id)            # returns sorted list of 3 Path objects
    img = cv2.imread(str(paths[page]), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"cv2 could not read {paths[page]}"
    return img