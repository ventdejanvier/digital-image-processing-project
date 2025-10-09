import os
import cv2
import numpy as np
from typing import Tuple

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_image(img: np.ndarray, path: str):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def pad_to_shape(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:

    H, W = img.shape[:2]
    TH, TW = target_hw
    out = np.zeros((TH, TW), dtype=img.dtype)
    h = min(H, TH)
    w = min(W, TW)
    out[:h, :w] = img[:h, :w]
    return out
