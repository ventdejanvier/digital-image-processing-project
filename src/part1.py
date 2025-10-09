# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_to_grayscale(img_color: np.ndarray) -> np.ndarray:
    if img_color is None:
        raise ValueError("img_color is None")
    if img_color.ndim == 3:
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_color.copy()
    if gray.dtype != np.uint8:
        gmin, gmax = float(gray.min()), float(gray.max())
        if gmax > gmin:
            gray = ((gray - gmin) / (gmax - gmin) * 255.0).astype(np.uint8)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)
    return gray

def calc_histogram(img_gray: np.ndarray) -> np.ndarray:
    if img_gray is None or img_gray.ndim != 2:
        raise ValueError("img_gray phải là ảnh xám 2D")
    if img_gray.dtype != np.uint8:
        img_gray = img_gray.astype(np.uint8)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    return hist

def plot_histogram(img_gray: np.ndarray, title: str, output_path: str) -> np.ndarray:
    hist = calc_histogram(img_gray)
    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.xlim([0, 255])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return hist

def equalize_histogram(img_gray: np.ndarray) -> np.ndarray:
    if img_gray is None or img_gray.ndim != 2:
        raise ValueError("img_gray phải là ảnh xám 2D")
    if img_gray.dtype != np.uint8:
        img_gray = img_gray.astype(np.uint8)
    return cv2.equalizeHist(img_gray)

def adjust_histogram(img_gray: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
    if img_gray is None or img_gray.ndim != 2:
        raise ValueError("img_gray phải là ảnh xám 2D")
    if min_val > max_val:
        min_val, max_val = max_val, min_val

    g = img_gray.astype(np.float32)
    gmin = float(g.min())
    gmax = float(g.max())

    if gmax == gmin:
        val = int(round((min_val + max_val) / 2.0))
        out = np.full_like(img_gray, val, dtype=np.uint8)
        return out

    out = (g - gmin) / (gmax - gmin)
    out = out * (max_val - min_val) + float(min_val)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
