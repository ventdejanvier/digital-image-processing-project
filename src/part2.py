import cv2
import numpy as np

def convolve(img_gray: np.ndarray, kernel: np.ndarray, padding: int, stride: int = 1) -> np.ndarray:
    if img_gray.ndim != 2:
        raise ValueError("img_gray phải là ảnh xám 2D")
    g = img_gray.astype(np.float32)
    kh, kw = kernel.shape
    H, W   = g.shape

    # zero padding
    pad_g = np.pad(g, ((padding, padding), (padding, padding)), mode="constant", constant_values=0)

    # output size theo công thức conv
    out_H = (H + 2*padding - kh) // stride + 1
    out_W = (W + 2*padding - kw) // stride + 1
    out   = np.zeros((out_H, out_W), dtype=np.float32)

    # lật kernel 180° cho đúng convolution
    k = np.flipud(np.fliplr(kernel.astype(np.float32)))

    for i_out in range(out_H):
        i = i_out * stride
        for j_out in range(out_W):
            j = j_out * stride
            patch = pad_g[i:i+kh, j:j+kw]
            out[i_out, j_out] = float(np.sum(patch * k))

    return np.clip(out, 0, 255).astype(np.uint8)


def median_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("ksize phải là số lẻ >= 1")
    return cv2.medianBlur(img, ksize)


def threshold_images(I4: np.ndarray, I5: np.ndarray) -> np.ndarray:
    a = I4.astype(np.int16)
    b = I5.astype(np.int16)
    out = np.where(a > b, 0, b)
    return np.clip(out, 0, 255).astype(np.uint8)
