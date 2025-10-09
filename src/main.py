import os, glob, cv2
import numpy as np
from utils import ensure_dir, save_image, pad_to_shape
from part1 import convert_to_grayscale, plot_histogram, equalize_histogram, adjust_histogram
from part2 import convolve, median_filter, threshold_images

BASE        = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR   = os.path.join(BASE, "..", "input")
OUTPUT_DIR  = os.path.join(BASE, "..", "output")

def process_one_image(img_path: str):
    name = os.path.splitext(os.path.basename(img_path))[0]
    out_dir_img  = os.path.join(OUTPUT_DIR, name)
    out_dir_hist = os.path.join(out_dir_img, "hist")
    ensure_dir(out_dir_img); ensure_dir(out_dir_hist)

    # ===== Phần 1 =====
    color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if color is None:
        print(f"[WARN] Không đọc được ảnh: {img_path}")
        return
    gray = convert_to_grayscale(color)
    save_image(gray, os.path.join(out_dir_img, "gray.png"))

    # H1
    plot_histogram(gray, f"{name} - H1", os.path.join(out_dir_hist, "H1.png"))
    # H2 (equalize)
    eq = equalize_histogram(gray)
    plot_histogram(eq, f"{name} - H2", os.path.join(out_dir_hist, "H2.png"))
    # H2 điều chỉnh [30,80]
    adj = adjust_histogram(eq, 30, 80)
    plot_histogram(adj, f"{name} - H2 Adjust [30,80]", os.path.join(out_dir_hist, "H2_adjust_30_80.png"))

    # ===== Phần 2 =====
    # Kernels: dùng box filter chuẩn (1/n) đúng yêu cầu "kernel 3x3/5x5/7x7"
    k3 = np.ones((3,3), np.float32) / 9.0
    k5 = np.ones((5,5), np.float32) / 25.0
    k7 = np.ones((7,7), np.float32) / 49.0

    # I1: 3x3, padding=1, stride=1
    I1 = convolve(gray, k3, padding=1, stride=1)
    save_image(I1, os.path.join(out_dir_img, "I1.png"))

    # I2: 5x5, padding=2, stride=1
    I2 = convolve(gray, k5, padding=2, stride=1)
    save_image(I2, os.path.join(out_dir_img, "I2.png"))

    # I3: 7x7, padding=3, stride=2
    I3 = convolve(gray, k7, padding=3, stride=2)
    save_image(I3, os.path.join(out_dir_img, "I3.png"))

    # I4: median 3x3 trên I3
    I4 = median_filter(I3, 3)
    save_image(I4, os.path.join(out_dir_img, "I4.png"))

    # I5: median 5x5 trên I1
    I5 = median_filter(I1, 5)
    save_image(I5, os.path.join(out_dir_img, "I5.png"))

    # I6: ngưỡng theo luật — pad về cùng kích thước trước khi so sánh (nếu cần)
    Ht = max(I4.shape[0], I5.shape[0])
    Wt = max(I4.shape[1], I5.shape[1])
    I4p = pad_to_shape(I4, (Ht, Wt))
    I5p = pad_to_shape(I5, (Ht, Wt))
    I6  = threshold_images(I4p, I5p)
    save_image(I6, os.path.join(out_dir_img, "I6.png"))

    print(f"[OK] {name}: Bài 1 + Bài 2 done. (I1..I6 @ {out_dir_img})")

def main():
    exts = ("*.png","*.jpg","*.jpeg","*.bmp")
    images = []
    for e in exts:
        images.extend(glob.glob(os.path.join(INPUT_DIR, e)))
    images = sorted(images)
    if not images:
        print("Hãy bỏ ảnh vào thư mục input/ rồi chạy lại.")
        return
    ensure_dir(OUTPUT_DIR)
    for p in images:
        process_one_image(p)
    print("\nHoàn tất Phần 1 + Phần 2. Kết quả trong: output")

if __name__ == "__main__":
    main()
