
# Digital Image Processing Project

## Giới thiệu

Đề tài tập trung vào các kỹ thuật xử lý ảnh số cơ bản và nâng cao, sử dụng các thư viện như OpenCV và Python.
Nó bao gồm các thuật toán phổ biến như lọc ảnh, tăng cường histogram, phân đoạn ảnh và biến đổi tần số.

## Tính năng chính

- Xử lý histogram: cân bằng hóa và biến đổi cục bộ.
- Lọc ảnh: lọc trung bình, Gaussian, median.
- Phân đoạn ảnh: ngưỡng hóa, watershed.
- Biến đổi miền tần số: FFT, lọc thông cao/thấp.


## Yêu cầu hệ thống

- Python 3.8+
- OpenCV: `pip install opencv-python`
- NumPy: `pip install numpy`
- Matplotlib: `pip install matplotlib`


## Hướng dẫn cài đặt

1. Clone repository:
`git clone https://github.com/ventdejanvier/digital-image-processing-project.git`
2. Di chuyển vào thư mục: `cd digital-image-processing-project`
3. Cài đặt dependencies: `pip install -r requirements.txt`
4. Chạy demo: `python main.py`

## Cấu trúc thư mục

```
digital-image-processing-project/
├── src/
│   ├── filters.py
│   ├── histogram.py
│   └── segmentation.py
├── examples/
│   └── images/
├── main.py
├── requirements.txt
└── README.md
```


## Cách sử dụng

Chạy file chính để xem demo:
`python main.py --input examples/images/lena.jpg --filter gaussian`
Thay đổi tham số để thử các thuật toán khác nhau.[^3]

