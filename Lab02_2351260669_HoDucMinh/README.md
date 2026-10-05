# Lab 02: Đặc trưng tiếng nói và nhận dạng bằng DTW
**Học phần**: CSE457 – Xử lý âm thanh và tiếng nói  
**Sinh viên thực hiện**: Hồ Đức Minh (MSSV: 2351260669)  
**Trường Đại học Thủy Lợi**

---

## 1. Cấu trúc thư mục

```text
Lab02_2351260669_HoDucMinh/
├── Lab02_2351260669.ipynb      # Notebook chính chạy end-to-end với đầy đủ đồ thị & kết quả
├── Lab2_2351260669.ipynb       # Bản sao notebook theo quy cách đặt tên Lab2_<MSSV>.ipynb
├── report_Lab02.md             # Báo cáo chuyên sâu và giải đáp 9 câu hỏi học thuật
├── README.md                   # Hướng dẫn tái lập môi trường và cấu trúc tệp
├── pyproject.toml              # Quản lý phụ thuộc gói UV / Python
├── generate_dataset.py         # Mã nguồn tổng hợp tập âm thanh chuẩn (25 file x 2 người nói)
├── pipeline.py                 # Mã nguồn pipeline tự động xuất biểu đồ và kết quả
├── results.csv                 # Bảng kết quả nhận dạng chi tiết trên tập test
├── dataset/                    # Tập dữ liệu chính (Speaker 1): 5 từ x 5 lần lặp = 25 WAV
│   ├── khong/                  # khong_01.wav ... khong_05.wav
│   ├── mot/                    # mot_01.wav   ... mot_05.wav
│   ├── hai/                    # hai_01.wav   ... hai_05.wav
│   ├── ba/                     # ba_01.wav    ... ba_05.wav
│   └── bon/                    # bon_01.wav   ... bon_05.wav
├── dataset_speaker2/           # Tập dữ liệu người nói 2 phục vụ thí nghiệm Cross-speaker (E4)
└── figures/                    # Toàn bộ biểu đồ phân tích kỹ thuật độ phân giải cao
    ├── waveform_examples.png
    ├── energy_zcr_analysis.png
    ├── endpoint_detection.png
    ├── mfcc_heatmaps.png
    ├── dtw_same_word.png
    ├── dtw_diff_word.png
    └── confusion_matrix_experiments.png
```

---

## 2. Hướng dẫn chạy chương trình

### 2.1. Cài đặt môi trường
Môi trường sử dụng Python `>=3.12` với trình quản lý gói `uv`:

```bash
cd Lab02_2351260669_HoDucMinh
uv venv
uv pip install numpy scipy matplotlib librosa soundfile scikit-learn pandas seaborn requests ipykernel nbconvert
```

### 2.2. Sinh tập dữ liệu (Dataset Generation)
Để tự động tải và sinh các biến thể phát âm chuẩn (tốc độ nói, cao độ, khoảng lặng đầu/cuối):
```bash
python generate_dataset.py
```

### 2.3. Chạy toàn bộ Pipeline và xuất kết quả
```bash
python pipeline.py
```
Lệnh này sẽ tự động:
- Trích xuất đặc trưng và vẽ toàn bộ biểu đồ vào thư mục `figures/`.
- Thực thi các bài toán đối sánh mẫu DTW.
- Đánh giá trên tập kiểm thử và xuất file `results.csv`.
- Chạy toàn bộ các thí nghiệm kiểm chứng E1, E2, E3, E4.

### 2.4. Chạy Jupyter Notebook
Mở và chạy file [Lab02_2351260669.ipynb](file:///home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab02_2351260669_HoDucMinh/Lab02_2351260669.ipynb) bằng VS Code hoặc Jupyter Lab:
```bash
jupyter notebook Lab02_2351260669.ipynb
```

---

## 3. Tóm tắt kết quả chính
- **Độ chính xác (Accuracy Baseline)**: **100.0%** (10/10 mẫu kiểm thử).
- **Thí nghiệm E1 (Endpoint Detection)**: Cắt bỏ 45.2% thời lượng khoảng lặng, loại trừ nhiễu nền giả tạo.
- **Thí nghiệm E3 (Đa mẫu)**: 3 templates/từ đạt 100% so với chỉ 60% khi dùng 1 template duy nhất.
- **Thí nghiệm E4 (Người nói mới)**: Chi phí khoảng cách tăng 37%, thể hiện tính chất phụ thuộc người nói của DTW.
