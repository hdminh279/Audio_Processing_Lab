# LAB 3: NHẬN DẠNG TIẾNG NÓI VỚI HMM
## CSE457 • ĐẠI HỌC THỦY LỢI

* **Sinh viên**: Hồ Đức Minh (MSSV: 2351260669)
* **Báo cáo**: [`report_Lab03.md`](./report_Lab03.md)
* **Notebook**: [`Lab03_2351260669.ipynb`](./Lab03_2351260669.ipynb) | [`Lab03.ipynb`](./Lab03.ipynb)

---

## CẤU TRÚC THƯ MỤC

```
Lab03_2351260669_HoDucMinh/
├── dataset/                     # 5 lệnh x 8 files = 40 WAV (16kHz mono PCM 16-bit)
│   ├── train/                   # 6 files/lớp (mo, dong, len, xuong, dung)
│   └── test/                    # 2 files/lớp
├── dataset_speaker2/            # Test set người nói 2 (thí nghiệm mismatch E5)
├── figures/                     # Đồ thị sinh tự động
│   ├── heatmap_mfcc_mo.png
│   ├── heatmap_mfcc_dong.png
│   ├── viterbi_alignment_mo.png
│   ├── viterbi_alignment_dong.png
│   ├── confusion_matrix_hmm.png
│   └── confusion_matrix_dtw.png
├── models/                      # 5 HMM models (.pkl)
├── lexicon.txt                  # Pronunciation lexicon format WORD PHONE_1...
├── generate_dataset.py          # Script tải và sinh dataset WAV
├── pipeline.py                  # Pipeline huấn luyện, nhận dạng, E1-E6
├── create_full_notebook.py      # Script sinh notebook
├── Lab03_2351260669.ipynb       # Notebook đã thực thi đầy đủ
├── Lab03.ipynb                  # Bản sao notebook
├── results.json                 # Số liệu định lượng đo đạc từ E1-E6
├── report_Lab03.md              # Báo cáo chi tiết + giải đáp 10 câu hỏi
└── README.md
```

---

## HƯỚNG DẪN THỰC THI

### 1. Cài đặt thư viện
```bash
uv add numpy scipy matplotlib librosa soundfile scikit-learn hmmlearn pandas seaborn jupyter ipykernel
# hoặc: pip install numpy scipy matplotlib librosa soundfile scikit-learn hmmlearn pandas seaborn jupyter ipykernel
```

### 2. Sinh dữ liệu (nếu cần tạo lại từ đầu)
```bash
python generate_dataset.py
```

### 3. Chạy pipeline huấn luyện và thí nghiệm E1–E6
```bash
python pipeline.py
```
Lệnh tự sinh models vào `models/`, hình vẽ vào `figures/`, số liệu vào `results.json`.

### 4. Chạy Jupyter Notebook
```bash
jupyter notebook Lab03_2351260669.ipynb
```

---

## KẾT QUẢ ĐO LƯỜNG CHÍNH

- **Accuracy HMM**: 100.0% (10/10 test mẫu đúng).
- **Tốc độ test**: HMM 9.59 ms/trial vs DTW 790.12 ms/trial (HMM nhanh hơn 82.4 lần).
- **Thực nghiệm E1–E6**:
  - E1: 5 states là cấu hình tối ưu cho từ đơn ngắn.
  - E2: 39-D ($\Delta + \Delta^2$) tăng margin phân tách so với 13-D.
  - E3: Full covariance bị overfit (test acc rơi xuống 40%), Diagonal ổn định (100%).
  - E4: DTW tốt hơn khi chỉ có 1 mẫu train (100% vs 40%), HMM vượt trội khi có đủ dữ liệu.
  - E5: Lệch người nói giảm còn 60%, nhiễu giảm còn 40%.
  - E6: $\lambda_{LM} \in [0.1, 0.3]$ củng cố quyết định âm học; $\lambda_{LM} > 1.0$ lấn át âm học.
