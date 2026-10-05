# CSE457 - Xử lý âm thanh và tiếng nói
## Lab 1: Phân tích và Xử lý Tín hiệu Âm thanh Số

* **Sinh viên**: Hồ Đức Minh
* **Mã số sinh viên (MSSV)**: 2351260669
* **Trường**: Đại học Thủy Lợi (TLU)
* **Khoa**: Công nghệ Thông tin

---

## 1. Giới thiệu dự án
Dự án hoàn thành toàn diện các yêu cầu của **Bài thực hành số 1 (Lab 1)** theo đề cương chi tiết học phần **CSE457 - Xử lý âm thanh và tiếng nói**:
- **Khối A**: Đọc tệp âm thanh, kiểm tra thông số kỹ thuật ($F_s$, channels, duration, dtype), chuyển đổi Stereo $\rightarrow$ Mono và chuẩn hóa biên độ $[-1, 1]$.
- **Khối B**: Phân tích miền thời gian, tính toán Peak, RMS, Năng lượng, kiểm tra clipping và đối chứng 2 đoạn có đặc tính khác biệt (Nhiễu nền vs Tiếng nói).
- **Khối C**: Phân tích phổ biến đổi Fourier nhanh (FFT), nhân cửa sổ Hamming, xác định $\ge 3$ đỉnh hài formant, so sánh 2 kích thước $N_{\text{FFT}} = 2048$ và $16384$, làm rõ bản chất khoảng cách bin $\Delta f$ và độ phân giải vật lý thực tế.
- **Khối D**: Biến đổi Fourier ngắn hạn (STFT) và biểu đồ Spectrogram, phân tích đánh đổi Heisenberg-Gabor với 3 cấu hình cửa sổ ($10\text{ ms}$, $25\text{ ms}$, $50\text{ ms}$) trên cùng một dải màu chuẩn hóa.
- **Khối E**: Thí nghiệm đối chứng cửa sổ Rectangular vs Hamming trên cùng 1 frame $25\text{ ms}$, phân tích búp chính, búp phụ và hiện tượng rò rỉ phổ (Spectral Leakage).
- **Khối F**: Thiết kế bộ lọc số FIR Low-pass ($2500\text{ Hz}$) và High-pass ($3000\text{ Hz}$) bậc 200 taps bằng phương pháp cửa sổ, phân tích Group Delay ($2.27\text{ ms}$), áp dụng lọc âm thanh và kiểm chứng phổ có bù trừ trễ pha.
- **Khối G**: Lượng tử hóa đều ($4, 6, 8, 12, 16\text{ bit}$), kiểm chứng công thức lý thuyết Rabiner-Schafer với hệ số Crest Factor; Resampling về $16\text{ kHz}$ và $8\text{ kHz}$; phân tích tốc độ bit PCM và tỷ lệ nén MP3 ($10.97 : 1$, tiết kiệm $90.88\%$).
- **Báo cáo lý thuyết**: Lời giải chi tiết, chính xác 7 câu hỏi khoa học tại Mục 6 trong đề bài.

---

## 2. Cấu trúc thư mục

```text
Lab01_2351260669_HoDucMinh/
├── Lab01_2351260669.ipynb      # Jupyter Notebook hoàn chỉnh, chạy độc lập từ đầu đến cuối
├── report_Lab01.md             # Báo cáo kỹ thuật chi tiết theo chuẩn GitHub Markdown
├── README.md                   # Hướng dẫn tái lập môi trường và cấu trúc dự án
├── audio/                      # Thư mục chứa toàn bộ tệp âm thanh đầu vào và đầu ra
│   ├── input_speech.wav        # Tệp âm thanh gốc (Stereo 44.1 kHz, kèm nhiễu nền 18 dB)
│   ├── input_speech.mp3        # Tệp nén MP3 128 kbps đối chứng nén
│   ├── filtered_lpf.wav        # Âm thanh sau khi qua bộ lọc FIR Low-pass 2500 Hz
│   ├── filtered_hpf.wav        # Âm thanh sau khi qua bộ lọc FIR High-pass 3000 Hz
│   ├── quantized_4bit.wav      # Âm thanh lượng tử hóa 4-bit (nghe rõ tiếng xào xạc)
│   ├── quantized_8bit.wav      # Âm thanh lượng tử hóa 8-bit
│   ├── quantized_16bit.wav     # Âm thanh lượng tử hóa 16-bit
│   ├── resampled_16k.wav       # Âm thanh sau khi hạ tần số lấy mẫu về 16 kHz
│   └── resampled_8k.wav        # Âm thanh sau khi hạ tần số lấy mẫu về 8 kHz
└── figures/                    # Thư mục lưu trữ toàn bộ các đồ thị khoa học xuất bản
    ├── waveform.png            # Đồ thị dạng sóng toàn tệp và zoom chi tiết
    ├── fft.png                 # Đồ thị phổ biên độ FFT và so sánh NFFT
    ├── spectrogram.png         # Spectrogram 3 panel so sánh frame length
    ├── window_comparison.png   # So sánh phổ cửa sổ Rectangular vs Hamming
    ├── filter_response.png     # Đáp ứng biên độ |H(f)| và phổ trước/sau lọc
    ├── quantization_snr.png    # Đường cong SNR thực nghiệm vs lý thuyết Rabiner
    └── resampling_comparison.png # Mật độ phổ công suất PSD so sánh 3 tần số lấy mẫu
```

---

## 3. Hướng dẫn cài đặt và tái lập thực nghiệm

Dự án được quản lý bằng trình quản lý gói hiện đại `uv`:

### 3.1. Cài đặt thư viện
```bash
# Cài đặt các gói phụ thuộc cần thiết
uv sync
# hoặc cài thủ công
uv add scipy pydub soundfile librosa matplotlib jupyter
```

### 3.2. Mở và chạy Notebook
```bash
# Khởi động Jupyter Notebook / JupyterLab
uv run jupyter lab
# hoặc
uv run jupyter notebook
```
Sau đó mở tệp `Lab01_2351260669.ipynb` và chọn `Kernel` $\rightarrow$ `Restart Kernel and Run All Cells`.

### 3.3. Tái lập chạy tự động không cần giao diện
```bash
uv run jupyter nbconvert --to notebook --execute --inplace Lab01_2351260669.ipynb
```
Toàn bộ biểu đồ mới sẽ được lưu vào thư mục `figures/` và toàn bộ âm thanh sẽ được xuất vào thư mục `audio/`.
