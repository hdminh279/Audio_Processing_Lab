# CSE457 Lab 4: Speech Synthesis (Tổng Hợp Tiếng Nói)
**Sinh viên:** Hồ Đức Minh — **MSSV:** 2351260669  
**Bộ môn:** Trí tuệ nhân tạo - Khoa CNTT - Đại học Thủy Lợi  

---

## 1. Cấu Trúc Thư Mục
```
Lab04_2351260669_HoDucMinh/
├── Lab04.py                  # Script thực thi toàn bộ luồng bài lab từ A -> G
├── Lab04.ipynb               # Jupyter Notebook tương tác trực quan
├── hmm_tts_backend.py        # Backend tổng hợp HMM & Resynthesis
├── neural_tts_backend.py     # Backend tổng hợp Deep Learning Neural TTS
├── generate_inventory.py     # Script khởi tạo kho ngữ liệu 16kHz mono WAV
├── generate_ratings.py       # Script sinh dữ liệu listening test MOS & A/B
├── report_Lab04.md           # Báo cáo kỹ thuật chi tiết (7 mục, trả lời 10 câu hỏi)
├── ratings.csv               # Bảng điểm đánh giá nghe chủ quan 3 phương pháp
├── lab4_data/
│   ├── metadata.csv          # Metadata nhãn, duration, RMS
│   └── units/                # 32 file WAV (16 từ x 2 candidates)
├── audio_output/             # Kết quả âm thanh các thí nghiệm
│   ├── raw_concat.wav        # Ghép thẳng 0ms
│   ├── best_concat.wav       # Unit selection tối ưu + cross-fade 15ms
│   ├── concat_xfade_*.wav    # Thí nghiệm cross-fade 0, 5, 15, 30 ms
│   ├── hmm_tts.wav           # HMM parametric synthesis
│   ├── neural_tts.wav        # Deep learning neural TTS
│   ├── neural_tts_s*.wav     # 3 câu kiểm thử độ trễ neural TTS
│   ├── neural_latency.json   # Đo đạc thời gian suy luận và RTF
│   └── prosody_*.wav         # Thí nghiệm pitch shift & time stretch
└── figures/                  # 5 biểu đồ yêu cầu bài Lab
    ├── fig1_inventory_stats.png
    ├── fig2_boundary_spectrogram.png
    ├── fig3_unit_selection_cost.png
    ├── fig4_hmm_parameter_tracks.png
    └── fig5_output_comparison.png
```

## 2. Hướng Dẫn Chạy Mã Nguồn
Toàn bộ môi trường đã được cấu hình tự động thông qua `uv`:
```bash
# Kích hoạt môi trường ảo
source ../.venv/bin/activate

# Chạy toàn bộ bài lab (sinh audio và xuất đồ thị)
python Lab04.py

# Hoặc mở Jupyter Notebook
jupyter notebook Lab04.ipynb
```
