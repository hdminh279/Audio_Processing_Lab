# BÁO CÁO THỰC HÀNH CHI TIẾT LAB 4
**MÔN HỌC: XỬ LÝ ÂM THANH VÀ TIẾNG NÓI (CSE457)**  
**Sinh viên thực hiện:** Hồ Đức Minh — MSSV: 2351260669  
**Chủ đề:** TỔNG HỢP TIẾNG NÓI: Từ kiến trúc TTS, ghép nối và Unit Selection đến tổng hợp tham số HMM và TTS học sâu  

---

## MỤC LỤC
1. [Mục tiêu và Chuẩn đầu ra](#1-mục-tiêu-và-chuẩn-đầu-ra)
2. [Kiến trúc TTS và Cơ sở Lý thuyết](#2-kiến-trúc-tts-và-cơ-sở-lý-thuyết)
3. [Thiết kế Thực nghiệm và Cài đặt Pipeline](#3-thiết-kế-thực-nghiệm-và-cài-đặt-pipeline)
4. [Kết quả Thực nghiệm Chi tiết](#4-kết-quả-thực-nghiệm-chi-tiết)
   - 4.1. Text Normalization và Thống kê Inventory
   - 4.2. Baseline Concatenation và Hiện tượng Gián đoạn Biên
   - 4.3. Unit Selection và Cross-fade Smoothing
   - 4.4. Điều khiển Ngữ điệu (Prosody) và Source-Filter
   - 4.5. HMM Parametric Synthesis và Phân tích Trajectory
   - 4.6. Neural TTS Pretrained Inference và Đo lường Latency
5. [Đánh giá So sánh 3 Mô hình Tổng hợp](#5-đánh-giá-so-sánh-3-mô-hình-tổng-hợp)
6. [Trả lời Chi tiết 10 Câu hỏi Báo cáo](#6-trả-lời-chi-tiết-10-câu-hỏi-báo-cáo)
7. [Kết luận và Hướng phát triển](#7-kết-luận-và-hướng-phát-triển)

---

## 1. MỤC TIÊU VÀ CHUẨN ĐẦU RA

- **Kiến trúc tổng thể:** Làm chủ chuỗi xử lý TTS gồm 4 khối chức năng kinh điển: Text Analysis / Normalization $\rightarrow$ Phonetic Analysis $\rightarrow$ Prosodic Analysis $\rightarrow$ Synthesis Backend $\rightarrow$ Waveform.
- **Concatenative TTS:** Xây dựng kho đơn vị phát âm (speech inventory) cho miền hẹp "thông báo xếp hàng", cài đặt thuật toán chọn đơn vị (Unit Selection) dựa trên hàm chi phí đích (Target Cost) và chi phí nối (Join Cost) tối ưu bằng quy hoạch động Viterbi.
- **Xử lý tín hiệu biên nối:** Áp dụng kỹ thuật Cross-fade / Overlap-Add (OLA) để triệt tiêu tiếng click biên, phân tích sự thay đổi phổ và miền thời gian.
- **Parametric HMM Synthesis:** Khảo sát cơ chế phát sinh chuỗi tham số âm học (Acoustic Parameter Generation) có ràng buộc động (dynamic feature / delta) từ mô hình Markov ẩn.
- **Deep Learning TTS:** Chạy suy luận offline hệ thống TTS nơ-ron 2 giai đoạn (Acoustic Feature Generator + Neural Vocoder), đo kiểm độ trễ phát sinh tín hiệu (Inference Latency và Real-Time Factor).

---

## 2. KIẾN TRÚC TTS VÀ CƠ SỞ LÝ THUYẾT

### 2.1. Chuỗi biến đổi tín hiệu
$$ \text{Text Input } T \xrightarrow{\text{Normalization}} \text{Tokens } W \xrightarrow{\text{G2P / Lexicon}} \text{Phonemes } P \xrightarrow{\text{Prosody}} (F_0, \Delta t, E) \xrightarrow{\text{Backend}} y(t) $$

### 2.2. Hàm chi phí trong Unit Selection
Thuật toán Unit Selection lựa chọn chuỗi đơn vị âm thanh $\hat{u}_{1:N}$ từ tập ứng viên để cực tiểu hóa tổng chi phí:
$$ C(u_{1:N}) = \sum_{i=1}^N C_t(t_i, u_i) + \sum_{i=1}^{N-1} C_j(u_i, u_{i+1}) $$
- **Target cost ($C_t$):** Độ sai lệch giữa đơn vị ứng viên $u_i$ với đích ngữ cảnh $t_i$ (về pitch, duration, ngữ cảnh âm vị trái/phải, vị trí trọng âm).
- **Join cost ($C_j$):** Đo mức độ bất liên tục tại biên ghép giữa điểm cuối của $u_i$ và điểm đầu của $u_{i+1}$. Trong bài thực hành, sai khác phổ log-magnitude được đo bằng khoảng cách Euclid:
$$ D_{\text{spec}} = \frac{1}{K} \sum_{k=1}^K \left[ L_{\text{left}}[k] - L_{\text{right}}[k] \right]^2 $$
với $L = 20 \log_{10} |X[k]|$, tính trên cửa sổ 20–30 ms xung quanh biên nối.

### 2.3. Mô hình Nguồn – Bộ lọc (Source–Filter Model)
Tín hiệu tiếng nói $s(t)$ là tích chập của nguồn kích thích $e(t)$ với đáp ứng xung của tuyến âm $v(t)$ và bức xạ môi $r(t)$:
$$ S(z) = E(z) \cdot V(z) \cdot R(z) $$
- **Nguồn kích thích ($E(z)$):** Xung thanh môn (Glottal pulses) có chu kỳ $T_0 = 1/F_0$ cho âm hữu thanh (Voiced) hoặc nhiễu ngẫu nhiên trắng cho âm vô thanh (Unvoiced).
- **Bộ lọc tuyến âm ($V(z)$):** Xác định các đỉnh cộng hưởng tần số (Formants $F_1, F_2, F_3$), quyết định bản sắc âm vị của nguyên âm và phụ âm.
- **Tính độc lập:** Thay đổi tần số cơ bản $F_0$ chỉ làm thay đổi khoảng cách giữa các vạch điều hòa (harmonic spacing) của nguồn $E(z)$, hoàn toàn không làm dịch chuyển phong bì phổ (spectral envelope / formant locations) của bộ lọc $V(z)$.

---

## 3. THIẾT KẾ THỰC NGHIỆM VÀ CÀI ĐẶT PIPELINE

Cấu trúc dự án được tổ chức chuẩn xác, tách biệt mã nguồn xử lý và dữ liệu:
```
Lab04_2351260669_HoDucMinh/
├── Lab04.py                  # Script thực thi toàn bộ luồng bài lab
├── Lab04.ipynb               # Jupyter Notebook tương tác trực quan
├── hmm_tts_backend.py        # Backend tổng hợp tham số HMM & Resynthesis
├── neural_tts_backend.py     # Backend tổng hợp tiếng nói Deep Learning offline
├── generate_inventory.py     # Khởi tạo kho dữ liệu từ chuẩn 16kHz mono
├── generate_ratings.py       # Bộ đánh giá listening test A/B và MOS
├── lab4_data/
│   ├── metadata.csv          # Thuộc tính nhãn, speaker, duration, RMS
│   └── units/                # 32 file WAV (16 từ x 2 ứng viên)
├── audio_output/             # File âm thanh xuất ra cho từng phương pháp
├── figures/                  # 5 biểu đồ phân tích kỹ thuật
└── ratings.csv               # Bảng điểm nghe kiểm tra chủ quan
```

---

## 4. KẾT QUẢ THỰC NGHIỆM CHI TIẾT

### 4.1. Text Normalization và Thống kê Inventory
- Quy tắc chuẩn hóa miền hẹp được áp dụng cho văn bản: *"Mời khách hàng số 23 đến quầy 5."*
- Chuỗi token đầu ra: `['mời', 'khách', 'hàng', 'số', 'hai', 'ba', 'đến', 'quầy', 'năm']`.
- Toàn bộ 16 từ vựng được xây dựng với 2 ứng viên mỗi từ (Candidate 01: giọng chuẩn, Candidate 02: biến thiên tốc độ +6% và micro-pitch +0.4 semitone để tạo tập ứng viên đa dạng cho Unit Selection).
- Tần số lấy mẫu: $F_s = 16\,000$ Hz, đơn kênh mono, biên độ năng lượng chuẩn hóa $\text{RMS} = 0.0800$.
- Phân bố duration tập đơn vị: trung bình $0.533$ s (ngắn nhất: từ *khách* $0.342$ s, dài nhất: từ *mời* $0.680$ s).
![Inventory Stats](file:///home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab04_2351260669_HoDucMinh/figures/fig1_inventory_stats.png)

### 4.2. Baseline Concatenation và Hiện tượng Gián đoạn Biên
- Ghép chuỗi trực tiếp ứng viên cố định `01` tại $0$ ms cross-fade tạo ra file [`raw_concat.wav`](file:///home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab04_2351260669_HoDucMinh/audio_output/raw_concat.wav).
- **Phân tích hiện tượng:** Tại các biên tiếp giáp giữa các từ (ví dụ giữa *mời* và *khách* tại thời điểm $0.680$ s), xuất hiện bước nhảy biên độ tức thời (DC offset / phase jump). Trên phổ spectrogram, xuất hiện các vệt năng lượng thẳng đứng trải rộng trên toàn dải tần số (spectral discontinuity), gây ra tiếng "click/pop" chói tai khi nghe thực tế.

### 4.3. Unit Selection và Cross-fade Smoothing
- **Tối ưu hóa Unit Selection:** Sử dụng thuật toán quy hoạch động Viterbi để dò đường tìm kiếm chuỗi ứng viên cực tiểu hóa tổng join cost $D_{\text{spec}}$ qua ma trận $2 \times 9$ trạng thái.
  - Chi phí nối khi chọn toàn bộ Candidate `01`: $5\,179.852\text{ dB}^2$.
  - Chi phí nối khi chọn toàn bộ Candidate `02`: $4\,597.711\text{ dB}^2$.
  - Chi phí nối tối ưu bởi Viterbi Unit Selection: **$3\,445.917\text{ dB}^2$** (chuỗi chọn: `['02', '01', '02', '02', '01', '01', '01', '01', '02']`). Giảm hơn $33.5\%$ sai lệch phổ so với baseline cố định.
![Unit Selection Cost](file:///home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab04_2351260669_HoDucMinh/figures/fig3_unit_selection_cost.png)

- **Thí nghiệm Cross-fade (0, 5, 15, 30 ms):**
  - Mức $0$ ms: Xuất hiện tiếng click rõ rệt.
  - Mức $5$ ms: Giảm bớt click nhẹ nhưng pha tại biên vẫn có gợn.
  - Mức $15$ ms: **Tối ưu nhất.** Đường dốc trọng số tuyến tính $\alpha$ triệt tiêu hoàn toàn bước nhảy biên độ mà không làm mờ phụ âm đầu/cuối của từ tiếp giáp. Lưu thành [`best_concat.wav`](file:///home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab04_2351260669_HoDucMinh/audio_output/best_concat.wav).
  - Mức $30$ ms: Vùng overlap quá dài làm năng lượng phụ âm vô thanh $/k^h/$, $/s/$ bị mờ và có hiện tượng chồng âm (smearing artifact).
![Boundary Spectrogram](file:///home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab04_2351260669_HoDucMinh/figures/fig2_boundary_spectrogram.png)

### 4.4. Điều khiển Ngữ điệu (Prosody) và Source-Filter
- Đo tần số cơ bản của đơn vị mẫu *mời*: Thu được $\bar{F}_0 \approx 201.1$ Hz trên đoạn bán tuần hoàn.
- Dịch cao độ pitch $\Delta s = \pm 2$ semitone bằng phương pháp chia pha phase vocoder (tỷ số tần số $r = 2^{\Delta s / 12} = 2^{\pm 2/12} \approx 0.8909$ và $1.1225$).
- Biến đổi tốc độ phát: co giãn duration $0.8\times$ và $1.2\times$ mà không làm thay đổi tần số cơ bản $F_0$.
- **Nhận xét quan trọng:** Việc co giãn thời gian hoặc dịch pitch độc lập bảo toàn vị trí các formant $F_1, F_2$, tránh được hiện tượng giọng nói bị biến dạng thành "giọng sóc chuột" (chipmunk effect).

### 4.5. HMM Parametric Synthesis và Phân tích Trajectory
- Mô hình HMM mô phỏng cơ chế chuyển trạng thái 5-state left-to-right cho mỗi âm vị/từ.
- Sinh chuỗi tham số âm học liên tục qua thuật toán Maximum Likelihood Parameter Generation (MLPG) kết hợp đặc trưng động ($\Delta$ feature).
- Quỹ đạo $F_0$ contour trơn tru, không có bước nhảy đột ngột tại biên từ. Formant $F_1$ (đại diện độ mở hàm) và $F_2$ (đại diện độ tiến/lùi của lưỡi) biến thiên liên tục.
- Tái tạo waveform bằng bộ lọc cộng hưởng số 2-pole (Formant Filter) kết hợp kích thích glottal pulse train, lưu tại [`hmm_tts.wav`](file:///home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab04_2351260669_HoDucMinh/audio_output/hmm_tts.wav). Âm thanh rất liên tục, không có tiếng click, nhưng có tính chất hơi nghẹt (muffled/buzzy) do bộ lọc tham số đơn giản hóa phổ.
![HMM Parameter Tracks](file:///home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab04_2351260669_HoDucMinh/figures/fig4_hmm_parameter_tracks.png)

### 4.6. Neural TTS Pretrained Inference và Đo lường Latency
- Hệ thống Deep Learning 2-stage thực hiện biến đổi từ chuỗi token sang Mel-Spectrogram (80 bins, $N_{\text{fft}}=1024, H=256$) và qua khối PostNet Convolutional để làm mịn không gian biểu diễn thời gian - tần số.
- Vocoder pha thực hiện chuyển đổi Mel về tín hiệu sóng miền thời gian, xuất ra [`neural_tts.wav`](file:///home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab04_2351260669_HoDucMinh/audio_output/neural_tts.wav).
- Đo đạc hiệu năng suy luận (Inference Latency) trên 3 câu kiểm thử:
  - Câu 1: *“Mời khách hàng số 23 đến quầy 5.”* — Thời gian: $951.3$ ms, Thời lượng audio: $5.06$ s, $\text{RTF} = 0.188$.
  - Câu 2: *“Mời khách hàng số 104 đến quầy 2.”* — Thời gian: $1\,190.3$ ms, Thời lượng audio: $5.46$ s, $\text{RTF} = 0.218$.
  - Câu 3: *“Mời khách hàng số 89 đến quầy 7.”* — Thời gian: $1\,175.0$ ms, Thời lượng audio: $5.09$ s, $\text{RTF} = 0.231$.
- Chỉ số Real-Time Factor (RTF) $< 0.25$ chứng minh khả năng chạy nhanh hơn thời gian thực $4\times$ trên CPU thông thường.

### 4.7. So sánh Trực quan Dạng sóng Toàn phần
![Output Comparison](file:///home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab04_2351260669_HoDucMinh/figures/fig5_output_comparison.png)

---

## 5. ĐÁNH GIÁ SO SÁNH 3 MÔ HÌNH TỔNG HỢP

Bảng so sánh tổng hợp định lượng và định tính theo rubric yêu cầu của bài Lab (thang đo MOS từ 1 đến 5 dựa trên file kiểm thử [`ratings.csv`](file:///home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab04_2351260669_HoDucMinh/ratings.csv)):

| Tiêu chí | Concatenative TTS (Ghép nối) | HMM Parametric TTS (Tham số) | Neural TTS (Học sâu) |
| :--- | :--- | :--- | :--- |
| **Độ rõ ràng (Intelligibility)** | Rất cao ($4.77 / 5.0$) — phát âm chuẩn từng từ gốc đã thu | Khá tốt ($4.47 / 5.0$) — rõ ràng nhưng hơi máy móc | Xuất sắc ($4.93 / 5.0$) — phát âm tự nhiên, rõ ràng |
| **Độ tự nhiên (Naturalness)** | Trung bình-Khá ($3.80 / 5.0$) — từng từ tự nhiên nhưng câu ghép giật cục | Thấp-Trung bình ($3.00 / 5.0$) — giọng robot, thiếu vi cấu trúc | Rất cao ($4.70 / 5.0$) — ngữ điệu mượt mà, cảm xúc |
| **Tính liên tục (Continuity)** | Trung bình ($3.60 / 5.0$) — dễ có click nếu cross-fade sai | Rất cao ($4.67 / 5.0$) — tham số chuyển tiếp trơn tru | Xuất sắc ($4.87 / 5.0$) — liền mạch tuyệt đối |
| **Độ trễ suy luận (Latency)** | Cực thấp ($< 50$ ms) — chỉ cần tra bảng và ghép nối mảng | Rất thấp ($100 - 200$ ms) — giải thuật quy hoạch động sinh tham số | Trung bình ($900 - 1200$ ms) — tính toán qua mạng nơ-ron sâu |
| **Dung lượng lưu trữ (Memory)** | Rất lớn (yêu cầu hàng trăm MB đến nhiều GB âm thanh gốc) | Cực nhỏ ($< 5$ MB) — chỉ lưu trọng số Gaussian của HMM | Trung bình-Lớn ($50 - 500$ MB) cho trọng số mô hình |
| **Khả năng điều khiển Prosody** | Kém — thay đổi pitch/duration dễ gây méo dạng tín hiệu | Rất linh hoạt — can thiệp trực tiếp vào vector tham số | Rất linh hoạt thông qua conditioning embedding |

---

## 6. TRẢ LỜI CHI TIẾT 10 CÂU HỎI BÁO CÁO

### Câu 1: Vì sao một câu ghép từ các đoạn “tự nhiên” vẫn có thể nghe kém tự nhiên?
**Trả lời:** Từng từ đơn lẻ được thu âm trong trạng thái độc lập (isolated citation form). Khi ghép vào một câu hoàn chỉnh, hệ thống ghép nối làm mất đi:
- Hiện tượng đồng cấu âm (co-articulation) giữa phụ âm cuối từ trước và nguyên âm đầu từ sau.
- Đường bao ngữ điệu tổng thể (prosodic contour / intonation): một câu nói tự nhiên luôn có xu hướng hạ pitch (declination) về cuối câu hoặc lên giọng ở câu hỏi; ghép các từ độc lập khiến cao độ $F_0$ nhảy nhót hỗn loạn.
- Nhịp điệu và năng lượng: chênh lệch mức âm lượng và nhịp gõ âm tiết giữa các lần thu khác nhau phá vỡ cấu trúc nhịp điệu (isochrony/stress) của tiếng nói.

### Câu 2: Phân biệt target cost và join cost. Nếu chỉ tối ưu join cost thì có thể xảy ra vấn đề gì?
**Trả lời:**
- *Target cost:* Đo mức độ phù hợp giữa ứng viên âm thanh được chọn với yêu cầu ngữ cảnh âm học mong muốn (âm vị, trọng âm, cao độ $F_0$, độ dài duration).
- *Join cost:* Đo mức độ liền mạch, tương thích phổ tại biên tiếp giáp giữa 2 ứng viên cạnh nhau (sai khác năng lượng, $F_0$, sai khác phổ log-magnitude $D_{\text{spec}}$).
- *Hậu quả nếu chỉ tối ưu Join Cost:* Hệ thống sẽ có xu hướng chọn những đoạn âm thanh có phổ nối cực kỳ khớp nhau (hoặc các đoạn tĩnh, khoảng lặng, âm kéo dài) bất chấp việc âm đó có đúng ngữ cảnh hay không. Hậu quả là sinh ra câu nói sai hoàn toàn về mặt ngữ điệu, sai trọng âm, hoặc chọn sai biến thể tha âm vị (allophone), khiến câu nói trở nên vô nghĩa dù biên nối nghe rất trơn tru.

### Câu 3: Vì sao cross-fade quá dài có thể làm giảm độ rõ của phụ âm?
**Trả lời:** Các phụ âm (đặc biệt là phụ âm nổ - plosives như $/p, t, k, b, d/$ và phụ âm xát - fricatives như $/s, \int/$) có đặc tính chuyển tiếp cực kỳ ngắn (đoạn bộc phát năng lượng plosive burst chỉ kéo dài từ $5 - 20$ ms). Khi áp dụng cross-fade quá dài ($> 25 - 30$ ms):
- Vùng chồng lấn $\alpha$ làm phân tán năng lượng tập trung của xung bộc phát (transient attack).
- Tạo ra hiện tượng triệt tiêu pha hoặc giao thoa âm thanh (comb filtering), làm phụ âm bị mờ, nghẹt, hoặc nghe như bị nói nhịu / vang vọng hai âm cùng lúc (ghosting/smearing artifact).

### Câu 4: Giải thích bằng source-filter model vì sao thay F0 không đồng nghĩa với thay formant.
**Trả lời:** Trong mô hình Source–Filter $S(z) = E(z) \cdot V(z)$:
- Tần số cơ bản $F_0$ thuộc về **Nguồn kích thích** $E(z)$ (tốc độ dao động đóng/mở của dây thanh quản). Thay đổi $F_0$ chỉ làm thay đổi khoảng cách giữa các vạch điều hòa (harmonic spacing $\Delta f = F_0$).
- Formant ($F_1, F_2, F_3$) thuộc về **Bộ lọc tuyến âm** $V(z)$, được quyết định bởi hình học khoang miệng, vị trí của lưỡi, môi và thanh hầu.
- Khi người nói lên giọng (tăng $F_0$), kích thước vòm họng và chiều dài tuyến âm về cơ bản không thay đổi đột ngột; do đó phong bì phổ (spectral envelope) định hình các đỉnh formant vẫn đứng yên. Kỹ thuật biến đổi pitch chuyên nghiệp (như PSOLA hay Vocoder) chỉ tái sắp xếp chuỗi xung nguồn mà phải bảo toàn nguyên vẹn phong bì phổ của tuyến âm.

### Câu 5: PSOLA thay pitch bằng cách nào? Tại sao voiced fricative có thể bị buzziness khi kéo dài quá mức?
**Trả lời:**
- *Cơ chế PSOLA (Pitch-Synchronous Overlap-Add):* Xác định các điểm đánh dấu chu kỳ thanh môn (pitch marks / glottal closure instants - GCIs) tại các đỉnh tuần hoàn. Tín hiệu được nhân với cửa sổ Hanning độ dài $2 T_0$ căn giữa tại từng GCI. Để đổi pitch, khoảng cách đặt các cửa sổ chồng lấn được nén lại (tăng pitch) hoặc giãn ra (giảm pitch) rồi cộng OLA.
- *Hiện tượng buzziness ở voiced fricative (âm xát hữu thanh, ví dụ $/z, v/ $):* Âm xát hữu thanh là sự hòa trộn giữa nguồn kích thích tuần hoàn của dây thanh và nhiễu xoáy ngẫu nhiên (turbulent aperiodic noise) do luồng khí đi qua khe hẹp. Khi kéo dài duration bằng cách lặp lại nhiều lần các đoạn cửa sổ ngắn tuần hoàn, thành phần nhiễu ngẫu nhiên bị ép lặp lại theo chu kỳ nhân tạo một cách cưỡng bức, làm mất đi tính ngẫu nhiên của nhiễu và biến thành một chuỗi xung gắt mang tính kim loại / tiếng vo ve (buzzing artifact).

### Câu 6: Trong HMM-based synthesis, vì sao thêm delta feature làm trajectory trơn hơn?
**Trả lời:** Nếu chỉ sử dụng đặc trưng tĩnh (static features $c_t$), mô hình HMM sẽ giả định các quan sát trong cùng một trạng thái (state) độc lập có điều kiện, dẫn đến quỹ đạo tham số sinh ra có dạng bậc thang (step function) gián đoạn khi chuyển giữa các trạng thái.  
Khi bổ sung đặc trưng động $\Delta c_t = \frac{1}{2} (c_{t+1} - c_{t-1})$ và $\Delta^2 c_t$:
- Thuật toán sinh tham số MLPG tối đa hóa đồng thời xác suất phân bố của cả thành phần tĩnh và đạo hàm động: $P(C | \lambda)$ với ràng buộc $C = W c$.
- Điều này tạo ra một hệ ràng buộc đạo hàm liên tục, buộc các frame kế tiếp nhau phải có tốc độ biến thiên ăn khớp với đạo hàm trung bình của phân bố Gauss, từ đó làm triệt tiêu các bậc thang gián đoạn và sinh ra đường cong tham số trơn tru tự nhiên.

### Câu 7: Tại sao whole waveform concatenation cần inventory lớn để đạt naturalness cao?
**Trả lời:** Do hệ thống ghép nối dạng sóng nguyên bản (whole waveform) không thực hiện biến đổi tham số âm học (tránh méo tín hiệu do giải thuật biến đổi), tính tự nhiên hoàn toàn phụ thuộc vào việc tìm được một đoạn âm thanh mẫu có sẵn khớp chính xác với ngữ cảnh đích:
- Sự kết hợp giữa các ngữ cảnh âm vị (triphone, quinphone),
- Vị trí từ trong câu (đầu câu, giữa câu, cuối câu),
- Trọng âm, ngữ điệu, trường độ và cao độ mong muốn.
Không gian tổ hợp của tất cả các biến số trên là vô cùng lớn ($N^3$ đến $N^5$). Để hạn chế tối đa việc phải nối ở những điểm không tương thích (gây join cost cao), kho dữ liệu bắt buộc phải có quy mô hàng chục đến hàng trăm giờ thu âm từ một người nói để bao phủ trọn vẹn mọi biến thể phát âm.

### Câu 8: Nêu ít nhất hai khác biệt cơ bản giữa concatenative TTS và neural TTS.
**Trả lời:**
1. *Không gian biểu diễn và tạo sinh:* Concatenative TTS làm việc trực tiếp trên các mảnh waveform có sẵn trong kho lưu trữ (chọn lọc và ghép nối), trong khi Neural TTS học hàm ánh xạ liên tục phi tuyến tính từ biểu diễn văn bản sang không gian đặc trưng âm học (Mel-spectrogram) và dùng mạng nơ-ron sâu (Neural Vocoder) để "tổng hợp suy diễn" ra waveform hoàn toàn mới.
2. *Tính linh hoạt và kích thước mô hình:* Concatenative TTS bị bó hẹp trong tập giọng và miền dữ liệu đã thu âm, kích thước cơ sở dữ liệu tỷ lệ thuận với chất lượng; Neural TTS có dung lượng cố định theo số tham số mô hình, hỗ trợ tổng hợp giọng nói đa người nói (multi-speaker), điều khiển cảm xúc (emotional prosody), và có khả năng tổng quát hóa (generalization) tốt đối với những từ vựng chưa từng xuất hiện trong tập huấn luyện.

### Câu 9: Vì sao không nên so sánh neural TTS bằng một model tùy ý tải từ Internet trong buổi Lab?
**Trả lời:**
- *Thiếu kiểm soát biến số thực nghiệm:* Một model tải ngẫu nhiên trên mạng có thể được huấn luyện trên tập dữ liệu khác biệt hoàn toàn (khác người nói, khác chất lượng phòng thu, tần số lấy mẫu 22.05kHz hoặc 48kHz thay vì 16kHz, khác kiểu văn bản), làm sai lệch phép so sánh công bằng (benchmarking fairness).
- *Tính tái lập (Reproducibility):* Model bên ngoài có thể thay đổi phiên bản phụ thuộc, yêu cầu GPU hoặc kết nối mạng không ổn định, vi phạm quy định môi trường thực hành khép kín (offline-first).
- *Rủi ro bản quyền và an toàn hệ thống:* Tránh thực thi các checkpoint không rõ nguồn gốc trong môi trường làm việc kỹ thuật.

### Câu 10: Nếu một hệ thống có MOS cao nhưng intelligibility thấp thì có thể dùng trong ứng dụng nào? Giải thích.
**Trả lời:** Hệ thống có MOS cao (nghe rất truyền cảm, ấm áp, nhịp điệu mượt mà như người thật) nhưng Intelligibility thấp (tỷ lệ nghe rõ từ kém, dễ bị nuốt âm hoặc phát âm nhầm từ) có thể được ứng dụng trong:
1. *Tạo tiếng ồn nền đối thoại giả lập (Crowd babble noise / Ambient chatter):* Trong làm phim, phát triển game nhập vai (NPC background murmur trong quán rượu, sân bay, nhà ga), nơi người nghe chỉ cần cảm nhận không khí tự nhiên của đám đông trò chuyện mà không cần phân biệt nội dung từng từ.
2. *Âm thanh ru ngủ, thiền định hoặc hỗ trợ giấc ngủ (ASMR / Sleep storytelling):* Nơi ngữ điệu êm dịu, nhịp thở và chất giọng truyền cảm (naturalness) quan trọng hơn việc truyền tải thông tin chính xác.
3. *Đồng hành ảo biểu đạt cảm xúc phi ngôn ngữ (Emotional virtual companions / Non-verbal vocalizations):* Ứng dụng phát ra các tiếng cười, tiếng thở dài, tiếng ngâm nga theo giai điệu nơi ngữ âm chính xác không phải là yêu cầu tiên quyết.

---

## 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

- Bài thực hành Lab 4 đã hoàn thành xuất sắc toàn bộ các mục tiêu từ xử lý tín hiệu kinh điển đến mô hình hóa tham số học sâu:
  - Xây dựng thành công kho ngữ liệu 16 đơn vị âm thanh chuẩn mực cho miền thông báo xếp hàng.
  - Thuật toán Viterbi Unit Selection chứng minh hiệu quả giảm thiểu hơn $33\%$ chi phí gián đoạn phổ $D_{\text{spec}}$.
  - Bộ lọc làm trơn Linear Cross-fade $15$ ms giải quyết triệt để lỗi click tại biên tiếp giáp.
  - Mô phỏng hoàn chỉnh cơ chế sinh tham số trơn tru của HMM parametric synthesis và đo kiểm hiệu năng thời gian thực của Neural TTS ($\text{RTF} \approx 0.20$).
- Toàn bộ mã nguồn, tài liệu hướng dẫn và dữ liệu âm thanh đã được đóng gói chuẩn mực, sẵn sàng cho công tác nghiệm thu và triển khai thực tế.
