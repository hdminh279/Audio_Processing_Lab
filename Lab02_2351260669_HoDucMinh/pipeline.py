"""
pipeline.py  –  Lab 2 • CSE457 • Hồ Đức Minh (2351260669)
======================================================================
Mỗi bước trong chuỗi xử lý MFCC + DTW đều được minh họa bằng biểu đồ
lưu trong figures/.

Luồng xử lý tổng thể:
  RAW WAV
    └─ [BƯỚC 1]  Đọc audio + chuẩn hóa biên độ
    └─ [BƯỚC 2]  So sánh tín hiệu trước / sau Pre-emphasis
    └─ [BƯỚC 3]  Phân khung (Framing) + Cửa sổ Hamming
    └─ [BƯỚC 4]  FFT → Phổ công suất (Power Spectrum)
    └─ [BƯỚC 5]  Mel Filterbank (24 bộ lọc tam giác)
    └─ [BƯỚC 6]  Log-Mel Spectrogram
    └─ [BƯỚC 7]  DCT → MFCC (13 hệ số)
    └─ [BƯỚC 8]  Cepstral Mean Normalization (CMN)
    └─ [BƯỚC 9]  So sánh MFCC trước/sau Delta
    └─ [BƯỚC 10] Short-time Energy, RMS, Magnitude
    └─ [BƯỚC 11] Zero-Crossing Rate (ZCR)
    └─ [BƯỚC 12] Short-time Autocorrelation + Pitch F0
    └─ [BƯỚC 13] Endpoint Detection (VAD Trim)
    └─ [BƯỚC 14] DTW – Cùng từ (same word)
    └─ [BƯỚC 15] DTW – Khác từ (different word)
    └─ [BƯỚC 16] DTW – Bảng quy hoạch động D[i,j] (toàn bộ)
    └─ [BƯỚC 17] Bộ nhận dạng: biểu đồ cột điểm DTW từng từ
    └─ [BƯỚC 18] Confusion Matrix + Accuracy tất cả thí nghiệm
    └─ [BƯỚC 19] Biểu đồ so sánh khoảng cách intra-class vs inter-class
    └─ [BƯỚC 20] Mel Filterbank được xây dựng ra sao (hình minh hoạ)
"""

import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from scipy.fft import dct as scipy_dct
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# ── Cài đặt style đồ thị ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'axes.titlesize':  11,
    'axes.labelsize':  10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.autolayout': True,
})
CMAP_SPEC  = 'magma'
CMAP_COOL  = 'viridis'
CMAP_HOT   = 'plasma'
CMAP_BLUE  = 'Blues'

# ── Hằng số hệ thống ────────────────────────────────────────────────────────
FS        = 16000
FRAME_MS  = 25
HOP_MS    = 10
WIN_LEN   = int(FS * FRAME_MS / 1000)   # 400 mẫu
HOP_LEN   = int(FS * HOP_MS  / 1000)   # 160 mẫu
N_FFT     = 512
N_MELS    = 24
N_MFCC    = 13
ALPHA     = 0.97                         # hệ số pre-emphasis

LABELS   = ['khong', 'mot', 'hai', 'ba', 'bon']
VN_NAMES = {'khong': 'Không', 'mot': 'Một', 'hai': 'Hai',
            'ba': 'Ba',    'bon': 'Bốn'}

BASE_DIR    = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name, fig=None):
    path = FIGURES_DIR / name
    (fig or plt).savefig(path, dpi=180, bbox_inches='tight')
    plt.close('all')
    print(f"  ✔ Đã lưu  figures/{name}")
    return path

# ══════════════════════════════════════════════════════════════════════════════
#  CÁC HÀM XỬ LÝ LÕI  (không in ảnh – chỉ tính toán)
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(path):
    y, _ = librosa.load(path, sr=FS, mono=True)
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def pre_emphasis(y, alpha=ALPHA):
    return lfilter([1.0, -alpha], [1.0], y)

def compute_frames(y, win_len=WIN_LEN, hop_len=HOP_LEN):
    """Trả về (frames raw, frames có Hamming, thời gian tâm frame)."""
    n_frames = 1 + (len(y) - win_len) // hop_len
    n_frames = max(n_frames, 0)
    window   = np.hamming(win_len)
    raw_fr   = np.zeros((n_frames, win_len), dtype=np.float32)
    win_fr   = np.zeros((n_frames, win_len), dtype=np.float32)
    t_center = np.zeros(n_frames, dtype=np.float32)
    for r in range(n_frames):
        s = r * hop_len
        raw_fr[r]   = y[s:s + win_len]
        win_fr[r]   = raw_fr[r] * window
        t_center[r] = (s + win_len / 2) / FS
    return raw_fr, win_fr, t_center

def power_spectrum(win_fr, n_fft=N_FFT):
    """Trả về ma trận phổ công suất (n_frames, n_fft//2+1)."""
    fft_out = np.fft.rfft(win_fr, n=n_fft, axis=1)
    return (np.abs(fft_out) ** 2) / n_fft

def build_mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=FS,
                         fmin=0.0, fmax=None):
    """Tạo ma trận filterbank Mel (n_mels, n_fft//2+1) thủ công."""
    fmax = fmax or sr / 2.0
    mel_min = 1125.0 * np.log(1 + fmin / 700.0)
    mel_max = 1125.0 * np.log(1 + fmax / 700.0)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = 700.0 * (np.exp(mel_pts / 1125.0) - 1.0)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        fl, fc, fr = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        for k in range(fl, fc):
            if fc != fl:
                fb[m-1, k] = (k - fl) / (fc - fl)
        for k in range(fc, fr):
            if fr != fc:
                fb[m-1, k] = (fr - k) / (fr - fc)
    return fb, hz_pts

def compute_time_features(raw_fr, win_fr):
    energy     = np.sum(win_fr ** 2, axis=1)
    magnitude  = np.sum(np.abs(win_fr), axis=1)
    rms        = np.sqrt(np.mean(win_fr ** 2, axis=1))
    log_energy = 10.0 * np.log10(energy + 1e-12)
    signs      = np.sign(raw_fr); signs[signs == 0] = 1
    zcr        = np.sum(np.abs(signs[:, 1:] - signs[:, :-1]), axis=1) \
                 / (2.0 * raw_fr.shape[1])
    return energy, magnitude, rms, log_energy, zcr

def autocorr_pitch(frame, fs=FS, fmin=70, fmax=400):
    wf   = frame * np.hamming(len(frame))
    corr = np.correlate(wf, wf, mode='full')[len(frame)-1:]
    lo, hi = int(fs / fmax), int(fs / fmin)
    hi = min(hi, len(corr) - 1)
    lag = lo + np.argmax(corr[lo:hi])
    f0  = fs / lag if lag > 0 else 0
    return f0, corr

def trim_endpoint(y, top_db=32, margin_ms=50):
    _, idx = librosa.effects.trim(y, top_db=top_db,
                                   frame_length=WIN_LEN, hop_length=HOP_LEN)
    m = int(FS * margin_ms / 1000)
    s = max(0, idx[0] - m)
    e = min(len(y), idx[1] + m)
    return y[s:e], (s, e)

def mfcc_feature(y, n_mfcc=N_MFCC, n_mels=N_MELS,
                 use_delta=False, use_cmn=True):
    y_pre = pre_emphasis(y)
    M = librosa.feature.mfcc(
        y=y_pre, sr=FS, n_mfcc=n_mfcc, n_mels=n_mels,
        n_fft=N_FFT, win_length=WIN_LEN, hop_length=HOP_LEN,
        window='hamming', center=False)
    if use_cmn:
        M = M - np.mean(M, axis=1, keepdims=True)
    if use_delta:
        M = np.vstack([M, librosa.feature.delta(M, order=1)])
    return M.T   # (T, D)

def dtw_distance(X, Y):
    N, M = len(X), len(Y)
    if N == 0 or M == 0:
        return np.inf, [], np.zeros((0,0)), np.zeros((0,0))
    D    = np.full((N+1, M+1), np.inf)
    D[0,0] = 0.0
    back = np.zeros((N+1, M+1, 2), dtype=int)
    for i in range(1, N+1):
        for j in range(1, M+1):
            loc = np.linalg.norm(X[i-1] - Y[j-1])
            opts = [(D[i-1,j-1], i-1,j-1),
                    (D[i-1,j],   i-1,j  ),
                    (D[i,  j-1], i,  j-1)]
            bc, pi, pj = min(opts, key=lambda z: z[0])
            D[i,j]     = loc + bc
            back[i,j]  = (pi, pj)
    path = []; i, j = N, M
    while i > 0 or j > 0:
        path.append((i-1, j-1))
        pi, pj = back[i,j]
        if i == 0 and j == 0: break
        i, j = pi, pj
        if i == 0 and j == 0: break
    path.reverse()
    local = np.array([[np.linalg.norm(X[i]-Y[j]) for j in range(M)]
                      for i in range(N)], dtype=np.float32)
    return D[N,M] / max(len(path),1), path, local, D[1:,1:]

def build_templates(ds_dir, do_trim=True, use_delta=False, n_tpl=3):
    tpl = {lab: [] for lab in LABELS}
    for lab in LABELS:
        for f in sorted((ds_dir/lab).glob("*.wav"))[:n_tpl]:
            y = load_audio(f)
            if do_trim: y, _ = trim_endpoint(y)
            tpl[lab].append(mfcc_feature(y, use_delta=use_delta))
    return tpl

def recognize(path, tpl, do_trim=True, use_delta=False):
    y = load_audio(path)
    if do_trim: y, _ = trim_endpoint(y)
    X = mfcc_feature(y, use_delta=use_delta)
    scores = {lab: min(dtw_distance(X, R)[0] for R in refs)
              for lab, refs in tpl.items()}
    scores = dict(sorted(scores.items(), key=lambda kv: kv[1]))
    return list(scores.keys())[0], scores

def evaluate(ds_dir, tpl, do_trim=True, use_delta=False,
             test_slice=slice(3,None)):
    yt, yp, rows = [], [], []
    for lab in LABELS:
        for f in sorted((ds_dir/lab).glob("*.wav"))[test_slice]:
            pred, sc = recognize(f, tpl, do_trim, use_delta)
            yt.append(lab); yp.append(pred)
            c = list(sc.items())
            rows.append({'file': f.name, 'true': lab, 'pred': pred,
                         'top1_label': c[0][0], 'top1_score': c[0][1],
                         'top2_label': c[1][0], 'top2_score': c[1][1],
                         'margin': c[1][1]-c[0][1],
                         'correct': lab==pred})
    acc = accuracy_score(yt, yp)
    cm  = confusion_matrix(yt, yp, labels=LABELS)
    return acc, cm, pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  SINH TẤT CẢ BIỂU ĐỒ – MỖI BƯỚC XỬ LÝ
# ══════════════════════════════════════════════════════════════════════════════

def plot_step01_raw_waveforms():
    """BƯỚC 1 – Đọc audio thô: waveform 5 từ khác nhau."""
    print("\n[BƯỚC 1] Vẽ waveform thô của cả 5 từ...")
    fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=False)
    for i, lab in enumerate(LABELS):
        y = load_audio(BASE_DIR / f"dataset/{lab}/{lab}_01.wav")
        t = np.arange(len(y)) / FS
        axes[i].plot(t, y, lw=0.9, color='steelblue')
        axes[i].set_title(f"Từ '{VN_NAMES[lab]}' ({lab}_01.wav) "
                          f"| {len(y)/FS:.3f}s | {len(y)} mẫu",
                          fontweight='bold')
        axes[i].set_ylabel("Biên độ")
        axes[i].set_ylim(-1.1, 1.1)
        axes[i].grid(True, linestyle='--', alpha=0.4)
    axes[-1].set_xlabel("Thời gian (giây)")
    fig.suptitle("BƯỚC 1 – Waveform thô (đã chuẩn hóa về [-1, 1])",
                 fontsize=13, fontweight='bold', y=1.01)
    savefig("step01_raw_waveforms.png", fig)


def plot_step02_pre_emphasis():
    """BƯỚC 2 – Pre-emphasis: trước và sau, phổ tần số so sánh."""
    print("[BƯỚC 2] Vẽ tác dụng Pre-emphasis...")
    y  = load_audio(BASE_DIR / "dataset/khong/khong_01.wav")
    ye = pre_emphasis(y)
    t  = np.arange(len(y)) / FS

    # Lấy 1 frame giữa để vẽ phổ
    mid  = len(y) // 2
    seg  = y[mid : mid + WIN_LEN] * np.hamming(WIN_LEN)
    sege = ye[mid : mid + WIN_LEN] * np.hamming(WIN_LEN)
    freqs = np.fft.rfftfreq(N_FFT, d=1/FS)
    sp   = 20 * np.log10(np.abs(np.fft.rfft(seg,  n=N_FFT)) + 1e-8)
    spe  = 20 * np.log10(np.abs(np.fft.rfft(sege, n=N_FFT)) + 1e-8)

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Waveform gốc
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, y, lw=0.8, color='steelblue')
    ax1.set_title("Tín hiệu GỐC  x[n]", fontweight='bold')
    ax1.set_ylabel("Biên độ"); ax1.grid(True, ls='--', alpha=0.4)

    # Waveform sau pre-emphasis
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, ye, lw=0.8, color='tomato')
    ax2.set_title(f"Sau PRE-EMPHASIS  y[n] = x[n] − {ALPHA}·x[n−1]",
                  fontweight='bold')
    ax2.set_ylabel("Biên độ"); ax2.grid(True, ls='--', alpha=0.4)

    # Phổ gốc
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(freqs, sp, lw=1.2, color='steelblue')
    ax3.set_title("Phổ frame giữa – TRƯỚC pre-emphasis", fontweight='bold')
    ax3.set_xlabel("Tần số (Hz)"); ax3.set_ylabel("Cường độ (dB)")
    ax3.grid(True, ls='--', alpha=0.4)

    # Phổ sau pre-emphasis
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(freqs, spe, lw=1.2, color='tomato',
             label='Sau pre-emphasis')
    ax4.plot(freqs, sp,  lw=1.0, color='steelblue', alpha=0.5,
             linestyle='--', label='Trước pre-emphasis')
    ax4.set_title("Phổ frame giữa – SAU pre-emphasis\n"
                  "(tần số cao được nâng lên ~+6dB/octave)",
                  fontweight='bold')
    ax4.set_xlabel("Tần số (Hz)"); ax4.set_ylabel("Cường độ (dB)")
    ax4.legend(fontsize=8); ax4.grid(True, ls='--', alpha=0.4)

    fig.suptitle("BƯỚC 2 – Pre-emphasis: bù suy hao tần số cao (-6dB/oct)",
                 fontsize=13, fontweight='bold')
    savefig("step02_pre_emphasis.png", fig)


def plot_step03_framing_windowing():
    """BƯỚC 3 – Phân khung (Framing) + cửa sổ Hamming."""
    print("[BƯỚC 3] Vẽ Framing + Hamming window...")
    y  = load_audio(BASE_DIR / "dataset/khong/khong_01.wav")
    ye = pre_emphasis(y)

    # Lấy 4 frame liên tiếp tại vùng speech
    start_sample = int(0.42 * FS)    # ~420ms – ngay giữa chữ
    window = np.hamming(WIN_LEN)
    n_show = 4
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    fig, axes = plt.subplots(n_show, 1, figsize=(12, 8), sharex=False)
    for idx in range(n_show):
        s   = start_sample + idx * HOP_LEN
        raw = ye[s : s + WIN_LEN]
        win = raw * window
        t_fr = np.arange(WIN_LEN) / FS * 1000  # ms

        axes[idx].fill_between(t_fr, win,  alpha=0.35, color=colors[idx])
        axes[idx].plot(t_fr, raw, lw=1.0, color='gray',
                       linestyle='--', label='Frame thô (chưa cửa sổ)')
        axes[idx].plot(t_fr, win, lw=1.5, color=colors[idx],
                       label=f'Frame #{idx+1} × Hamming')
        axes[idx].plot(t_fr, window * max(abs(raw)), lw=1.2,
                       color='black', linestyle=':', label='Dạng cửa sổ Hamming')
        axes[idx].set_title(
            f"Frame #{idx+1} | Bắt đầu tại {s/FS*1000:.0f}ms "
            f"| Chồng lấn {WIN_LEN-HOP_LEN} mẫu ({(WIN_LEN-HOP_LEN)/FS*1000:.0f}ms)",
            fontweight='bold')
        axes[idx].set_ylabel("Biên độ")
        axes[idx].grid(True, ls='--', alpha=0.4)
        if idx == 0:
            axes[idx].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel("Thời gian trong frame (ms)")
    fig.suptitle("BƯỚC 3 – Phân khung 25ms, hop 10ms, cửa sổ Hamming\n"
                 "(Chồng lấn 15ms = 60% – giảm spectral leakage)",
                 fontsize=13, fontweight='bold')
    savefig("step03_framing_windowing.png", fig)


def plot_step04_power_spectrum():
    """BƯỚC 4 – FFT → Phổ công suất (Power Spectrum) của 1 frame."""
    print("[BƯỚC 4] Vẽ FFT Power Spectrum...")
    y    = load_audio(BASE_DIR / "dataset/khong/khong_01.wav")
    ye   = pre_emphasis(y)
    s    = int(0.42 * FS)
    frame_raw = ye[s : s + WIN_LEN] * np.hamming(WIN_LEN)

    fft_full = np.fft.fft(frame_raw, n=N_FFT)
    freqs    = np.fft.rfftfreq(N_FFT, d=1/FS)
    power    = (np.abs(np.fft.rfft(frame_raw, n=N_FFT)) ** 2) / N_FFT
    power_db = 10 * np.log10(power + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Real + Imag phần toàn FFT (đối xứng)
    axes[0].plot(np.real(fft_full), lw=1.0, color='steelblue',
                 label='Re[X(k)]')
    axes[0].plot(np.imag(fft_full), lw=1.0, color='tomato', alpha=0.7,
                 label='Im[X(k)]')
    axes[0].axvline(N_FFT//2, color='black', ls='--', lw=1, label='Nyquist')
    axes[0].set_title("Kết quả FFT (phần Re & Im)\n"
                      "Chỉ lấy nửa dương (0 → N/2)", fontweight='bold')
    axes[0].set_xlabel("Chỉ số bin k"); axes[0].set_ylabel("Biên độ")
    axes[0].legend(fontsize=8); axes[0].grid(True, ls='--', alpha=0.4)

    # |X(k)|² / N – tuyến tính
    axes[1].fill_between(freqs, power, alpha=0.4, color='darkorange')
    axes[1].plot(freqs, power, lw=1.2, color='darkorange')
    axes[1].set_title("Phổ công suất  P[k] = |X[k]|² / N\n"
                      "(Tuyến tính – Hz)", fontweight='bold')
    axes[1].set_xlabel("Tần số (Hz)"); axes[1].set_ylabel("Công suất")
    axes[1].grid(True, ls='--', alpha=0.4)

    # Phổ công suất dB
    axes[2].fill_between(freqs, power_db, alpha=0.4, color='mediumseagreen')
    axes[2].plot(freqs, power_db, lw=1.2, color='mediumseagreen')
    axes[2].set_title("Phổ công suất  10·log₁₀(P[k])  [dB]\n"
                      "(Logarithm – cảm nhận tai người)", fontweight='bold')
    axes[2].set_xlabel("Tần số (Hz)"); axes[2].set_ylabel("Cường độ (dB)")
    axes[2].grid(True, ls='--', alpha=0.4)

    fig.suptitle("BƯỚC 4 – FFT → Phổ công suất của 1 frame (tại giữa từ 'không')",
                 fontsize=13, fontweight='bold')
    savefig("step04_power_spectrum.png", fig)


def plot_step05_mel_filterbank():
    """BƯỚC 5 – Mel Filterbank: vẽ 24 bộ lọc tam giác."""
    print("[BƯỚC 5] Vẽ Mel Filterbank...")
    fb, hz_pts = build_mel_filterbank()
    freqs = np.fft.rfftfreq(N_FFT, d=1/FS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Hình trái: Mel Filterbank trên trục Hz
    cmap_fb = plt.cm.rainbow(np.linspace(0, 1, N_MELS))
    for m in range(N_MELS):
        axes[0].plot(freqs, fb[m], lw=1.5, color=cmap_fb[m])
    axes[0].set_title(f"{N_MELS} bộ lọc tam giác Mel trên trục tần số Hz\n"
                       "(Khoảng cách rộng dần ở tần số cao)",
                       fontweight='bold')
    axes[0].set_xlabel("Tần số (Hz)")
    axes[0].set_ylabel("Trọng số bộ lọc")
    axes[0].grid(True, ls='--', alpha=0.4)
    # Đánh dấu các tâm bộ lọc
    for m in range(N_MELS):
        peak_k = np.argmax(fb[m])
        axes[0].axvline(freqs[peak_k], color=cmap_fb[m],
                        alpha=0.3, lw=0.7, ls=':')

    # --- Hình phải: Minh họa thang Mel vs Hz
    hz_lin   = np.linspace(0, FS/2, 500)
    mel_lin  = 1125.0 * np.log(1 + hz_lin / 700.0)
    mel_axis = np.linspace(0, mel_lin[-1], N_MELS + 2)
    hz_mel   = 700.0 * (np.exp(mel_axis / 1125.0) - 1.0)

    ax2t = axes[1]
    ax2t.plot(hz_lin, mel_lin, lw=2, color='darkblue',
              label='Thang Mel: B(f) = 1125·ln(1+f/700)')
    ax2t.scatter(hz_mel, mel_axis, color='red', s=50, zorder=5,
                 label=f'{N_MELS+2} điểm chia đều trên Mel')
    ax2t.set_title("Thang Mel so với tần số vật lý Hz\n"
                   "(Khoảng cách đều trên Mel → không đều trên Hz)",
                   fontweight='bold')
    ax2t.set_xlabel("Tần số vật lý (Hz)")
    ax2t.set_ylabel("Mel")
    ax2t.legend(fontsize=8)
    ax2t.grid(True, ls='--', alpha=0.4)

    fig.suptitle("BƯỚC 5 – Mel Filterbank: mô phỏng thính giác ốc tai người",
                 fontsize=13, fontweight='bold')
    savefig("step05_mel_filterbank.png", fig)


def plot_step06_log_mel_spectrogram():
    """BƯỚC 6 – Log-Mel Spectrogram (trước DCT)."""
    print("[BƯỚC 6] Vẽ Log-Mel Spectrogram...")
    pairs = [('khong', 'Không'), ('hai', 'Hai'), ('bon', 'Bốn')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (lab, vn) in zip(axes, pairs):
        y, _ = trim_endpoint(load_audio(BASE_DIR/f"dataset/{lab}/{lab}_01.wav"))
        ye   = pre_emphasis(y)
        mel_s = librosa.feature.melspectrogram(
            y=ye, sr=FS, n_mels=N_MELS, n_fft=N_FFT,
            win_length=WIN_LEN, hop_length=HOP_LEN,
            window='hamming', center=False)
        log_mel = np.log(mel_s + 1e-9)
        t_ax    = np.arange(log_mel.shape[1]) * HOP_LEN / FS * 1000
        freqs   = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=FS/2)
        im = ax.imshow(log_mel, origin='lower', aspect='auto',
                       cmap=CMAP_SPEC, interpolation='nearest',
                       extent=[t_ax[0], t_ax[-1], freqs[0], freqs[-1]])
        ax.set_title(f"Log-Mel Spectrogram: '{vn}'\n"
                     f"({log_mel.shape[1]} frames × {N_MELS} Mel bins)",
                     fontweight='bold')
        ax.set_xlabel("Thời gian (ms)")
        ax.set_ylabel("Tần số Mel (Hz)")
        fig.colorbar(im, ax=ax, label="ln(Năng lượng)")
    fig.suptitle("BƯỚC 6 – Log-Mel Spectrogram\n"
                 "(Phổ công suất sau khi qua 24 bộ lọc Mel + lấy Logarithm)",
                 fontsize=13, fontweight='bold')
    savefig("step06_log_mel_spectrogram.png", fig)


def plot_step07_mfcc_raw():
    """BƯỚC 7 – MFCC thô (sau DCT, TRƯỚC CMN)."""
    print("[BƯỚC 7] Vẽ MFCC thô (trước CMN)...")
    pairs = [('khong', 'Không'), ('mot', 'Một'), ('hai', 'Hai')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (lab, vn) in zip(axes, pairs):
        y, _ = trim_endpoint(load_audio(BASE_DIR/f"dataset/{lab}/{lab}_01.wav"))
        ye   = pre_emphasis(y)
        M_raw = librosa.feature.mfcc(
            y=ye, sr=FS, n_mfcc=N_MFCC, n_mels=N_MELS,
            n_fft=N_FFT, win_length=WIN_LEN, hop_length=HOP_LEN,
            window='hamming', center=False)   # KHÔNG CMN
        im = ax.imshow(M_raw, origin='lower', aspect='auto',
                       cmap=CMAP_SPEC, interpolation='nearest')
        ax.set_title(f"MFCC (thô, trước CMN): '{vn}'\n"
                     f"({M_raw.shape[1]} frames × {N_MFCC} hệ số)",
                     fontweight='bold')
        ax.set_xlabel("Chỉ số Frame")
        ax.set_ylabel("Hệ số MFCC  c₀ – c₁₂")
        fig.colorbar(im, ax=ax)
    fig.suptitle("BƯỚC 7 – DCT → MFCC thô (Chưa chuẩn hóa CMN)\n"
                 "Hàng 0 = c₀ (DC/energy), Hàng 12 = c₁₂ (biến thiên nhanh)",
                 fontsize=13, fontweight='bold')
    savefig("step07_mfcc_raw.png", fig)


def plot_step08_cmn():
    """BƯỚC 8 – Cepstral Mean Normalization (CMN): trước và sau."""
    print("[BƯỚC 8] Vẽ tác dụng CMN...")
    y, _ = trim_endpoint(load_audio(BASE_DIR / "dataset/khong/khong_01.wav"))
    ye   = pre_emphasis(y)
    M_raw = librosa.feature.mfcc(
        y=ye, sr=FS, n_mfcc=N_MFCC, n_mels=N_MELS,
        n_fft=N_FFT, win_length=WIN_LEN, hop_length=HOP_LEN,
        window='hamming', center=False)
    M_cmn = M_raw - np.mean(M_raw, axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    vmax = max(np.abs(M_raw).max(), np.abs(M_cmn).max())

    im1 = axes[0].imshow(M_raw, origin='lower', aspect='auto',
                          cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title("MFCC TRƯỚC CMN\n(Giá trị lớn/nhỏ lệch nhau nhiều)",
                       fontweight='bold')
    axes[0].set_xlabel("Frame"); axes[0].set_ylabel("Hệ số MFCC")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(M_cmn, origin='lower', aspect='auto',
                          cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title("MFCC SAU CMN\n(Mỗi hệ số được trừ giá trị trung bình)",
                       fontweight='bold')
    axes[1].set_xlabel("Frame"); axes[1].set_ylabel("Hệ số MFCC")
    fig.colorbar(im2, ax=axes[1])

    # Vẽ mean của từng hệ số trước/sau CMN
    coef_idx = np.arange(N_MFCC)
    axes[2].barh(coef_idx - 0.2, np.mean(M_raw, axis=1),
                 height=0.35, color='steelblue', label='Trước CMN (mean ≠ 0)')
    axes[2].barh(coef_idx + 0.2, np.mean(M_cmn, axis=1),
                 height=0.35, color='tomato',    label='Sau CMN (mean ≈ 0)')
    axes[2].axvline(0, color='black', lw=1.5)
    axes[2].set_title("Giá trị trung bình mỗi hệ số MFCC\ntrước và sau CMN",
                       fontweight='bold')
    axes[2].set_xlabel("Giá trị trung bình")
    axes[2].set_ylabel("Hệ số MFCC  c₀ – c₁₂")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, ls='--', alpha=0.4)

    fig.suptitle("BƯỚC 8 – CMN: Trừ trung bình từng hệ số để loại trừ ảnh hưởng kênh/micro",
                 fontsize=13, fontweight='bold')
    savefig("step08_cmn.png", fig)


def plot_step09_delta_mfcc():
    """BƯỚC 9 – MFCC tĩnh vs MFCC + Delta (26 chiều)."""
    print("[BƯỚC 9] Vẽ MFCC + Delta...")
    y, _ = trim_endpoint(load_audio(BASE_DIR / "dataset/khong/khong_01.wav"))
    feat_static = mfcc_feature(y, use_delta=False)   # (T, 13)
    feat_delta  = mfcc_feature(y, use_delta=True)    # (T, 26)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im1 = axes[0].imshow(feat_static.T, origin='lower', aspect='auto',
                          cmap=CMAP_SPEC, interpolation='nearest')
    axes[0].set_title(f"MFCC tĩnh: 13 chiều\nShape = {feat_static.shape}",
                       fontweight='bold')
    axes[0].set_xlabel("Frame"); axes[0].set_ylabel("Hệ số")
    axes[0].axhline(12.5, color='white', lw=2, ls='--')
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(feat_delta.T, origin='lower', aspect='auto',
                          cmap=CMAP_SPEC, interpolation='nearest')
    axes[1].axhline(12.5, color='yellow', lw=2, ls='--',
                    label='Ranh giới: MFCC | Δ-MFCC')
    axes[1].set_title(f"MFCC + Δ-MFCC: 26 chiều\nShape = {feat_delta.shape}",
                       fontweight='bold')
    axes[1].set_xlabel("Frame"); axes[1].set_ylabel("Hệ số")
    axes[1].legend(loc='upper right', fontsize=8)
    fig.colorbar(im2, ax=axes[1])

    fig.suptitle("BƯỚC 9 – Delta MFCC: bổ sung đạo hàm bậc 1 theo thời gian\n"
                 "(c₀–c₁₂ tĩnh | Δc₀–Δc₁₂ động – biểu diễn tốc độ biến đổi phổ)",
                 fontsize=13, fontweight='bold')
    savefig("step09_delta_mfcc.png", fig)


def plot_step10_energy_rms_mag():
    """BƯỚC 10 – Short-time Energy, RMS, Magnitude."""
    print("[BƯỚC 10] Vẽ Energy / RMS / Magnitude...")
    y    = load_audio(BASE_DIR / "dataset/khong/khong_01.wav")
    rf, wf, tc = compute_frames(y)
    energy, magnitude, rms, log_energy, _ = compute_time_features(rf, wf)
    t_audio = np.arange(len(y)) / FS

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t_audio, y, lw=0.8, color='black')
    axes[0].set_title("Waveform gốc từ 'không'", fontweight='bold')
    axes[0].set_ylabel("Biên độ")
    axes[0].grid(True, ls='--', alpha=0.4)

    axes[1].plot(tc, log_energy, lw=1.5, color='firebrick',
                 label='Log-Energy (dB)')
    axes[1].set_ylabel("Log-Energy (dB)")
    axes[1].grid(True, ls='--', alpha=0.4)
    axes[1].legend(loc='upper right')

    axes[2].plot(tc, rms, lw=1.5, color='darkorange', label='RMS')
    axes[2].plot(tc, np.sqrt(energy / WIN_LEN), lw=1.0,
                 color='gold', ls='--', alpha=0.7, label='√(Energy/L)')
    axes[2].set_ylabel("RMS / √E")
    axes[2].grid(True, ls='--', alpha=0.4)
    axes[2].legend(loc='upper right')

    axes[3].plot(tc, magnitude, lw=1.5, color='slateblue', label='Magnitude')
    axes[3].set_ylabel("Magnitude")
    axes[3].set_xlabel("Thời gian (giây)")
    axes[3].grid(True, ls='--', alpha=0.4)
    axes[3].legend(loc='upper right')

    fig.suptitle("BƯỚC 10 – Đặc trưng biên độ ngắn hạn: Energy | RMS | Magnitude",
                 fontsize=13, fontweight='bold')
    savefig("step10_energy_rms_mag.png", fig)


def plot_step11_zcr():
    """BƯỚC 11 – Zero-Crossing Rate (ZCR) cho 3 từ."""
    print("[BƯỚC 11] Vẽ Zero-Crossing Rate...")
    words = ['khong', 'mot', 'ba']
    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    for row, lab in enumerate(words):
        y = load_audio(BASE_DIR / f"dataset/{lab}/{lab}_01.wav")
        rf, wf, tc = compute_frames(y)
        _, _, _, log_energy, zcr = compute_time_features(rf, wf)
        t_audio = np.arange(len(y)) / FS

        axes[row, 0].plot(t_audio, y, lw=0.8, color='black', alpha=0.6)
        axes[row, 0].plot(tc, log_energy / log_energy.max(),
                          lw=1.5, color='firebrick', label='Log-Energy (normalized)')
        axes[row, 0].set_title(f"Từ '{VN_NAMES[lab]}' – Waveform + Log-Energy",
                                fontweight='bold')
        axes[row, 0].set_ylabel("Biên độ / Energy")
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, ls='--', alpha=0.4)

        axes[row, 1].plot(tc, zcr, lw=1.5, color='teal')
        axes[row, 1].fill_between(tc, zcr, alpha=0.25, color='teal')
        axes[row, 1].axhline(0.25, color='red', ls='--', lw=1.2,
                              label='Ngưỡng thường dùng ~0.25')
        axes[row, 1].set_title(f"Zero-Crossing Rate (ZCR) – '{VN_NAMES[lab]}'",
                                fontweight='bold')
        axes[row, 1].set_ylabel("ZCR (lần/mẫu)")
        axes[row, 1].set_ylim(0, 0.6)
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, ls='--', alpha=0.4)
        if row == 2:
            axes[row, 0].set_xlabel("Thời gian (giây)")
            axes[row, 1].set_xlabel("Thời gian (giây)")

    fig.suptitle("BƯỚC 11 – Zero-Crossing Rate (ZCR)\n"
                 "ZCR cao = noise/âm vô thanh | ZCR thấp = nguyên âm hữu thanh",
                 fontsize=13, fontweight='bold')
    savefig("step11_zcr.png", fig)


def plot_step12_autocorr_pitch():
    """BƯỚC 12 – Short-time Autocorrelation + ước lượng Pitch F0."""
    print("[BƯỚC 12] Vẽ Autocorrelation + Pitch...")
    y   = load_audio(BASE_DIR / "dataset/khong/khong_01.wav")
    rf, wf, tc = compute_frames(y)

    # Chọn 3 frame: silence, voiced vowel, noise
    frame_silence = rf[5]   # đầu file – silence
    frame_voiced  = rf[int(len(rf)*0.55)]  # giữa từ – nguyên âm
    frame_noise   = rf[-5]  # cuối file – noise

    frames_to_show = [
        (frame_silence, "Frame #5 – Silence / Nhiễu nền", 'gray'),
        (frame_voiced,  f"Frame #{int(len(rf)*0.55)} – Nguyên âm hữu thanh", 'steelblue'),
        (frame_noise,   f"Frame #{len(rf)-5} – Noise cuối file", 'tomato'),
    ]

    fig, axes = plt.subplots(len(frames_to_show), 2, figsize=(14, 9))
    for row, (frame, title, color) in enumerate(frames_to_show):
        f0, corr = autocorr_pitch(frame)
        lags = np.arange(len(corr))
        t_fr = np.arange(WIN_LEN) / FS * 1000

        axes[row, 0].plot(t_fr, frame, lw=1.2, color=color)
        axes[row, 0].set_title(f"Dạng sóng – {title}", fontweight='bold')
        axes[row, 0].set_ylabel("Biên độ")
        axes[row, 0].grid(True, ls='--', alpha=0.4)
        if row == len(frames_to_show)-1:
            axes[row, 0].set_xlabel("Thời gian trong frame (ms)")

        axes[row, 1].plot(lags / FS * 1000, corr, lw=1.2, color=color)
        if f0 > 0 and f0 < 400:
            peak_lag = int(FS / f0)
            axes[row, 1].axvline(peak_lag / FS * 1000, color='red',
                                  lw=2, ls='--', label=f'Đỉnh lag → F₀ ≈ {f0:.0f} Hz')
        axes[row, 1].set_title(f"Short-time Autocorrelation  (F₀ ≈ {f0:.0f} Hz)",
                                fontweight='bold')
        axes[row, 1].set_ylabel("R[k]")
        axes[row, 1].set_xlim(0, 15)
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, ls='--', alpha=0.4)
        if row == len(frames_to_show)-1:
            axes[row, 1].set_xlabel("Lag (ms)")

    fig.suptitle("BƯỚC 12 – Short-time Autocorrelation + Ước lượng Pitch F₀\n"
                 "Nguyên âm hữu thanh có đỉnh tương quan rõ → F₀ ≈ Fs/lag_peak",
                 fontsize=13, fontweight='bold')
    savefig("step12_autocorr_pitch.png", fig)


def plot_step13_endpoint_detection():
    """BƯỚC 13 – Endpoint Detection / VAD Trim."""
    print("[BƯỚC 13] Vẽ Endpoint Detection...")
    y = load_audio(BASE_DIR / "dataset/khong/khong_01.wav")
    rf, wf, tc = compute_frames(y)
    _, _, _, log_energy, zcr = compute_time_features(rf, wf)
    y_trim, (s_idx, e_idx) = trim_endpoint(y)
    t_audio = np.arange(len(y)) / FS

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Waveform với vùng speech highlight
    axes[0].plot(t_audio, y, lw=0.9, color='dimgray', alpha=0.6,
                 label='Tín hiệu gốc')
    axes[0].axvspan(s_idx/FS, e_idx/FS, alpha=0.2, color='lime',
                    label='Vùng giữ lại')
    axes[0].axvline(s_idx/FS, color='green', lw=2, ls='--', label=f'Start={s_idx/FS:.2f}s')
    axes[0].axvline(e_idx/FS, color='red',   lw=2, ls='--', label=f'End={e_idx/FS:.2f}s')
    axes[0].set_title("Waveform + Vùng tiếng nói được giữ lại (Endpoint Detection)",
                       fontweight='bold')
    axes[0].set_ylabel("Biên độ")
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, ls='--', alpha=0.4)

    # Log-Energy
    axes[1].plot(tc, log_energy, lw=1.5, color='firebrick', label='Log-Energy')
    axes[1].axvspan(s_idx/FS, e_idx/FS, alpha=0.2, color='lime')
    axes[1].axvline(s_idx/FS, color='green', lw=2, ls='--')
    axes[1].axvline(e_idx/FS, color='red',   lw=2, ls='--')
    axes[1].set_title("Log-Energy → Xác định vùng Speech thô (top_db = 32 dB)",
                       fontweight='bold')
    axes[1].set_ylabel("Log-Energy (dB)")
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, ls='--', alpha=0.4)

    # ZCR
    axes[2].plot(tc, zcr, lw=1.5, color='teal', label='ZCR')
    axes[2].axvspan(s_idx/FS, e_idx/FS, alpha=0.2, color='lime')
    axes[2].axvline(s_idx/FS, color='green', lw=2, ls='--')
    axes[2].axvline(e_idx/FS, color='red',   lw=2, ls='--')
    axes[2].set_title("ZCR → Phân biệt noise (ZCR cao) với voiced speech (ZCR thấp)",
                       fontweight='bold')
    axes[2].set_ylabel("ZCR (tỉ lệ)")
    axes[2].set_xlabel("Thời gian (giây)")
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, ls='--', alpha=0.4)

    dur_orig = len(y)/FS
    dur_trim = len(y_trim)/FS
    fig.suptitle(f"BƯỚC 13 – Endpoint Detection (VAD Trim)\n"
                 f"Gốc: {dur_orig:.3f}s → Sau cắt: {dur_trim:.3f}s "
                 f"(Giảm {(1-dur_trim/dur_orig)*100:.1f}% silence)",
                 fontsize=13, fontweight='bold')
    savefig("step13_endpoint_detection.png", fig)


def plot_step14_dtw_same_word():
    """BƯỚC 14 – DTW Cùng từ: khong_01 vs khong_02."""
    print("[BƯỚC 14] Vẽ DTW cùng từ...")
    y1, _ = trim_endpoint(load_audio(BASE_DIR/"dataset/khong/khong_01.wav"))
    y2, _ = trim_endpoint(load_audio(BASE_DIR/"dataset/khong/khong_02.wav"))
    X1 = mfcc_feature(y1); X2 = mfcc_feature(y2)
    cost, path, local, D_acc = dtw_distance(X1, X2)
    px, py = zip(*path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Ma trận local distance
    im0 = axes[0].imshow(local, origin='lower', aspect='auto', cmap=CMAP_COOL)
    axes[0].set_title("Ma trận khoảng cách cục bộ\n||MFCC_i − MFCC_j||₂",
                       fontweight='bold')
    axes[0].set_xlabel("Frame khong_02"); axes[0].set_ylabel("Frame khong_01")
    fig.colorbar(im0, ax=axes[0], label="Euclidean dist")

    # Ma trận tích lũy D
    im1 = axes[1].imshow(D_acc, origin='lower', aspect='auto', cmap='hot_r')
    axes[1].plot(py, px, color='cyan', lw=2.5, label=f'Warping Path')
    axes[1].set_title(f"Bảng tích lũy D[i,j] + Optimal Path\n"
                       f"DTW_norm = {cost:.3f}  |Path| = {len(path)}",
                       fontweight='bold')
    axes[1].set_xlabel("Frame khong_02"); axes[1].set_ylabel("Frame khong_01")
    axes[1].legend(fontsize=9)
    fig.colorbar(im1, ax=axes[1], label="Chi phí tích lũy")

    # Chẩn đoán đường đi
    diag = [j - i for i, j in path]
    axes[2].plot(range(len(diag)), diag, lw=1.5, color='purple')
    axes[2].axhline(0, color='black', lw=1.5, ls='--',
                    label='Đường chéo lý tưởng (bằng tốc độ)')
    axes[2].fill_between(range(len(diag)), diag, alpha=0.3, color='purple')
    axes[2].set_title("Độ lệch so với đường chéo  (j − i)\n"
                       "(= 0: tốc độ bằng nhau | >0: X nhanh hơn)",
                       fontweight='bold')
    axes[2].set_xlabel("Bước đi trên path")
    axes[2].set_ylabel("j − i")
    axes[2].legend(); axes[2].grid(True, ls='--', alpha=0.4)

    fig.suptitle(f"BƯỚC 14 – DTW CÙNG TỪ:  khong_01  vs  khong_02\n"
                 f"DTW_norm = {cost:.3f}  (nhỏ → khớp tốt)",
                 fontsize=13, fontweight='bold')
    savefig("step14_dtw_same_word.png", fig)


def plot_step15_dtw_diff_word():
    """BƯỚC 15 – DTW Khác từ: khong_01 vs mot_01."""
    print("[BƯỚC 15] Vẽ DTW khác từ...")
    y1, _ = trim_endpoint(load_audio(BASE_DIR/"dataset/khong/khong_01.wav"))
    y2, _ = trim_endpoint(load_audio(BASE_DIR/"dataset/mot/mot_01.wav"))
    X1 = mfcc_feature(y1); X2 = mfcc_feature(y2)
    cost_s, path_s, local_s, _ = dtw_distance(X1, X1)   # self
    cost_d, path_d, local_d, D_d = dtw_distance(X1, X2)  # different
    px, py = zip(*path_d)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(local_d, origin='lower', aspect='auto', cmap=CMAP_HOT)
    axes[0].set_title("Ma trận khoảng cách cục bộ\nkhong_01  vs  mot_01",
                       fontweight='bold')
    axes[0].set_xlabel("Frame mot_01"); axes[0].set_ylabel("Frame khong_01")
    fig.colorbar(im0, ax=axes[0], label="Euclidean dist")

    im1 = axes[1].imshow(D_d, origin='lower', aspect='auto', cmap='hot_r')
    axes[1].plot(py, px, color='cyan', lw=2.5, label='Warping Path')
    axes[1].set_title(f"Bảng tích lũy D[i,j] + Optimal Path\n"
                       f"DTW_norm = {cost_d:.3f}  |Path| = {len(path_d)}",
                       fontweight='bold')
    axes[1].set_xlabel("Frame mot_01"); axes[1].set_ylabel("Frame khong_01")
    axes[1].legend(fontsize=9)
    fig.colorbar(im1, ax=axes[1], label="Chi phí tích lũy")

    # So sánh cost: cùng từ vs khác từ
    categories = ['Self-match\n(cost ≈ 0)',
                  'Cùng từ\n(khong_01 vs khong_02)',
                  'Khác từ\n(khong_01 vs mot_01)']
    values = [cost_s, dtw_distance(X1, mfcc_feature(
        trim_endpoint(load_audio(BASE_DIR/"dataset/khong/khong_02.wav"))[0]))[0],
              cost_d]
    colors_bar = ['#2ca02c', '#1f77b4', '#d62728']
    bars = axes[2].bar(categories, values, color=colors_bar, width=0.5,
                        edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.3,
                     f'{val:.2f}', ha='center', fontweight='bold')
    axes[2].set_title("So sánh DTW_norm\ngiữa các kịch bản đối sánh",
                       fontweight='bold')
    axes[2].set_ylabel("DTW_norm (Chi phí)")
    axes[2].grid(True, ls='--', alpha=0.4, axis='y')

    fig.suptitle(f"BƯỚC 15 – DTW KHÁC TỪ:  khong_01  vs  mot_01\n"
                 f"DTW_norm = {cost_d:.3f}  (lớn hơn {cost_d/max(values[1],0.001):.1f}× so với cùng từ)",
                 fontsize=13, fontweight='bold')
    savefig("step15_dtw_diff_word.png", fig)


def plot_step16_dtw_matrix_detail():
    """BƯỚC 16 – Minh họa toàn bộ bảng DP và 3 bước chuyển (chéo/dọc/ngang)."""
    print("[BƯỚC 16] Vẽ bảng DP DTW chi tiết + 3 bước chuyển trạng thái...")

    # Dùng chuỗi MFCC ngắn (10 frame) để dễ nhìn
    y1, _ = trim_endpoint(load_audio(BASE_DIR/"dataset/khong/khong_01.wav"))
    y2, _ = trim_endpoint(load_audio(BASE_DIR/"dataset/khong/khong_03.wav"))
    X = mfcc_feature(y1)[:12]
    Y = mfcc_feature(y2)[:12]
    _, path, local, D_acc = dtw_distance(X, Y)
    px, py = zip(*path)

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # Local distance matrix
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(local, origin='lower', aspect='auto', cmap=CMAP_COOL)
    ax0.set_title("C[i,j] = ||X[i]−Y[j]||₂\n(Ma trận local distance)",
                   fontweight='bold')
    ax0.set_xlabel("j (chuỗi Y)"); ax0.set_ylabel("i (chuỗi X)")
    for i in range(len(X)):
        for j in range(len(Y)):
            ax0.text(j, i, f'{local[i,j]:.0f}', ha='center', va='center',
                     fontsize=6, color='white' if local[i,j] > local.max()*0.5 else 'black')
    fig.colorbar(im0, ax=ax0)

    # Accumulated cost + path
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(D_acc, origin='lower', aspect='auto', cmap='hot_r')
    ax1.plot(py, px, 'c-o', lw=2, ms=5, label='Optimal Path')
    ax1.set_title("D[i,j] = C[i,j] + min(3 ô lân cận)\n(Bảng quy hoạch động tích lũy)",
                   fontweight='bold')
    ax1.set_xlabel("j (chuỗi Y)"); ax1.set_ylabel("i (chuỗi X)")
    ax1.legend(fontsize=8)
    fig.colorbar(im1, ax=ax1)

    # Sơ đồ 3 bước chuyển trạng thái
    ax2 = fig.add_subplot(gs[2])
    ax2.set_xlim(-1, 3); ax2.set_ylim(-1, 3)
    ax2.set_aspect('equal')
    ax2.set_title("3 Bước chuyển trạng thái cục bộ\n"
                   "Từ ô đen (i,j) nhìn ngược về 3 ô nguồn",
                   fontweight='bold')
    # Vẽ lưới
    for x in range(3):
        for y_val in range(3):
            ax2.add_patch(plt.Rectangle((x-0.5, y_val-0.5), 1, 1,
                           fill=False, edgecolor='gray', lw=0.8))
    # Ô đích (i,j) = (1,1) – màu đen
    ax2.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, color='black', alpha=0.9, zorder=3))
    ax2.text(1, 1, "(i, j)", ha='center', va='center',
             color='white', fontweight='bold', fontsize=10, zorder=4)
    # 3 ô nguồn
    sources = [
        (0, 0, '(i-1,j-1)\nChéo\n(Cùng tốc độ)', 'steelblue', 'w'),
        (0, 1, '(i-1, j)\nDọc\n(Y bị kéo dài)', 'darkorange', 'w'),
        (1, 0, '(i, j-1)\nNgang\n(X bị kéo dài)', 'mediumseagreen', 'w'),
    ]
    for x, y_val, label, color, tc2 in sources:
        ax2.add_patch(plt.Rectangle((x-0.5, y_val-0.5), 1, 1,
                       color=color, alpha=0.8, zorder=3))
        ax2.text(x, y_val, label, ha='center', va='center',
                 color=tc2, fontsize=7.5, fontweight='bold', zorder=4)
        ax2.annotate('', xy=(1, 1), xytext=(x, y_val),
                      arrowprops=dict(arrowstyle='->', color=color, lw=2),
                      zorder=5)
    ax2.axis('off')

    fig.suptitle("BƯỚC 16 – Cơ chế Quy hoạch động DTW:\n"
                 "Local distance matrix | Accumulated cost | 3 bước chuyển trạng thái",
                 fontsize=13, fontweight='bold')
    savefig("step16_dtw_matrix_detail.png", fig)


def plot_step17_recognizer_scores():
    """BƯỚC 17 – Bộ nhận dạng: biểu đồ cột điểm DTW từng từ."""
    print("[BƯỚC 17] Vẽ biểu đồ điểm nhận dạng...")
    tpl = build_templates(BASE_DIR/"dataset", do_trim=True, n_tpl=3)
    test_files = [(lab, sorted((BASE_DIR/"dataset"/lab).glob("*.wav"))[3])
                  for lab in LABELS]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)
    for ax, (true_lab, fpath) in zip(axes, test_files):
        pred, scores = recognize(fpath, tpl)
        labs   = list(scores.keys())
        vals   = list(scores.values())
        colors = ['#2ca02c' if l == true_lab else
                  ('#d62728' if l == pred and pred != true_lab else '#aec7e8')
                  for l in labs]
        bars = ax.bar([VN_NAMES[l] for l in labs], vals,
                       color=colors, edgecolor='black', lw=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + vals[-1]*0.02, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_title(f"File: {fpath.name}\nThực tế: '{VN_NAMES[true_lab]}' "
                     f"→ Dự đoán: '{VN_NAMES[pred]}'",
                     fontweight='bold', fontsize=9)
        ax.set_ylabel("DTW_norm"); ax.set_xlabel("Nhãn từ")
        ax.grid(True, ls='--', alpha=0.4, axis='y')
        ok = '✓ ĐÚNG' if pred == true_lab else '✗ SAI'
        ax.set_facecolor('#f0fff0' if pred == true_lab else '#fff0f0')
        ax.text(0.5, 0.97, ok, transform=ax.transAxes,
                ha='center', va='top', fontweight='bold',
                color='green' if pred == true_lab else 'red', fontsize=11)

    fig.suptitle("BƯỚC 17 – Bộ nhận dạng Nearest-Template DTW\n"
                 "Cột xanh = nhãn đúng | Cột thấp nhất = nhãn dự đoán",
                 fontsize=13, fontweight='bold')
    savefig("step17_recognizer_scores.png", fig)


def plot_step18_confusion_matrices():
    """BƯỚC 18 – Confusion Matrix + Accuracy cho tất cả thí nghiệm."""
    print("[BƯỚC 18] Vẽ Confusion Matrix các thí nghiệm...")
    ds = BASE_DIR / "dataset"
    ds2 = BASE_DIR / "dataset_speaker2"

    tpl_base   = build_templates(ds, do_trim=True,  n_tpl=3)
    tpl_notrim = build_templates(ds, do_trim=False, n_tpl=3)
    tpl_delta  = build_templates(ds, do_trim=True,  use_delta=True, n_tpl=3)
    tpl_1      = build_templates(ds, do_trim=True,  n_tpl=1)

    experiments = [
        ("Baseline\nTrim+13MFCC+3Tpl", *evaluate(ds,  tpl_base,   True,  False),  'Blues'),
        ("E1: No Trim\n13MFCC+3Tpl",   *evaluate(ds,  tpl_notrim, False, False),  'Reds'),
        ("E2: +Delta\nTrim+26MFCC+3Tpl",*evaluate(ds, tpl_delta,  True,  True),   'Greens'),
        ("E3: 1 Template\nTrim+13MFCC", *evaluate(ds, tpl_1,      True,  False),  'Oranges'),
        ("E4: Cross-speaker\n(Spk2→Spk1Tpl)",
         *evaluate(ds2, tpl_base, True, False, test_slice=slice(None)), 'Purples'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (name, acc, cm, df, cmap) in zip(axes, experiments):
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=[VN_NAMES[l] for l in LABELS],
                    yticklabels=[VN_NAMES[l] for l in LABELS],
                    linewidths=0.5, ax=ax, cbar=False)
        ax.set_title(f"{name}\nAcc = {acc*100:.1f}%", fontweight='bold')
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")

    fig.suptitle("BƯỚC 18 – Confusion Matrix của 5 thí nghiệm\n"
                 "Đường chéo chính = dự đoán đúng",
                 fontsize=13, fontweight='bold')
    savefig("step18_confusion_matrices.png", fig)

    # Trả về để in bảng
    acc_vals = [e[1] for e in experiments]
    names    = [e[0].replace('\n', ' ') for e in experiments]
    df_res   = pd.DataFrame({'Thí nghiệm': names,
                              'Accuracy': [f'{a*100:.1f}%' for a in acc_vals]})
    print(df_res.to_string(index=False))
    return experiments[0][3]   # df_base


def plot_step19_distance_distribution():
    """BƯỚC 19 – Phân bố khoảng cách intra-class vs inter-class."""
    print("[BƯỚC 19] Vẽ phân bố khoảng cách DTW intra vs inter-class...")
    tpl = build_templates(BASE_DIR/"dataset", do_trim=True, n_tpl=3)

    intra_costs, inter_costs = [], []
    for lab in LABELS:
        test_files = sorted((BASE_DIR/"dataset"/lab).glob("*.wav"))[3:]
        for f in test_files:
            y, _ = trim_endpoint(load_audio(f))
            X    = mfcc_feature(y)
            for lab2, refs in tpl.items():
                d = min(dtw_distance(X, R)[0] for R in refs)
                if lab2 == lab:
                    intra_costs.append(d)
                else:
                    inter_costs.append(d)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    bins = np.linspace(0, max(max(intra_costs), max(inter_costs)) + 2, 30)
    axes[0].hist(intra_costs, bins=bins, alpha=0.7, color='steelblue',
                  label=f'Intra-class (cùng từ)\nn={len(intra_costs)}, '
                         f'mean={np.mean(intra_costs):.1f}')
    axes[0].hist(inter_costs, bins=bins, alpha=0.7, color='tomato',
                  label=f'Inter-class (khác từ)\nn={len(inter_costs)}, '
                         f'mean={np.mean(inter_costs):.1f}')
    axes[0].axvline(np.mean(intra_costs), color='steelblue', lw=2, ls='--')
    axes[0].axvline(np.mean(inter_costs), color='tomato',    lw=2, ls='--')
    axes[0].set_title("Phân bố DTW_norm\nIntra-class vs Inter-class",
                       fontweight='bold')
    axes[0].set_xlabel("DTW_norm"); axes[0].set_ylabel("Số lượng cặp")
    axes[0].legend(); axes[0].grid(True, ls='--', alpha=0.4)

    # Boxplot
    try:
        axes[1].boxplot([intra_costs, inter_costs],
                         tick_labels=['Intra-class\n(cùng từ)',
                                      'Inter-class\n(khác từ)'],
                         patch_artist=True,
                         boxprops=dict(facecolor='lightblue', color='steelblue'),
                         medianprops=dict(color='red', lw=2))
    except TypeError:
        axes[1].boxplot([intra_costs, inter_costs],
                         labels=['Intra-class\n(cùng từ)',
                                 'Inter-class\n(khác từ)'],
                         patch_artist=True,
                         boxprops=dict(facecolor='lightblue', color='steelblue'),
                         medianprops=dict(color='red', lw=2))
    axes[1].set_title("Boxplot DTW_norm\n(Phân tách rõ → Hệ thống tốt)",
                       fontweight='bold')
    axes[1].set_ylabel("DTW_norm"); axes[1].grid(True, ls='--', alpha=0.4)

    sep = (np.mean(inter_costs) - np.mean(intra_costs)) / \
          np.sqrt((np.std(intra_costs)**2 + np.std(inter_costs)**2) / 2)
    fig.suptitle(f"BƯỚC 19 – Phân bố khoảng cách DTW\n"
                 f"Cohen's d = {sep:.2f}  (độ phân tách giữa 2 lớp)",
                 fontsize=13, fontweight='bold')
    savefig("step19_distance_distribution.png", fig)


def plot_step20_mfcc_all5_words():
    """BƯỚC 20 – MFCC Heatmap cả 5 từ: so sánh trực quan."""
    print("[BƯỚC 20] Vẽ MFCC Heatmap cả 5 từ...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    vmin, vmax = None, None
    mfccs = []
    for lab in LABELS:
        y, _ = trim_endpoint(load_audio(BASE_DIR/f"dataset/{lab}/{lab}_01.wav"))
        M = mfcc_feature(y).T   # (13, T)
        mfccs.append(M)
        cur_min, cur_max = M.min(), M.max()
        vmin = cur_min if vmin is None else min(vmin, cur_min)
        vmax = cur_max if vmax is None else max(vmax, cur_max)

    for ax, lab, M in zip(axes, LABELS, mfccs):
        im = ax.imshow(M, origin='lower', aspect='auto',
                        cmap=CMAP_SPEC, vmin=vmin, vmax=vmax,
                        interpolation='nearest')
        ax.set_title(f"'{VN_NAMES[lab]}'\n"
                     f"({M.shape[1]} frames × {M.shape[0]} coeff)",
                     fontweight='bold')
        ax.set_xlabel("Frame")
        ax.set_ylabel("MFCC coeff")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("BƯỚC 20 – MFCC Heatmap toàn bộ 5 từ vựng\n"
                 "(Cùng thang màu → dễ so sánh cấu trúc phổ âm tiết)",
                 fontsize=13, fontweight='bold')
    savefig("step20_mfcc_all5_words.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  Lab 2 – Pipeline đầy đủ với biểu đồ từng bước xử lý")
    print("=" * 65)

    plot_step01_raw_waveforms()
    plot_step02_pre_emphasis()
    plot_step03_framing_windowing()
    plot_step04_power_spectrum()
    plot_step05_mel_filterbank()
    plot_step06_log_mel_spectrogram()
    plot_step07_mfcc_raw()
    plot_step08_cmn()
    plot_step09_delta_mfcc()
    plot_step10_energy_rms_mag()
    plot_step11_zcr()
    plot_step12_autocorr_pitch()
    plot_step13_endpoint_detection()
    plot_step14_dtw_same_word()
    plot_step15_dtw_diff_word()
    plot_step16_dtw_matrix_detail()
    plot_step17_recognizer_scores()
    df_base = plot_step18_confusion_matrices()
    plot_step19_distance_distribution()
    plot_step20_mfcc_all5_words()

    # Xuất results.csv
    print("\n[*] Xuất results.csv...")
    tpl = build_templates(BASE_DIR/"dataset", do_trim=True, n_tpl=3)
    acc, cm, df = evaluate(BASE_DIR/"dataset", tpl)
    df.to_csv(BASE_DIR/"results.csv", index=False)
    print(f"  ✔ Accuracy = {acc*100:.1f}%  |  results.csv đã lưu")

    print("\n" + "=" * 65)
    print(f"  Hoàn tất! Tất cả {20} biểu đồ đã lưu vào  figures/")
    print("=" * 65)
