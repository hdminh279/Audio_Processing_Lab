"""
Lab04.py: Full pipeline executing all requirements and experiments of Lab 4:
  4.1. Text normalization & inventory stats
  4.2. Baseline concatenation (raw_concat.wav, boundary markers, waveform & spectrogram)
  4.3. Unit selection & Smoothing (Acoustic pause & micro-fade) -> Crystal smooth best_concat.wav
  4.4. Prosody & Source-Filter analysis (High-resolution F0 estimation, pitch shifting, time stretching)
  4.5. HMM-based synthesis (State sequence, F0 & Formant trajectories inspection)
  4.6. Neural TTS offline inference (3 test sentences, mel-spectrogram, latency)
  4.7. Evaluation & Comparison table generation
"""

import os
import time
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

# Import local backends
import hmm_tts_backend
import neural_tts_backend

FS = 16000
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "lab4_data"
UNITS_DIR = DATA_DIR / "units"
AUDIO_OUT_DIR = BASE_DIR / "audio_output"
FIG_DIR = BASE_DIR / "figures"

AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}

def normalize_queue_text(text: str):
    clean = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
    tokens = []
    for tok in clean.split():
        if tok.isdigit():
            tokens.extend([DIGIT_MAP[d] for d in tok])
        else:
            tokens.append(tok)
    return tokens

def load_unit(path: Path, sr: int = FS):
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != sr:
        x = librosa.resample(x, orig_sr=fs, target_sr=sr)
    return x

def rms(x):
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def match_rms(x, target=0.08):
    cur = rms(x)
    if cur > 0:
        y = x * (target / cur)
    else:
        y = x
    return np.clip(y, -0.95, 0.95).astype(np.float32)

def crossfade(a, b, n):
    """
    Standard linear crossfade as described in section 5.3 of Lab 4
    """
    n = min(n, len(a), len(b))
    if n == 0:
        return np.concatenate([a, b])
    alpha = np.linspace(0.0, 1.0, n, endpoint=True)
    mid = (1 - alpha) * a[-n:] + alpha * b[:n]
    return np.concatenate([a[:-n], mid, b[n:]])

def concat_with_natural_cadence(units, pause_ms=25, fade_ms=5):
    """
    Combines units with micro natural inter-word cadence (25ms) and 5ms boundary softening.
    Completely eliminates comb-filtering, tremolo phase cancellation, and boundary pops.
    """
    pause = np.zeros(int(FS * pause_ms / 1000), dtype=np.float32)
    fade_len = int(FS * fade_ms / 1000)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))
    
    segs = []
    for i, u in enumerate(units):
        u_clean = u.copy()
        if len(u_clean) > 2 * fade_len:
            u_clean[:fade_len] *= ramp
            u_clean[-fade_len:] *= ramp[::-1]
        segs.append(u_clean)
        if i < len(units) - 1:
            segs.append(pause)
    return np.concatenate(segs)

def join_cost(a, b, sr=16000, ms=25, nfft=512):
    n = int(sr * ms / 1000)
    n = min(n, len(a), len(b))
    wa = np.hanning(n)
    wb = np.hanning(n)
    A = np.fft.rfft(a[-n:] * wa, n=nfft)
    B = np.fft.rfft(b[:n] * wb, n=nfft)
    La = 20 * np.log10(np.maximum(np.abs(A), 1e-8))
    Lb = 20 * np.log10(np.maximum(np.abs(B), 1e-8))
    return float(np.mean((La - Lb) ** 2))

TOKEN_TO_BASE = {
    "mời": "moi", "khách": "khach", "hàng": "hang", "số": "so",
    "không": "khong", "một": "mot", "hai": "hai", "ba": "ba", "bốn": "bon",
    "năm": "nam", "sáu": "sau", "bảy": "bay", "tám": "tam", "chín": "chin",
    "đến": "den", "quầy": "quay"
}

# -------------------------------------------------------------
# Module A: Inventory Analysis & Statistics
# -------------------------------------------------------------
def run_inventory_analysis():
    print("\n--- [A] INVENTORY ANALYSIS & STATS ---")
    meta_path = DATA_DIR / "metadata.csv"
    durations = []
    rmss = []
    with open(meta_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            durations.append(float(row["duration"]))
            rmss.append(float(row["rms"]))
            
    print(f"Total units cataloged: {len(durations)}")
    print(f"Duration: mean={np.mean(durations):.3f}s, min={np.min(durations):.3f}s, max={np.max(durations):.3f}s")
    print(f"RMS Energy: mean={np.mean(rmss):.4f}, min={np.min(rmss):.4f}, max={np.max(rmss):.4f}")
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(durations, bins=10, color='skyblue', edgecolor='black', alpha=0.8)
    axs[0].set_title("Unit Duration Distribution (s)")
    axs[0].set_xlabel("Duration (s)")
    axs[0].set_ylabel("Count")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    axs[1].hist(rmss, bins=8, color='salmon', edgecolor='black', alpha=0.8)
    axs[1].set_title("Unit RMS Energy Distribution")
    axs[1].set_xlabel("RMS")
    axs[1].set_ylabel("Count")
    axs[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig1_path = FIG_DIR / "fig1_inventory_stats.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

# -------------------------------------------------------------
# Module B: Baseline Concatenation
# -------------------------------------------------------------
def run_baseline_concatenation(case_text="Mời khách hàng số 23 đến quầy 5."):
    print("\n--- [B] BASELINE CONCATENATION ---")
    tokens = normalize_queue_text(case_text)
    print("Tokens normalized:", tokens)
    
    paths = [UNITS_DIR / f"{TOKEN_TO_BASE[tok]}_01.wav" for tok in tokens]
    units_audio = [match_rms(load_unit(p)) for p in paths]
    
    raw_audio = np.concatenate(units_audio)
    raw_path = AUDIO_OUT_DIR / "raw_concat.wav"
    sf.write(raw_path, raw_audio, FS, subtype='PCM_16')
    print(f"Saved baseline: {raw_path} ({len(raw_audio)/FS:.2f}s)")
    
    boundaries = [0]
    for u in units_audio:
        boundaries.append(boundaries[-1] + len(u))
    boundaries = np.array(boundaries)
    
    return tokens, paths, units_audio, raw_audio, boundaries

# -------------------------------------------------------------
# Module C: Unit Selection & Smoothing (Cross-fade)
# -------------------------------------------------------------
def run_unit_selection_and_smoothing(tokens, raw_audio, boundaries):
    print("\n--- [C] UNIT SELECTION & SMOOTHING ---")
    n_tokens = len(tokens)
    cands = ['01', '02']
    
    cand_audios = {}
    for tok in tokens:
        base = TOKEN_TO_BASE[tok]
        cand_audios[(tok, '01')] = match_rms(load_unit(UNITS_DIR / f"{base}_01.wav"))
        cand_audios[(tok, '02')] = match_rms(load_unit(UNITS_DIR / f"{base}_02.wav"))
        
    dp = np.zeros((n_tokens, 2))
    backpointer = np.zeros((n_tokens, 2), dtype=int)
    
    for t in range(1, n_tokens):
        prev_tok = tokens[t - 1]
        curr_tok = tokens[t]
        for curr_i, curr_c in enumerate(cands):
            best_prev_cost = float('inf')
            best_prev_idx = 0
            for prev_i, prev_c in enumerate(cands):
                cost = join_cost(cand_audios[(prev_tok, prev_c)], cand_audios[(curr_tok, curr_c)])
                total = dp[t - 1, prev_i] + cost
                if total < best_prev_cost:
                    best_prev_cost = total
                    best_prev_idx = prev_i
            dp[t, curr_i] = best_prev_cost
            backpointer[t, curr_i] = best_prev_idx
            
    best_cand_seq = [0] * n_tokens
    best_cand_seq[-1] = int(np.argmin(dp[-1]))
    for t in range(n_tokens - 1, 0, -1):
        best_cand_seq[t - 1] = backpointer[t, best_cand_seq[t]]
    opt_candidates = [cands[idx] for idx in best_cand_seq]
    
    def get_seq_cost(seq_cands):
        tot = 0.0
        for t in range(n_tokens - 1):
            tot += join_cost(cand_audios[(tokens[t], seq_cands[t])], cand_audios[(tokens[t+1], seq_cands[t+1])])
        return tot
        
    cost_all_01 = get_seq_cost(['01'] * n_tokens)
    cost_all_02 = get_seq_cost(['02'] * n_tokens)
    cost_optimal = get_seq_cost(opt_candidates)
    
    print(f"Join Cost (Candidate '01' only)  : {cost_all_01:.3f} dB^2")
    print(f"Join Cost (Candidate '02' only)  : {cost_all_02:.3f} dB^2")
    print(f"Join Cost (Unit Selection optimal): {cost_optimal:.3f} dB^2 | Path: {opt_candidates}")
    
    opt_units = [cand_audios[(tokens[i], opt_candidates[i])] for i in range(n_tokens)]
    
    # Save standard xfade experiments as required by section 5.3 & 6
    for ms_val in [0, 5, 15, 30]:
        y = opt_units[0]
        n_fade = int(FS * ms_val / 1000)
        for u in opt_units[1:]:
            y = crossfade(y, u, n_fade)
        out_name = f"concat_xfade_{ms_val}ms.wav"
        sf.write(AUDIO_OUT_DIR / out_name, y, FS, subtype='PCM_16')
        
    # High-quality best_concat.wav using natural inter-word cadence
    y_best = concat_with_natural_cadence(opt_units, pause_ms=25, fade_ms=5)
    best_path = AUDIO_OUT_DIR / "best_concat.wav"
    sf.write(best_path, y_best, FS, subtype='PCM_16')
    print(f"Saved crystal-clear smooth best_concat.wav: {best_path} ({len(y_best)/FS:.2f}s)")
            
    # Figure 2: Spectrogram and Waveform Boundary Inspection
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex='col')
    b0 = boundaries[1]
    zoom_range = slice(max(0, b0 - int(0.12 * FS)), min(len(raw_audio), b0 + int(0.12 * FS)))
    time_ax = np.arange(zoom_range.start, zoom_range.stop) / FS
    
    axs[0, 0].plot(time_ax, raw_audio[zoom_range], color='royalblue', lw=1)
    axs[0, 0].axvline(b0 / FS, color='red', linestyle='--', label='Boundary (0ms)')
    axs[0, 0].set_title("Boundary Zoom: Raw Concat (0ms Click Discontinuity)")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    zoom_best = y_best[zoom_range]
    axs[1, 0].plot(time_ax[:len(zoom_best)], zoom_best, color='darkgreen', lw=1)
    axs[1, 0].axvline(b0 / FS, color='red', linestyle='--', label='Boundary (Smooth Cadence)')
    axs[1, 0].set_title("Boundary Zoom: Smooth Concat (Cadence Windowed)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    D_raw = librosa.amplitude_to_db(np.abs(librosa.stft(raw_audio[zoom_range], n_fft=512, hop_length=64)), ref=np.max)
    D_smooth = librosa.amplitude_to_db(np.abs(librosa.stft(zoom_best, n_fft=512, hop_length=64)), ref=np.max)
    
    axs[0, 1].imshow(D_raw, origin='lower', aspect='auto', cmap='magma')
    axs[0, 1].set_title("Spectrogram (0ms Click Discontinuity)")
    axs[0, 1].set_ylabel("Freq Bins")
    
    axs[1, 1].imshow(D_smooth, origin='lower', aspect='auto', cmap='magma')
    axs[1, 1].set_title("Spectrogram (Smooth Boundary Transition)")
    axs[1, 1].set_xlabel("STFT Frames")
    axs[1, 1].set_ylabel("Freq Bins")

    plt.tight_layout()
    fig2_path = FIG_DIR / "fig2_boundary_spectrogram.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = ["All Cand 01", "All Cand 02", "Unit Selection (Viterbi)"]
    costs = [cost_all_01, cost_all_02, cost_optimal]
    bars = ax.bar(methods, costs, color=['#4a90e2', '#50e3c2', '#f5a623'], edgecolor='black')
    ax.set_ylabel("Total Join Cost (dB²)")
    ax.set_title("Unit Selection: Join Cost Comparison")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig3_path = FIG_DIR / "fig3_unit_selection_cost.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved: {fig3_path}")
    
    return opt_candidates, y_best

# -------------------------------------------------------------
# Module D: Prosody & Source-Filter Experiments
# -------------------------------------------------------------
def run_prosody_experiments(test_unit_name="moi_01.wav"):
    print("\n--- [D] PROSODY & SOURCE-FILTER EXPERIMENTS (PRISTINE) ---")
    u_path = UNITS_DIR / test_unit_name
    wav, sr = sf.read(u_path, dtype="float32")
    wav = match_rms(wav)
    
    f0, voiced_flag, voiced_probs = librosa.pyin(wav, fmin=80, fmax=400, sr=sr)
    valid_f0 = f0[voiced_flag]
    mean_f0 = np.nanmean(valid_f0)
    print(f"Pitch measurement on '{test_unit_name}': Mean F0 = {mean_f0:.1f} Hz")
    
    # Pristine Pitch Shifting (n_fft=2048, bins_per_octave=24)
    wav_p_minus = librosa.effects.pitch_shift(wav, sr=sr, n_steps=-2, n_fft=2048, bins_per_octave=24)
    wav_p_plus = librosa.effects.pitch_shift(wav, sr=sr, n_steps=+2, n_fft=2048, bins_per_octave=24)
    wav_p_minus = match_rms(wav_p_minus)
    wav_p_plus = match_rms(wav_p_plus)
    sf.write(AUDIO_OUT_DIR / "prosody_pitch_minus2.wav", wav_p_minus, sr)
    sf.write(AUDIO_OUT_DIR / "prosody_pitch_plus2.wav", wav_p_plus, sr)
    
    # Pristine Time Stretching (n_fft=2048)
    wav_s_slow = librosa.effects.time_stretch(wav, rate=0.8, n_fft=2048)
    wav_s_fast = librosa.effects.time_stretch(wav, rate=1.2, n_fft=2048)
    wav_s_slow = match_rms(wav_s_slow)
    wav_s_fast = match_rms(wav_s_fast)
    sf.write(AUDIO_OUT_DIR / "prosody_speed_0.8x.wav", wav_s_slow, sr)
    sf.write(AUDIO_OUT_DIR / "prosody_speed_1.2x.wav", wav_s_fast, sr)
    print("Saved pristine prosody shifted audios to audio_output/")

# -------------------------------------------------------------
# Module E: HMM-based Synthesis Backend & Parameter Inspection
# -------------------------------------------------------------
def run_hmm_synthesis(case_text="Mời khách hàng số 23 đến quầy 5"):
    print("\n--- [E] HMM PARAMETRIC TTS SYNTHESIS ---")
    y_hmm, sr, tracks = hmm_tts_backend.synthesize_hmm(case_text)
    hmm_path = AUDIO_OUT_DIR / "hmm_tts.wav"
    sf.write(hmm_path, y_hmm, sr, subtype='PCM_16')
    print(f"Saved HMM TTS audio: {hmm_path} ({len(y_hmm)/sr:.2f}s)")
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    frames = np.arange(len(tracks["f0"]))
    time_ax = frames * tracks["frame_shift_ms"] / 1000.0
    
    axs[0].plot(time_ax, tracks["f0"], color='crimson', lw=2)
    axs[0].set_ylabel("F0 (Hz)")
    axs[0].set_title("HMM Synthesis Parameter Trajectories: Pitch (F0)")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    
    axs[1].plot(time_ax, tracks["f1"], color='purple', lw=1.8, label='F1 (Vocal tract opening)')
    axs[1].plot(time_ax, tracks["f2"], color='teal', lw=1.8, label='F2 (Tongue advancement)')
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_title("HMM Formant Trajectories (Spectral Envelope)")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.5)
    
    axs[2].plot(time_ax, tracks["energy"], color='darkgoldenrod', lw=1.8)
    axs[2].set_ylabel("Gain")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("Acoustic Energy Contour")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig4_path = FIG_DIR / "fig4_hmm_parameter_tracks.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"Saved: {fig4_path}")
    
    return y_hmm, tracks

# -------------------------------------------------------------
# Module F: Neural TTS Offline Inference
# -------------------------------------------------------------
def run_neural_tts(test_sentences):
    print("\n--- [F] DEEP LEARNING NEURAL TTS INFERENCE ---")
    neural_results = []
    
    for idx, text in enumerate(test_sentences):
        y_nn, sr, extra = neural_tts_backend.synthesize(text)
        out_file = AUDIO_OUT_DIR / f"neural_tts_s{idx+1}.wav"
        sf.write(out_file, y_nn, sr, subtype='PCM_16')
        if idx == 0:
            sf.write(AUDIO_OUT_DIR / "neural_tts.wav", y_nn, sr, subtype='PCM_16')
            
        neural_results.append({
            "sentence": text,
            "latency_ms": round(extra["latency_ms"], 2),
            "duration_s": round(extra["duration_s"], 2),
            "rtf": round((extra["latency_ms"] / 1000.0) / extra["duration_s"], 3)
        })
        print(f"  Sentence {idx+1}: Latency = {extra['latency_ms']:.1f}ms | Audio Dur = {extra['duration_s']:.2f}s | RTF = {neural_results[-1]['rtf']}")
        
    with open(AUDIO_OUT_DIR / "neural_latency.json", "w", encoding="utf-8") as f:
        json.dump(neural_results, f, ensure_ascii=False, indent=2)
        
    return neural_results

# -------------------------------------------------------------
# Module G: Multi-Backend Comparison & Figure 5
# -------------------------------------------------------------
def run_comparison_and_figure5(raw_audio, best_concat_audio, hmm_audio, neural_audio):
    print("\n--- [G] EVALUATION & COMPARISON FIGURE 5 ---")
    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=False)
    
    labels = [
        ("Raw Concatenation (0ms)", raw_audio, 'midnightblue'),
        ("Best Unit Selection (Natural Cadence)", best_concat_audio, 'forestgreen'),
        ("Statistical Parametric (HMM + Resynthesis)", hmm_audio, 'firebrick'),
        ("Pretrained Deep Learning Neural TTS", neural_audio, 'darkviolet')
    ]
    
    for i, (title, sig, col) in enumerate(labels):
        t = np.arange(len(sig)) / FS
        axs[i].plot(t, sig, color=col, lw=0.8)
        axs[i].set_title(title, fontsize=11, fontweight='bold')
        axs[i].set_ylabel("Amplitude")
        axs[i].set_ylim(-1.0, 1.0)
        axs[i].grid(True, alpha=0.3)
        
    axs[3].set_xlabel("Time (seconds)")
    plt.tight_layout()
    fig5_path = FIG_DIR / "fig5_output_comparison.png"
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f"Saved: {fig5_path}")

def main():
    print("=================================================================")
    print("STARTING FULL LAB 4 SPEECH SYNTHESIS PIPELINE EXECUTION (PRISTINE)")
    print("=================================================================")
    
    run_inventory_analysis()
    
    case_phrase = "Mời khách hàng số 23 đến quầy 5."
    tokens, paths, units_audio, raw_audio, boundaries = run_baseline_concatenation(case_phrase)
    
    opt_cands, best_concat_audio = run_unit_selection_and_smoothing(tokens, raw_audio, boundaries)
    
    run_prosody_experiments("moi_01.wav")
    
    y_hmm, tracks = run_hmm_synthesis(case_phrase)
    
    test_phrases = [
        "Mời khách hàng số 23 đến quầy 5.",
        "Mời khách hàng số 104 đến quầy 2.",
        "Mời khách hàng số 89 đến quầy 7."
    ]
    neural_stats = run_neural_tts(test_phrases)
    y_neural, _, _ = neural_tts_backend.synthesize(case_phrase)
    
    run_comparison_and_figure5(raw_audio, best_concat_audio, y_hmm, y_neural)
    
    print("\n=================================================================")
    print("ALL EXPERIMENTS EXECUTED WITH PURE NATURAL CADENCE!")
    print("=================================================================")

if __name__ == "__main__":
    main()
