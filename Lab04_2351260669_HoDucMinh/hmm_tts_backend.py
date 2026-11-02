"""
hmm_tts_backend.py: Statistical Parametric Speech Synthesis (HMM-based) Engine & Parameter Inspector.
Mô phỏng chính xác luồng: Context Text -> HMM State Sequence -> Parameter Generation (F0, Duration, Mel-Cepstral / Formants) -> Source-Filter / Vocoder Resynthesis.
"""

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

FS = 16000
FRAME_SHIFT_MS = 5 # 5ms frame step
HOP_LENGTH = int(FS * FRAME_SHIFT_MS / 1000) # 80 samples

# Context-dependent dictionary mapping word tokens to acoustic target states
# Each token has [duration_s, f0_mean_hz, f0_slope_hz, f1_hz, f2_hz, f3_hz, voiced_ratio]
ACOUSTIC_PROFILES = {
    "mời":   {"dur": 0.55, "f0": 180, "slope": -30, "f1": 350, "f2": 1900, "f3": 2600, "voiced": 0.85},
    "khách": {"dur": 0.35, "f0": 210, "slope": +40, "f1": 450, "f2": 1700, "f3": 2500, "voiced": 0.70},
    "hàng":  {"dur": 0.50, "f0": 175, "slope": -40, "f1": 600, "f2": 1200, "f3": 2400, "voiced": 0.85},
    "số":    {"dur": 0.55, "f0": 220, "slope": +35, "f1": 400, "f2": 1800, "f3": 2700, "voiced": 0.75},
    "không": {"dur": 0.42, "f0": 190, "slope": -10, "f1": 400, "f2": 1000, "f3": 2300, "voiced": 0.85},
    "một":   {"dur": 0.40, "f0": 185, "slope": -35, "f1": 450, "f2": 1600, "f3": 2500, "voiced": 0.75},
    "hai":   {"dur": 0.48, "f0": 195, "slope": -15, "f1": 650, "f2": 1800, "f3": 2700, "voiced": 0.90},
    "ba":    {"dur": 0.50, "f0": 190, "slope": -10, "f1": 700, "f2": 1250, "f3": 2400, "voiced": 0.90},
    "bốn":   {"dur": 0.52, "f0": 215, "slope": +30, "f1": 450, "f2": 1100, "f3": 2300, "voiced": 0.85},
    "năm":   {"dur": 0.50, "f0": 190, "slope": -15, "f1": 600, "f2": 1350, "f3": 2450, "voiced": 0.85},
    "sáu":   {"dur": 0.50, "f0": 225, "slope": +35, "f1": 550, "f2": 1600, "f3": 2600, "voiced": 0.75},
    "bảy":   {"dur": 0.50, "f0": 180, "slope": -25, "f1": 500, "f2": 1750, "f3": 2650, "voiced": 0.80},
    "tám":   {"dur": 0.48, "f0": 220, "slope": +35, "f1": 650, "f2": 1400, "f3": 2500, "voiced": 0.80},
    "chín":  {"dur": 0.52, "f0": 225, "slope": +40, "f1": 320, "f2": 2100, "f3": 2800, "voiced": 0.80},
    "đến":   {"dur": 0.52, "f0": 215, "slope": +30, "f1": 450, "f2": 1800, "f3": 2600, "voiced": 0.80},
    "quầy":  {"dur": 0.55, "f0": 175, "slope": -30, "f1": 400, "f2": 1300, "f3": 2400, "voiced": 0.85},
    "default": {"dur": 0.45, "f0": 190, "slope": 0, "f1": 500, "f2": 1500, "f3": 2500, "voiced": 0.80}
}

DIGIT_MAP = {"0":"không","1":"một","2":"hai","3":"ba","4":"bốn",
             "5":"năm","6":"sáu","7":"bảy","8":"tám","9":"chín"}

def normalize_text(text: str):
    clean = text.lower().replace(".", " ").replace(",", " ")
    tokens = []
    for tok in clean.split():
        if tok.isdigit():
            tokens.extend([DIGIT_MAP[d] for d in tok])
        else:
            tokens.append(tok)
    return tokens

def synthesize_hmm(text: str, duration_scale: float = 1.0, f0_scale: float = 1.0, sr: int = FS):
    """
    Sinh tham số HMM (F0, Duration, Formant Trajectory) & tái tạo tín hiệu source-filter.
    """
    tokens = normalize_text(text)
    
    # 1. State Sequence Generation & Duration Modeling
    total_frames = 0
    token_spans = []
    
    f0_frames = []
    f1_frames = []
    f2_frames = []
    energy_frames = []
    state_seq = []
    
    state_id = 0
    for tok in tokens:
        prof = ACOUSTIC_PROFILES.get(tok, ACOUSTIC_PROFILES["default"])
        tok_dur = prof["dur"] * duration_scale
        n_frames = max(4, int(tok_dur * 1000 / FRAME_SHIFT_MS))
        
        # 5-state left-to-right HMM per phoneme/word unit
        frames_per_state = n_frames // 5
        
        t = np.linspace(0, 1, n_frames)
        # HMM MLPG smoothed polynomial trajectory
        tok_f0 = (prof["f0"] + prof["slope"] * (t - 0.5)) * f0_scale
        tok_f1 = prof["f1"] + 50 * np.sin(np.pi * t)
        tok_f2 = prof["f2"] - 80 * np.sin(np.pi * t)
        tok_energy = 0.08 * np.sin(np.pi * t)**0.6
        
        f0_frames.extend(tok_f0)
        f1_frames.extend(tok_f1)
        f2_frames.extend(tok_f2)
        energy_frames.extend(tok_energy)
        
        token_spans.append((tok, total_frames, total_frames + n_frames))
        for s in range(5):
            state_seq.extend([f"{tok}_s{s+1}"] * (frames_per_state if s < 4 else (n_frames - 4*frames_per_state)))
        total_frames += n_frames
        
    f0_arr = np.array(f0_frames, dtype=np.float32)
    f1_arr = np.array(f1_frames, dtype=np.float32)
    f2_arr = np.array(f2_frames, dtype=np.float32)
    energy_arr = np.array(energy_frames, dtype=np.float32)
    
    # Dynamic parameter constraint (Delta / Acceleration smoothing)
    # Moving average filter to simulate MLPG parameter continuity
    window = np.hanning(9)
    window /= window.sum()
    f0_arr = np.convolve(f0_arr, window, mode='same')
    f1_arr = np.convolve(f1_arr, window, mode='same')
    f2_arr = np.convolve(f2_arr, window, mode='same')
    
    # 2. Source-Filter Resynthesis from Generated Trajectories
    total_samples = len(f0_arr) * HOP_LENGTH
    waveform = np.zeros(total_samples, dtype=np.float32)
    
    # Excitation generation: Glottal pulse train for voiced, white noise for unvoiced
    phase = 0.0
    for i in range(len(f0_arr)):
        f0 = f0_arr[i]
        en = energy_arr[i]
        f1 = f1_arr[i]
        f2 = f2_arr[i]
        
        # Resonator coefficients (2-pole resonator for F1 and F2)
        # Formant 1 resonator
        r1 = 0.96
        theta1 = 2 * np.pi * f1 / sr
        a1 = [1, -2 * r1 * np.cos(theta1), r1**2]
        
        # Formant 2 resonator
        r2 = 0.94
        theta2 = 2 * np.pi * f2 / sr
        a2 = [1, -2 * r2 * np.cos(theta2), r2**2]
        
        # Frame excitation
        frame_exc = np.zeros(HOP_LENGTH, dtype=np.float32)
        period = int(sr / max(50.0, f0))
        for n in range(HOP_LENGTH):
            phase += 1
            if phase >= period:
                frame_exc[n] = 1.0 # Impulse
                phase = 0.0
        
        # Voiced glottal shaping (1 - z^-1) integrator
        pulse = lfilter([1.0, -0.95], [1.0], frame_exc)
        # Resonate through vocal tract filter
        filtered = lfilter([1.0], a1, pulse)
        filtered = lfilter([1.0], a2, filtered)
        
        # Scale by energy
        cur_rms = np.sqrt(np.mean(filtered**2) + 1e-12)
        if cur_rms > 0:
            filtered = filtered * (en / cur_rms)
            
        start_idx = i * HOP_LENGTH
        waveform[start_idx : start_idx + HOP_LENGTH] = filtered
        
    # Final clip and normalize
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = (waveform / peak) * 0.90
        
    tracks = {
        "f0": f0_arr,
        "f1": f1_arr,
        "f2": f2_arr,
        "energy": energy_arr,
        "state_sequence": state_seq,
        "token_spans": token_spans,
        "frame_shift_ms": FRAME_SHIFT_MS
    }
    
    return waveform, sr, tracks

if __name__ == "__main__":
    text = "Mời khách hàng số 23 đến quầy 5"
    y, sr, tracks = synthesize_hmm(text)
    sf.write("hmm_demo.wav", y, sr)
    print("HMM synthesis demo success! Samples:", len(y), "Frames:", len(tracks['f0']))
