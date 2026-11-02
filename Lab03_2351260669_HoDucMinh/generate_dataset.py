"""Tạo tập dữ liệu mẫu chuẩn cho Lab 3: 5 từ x 8 lần lặp x 16kHz mono WAV."""

import io
import urllib.request
import urllib.parse
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

FS = 16000
WORDS = {'mo': 'mở', 'dong': 'đóng', 'len': 'lên', 'xuong': 'xuống', 'dung': 'dừng'}

def download_tts(text):
    # Tải audio TTS tiếng Việt trực tiếp vào bộ nhớ
    url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={urllib.parse.quote(text)}&tl=vi&client=tw-ob"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    y, _ = librosa.load(io.BytesIO(data), sr=FS, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    return y

def pad_and_noise(audio, head_s, tail_s, snr_db=50):
    # Thêm khoảng lặng đầu/cuối và nhiễu nền phòng nhẹ (-50 dBFS)
    pad = np.concatenate([np.zeros(int(head_s * FS), dtype=np.float32),
                          audio,
                          np.zeros(int(tail_s * FS), dtype=np.float32)])
    noise = np.random.normal(0, 10 ** (-snr_db / 20.0), len(pad)).astype(np.float32)
    out = pad + noise
    peak = np.max(np.abs(out))
    return (out / peak) * 0.90 if peak > 0 else out

def make_variations(base_audio, spk='speaker1'):
    # 8 biến thể (6 train + 2 test): biến đổi tốc độ nói và cao độ
    configs = [
        (1.00,  0.0, 0.30, 0.32), (0.88, -0.7, 0.38, 0.28),
        (1.12, +0.5, 0.25, 0.38), (1.05, +1.0, 0.35, 0.26),
        (0.95, -0.8, 0.28, 0.35), (1.02, -0.3, 0.32, 0.30),
        (0.92, +0.3, 0.29, 0.34), (1.10, -0.5, 0.36, 0.27)
    ] if spk == 'speaker1' else [
        (1.03, +3.8, 0.28, 0.32), (0.90, +3.2, 0.36, 0.29),
        (1.15, +4.4, 0.24, 0.38), (1.07, +4.0, 0.33, 0.30),
        (0.96, +3.5, 0.29, 0.34), (1.01, +3.0, 0.31, 0.29),
        (0.93, +4.2, 0.27, 0.35), (1.12, +3.6, 0.35, 0.26)
    ]
    vars_list = []
    for rate, pitch, head, tail in configs:
        y = librosa.effects.time_stretch(base_audio, rate=rate)
        if pitch != 0:
            y = librosa.effects.pitch_shift(y, sr=FS, n_steps=pitch)
        vars_list.append(pad_and_noise(y, head, tail))
    return vars_list

def main():
    np.random.seed(42)
    for spk_name, spk_type, root in [("Speaker 1", "speaker1", Path("dataset")),
                                     ("Speaker 2", "speaker2", Path("dataset_speaker2"))]:
        print(f"Generating data: {spk_name} -> {root}")
        for key, text in WORDS.items():
            base = download_tts(text)
            vars_ = make_variations(base, spk=spk_type)
            train_dir = root / "train" / key
            test_dir = root / "test" / key
            train_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)
            for idx, wav in enumerate(vars_, 1):
                dst = (train_dir if idx <= 6 else test_dir) / f"{key}_{idx:02d}.wav"
                sf.write(dst, wav, FS, subtype='PCM_16')
    print("Done generating dataset.")

if __name__ == "__main__":
    main()
