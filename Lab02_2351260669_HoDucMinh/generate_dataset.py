"""
generate_dataset.py
Tạo tập dữ liệu phát âm mẫu chuẩn (dataset/) phục vụ thí nghiệm Lab 2:
- 5 lớp từ: khong, mot, hai, ba, bon
- Mỗi từ gồm 5 lần lặp (5 utterances), chuẩn mono WAV 16kHz, 16-bit PCM.
- Có khoảng lặng (silence margin) 0.2s - 0.45s ở đầu và cuối theo chuẩn yêu cầu.
- Các lần lặp mô phỏng chính xác sự biến thiên thực tế: tốc độ nói (time stretching),
  cao độ (pitch shifting), âm lượng và tạp âm nền phòng nhẹ (-55 dBFS).
- Đồng thời tạo tập dataset_speaker2/ để phục vụ thí nghiệm mở rộng Cross-speaker (E4).
"""

import os
import urllib.request
import urllib.parse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

FS = 16000
WORDS = {
    'khong': 'không',
    'mot': 'một',
    'hai': 'hai',
    'ba': 'ba',
    'bon': 'bốn'
}

def download_tts_base(word_text, cache_dir=Path("cache_tts")):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{urllib.parse.quote(word_text)}.mp3"
    if not cache_file.exists():
        encoded = urllib.parse.quote(word_text)
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded}&tl=vi&client=tw-ob"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(cache_file, "wb") as f:
            f.write(resp.read())
    y, sr = librosa.load(cache_file, sr=FS, mono=True)
    # Cắt gọn đoạn silence ban đầu của file TTS thô
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    return y_trimmed

def add_silence_and_noise(audio, head_sec, tail_sec, noise_snr_db=45):
    """Thêm silence đầu/cuối và một lượng nhiễu nền phòng nhẹ chân thực."""
    n_head = int(head_sec * FS)
    n_tail = int(tail_sec * FS)
    
    # Tạo tín hiệu đệm
    head_pad = np.zeros(n_head, dtype=np.float32)
    tail_pad = np.zeros(n_tail, dtype=np.float32)
    padded = np.concatenate([head_pad, audio, tail_pad])
    
    # Thêm nhiễu nền Gaussian cực nhỏ mô phỏng micro phòng học/phòng thu (-50dB)
    noise = np.random.normal(0, 10 ** (-noise_snr_db / 20.0), len(padded)).astype(np.float32)
    out = padded + noise
    
    # Chuẩn hóa biên độ cực đại đạt ~ 0.9 để tránh clipping
    peak = np.max(np.abs(out))
    if peak > 0:
        out = (out / peak) * 0.90
    return out

def generate_variations(base_audio, speaker_type="speaker1"):
    """
    Tạo 5 lần phát âm biến thiên cho một từ:
    - speaker1: Người nói chính (nam/chuẩn) với 5 tốc độ và ngữ điệu khác nhau
    - speaker2: Người nói thứ hai (cao độ cao hơn, nhịp điệu khác) cho Cross-speaker
    """
    variations = []
    
    if speaker_type == "speaker1":
        configs = [
            # rate, pitch_shift (semitones), head_sec, tail_sec
            (1.00,  0.0, 0.32, 0.35),   # 01: Chuẩn, tốc độ bình thường
            (0.88, -0.8, 0.40, 0.30),   # 02: Nói chậm, giọng trầm hơn chút
            (1.14, +0.6, 0.25, 0.42),   # 03: Nói nhanh, cao độ nhỉnh nhẹ
            (1.05, +1.2, 0.36, 0.28),   # 04: Nhấn giọng, tốc độ hơi nhanh
            (0.94, -1.0, 0.28, 0.38),   # 05: Thư thả, kéo dài đuôi
        ]
    else: # speaker2: giọng cao hơn (+3.5 to +4.5 semitones), tốc độ khác
        configs = [
            (1.02, +3.8, 0.30, 0.33),
            (0.90, +3.2, 0.38, 0.29),
            (1.16, +4.4, 0.24, 0.40),
            (1.08, +4.0, 0.35, 0.31),
            (0.96, +3.5, 0.29, 0.35),
        ]
        
    for rate, p_shift, head, tail in configs:
        # Biến đổi tốc độ
        y_mod = librosa.effects.time_stretch(base_audio, rate=rate)
        # Biến đổi cao độ
        if p_shift != 0.0:
            y_mod = librosa.effects.pitch_shift(y_mod, sr=FS, n_steps=p_shift)
        # Thêm khoảng lặng đầu/cuối và nhiễu nhẹ
        y_final = add_silence_and_noise(y_mod, head, tail, noise_snr_db=48)
        variations.append(y_final)
        
    return variations

def build_all_datasets():
    np.random.seed(42)
    base_dir = Path(__file__).resolve().parent
    
    datasets = [
        ("dataset", "speaker1"),
        ("dataset_speaker2", "speaker2")
    ]
    
    for folder_name, spk_type in datasets:
        target_root = base_dir / folder_name
        print(f"\n[*] Đang tạo dữ liệu cho {folder_name} ({spk_type})...")
        for label, word_text in WORDS.items():
            out_folder = target_root / label
            out_folder.mkdir(parents=True, exist_ok=True)
            
            # Tải âm chuẩn
            base_audio = download_tts_base(word_text, cache_dir=base_dir / "cache_tts")
            # Tạo 5 lần lặp biến thiên
            utts = generate_variations(base_audio, speaker_type=spk_type)
            
            for idx, y in enumerate(utts, 1):
                filename = f"{label}_{idx:02d}.wav"
                out_path = out_folder / filename
                sf.write(out_path, y, FS, subtype='PCM_16')
                print(f"    - Đã ghi: {out_path.relative_to(base_dir)} | Độ dài: {len(y)/FS:.3f}s")
                
    print("\n[SUCCESS] Hoàn thành sinh tập dữ liệu 5 từ x 5 lần lặp x 2 người nói!")

if __name__ == "__main__":
    build_all_datasets()
