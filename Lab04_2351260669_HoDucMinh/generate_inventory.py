import io
import urllib.request
import urllib.parse
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import csv

FS = 16000

VOCAB = [
    "mời", "khách", "hàng", "số",
    "không", "một", "hai", "ba", "bốn",
    "năm", "sáu", "bảy", "tám", "chín",
    "đến", "quầy"
]

MAPPING_NAMES = {
    "mời": "moi", "khách": "khach", "hàng": "hang", "số": "so",
    "không": "khong", "một": "mot", "hai": "hai", "ba": "ba", "bốn": "bon",
    "năm": "nam", "sáu": "sau", "bảy": "bay", "tám": "tam", "chín": "chin",
    "đến": "den", "quầy": "quay"
}

def download_tts(text):
    url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={urllib.parse.quote(text)}&tl=vi&client=tw-ob"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = resp.read()
    y, _ = librosa.load(io.BytesIO(data), sr=FS, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    return y

def rms(x):
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def match_rms(x, target=0.08):
    cur = rms(x)
    if cur > 0:
        y = x * (target / cur)
    else:
        y = x
    return np.clip(y, -0.95, 0.95).astype(np.float32)

def generate_clean_inventory(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    
    print("Generating pure natural acoustic inventory (no artificial phase vocoder artifacts)...")
    for word in VOCAB:
        base_name = MAPPING_NAMES[word]
        
        # Candidate 01: Standard natural pronunciation
        # Candidate 02: Continuation tone pronunciation (with slight punctuation cadence)
        wav_01 = download_tts(word)
        wav_02 = download_tts(f"{word},")
        
        for cand_id, raw_wav in [("01", wav_01), ("02", wav_02)]:
            # Smooth windowing on boundaries (5ms cosine ramp) to prevent clicks
            fade_len = int(0.005 * FS)
            w = raw_wav.copy()
            ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))
            w[:fade_len] *= ramp
            w[-fade_len:] *= ramp[::-1]
            w = match_rms(w, target=0.08)
            
            filename = f"{base_name}_{cand_id}.wav"
            file_path = output_dir / filename
            sf.write(file_path, w, FS, subtype='PCM_16')
            
            dur = len(w) / FS
            val_rms = rms(w)
            metadata.append({
                "label": word,
                "token_id": base_name,
                "candidate": cand_id,
                "speaker": "spk1",
                "duration": round(dur, 4),
                "rms": round(val_rms, 4),
                "file_path": str(file_path.relative_to(output_dir.parent.parent))
            })
            print(f"  [OK] {filename:15s} | dur={dur:.3f}s | rms={val_rms:.4f}")
            
    csv_path = output_dir.parent / "metadata.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "token_id", "candidate", "speaker", "duration", "rms", "file_path"])
        writer.writeheader()
        writer.writerows(metadata)
    print(f"Metadata saved to {csv_path}")

if __name__ == "__main__":
    out_dir = Path("/home/minh/Documents/Docs/Workspace/University_Subjects/Xử lý âm thanh/Labs/Lab04_2351260669_HoDucMinh/lab4_data/units")
    generate_clean_inventory(out_dir)
