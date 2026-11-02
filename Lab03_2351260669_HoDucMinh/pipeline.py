"""
pipeline.py
Hệ nhận dạng tiếng nói HMM cho 5 khẩu lệnh tiếng Việt (Lab 3 CSE457):
- Front-End: MFCC 39 chiều (13 static + 13 delta + 13 delta2) + CMVN
- Acoustic Model: Left-to-Right Gaussian HMM (Bakis topology)
- Giải mã: Maximum Likelihood + Viterbi state alignment
- Đối chứng: DTW baseline (Lab 2)
- Mở rộng: Lexicon + Bigram LM MAP re-ranking
- Thí nghiệm kiểm soát biến: E1 đến E6
"""

import time
import math
import pickle
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

FS = 16000
LABELS = ['mo', 'dong', 'len', 'xuong', 'dung']
LABEL_MAP_VI = {'mo': 'mở', 'dong': 'đóng', 'len': 'lên', 'xuong': 'xuống', 'dung': 'dừng'}

# -------------------------------------------------------------
# 1. TRÍCH XUẤT ĐẶC TRƯNG
# -------------------------------------------------------------
def extract_feat(path_or_audio, sr=FS, n_mfcc=13, use_dynamic=True, use_cmvn=True):
    # Nhận đường dẫn tệp hoặc mảng âm thanh thô trực tiếp từ bộ nhớ
    if isinstance(path_or_audio, (str, Path)):
        y, sr = librosa.load(path_or_audio, sr=sr, mono=True)
    else:
        y = np.asarray(path_or_audio, dtype=np.float32)
        
    y, _ = librosa.effects.trim(y, top_db=30)
    y = librosa.effects.preemphasis(y, coef=0.97)
    
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=512, win_length=int(0.025 * sr),
        hop_length=int(0.010 * sr), window='hamming',
        n_mels=26, fmin=20, fmax=sr / 2
    )
    
    if use_dynamic:
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)
        F = np.vstack([mfcc, d1, d2]).T  # Shape: (T, 39)
    else:
        F = mfcc.T  # Shape: (T, 13)
        
    if use_cmvn:
        mu = F.mean(axis=0, keepdims=True)
        sd = F.std(axis=0, keepdims=True) + 1e-8
        F = (F - mu) / sd
        
    return F

# -------------------------------------------------------------
# 2. KHỞI TẠO VÀ HUẤN LUYỆN HMM
# -------------------------------------------------------------
def make_lr_hmm(n_states=5, n_iter=80, covariance_type='diag', params='mc', random_state=42):
    # Bakis topology: chỉ self-loop (0.65) và chuyển trạng thái kế (0.35)
    # params='mc': cố định topology, chỉ ước lượng Means và Covariances bằng EM
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=1e-3,
        min_covar=1e-3,
        init_params='mc',
        params=params,
        random_state=random_state
    )
    start = np.zeros(n_states)
    start[0] = 1.0
    
    A = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        A[i, i] = 0.65
        A[i, i + 1] = 0.35
    A[-1, -1] = 1.0
    
    model.startprob_ = start
    model.transmat_ = A
    return model

def train_hmms(train_dir=Path('dataset/train'), labels=LABELS, n_states=5,
               covariance_type='diag', params='mc', use_dynamic=True, n_train_per_class=None):
    models = {}
    train_times = {}
    for label in labels:
        wav_files = sorted((train_dir / label).glob('*.wav'))
        if n_train_per_class is not None:
            wav_files = wav_files[:n_train_per_class]
            
        t0 = time.time()
        feats = [extract_feat(str(w), use_dynamic=use_dynamic) for w in wav_files]
        lengths = [len(f) for f in feats]
        X = np.vstack(feats)
        
        model = make_lr_hmm(n_states=n_states, covariance_type=covariance_type, params=params)
        model.fit(X, lengths)
        models[label] = model
        train_times[label] = time.time() - t0
        
    return models, train_times

# -------------------------------------------------------------
# 3. NHẬN DẠNG VÀ VITERBI DECODING
# -------------------------------------------------------------
def score_utterance(F, models):
    # Trả về điểm log-likelihood trên toàn bộ mô hình và nhãn argmax
    scores = {}
    for l, m in models.items():
        try:
            scores[l] = float(m.score(F))
        except Exception:
            scores[l] = -1e9
    pred = max(scores, key=scores.get)
    return pred, scores

def evaluate_hmm(models, test_dir=Path('dataset/test'), labels=LABELS, use_dynamic=True):
    true_y, pred_y = [], []
    all_scores = {}
    total_time = 0.0
    trials = 0
    
    for label in labels:
        for wav in sorted((test_dir / label).glob('*.wav')):
            F = extract_feat(str(wav), use_dynamic=use_dynamic)
            t0 = time.time()
            pred, scores = score_utterance(F, models)
            total_time += (time.time() - t0)
            
            true_y.append(label)
            pred_y.append(pred)
            all_scores[wav.name] = (label, pred, scores)
            trials += 1
            
    acc = accuracy_score(true_y, pred_y)
    avg_time = (total_time / trials) if trials > 0 else 0.0
    return acc, true_y, pred_y, all_scores, avg_time

# -------------------------------------------------------------
# 4. BASELINE DTW (LAB 2)
# -------------------------------------------------------------
def dtw_distance(X, Y):
    # Quy hoạch động DTW: tìm đường uốn thời gian cực tiểu hóa khoảng cách Euclid
    N, M = len(X), len(Y)
    if N == 0 or M == 0:
        return np.inf, []
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    back = np.zeros((N + 1, M + 1, 2), dtype=int)
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = np.linalg.norm(X[i - 1] - Y[j - 1])
            opts = [(D[i - 1, j - 1], i - 1, j - 1),
                    (D[i - 1, j],     i - 1, j),
                    (D[i, j - 1],     i,     j - 1)]
            bc, pi, pj = min(opts, key=lambda z: z[0])
            D[i, j] = cost + bc
            back[i, j] = (pi, pj)
            
    path = []
    i, j = N, M
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0 and j == 0:
            break
        i, j = back[i, j]
        if i == 0 and j == 0:
            break
    path.reverse()
    return D[N, M] / max(len(path), 1), path

def train_dtw_templates(train_dir=Path('dataset/train'), labels=LABELS, use_dynamic=True, n_train=None):
    # Lưu tập template từ tập train
    templates = {l: [] for l in labels}
    t0 = time.time()
    for label in labels:
        wavs = sorted((train_dir / label).glob('*.wav'))
        if n_train is not None:
            wavs = wavs[:n_train]
        for w in wavs:
            templates[label].append(extract_feat(str(w), use_dynamic=use_dynamic))
    return templates, time.time() - t0

def evaluate_dtw(templates, test_dir=Path('dataset/test'), labels=LABELS, use_dynamic=True):
    true_y, pred_y = [], []
    total_time = 0.0
    trials = 0
    
    for label in labels:
        for wav in sorted((test_dir / label).glob('*.wav')):
            F_test = extract_feat(str(wav), use_dynamic=use_dynamic)
            t0 = time.time()
            best_dist = np.inf
            best_label = None
            for cand_l, tpls in templates.items():
                for tpl in tpls:
                    d, _ = dtw_distance(F_test, tpl)
                    if d < best_dist:
                        best_dist = d
                        best_label = cand_l
            total_time += (time.time() - t0)
            true_y.append(label)
            pred_y.append(best_label)
            trials += 1
            
    acc = accuracy_score(true_y, pred_y)
    avg_time = (total_time / trials) if trials > 0 else 0.0
    return acc, true_y, pred_y, avg_time

# -------------------------------------------------------------
# 5. LEXICON & BIGRAM MAP RE-RANKING
# -------------------------------------------------------------
def parse_lexicon(lex_path="lexicon.txt"):
    lex_dict = {}
    with open(lex_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                lex_dict[parts[0]] = parts[1:]
    return lex_dict

CORPUS_COMMANDS = [
    ["mo", "len"], ["dong", "xuong"], ["mo", "dung"], ["dong", "dung"],
    ["len", "dung"], ["xuong", "dung"], ["len", "xuong"], ["xuong", "len"],
    ["mo", "dong"], ["dong", "mo"], ["mo", "len"], ["dong", "xuong"]
]

def build_bigram_counts(corpus=CORPUS_COMMANDS):
    c1, c2 = {}, {}
    for sent in corpus:
        for i, w in enumerate(sent):
            c1[w] = c1.get(w, 0) + 1
            if i > 0:
                pair = (sent[i - 1], w)
                c2[pair] = c2.get(pair, 0) + 1
    return c1, c2

def bigram_prob(prev, cur, c2, c1, V=len(LABELS), k=0.5):
    # Add-k smoothing
    return (c2.get((prev, cur), 0) + k) / (c1.get(prev, 0) + k * V)

def rerank_bayes(F, prev, models, c2, c1, lm_weight=0.3):
    # MAP score = log P(X|w)/T + lambda_LM * log P_LM(w|prev)
    T = len(F)
    total, acoustics, lm_scores = {}, {}, {}
    for w, m in models.items():
        a = m.score(F) / T
        lm = math.log(bigram_prob(prev, w, c2, c1))
        acoustics[w] = a
        lm_scores[w] = lm
        total[w] = a + lm_weight * lm
    return max(total, key=total.get), total, acoustics, lm_scores

# -------------------------------------------------------------
# 6. THỰC THI THÍ NGHIỆM CHÍNH (E1 - E6)
# -------------------------------------------------------------
def run_all():
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Heatmaps
    for word in ['mo', 'dong']:
        F = extract_feat(f"dataset/train/{word}/{word}_01.wav")
        plt.figure(figsize=(9, 5))
        plt.subplot(3, 1, 1); plt.imshow(F[:, :13].T, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"MFCC 13 Tĩnh - '{word}_01.wav'"); plt.colorbar()
        plt.subplot(3, 1, 2); plt.imshow(F[:, 13:26].T, aspect='auto', origin='lower', cmap='magma')
        plt.title("Delta MFCC (Vận tốc)"); plt.colorbar()
        plt.subplot(3, 1, 3); plt.imshow(F[:, 26:39].T, aspect='auto', origin='lower', cmap='coolwarm')
        plt.title("Delta2 MFCC (Gia tốc)"); plt.colorbar()
        plt.tight_layout()
        plt.savefig(fig_dir / f"heatmap_mfcc_{word}.png", dpi=150)
        plt.close()
        
    # Baseline HMM
    models_base, train_times = train_hmms()
    for l, m in models_base.items():
        with open(model_dir / f"hmm_{l}.pkl", "wb") as f:
            pickle.dump(m, f)
            
    acc_base, true_b, pred_b, scores_b, time_hmm = evaluate_hmm(models_base)
    cm_base = confusion_matrix(true_b, pred_b, labels=LABELS)
    ConfusionMatrixDisplay(cm_base, display_labels=LABELS).plot(cmap='Blues')
    plt.title("HMM Baseline Confusion Matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix_hmm.png", dpi=150)
    plt.close()
    
    # Viterbi alignment
    for target in ['mo', 'dong']:
        F = extract_feat(f"dataset/test/{target}/{target}_07.wav")
        logp, states = models_base[target].decode(F, algorithm='viterbi')
        plt.figure(figsize=(8, 3.5))
        plt.step(range(len(states)), states, where='post', color='navy', lw=2)
        plt.xlabel('Frame index (10ms)'); plt.ylabel('HMM State')
        plt.title(f"Viterbi Alignment: {target}_07.wav (logp: {logp:.1f})")
        plt.yticks(range(5)); plt.grid(True, ls='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig_dir / f"viterbi_alignment_{target}.png", dpi=150)
        plt.close()
        
    # E1: n_states
    res_e1 = {}
    for ns in [3, 5, 7]:
        m_ns, _ = train_hmms(n_states=ns)
        acc_ns, _, _, _, t_ns = evaluate_hmm(m_ns)
        ll = np.mean([m_ns[l].score(extract_feat(str(w)))
                      for l in LABELS for w in (Path('dataset/train') / l).glob('*.wav')])
        res_e1[f"n_states={ns}"] = {"accuracy": acc_ns, "avg_train_ll": float(ll), "test_time_ms": t_ns * 1000}
        
    # E2: MFCC 13 vs 39
    m_13, _ = train_hmms(use_dynamic=False)
    acc_13, _, _, _, _ = evaluate_hmm(m_13, use_dynamic=False)
    res_e2 = {"MFCC_13": float(acc_13), "MFCC_39": float(acc_base)}
    
    # E3: Diag vs Full
    m_full, _ = train_hmms(covariance_type='full')
    acc_full, _, _, _, _ = evaluate_hmm(m_full)
    tr_diag = np.mean([models_base[l].score(extract_feat(str(w)))
                       for l in LABELS for w in (Path('dataset/train') / l).glob('*.wav')])
    tr_full = np.mean([m_full[l].score(extract_feat(str(w)))
                       for l in LABELS for w in (Path('dataset/train') / l).glob('*.wav')])
    res_e3 = {
        "diag": {"train_score": float(tr_diag), "test_acc": float(acc_base)},
        "full": {"train_score": float(tr_full), "test_acc": float(acc_full)}
    }
    
    # E4: DTW vs HMM
    tpls, t_train_dtw = train_dtw_templates()
    acc_dtw, true_d, pred_d, t_test_dtw = evaluate_dtw(tpls)
    ConfusionMatrixDisplay(confusion_matrix(true_d, pred_d, labels=LABELS), display_labels=LABELS).plot(cmap='Oranges')
    plt.title("DTW Baseline Confusion Matrix"); plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix_dtw.png", dpi=150)
    plt.close()
    
    tpls_1, _ = train_dtw_templates(n_train=1)
    acc_dtw_1, _, _, _ = evaluate_dtw(tpls_1)
    m_1, _ = train_hmms(n_train_per_class=1)
    acc_hmm_1, _, _, _, _ = evaluate_hmm(m_1)
    
    res_e4 = {
        "HMM_full": {"accuracy": float(acc_base), "train_time_ms": sum(train_times.values()) * 1000, "test_time_ms": time_hmm * 1000},
        "DTW_full": {"accuracy": float(acc_dtw), "train_time_ms": t_train_dtw * 1000, "test_time_ms": t_test_dtw * 1000},
        "HMM_1sample": float(acc_hmm_1),
        "DTW_1sample": float(acc_dtw_1)
    }
    
    # E5: Mismatch
    acc_spk2, _, _, _, _ = evaluate_hmm(models_base, test_dir=Path('dataset_speaker2/test'))
    # In-memory noise: không tạo file tạm trên ổ đĩa
    true_n, pred_n = [], []
    for l in LABELS:
        for w in (Path('dataset/test') / l).glob('*.wav'):
            y, _ = librosa.load(w, sr=FS, mono=True)
            y_noise = y + np.random.normal(0, 0.05, len(y))
            F_n = extract_feat(y_noise)
            p_n, _ = score_utterance(F_n, models_base)
            true_n.append(l)
            pred_n.append(p_n)
    acc_noise = accuracy_score(true_n, pred_n)
    res_e5 = {"matched": float(acc_base), "speaker_mismatch": float(acc_spk2), "noise_mismatch": float(acc_noise)}
    
    # E6: LM Re-ranking
    c1, c2 = build_bigram_counts()
    F_rr = extract_feat("dataset/test/len/len_07.wav")
    res_e6 = {}
    for lam in [0.0, 0.1, 0.3, 1.0]:
        pred_rr, tot, ac_sc, lm_sc = rerank_bayes(F_rr, 'mo', models_base, c2, c1, lm_weight=lam)
        res_e6[f"lambda_{lam}"] = {"prediction": pred_rr, "total_scores": tot, "acoustic_scores": ac_sc, "lm_scores": lm_sc}
        
    out_data = {
        "E1_states": res_e1, "E2_features": res_e2, "E3_covariance": res_e3,
        "E4_dtw_vs_hmm": res_e4, "E5_mismatch": res_e5, "E6_language_model": res_e6
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
        
    print(f"[OK] Pipeline hoàn tất. HMM Test Acc: {acc_base*100:.1f}%. Đã lưu models/, figures/, results.json.")

if __name__ == "__main__":
    run_all()
