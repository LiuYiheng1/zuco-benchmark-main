"""SIED + SRGC Cascade: Subject-Invariant features + Source-Regularized Gaussian Calibration"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

os.chdir('d:/pycharmproject/zuco-benchmark-main/src')

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
FEATURES_DIR = 'features'

def load_eeg(subject):
    path = os.path.join(FEATURES_DIR, f'{subject}_electrode_features_all.npy')
    if not os.path.exists(path): return None, None
    data = np.load(path, allow_pickle=True).item()
    X, y = [], []
    for k, v in data.items():
        p = k.split('_')
        if len(p) >= 2 and p[1] == 'NR': lbl = 1
        elif len(p) >= 2 and p[1] == 'TSR': lbl = 0
        else: continue
        X.append(np.array(v[:-1], dtype=np.float64))
        y.append(lbl)
    return np.array(X), np.array(y)

class EEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.net(x)

class TaskClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class SubjectDiscriminator(nn.Module):
    def __init__(self, input_dim, n_subjects):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_subjects)
        )

    def forward(self, x):
        return self.net(x)

def train_sied(X_train, y_train, sub_ids, n_subjects, epochs=30, lambda_adv=0.01):
    device = 'cpu'
    input_dim = X_train.shape[1]
    encoder = EEGEncoder(input_dim).to(device)
    task_clf = TaskClassifier(encoder.output_dim).to(device)
    sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    sub_t = torch.LongTensor(sub_ids)

    for epoch in range(epochs):
        encoder.train()
        task_clf.train()
        sub_disc.train()

        z = encoder(X_t)
        task_logits = task_clf(z)
        sub_logits = sub_disc(z)

        task_loss = F.binary_cross_entropy_with_logits(task_logits.squeeze(), y_t.squeeze())
        sub_loss = F.cross_entropy(sub_logits, sub_t)
        loss = task_loss + lambda_adv * sub_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    encoder.eval()
    return encoder

def stats(X, y, c):
    Xc = X[y==c]
    if len(Xc) == 0: return None, None
    return np.mean(Xc, axis=0), np.std(Xc, axis=0) + 1e-8

def srgc_predict(X_cal, y_cal, X_test, mu0, sigma0, mu1, sigma1, alpha=0.75):
    mc0, sc0 = stats(X_cal, y_cal, 0)
    mc1, sc1 = stats(X_cal, y_cal, 1)
    if mc0 is None: return np.zeros(len(X_test))
    mb0, mb1 = alpha*mu0+(1-alpha)*mc0, alpha*mu1+(1-alpha)*mc1
    sb0, sb1 = alpha*sigma0+(1-alpha)*sc0, alpha*sigma1+(1-alpha)*sc1
    cov0 = np.linalg.inv(np.diag(sb0**2+1e-8))
    cov1 = np.linalg.inv(np.diag(sb1**2+1e-8))
    s0 = np.array([np.sqrt(np.dot(np.dot(x-mb0, cov0), x-mb0)) for x in X_test])
    s1 = np.array([np.sqrt(np.dot(np.dot(x-mb1, cov1), x-mb1)) for x in X_test])
    return (s1 < s0).astype(int)

def svm_predict(X_cal, y_cal, X_test):
    if len(np.unique(y_cal)) < 2: return np.zeros(len(X_test))
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(sc.fit_transform(X_cal), y_cal)
    return clf.predict(sc.transform(X_test))

def sample(y_pool, n):
    i0 = np.where(y_pool==0)[0]
    i1 = np.where(y_pool==1)[0]
    np.random.shuffle(i0); np.random.shuffle(i1)
    return np.concatenate([i0[:min(n,len(i0))], i1[:min(n,len(i1))]])

print('Loading data...')
cache = {s: load_eeg(s) for s in Y_SUBJECTS}
print('Data loaded')

results = []
shot_settings = [3, 5, 10, 20, 50]

for seed in [0, 1, 2]:
    print(f'\nSeed {seed}:', flush=True)
    np.random.seed(seed)

    for held_out_idx, held_out in enumerate(Y_SUBJECTS):
        X_test_orig, y_test_orig = cache.get(held_out, (None, None))
        if X_test_orig is None: continue

        X_train_all = []
        y_train_all = []
        sub_ids = []
        subj_idx_map = {}
        actual_idx = 0
        for subj_idx, subj in enumerate(Y_SUBJECTS):
            if subj != held_out and cache[subj][0] is not None:
                X, y = cache[subj]
                X_train_all.append(X)
                y_train_all.append(y)
                subj_idx_map[subj_idx] = actual_idx
                sub_ids.extend([actual_idx] * len(y))
                actual_idx += 1

        if not X_train_all: continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)
        sub_ids = np.array(sub_ids)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_all)

        n_subjects = actual_idx
        encoder = train_sied(X_tr, y_train_all, sub_ids, n_subjects, epochs=20, lambda_adv=0.005)
        encoder.eval()

        with torch.no_grad():
            X_test_scaled = torch.FloatTensor(scaler.transform(X_test_orig))
            test_features = encoder(X_test_scaled).numpy()

        mu0, sigma0, mu1, sigma1 = None, None, None, None
        with torch.no_grad():
            X_tr_enc = encoder(torch.FloatTensor(X_tr)).numpy()
            mu0, sigma0 = stats(X_tr_enc, y_train_all, 0)
            mu1, sigma1 = stats(X_tr_enc, y_train_all, 1)

        if mu0 is None: continue

        idx = np.random.permutation(len(y_test_orig))
        ts = len(y_test_orig) // 3
        X_test, y_test = X_test_orig[idx[:ts]], y_test_orig[idx[:ts]]
        X_cp, y_cp = X_test_orig[idx[ts:]], y_test_orig[idx[ts:]]

        with torch.no_grad():
            X_cp_t = torch.FloatTensor(scaler.transform(X_cp))
            X_cp_enc = encoder(X_cp_t).numpy()
            X_test_t = torch.FloatTensor(scaler.transform(X_test))
            X_test_enc = encoder(X_test_t).numpy()

        print(f' {held_out}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(X_cp): continue

            cal_idx = sample(y_cp, n_cal)
            Xc, yc = X_cp[cal_idx], y_cp[cal_idx]
            Xc_enc, yc_enc = X_cp_enc[cal_idx], y_cp[cal_idx]

            if len(np.unique(yc)) < 2: continue

            # Raw EEG + SVM
            svm_preds = svm_predict(Xc, yc, X_test)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                          'method': 'Raw_SVM', 'accuracy': accuracy_score(y_test, svm_preds)})

            # Raw EEG + SRGC
            mu0_raw, sigma0_raw = stats(X_train_all, y_train_all, 0)
            mu1_raw, sigma1_raw = stats(X_train_all, y_train_all, 1)
            if mu0_raw is not None:
                srgc_preds = srgc_predict(Xc, yc, X_test, mu0_raw, sigma0_raw, mu1_raw, sigma1_raw, alpha=0.75)
                results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                              'method': 'Raw_SRGC', 'accuracy': accuracy_score(y_test, srgc_preds)})

            # SIED features + SVM
            sied_svm_preds = svm_predict(Xc_enc, yc_enc, X_test_enc)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                          'method': 'SIED_SVM', 'accuracy': accuracy_score(y_test, sied_svm_preds)})

            # SIED features + SRGC
            sied_srgc_preds = srgc_predict(Xc_enc, yc_enc, X_test_enc, mu0, sigma0, mu1, sigma1, alpha=0.75)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                          'method': 'SIED_SRGC', 'accuracy': accuracy_score(y_test, sied_srgc_preds)})

        print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv('results/final/sied_srgc_cascade_results.csv', index=False)

print('')
print('\n' + '='*60)
print('SUMMARY: SIED + SRGC Cascade')
print('='*60)

df = pd.DataFrame(results)

print('\nShot | Raw_SVM | Raw_SRGC | SIED_SVM | SIED_SRGC')
print('-'*60)
for n in shot_settings:
    raw_svm = df[(df.method=='Raw_SVM') & (df.n_cal==n)]['accuracy'].mean()
    raw_srgc = df[(df.method=='Raw_SRGC') & (df.n_cal==n)]['accuracy'].mean()
    sied_svm = df[(df.method=='SIED_SVM') & (df.n_cal==n)]['accuracy'].mean()
    sied_srgc = df[(df.method=='SIED_SRGC') & (df.n_cal==n)]['accuracy'].mean()
    print(f'{n:4d} | {raw_svm:.4f} | {raw_srgc:.4f} | {sied_svm:.4f} | {sied_srgc:.4f}')

print('\nGap Analysis:')
for n in shot_settings:
    raw_srgc = df[(df.method=='Raw_SRGC') & (df.n_cal==n)]['accuracy'].mean()
    sied_srgc = df[(df.method=='SIED_SRGC') & (df.n_cal==n)]['accuracy'].mean()
    sied_svm = df[(df.method=='SIED_SVM') & (df.n_cal==n)]['accuracy'].mean()
    print(f'{n}-shot: SIED_SRGC vs Raw_SRGC = {sied_srgc-raw_srgc:+.4f}, SIED_SRGC vs SIED_SVM = {sied_srgc-sied_svm:+.4f}')

print('\nDone!')