"""PRDC: Prior-Regularized Discriminative Calibration"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
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

def stats(X, y, c):
    Xc = X[y==c]
    if len(Xc) == 0: return None, None
    return np.mean(Xc, axis=0), np.cov(Xc.T) + np.eye(Xc.shape[1]) * 1e-6

def compute_source_w(X_source, y_source):
    mu0, sigma0 = stats(X_source, y_source, 0)
    mu1, sigma1 = stats(X_source, y_source, 1)
    if mu0 is None: return None
    try:
        sigma_inv = np.linalg.inv((sigma0 + sigma1) / 2)
        w_source = sigma_inv @ (mu1 - mu0)
        return w_source
    except:
        return None

def prdc_predict(X_cal, y_cal, X_test, w_source, kappa=10):
    if w_source is None: return np.zeros(len(X_test))
    if len(np.unique(y_cal)) < 2: return np.zeros(len(X_test))

    n_total = len(y_cal)
    lambda_n = kappa / (kappa + n_total)

    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, random_state=42, penalty='l2', C=1.0)
    clf.fit(X_cal_s, y_cal)

    w_learned = clf.coef_.flatten()
    b_learned = clf.intercept_[0]

    w_reg = (1 - lambda_n) * w_learned + lambda_n * w_source

    logits = X_test_s @ w_reg + b_learned
    preds = (logits > 0).astype(int)
    return preds

def svm_predict(X_cal, y_cal, X_test):
    if len(np.unique(y_cal)) < 2: return np.zeros(len(X_test))
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(scaler.fit_transform(X_cal), y_cal)
    return clf.predict(scaler.transform(X_test))

def sample(y_pool, n):
    i0 = np.where(y_pool==0)[0]
    i1 = np.where(y_pool==1)[0]
    np.random.shuffle(i0); np.random.shuffle(i1)
    return np.concatenate([i0[:min(n,len(i0))], i1[:min(n,len(i1))]])

print('Loading data...')
cache = {s: load_eeg(s) for s in Y_SUBJECTS}
print('Data loaded')

results = []
kappa_values = [1, 5, 10, 20, 50]

for seed in [0, 1, 2]:
    print(f'\nSeed {seed}:', flush=True)
    np.random.seed(seed)

    for ho in Y_SUBJECTS:
        Xt, yt = cache[ho]
        if Xt is None: continue

        Xtr, ytr = [], []
        for s in Y_SUBJECTS:
            if s != ho and cache[s][0] is not None:
                Xtr.append(cache[s][0]); ytr.append(cache[s][1])
        if not Xtr: continue

        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        w_source = compute_source_w(Xtr, ytr)
        if w_source is None: continue

        idx = np.random.permutation(len(yt))
        ts = len(yt) // 3
        X_test, y_test = Xt[idx[:ts]], yt[idx[:ts]]
        Xcp, ycp = Xt[idx[ts:]], yt[idx[ts:]]

        print(f' {ho}', end='', flush=True)

        for n_cal in [3, 5, 10, 20, 50]:
            if n_cal * 2 > len(Xcp): continue
            ci = sample(ycp, n_cal)
            Xc, yc = Xcp[ci], ycp[ci]
            if len(np.unique(yc)) < 2: continue

            svm_preds = svm_predict(Xc, yc, X_test)
            results.append({
                'seed': seed, 'subject': ho, 'n_cal': n_cal,
                'method': 'EEG_SVM', 'kappa': np.nan,
                'accuracy': accuracy_score(y_test, svm_preds),
                'macro_f1': f1_score(y_test, svm_preds, average='macro'),
                'balanced_accuracy': balanced_accuracy_score(y_test, svm_preds)
            })

            for kappa in kappa_values:
                prdc_preds = prdc_predict(Xc, yc, X_test, w_source, kappa=kappa)
                results.append({
                    'seed': seed, 'subject': ho, 'n_cal': n_cal,
                    'method': 'PRDC', 'kappa': kappa,
                    'accuracy': accuracy_score(y_test, prdc_preds),
                    'macro_f1': f1_score(y_test, prdc_preds, average='macro'),
                    'balanced_accuracy': balanced_accuracy_score(y_test, prdc_preds)
                })

        print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv('results/final/prdc_results.csv', index=False)

print('')
print('\n' + '='*60)
print('SUMMARY')
print('='*60)

df = pd.DataFrame(results)

print('\nEEG_SVM:')
for n in [3, 5, 10, 20, 50]:
    d = df[(df.method == 'EEG_SVM') & (df.n_cal == n)]
    if len(d) > 0:
        print(f'  {n}-shot: {d.accuracy.mean():.4f} +/- {d.accuracy.std():.4f}')

print('\nPRDC best kappa by shot:')
for n in [3, 5, 10, 20, 50]:
    prdc = df[(df.method == 'PRDC') & (df.n_cal == n)]
    if len(prdc) > 0:
        best_kappa = prdc.groupby('kappa')['accuracy'].mean().idxmax()
        best_acc = prdc.groupby('kappa')['accuracy'].mean().max()
        avg_acc = prdc['accuracy'].mean()
        print(f'  {n}-shot: avg={avg_acc:.4f}, best={best_acc:.4f} (kappa={best_kappa})')

print('\nComparison:')
for n in [3, 5, 10, 20, 50]:
    svm = df[(df.method == 'EEG_SVM') & (df.n_cal == n)]['accuracy'].mean()
    prdc = df[(df.method == 'PRDC') & (df.n_cal == n)]
    if len(prdc) > 0:
        prdc_avg = prdc['accuracy'].mean()
        prdc_best = prdc.groupby('kappa')['accuracy'].mean().max()
        gap_avg = prdc_avg - svm
        gap_best = prdc_best - svm
        print(f'  {n}-shot: SVM={svm:.4f}, PRDC_avg={prdc_avg:.4f} ({gap_avg:+.4f}), PRDC_best={prdc_best:.4f} ({gap_best:+.4f})')

print('\nDone!')