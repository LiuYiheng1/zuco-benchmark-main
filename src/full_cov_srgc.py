"""Full-Covariance SRGC: Using Ledoit-Wolf covariance estimation"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score
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

def stats_diag(X, y, c):
    Xc = X[y==c]
    if len(Xc) == 0: return None, None
    return np.mean(Xc, axis=0), np.std(Xc, axis=0) + 1e-8

def stats_full(X, y, c):
    Xc = X[y==c]
    if len(Xc) < 2: return None, None
    lw = LedoitWolf()
    lw.fit(Xc)
    return np.mean(Xc, axis=0), lw.covariance_

def srgc_diag(X_cal, y_cal, X_test, mu0, sigma0, mu1, sigma1, alpha):
    mc0, sc0 = stats_diag(X_cal, y_cal, 0)
    mc1, sc1 = stats_diag(X_cal, y_cal, 1)
    if mc0 is None: return np.zeros(len(X_test))
    mb0, mb1 = alpha*mu0+(1-alpha)*mc0, alpha*mu1+(1-alpha)*mc1
    sb0, sb1 = alpha*sigma0+(1-alpha)*sc0, alpha*sigma1+(1-alpha)*sc1
    cov0 = np.linalg.inv(np.diag(sb0**2+1e-8))
    cov1 = np.linalg.inv(np.diag(sb1**2+1e-8))
    s0 = np.array([np.sqrt(np.dot(np.dot(x-mb0, cov0), x-mb0)) for x in X_test])
    s1 = np.array([np.sqrt(np.dot(np.dot(x-mb1, cov1), x-mb1)) for x in X_test])
    return (s1 < s0).astype(int)

def srgc_full(X_cal, y_cal, X_test, mu0, cov0, mu1, cov1, alpha):
    mc0, cc0 = stats_full(X_cal, y_cal, 0)
    mc1, cc1 = stats_full(X_cal, y_cal, 1)
    if mc0 is None: return np.zeros(len(X_test))
    mb0, mb1 = alpha*mu0+(1-alpha)*mc0, alpha*mu1+(1-alpha)*mc1
    cb0, cb1 = alpha*cov0+(1-alpha)*cc0, alpha*cov1+(1-alpha)*cc1
    try:
        cb0_inv = np.linalg.inv(cb0 + np.eye(cb0.shape[0])*1e-6)
        cb1_inv = np.linalg.inv(cb1 + np.eye(cb1.shape[0])*1e-6)
    except:
        return np.zeros(len(X_test))
    s0 = np.array([np.sqrt(np.dot(np.dot(x-mb0, cb0_inv), x-mb0)) for x in X_test])
    s1 = np.array([np.sqrt(np.dot(np.dot(x-mb1, cb1_inv), x-mb1)) for x in X_test])
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

    for ho in Y_SUBJECTS:
        Xt, yt = cache[ho]
        if Xt is None: continue

        Xtr, ytr = [], []
        for s in Y_SUBJECTS:
            if s != ho and cache[s][0] is not None:
                Xtr.append(cache[s][0]); ytr.append(cache[s][1])
        if not Xtr: continue
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)

        mu0_d, sigma0_d = stats_diag(Xtr, ytr, 0)
        mu1_d, sigma1_d = stats_diag(Xtr, ytr, 1)
        mu0_f, cov0_f = stats_full(Xtr, ytr, 0)
        mu1_f, cov1_f = stats_full(Xtr, ytr, 1)
        if mu0_d is None: continue

        idx = np.random.permutation(len(yt))
        ts = len(yt) // 3
        X_test, y_test = Xt[idx[:ts]], yt[idx[:ts]]
        X_cp, y_cp = Xt[idx[ts:]], yt[idx[ts:]]

        print(f' {ho}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(X_cp): continue
            cal_idx = sample(y_cp, n_cal)
            Xc, yc = X_cp[cal_idx], y_cp[cal_idx]
            if len(np.unique(yc)) < 2: continue

            svm_preds = svm_predict(Xc, yc, X_test)
            results.append({'seed':seed,'subject':ho,'n_cal':n_cal,'method':'SVM','accuracy':accuracy_score(y_test, svm_preds)})

            for alpha in [0.5, 0.75]:
                diag_preds = srgc_diag(Xc, yc, X_test, mu0_d, sigma0_d, mu1_d, sigma1_d, alpha)
                results.append({'seed':seed,'subject':ho,'n_cal':n_cal,'method':f'SRGC_diag_a{alpha}','accuracy':accuracy_score(y_test, diag_preds)})

                full_preds = srgc_full(Xc, yc, X_test, mu0_f, cov0_f, mu1_f, cov1_f, alpha)
                results.append({'seed':seed,'subject':ho,'n_cal':n_cal,'method':f'SRGC_full_a{alpha}','accuracy':accuracy_score(y_test, full_preds)})

        print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv('results/final/full_cov_srgc_results.csv', index=False)

print('')
print('\n' + '='*60)
print('SUMMARY: Diagonal vs Full Covariance SRGC')
print('='*60)

df = pd.DataFrame(results)

print('\nShot | SVM | SRGC_diag_a0.5 | SRGC_diag_a0.75 | SRGC_full_a0.5 | SRGC_full_a0.75')
print('-'*90)
for n in shot_settings:
    svm = df[(df.method=='SVM') & (df.n_cal==n)]['accuracy'].mean()
    da5 = df[(df.method=='SRGC_diag_a0.5') & (df.n_cal==n)]['accuracy'].mean()
    da75 = df[(df.method=='SRGC_diag_a0.75') & (df.n_cal==n)]['accuracy'].mean()
    fa5 = df[(df.method=='SRGC_full_a0.5') & (df.n_cal==n)]['accuracy'].mean()
    fa75 = df[(df.method=='SRGC_full_a0.75') & (df.n_cal==n)]['accuracy'].mean()
    print(f'{n:4d} | {svm:.4f} | {da5:.4f} | {da75:.4f} | {fa5:.4f} | {fa75:.4f}')

print('\nBest SRGC comparison:')
for n in shot_settings:
    diag_best = max(df[(df.method.str.startswith('SRGC_diag')) & (df.n_cal==n)].groupby('method')['accuracy'].mean().max())
    full_best = max(df[(df.method.str.startswith('SRGC_full')) & (df.n_cal==n)].groupby('method')['accuracy'].mean().max())
    print(f'{n}-shot: Diag_best={diag_best:.4f}, Full_best={full_best:.4f}, gap={full_best-diag_best:+.4f}')

print('\nDone!')