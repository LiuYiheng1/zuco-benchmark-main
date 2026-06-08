"""APR-SRGC: Minimal test with reduced combos"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

def stats(X, y, c):
    Xc = X[y==c]
    if len(Xc) == 0: return None, None
    return np.mean(Xc, axis=0), np.std(Xc, axis=0) + 1e-8

def apr_pred(Xc, yc, Xt, ms0, ss0, ms1, ss1, k, nu):
    n0, n1 = np.sum(yc==0), np.sum(yc==1)
    l0, l1 = k/(k+n0) if n0>0 else 0.5, k/(k+n1) if n1>0 else 0.5
    g0, g1 = nu/(nu+n0) if n0>0 else 0.5, nu/(nu+n1) if n1>0 else 0.5
    mc0, sc0 = stats(Xc, yc, 0)
    mc1, sc1 = stats(Xc, yc, 1)
    if mc0 is None: return np.zeros(len(Xt))
    mb0, mb1 = l0*ms0+(1-l0)*mc0, l1*ms1+(1-l1)*mc1
    sb0, sb1 = g0*ss0+(1-g0)*sc0, g1*ss1+(1-g1)*sc1
    cov0 = np.linalg.inv(np.diag(sb0**2+1e-8))
    cov1 = np.linalg.inv(np.diag(sb1**2+1e-8))
    s0 = np.array([np.sqrt(np.dot(np.dot(x-mb0, cov0), x-mb0)) for x in Xt])
    s1 = np.array([np.sqrt(np.dot(np.dot(x-mb1, cov1), x-mb1)) for x in Xt])
    return (s1 < s0).astype(int)

def fixed_pred(Xc, yc, Xt, ms0, ss0, ms1, ss1, a=0.75):
    mc0, sc0 = stats(Xc, yc, 0)
    mc1, sc1 = stats(Xc, yc, 1)
    if mc0 is None: return np.zeros(len(Xt))
    mb0, mb1 = a*ms0+(1-a)*mc0, a*ms1+(1-a)*mc1
    sb0, sb1 = a*ss0+(1-a)*sc0, a*ss1+(1-a)*sc1
    cov0 = np.linalg.inv(np.diag(sb0**2+1e-8))
    cov1 = np.linalg.inv(np.diag(sb1**2+1e-8))
    s0 = np.array([np.sqrt(np.dot(np.dot(x-mb0, cov0), x-mb0)) for x in Xt])
    s1 = np.array([np.sqrt(np.dot(np.dot(x-mb1, cov1), x-mb1)) for x in Xt])
    return (s1 < s0).astype(int)

def svm_pred(Xc, yc, Xt):
    if len(np.unique(yc)) < 2: return np.zeros(len(Xt))
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(sc.fit_transform(Xc), yc)
    return clf.predict(sc.transform(Xt))

def sample(y_pool, n):
    i0 = np.where(y_pool==0)[0]
    i1 = np.where(y_pool==1)[0]
    np.random.shuffle(i0); np.random.shuffle(i1)
    return np.concatenate([i0[:min(n,len(i0))], i1[:min(n,len(i1))]])

print('Loading data...', flush=True)
cache = {s: load_eeg(s) for s in Y_SUBJECTS}
print('Data loaded', flush=True)

results = []
kappa_nu_list = [(1,1), (10,10), (50,50)]

for seed in [0]:
    np.random.seed(seed)
    for ho in ['YAC']:
        Xt, yt = cache[ho]
        Xtr, ytr = [], []
        for s in Y_SUBJECTS:
            if s != ho:
                X, y = cache[s]
                if X is not None:
                    Xtr.append(X); ytr.append(y)
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        ms0, ss0 = stats(Xtr, ytr, 0); ms1, ss1 = stats(Xtr, ytr, 1)
        idx = np.random.permutation(len(yt))
        ts = len(yt) // 3
        X_test, y_test = Xt[idx[:ts]], yt[idx[:ts]]
        Xcp, ycp = Xt[idx[ts:]], yt[idx[ts:]]

        for n_cal in [3, 5, 10, 20, 50]:
            if n_cal * 2 > len(Xcp): continue
            ci = sample(ycp, n_cal)
            Xc, yc = Xcp[ci], ycp[ci]
            if len(np.unique(yc)) < 2: continue

            svm_acc = accuracy_score(y_test, svm_pred(Xc, yc, X_test))
            fixed_acc = accuracy_score(y_test, fixed_pred(Xc, yc, X_test, ms0, ss0, ms1, ss1))
            results.append({'n_cal': n_cal, 'method': 'SVM', 'acc': svm_acc})
            results.append({'n_cal': n_cal, 'method': 'SRGC', 'acc': fixed_acc})

            for k, nu in kappa_nu_list:
                apr_acc = accuracy_score(y_test, apr_pred(Xc, yc, X_test, ms0, ss0, ms1, ss1, k, nu))
                results.append({'n_cal': n_cal, 'method': f'APR_{k}_{nu}', 'acc': apr_acc})

df = pd.DataFrame(results)
print('\n' + '='*50)
print('Subject YAC, seed=0:')
print('='*50)
for n in [3, 5, 10, 20, 50]:
    print(f'\n{n}-shot:')
    d = df[df.n_cal == n]
    for m in ['SVM', 'SRGC'] + [f'APR_{k}_{nu}' for k, nu in kappa_nu_list]:
        row = d[d.method == m]
        if len(row) > 0:
            print(f'  {m}: {row.acc.values[0]:.4f}')

os.makedirs('results/final', exist_ok=True)
df.to_csv('results/final/apr_srgc_results.csv', index=False)
print('\nSaved!', flush=True)
print('Done!', flush=True)