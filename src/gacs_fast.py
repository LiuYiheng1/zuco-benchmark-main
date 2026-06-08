"""GACS: Fast version - SVM only"""
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

def compute_source_stats(X, y):
    mu0 = np.mean(X[y==0], axis=0)
    mu1 = np.mean(X[y==1], axis=0)
    sigma0 = np.std(X[y==0], axis=0) + 1e-8
    sigma1 = np.std(X[y==1], axis=0) + 1e-8
    return mu0, sigma0, mu1, sigma1

def gaussian_posterior(X_pool, mu0, sigma0, mu1, sigma1, temperature=1.0):
    cov_inv_0 = np.linalg.inv(np.diag(sigma0**2))
    cov_inv_1 = np.linalg.inv(np.diag(sigma1**2))
    scores_0 = np.array([np.sqrt(np.dot(np.dot(x-mu0, cov_inv_0), x-mu0)) for x in X_pool])
    scores_1 = np.array([np.sqrt(np.dot(np.dot(x-mu1, cov_inv_1), x-mu1)) for x in X_pool])
    logits = np.stack([-scores_0, -scores_1], axis=1)
    exp_logits = np.exp(logits / temperature)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pseudo_labels = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    boundary_score = np.abs(probs[:, 1] - 0.5)
    return pseudo_labels, confidence, boundary_score

def gacs_sample(X_pool, pseudo_labels, confidence, boundary_score, n_per_class, mode='anchor'):
    selected = []
    n_select = n_per_class * 2

    if mode == 'anchor':
        for c in [0, 1]:
            c_idx = np.where(pseudo_labels == c)[0]
            if len(c_idx) == 0: continue
            c_conf = confidence[c_idx]
            top_k = min(n_per_class, len(c_idx))
            top_idx = c_idx[np.argsort(-c_conf)[:top_k]]
            selected.extend(top_idx)
        while len(selected) < n_select:
            remaining = [i for i in range(len(X_pool)) if i not in selected]
            if not remaining: break
            selected.append(remaining[np.random.randint(len(remaining))])

    elif mode == 'boundary':
        for c in [0, 1]:
            c_idx = np.where(pseudo_labels == c)[0]
            if len(c_idx) == 0: continue
            n_c = min(n_per_class // 2, len(c_idx))
            c_conf = confidence[c_idx]
            top_idx = c_idx[np.argsort(-c_conf)[:n_c]]
            selected.extend(top_idx)
        boundary_idx = np.argsort(boundary_score)[:n_per_class]
        for idx in boundary_idx:
            if idx not in selected:
                selected.append(idx)
        while len(selected) < n_select:
            remaining = [i for i in range(len(X_pool)) if i not in selected]
            if not remaining: break
            selected.append(remaining[np.random.randint(len(remaining))])

    else:
        for c in [0, 1]:
            c_idx = np.where(pseudo_labels == c)[0]
            if len(c_idx) == 0: continue
            n_c = min(n_per_class // 2, len(c_idx))
            c_conf = confidence[c_idx]
            top_idx = c_idx[np.argsort(-c_conf)[:n_c]]
            selected.extend(top_idx)
        boundary_idx = np.argsort(boundary_score)[:n_per_class]
        for idx in boundary_idx:
            if idx not in selected:
                selected.append(idx)

    return np.array(selected[:n_select])

def svm_predict(X_cal, y_cal, X_test):
    if len(np.unique(y_cal)) < 2: return np.zeros(len(X_test))
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(sc.fit_transform(X_cal), y_cal)
    return clf.predict(sc.transform(X_test))

def random_sample(y_pool, n_per_class):
    idx0 = np.where(y_pool==0)[0]
    idx1 = np.where(y_pool==1)[0]
    np.random.shuffle(idx0)
    np.random.shuffle(idx1)
    n0 = min(n_per_class, len(idx0))
    n1 = min(n_per_class, len(idx1))
    selected = np.concatenate([idx0[:n0], idx1[:n1]])
    np.random.shuffle(selected)
    return selected

print('Loading data...')
cache = {s: load_eeg(s) for s in Y_SUBJECTS}
print('Data loaded')

results = []
shot_settings = [1, 3, 5, 10, 20, 50]

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
        mu0, sigma0, mu1, sigma1 = compute_source_stats(Xtr, ytr)

        idx = np.random.permutation(len(yt))
        ts = len(yt) // 3
        X_test, y_test = Xt[idx[:ts]], yt[idx[:ts]]
        X_cp, y_cp = Xt[idx[ts:]], yt[idx[ts:]]

        print(f' {ho}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(X_cp): continue

            pseudo_y, conf, boundary = gaussian_posterior(X_cp, mu0, sigma0, mu1, sigma1)

            for mode in ['anchor', 'boundary', 'balanced']:
                sel_idx = gacs_sample(X_cp, pseudo_y, conf, boundary, n_cal, mode)
                Xc, yc = X_cp[sel_idx], y_cp[sel_idx]
                if len(np.unique(yc)) < 2: continue
                preds = svm_predict(Xc, yc, X_test)
                results.append({
                    'seed': seed, 'subject': ho, 'n_cal': n_cal,
                    'sampling': f'GACS_{mode}',
                    'accuracy': accuracy_score(y_test, preds)
                })

            rand_idx = random_sample(y_cp, n_cal)
            Xc, yc = X_cp[rand_idx], y_cp[rand_idx]
            if len(np.unique(yc)) < 2: continue
            preds = svm_predict(Xc, yc, X_test)
            results.append({
                'seed': seed, 'subject': ho, 'n_cal': n_cal,
                'sampling': 'Random',
                'accuracy': accuracy_score(y_test, preds)
            })

        print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv('results/final/gacs_results.csv', index=False)

print('')
print('\n' + '='*60)
print('SUMMARY')
print('='*60)

df = pd.DataFrame(results)

print('\nGACS vs Random (SVM):')
for n in shot_settings:
    print(f'\n{n}-shot:')
    for s in ['Random', 'GACS_anchor', 'GACS_boundary', 'GACS_balanced']:
        d = df[(df.sampling == s) & (df.n_cal == n)]
        if len(d) > 0:
            print(f'  {s}: {d.accuracy.mean():.4f}')

print('\nDone!')