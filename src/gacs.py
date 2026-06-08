"""GACS: Gaussian-prior Active Calibration Sampling"""
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
    return pseudo_labels, confidence, boundary_score, probs

def sample_anchor(X_pool, pseudo_labels, confidence, target_class, n_select):
    class_indices = np.where(pseudo_labels == target_class)[0]
    if len(class_indices) == 0:
        return np.array([], dtype=int)
    class_confs = confidence[class_indices]
    top_k = min(n_select, len(class_indices))
    top_indices = class_indices[np.argsort(-class_confs)[:top_k]]
    return top_indices

def sample_boundary(X_pool, boundary_score, n_select):
    top_k = min(n_select, len(X_pool))
    indices = np.argsort(boundary_score)[:top_k]
    return indices

def sample_diversity(X_pool, selected_indices, n_select):
    if len(selected_indices) >= n_select:
        return selected_indices[:n_select]
    if len(selected_indices) == 0:
        return np.random.choice(len(X_pool), min(n_select, len(X_pool)), replace=False)
    remaining = np.array([i for i in range(len(X_pool)) if i not in selected_indices])
    if len(remaining) <= n_select - len(selected_indices):
        return np.concatenate([selected_indices, remaining])
    selected = list(selected_indices)
    for _ in range(n_select - len(selected_indices)):
        if len(remaining) == 0:
            break
        remaining = np.array([i for i in remaining if i not in selected])
        if len(remaining) == 0:
            break
        dists = np.array([np.min(np.linalg.norm(X_pool[i] - X_pool[s]) for s in selected) for i in remaining])
        next_idx = remaining[np.argmax(dists)]
        selected.append(next_idx)
    return np.array(selected)

def gacs_sampling(X_pool, pseudo_labels, confidence, boundary_score, n_per_class, strategy='anchor'):
    n_select = n_per_class
    if strategy == 'anchor':
        n_anchor_per_class = int(np.ceil(n_select * 0.7))
        n_diversity = int(n_select * 0.3)
        selected = []
        for c in [0, 1]:
            anchor_idx = sample_anchor(X_pool, pseudo_labels, confidence, c, n_anchor_per_class)
            selected.extend(anchor_idx)
        remaining_diversity = n_select * 2 - len(selected)
        if remaining_diversity > 0:
            div_idx = sample_boundary(X_pool, boundary_score, remaining_diversity)
            selected.extend(div_idx)
        selected = np.array(list(set(selected)))[:n_select*2]
    elif strategy == 'boundary':
        n_anchor_per_class = int(np.ceil(n_select * 0.25))
        n_boundary = int(n_select * 0.3)
        n_diversity = int(n_select * 0.2)
        selected = []
        for c in [0, 1]:
            anchor_idx = sample_anchor(X_pool, pseudo_labels, confidence, c, n_anchor_per_class)
            selected.extend(anchor_idx)
        boundary_idx = sample_boundary(X_pool, boundary_score, n_boundary)
        selected.extend(boundary_idx)
        remaining = n_select * 2 - len(selected)
        if remaining > 0:
            div_idx = sample_boundary(X_pool, boundary_score, remaining)
            selected.extend(div_idx)
        selected = np.array(list(set(selected)))[:n_select*2]
    else:
        n_anchor_per_class = int(np.ceil(n_select * 0.2))
        n_boundary = int(np.ceil(n_select * 0.4))
        n_diversity = n_select * 2 - 2 * n_anchor_per_class - n_boundary
        selected = []
        for c in [0, 1]:
            anchor_idx = sample_anchor(X_pool, pseudo_labels, confidence, c, n_anchor_per_class)
            selected.extend(anchor_idx)
        boundary_idx = sample_boundary(X_pool, boundary_score, n_boundary)
        selected.extend(boundary_idx)
        remaining = n_select * 2 - len(selected)
        if remaining > 0:
            div_idx = sample_boundary(X_pool, boundary_score, remaining)
            selected.extend(div_idx)
        selected = np.array(list(set(selected)))[:n_select*2]
    return selected

def svm_predict(X_cal, y_cal, X_test):
    if len(np.unique(y_cal)) < 2: return np.zeros(len(X_test))
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(sc.fit_transform(X_cal), y_cal)
    return clf.predict(sc.transform(X_test))

def srgc_predict(X_cal, y_cal, X_test, mu0, sigma0, mu1, sigma1, alpha=0.75):
    mc0, sc0 = np.mean(X_cal[y_cal==0], axis=0), np.std(X_cal[y_cal==0], axis=0) + 1e-8
    mc1, sc1 = np.mean(X_cal[y_cal==1], axis=0), np.std(X_cal[y_cal==1], axis=0) + 1e-8
    if mc0 is None: return np.zeros(len(X_test))
    mb0, mb1 = alpha*mu0+(1-alpha)*mc0, alpha*mu1+(1-alpha)*mc1
    sb0, sb1 = alpha*sigma0+(1-alpha)*sc0, alpha*sigma1+(1-alpha)*sc1
    cov0 = np.linalg.inv(np.diag(sb0**2+1e-8))
    cov1 = np.linalg.inv(np.diag(sb1**2+1e-8))
    s0 = np.array([np.sqrt(np.dot(np.dot(x-mb0, cov0), x-mb0)) for x in X_test])
    s1 = np.array([np.sqrt(np.dot(np.dot(x-mb1, cov1), x-mb1)) for x in X_test])
    return (s1 < s0).astype(int)

def random_sampling(y_pool, n_per_class):
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
strategies = ['anchor', 'boundary', 'balanced']
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

            for strat in strategies:
                sel_idx = gacs_sampling(X_cp, None, None, None, n_cal, strat)
                if len(sel_idx) < 2: continue
                Xc, yc = X_cp[sel_idx], y_cp[sel_idx]
                if len(np.unique(yc)) < 2: continue

                pseudo_y, conf, boundary, probs = gaussian_posterior(X_cp, mu0, sigma0, mu1, sigma1)
                sel_idx = gacs_sampling(X_cp, pseudo_y, conf, boundary, n_cal, strat)
                Xc, yc = X_cp[sel_idx], y_cp[sel_idx]

                if len(np.unique(yc)) < 2: continue

                for clf_name, clf_func in [('SVM', svm_predict), ('SRGC', srgc_predict)]:
                    if clf_name == 'SRGC':
                        preds = clf_func(Xc, yc, X_test, mu0, sigma0, mu1, sigma1)
                    else:
                        preds = clf_func(Xc, yc, X_test)
                    results.append({
                        'seed': seed, 'subject': ho, 'n_cal': n_cal,
                        'sampling': f'GACS_{strat}', 'classifier': clf_name,
                        'accuracy': accuracy_score(y_test, preds),
                        'pseudo_balance': np.mean(pseudo_y),
                        'conf_mean': conf.mean()
                    })

            rand_idx = random_sampling(y_cp, n_cal)
            Xc, yc = X_cp[rand_idx], y_cp[rand_idx]
            if len(np.unique(yc)) < 2: continue

            for clf_name, clf_func in [('SVM', svm_predict), ('SRGC', srgc_predict)]:
                if clf_name == 'SRGC':
                    preds = clf_func(Xc, yc, X_test, mu0, sigma0, mu1, sigma1)
                else:
                    preds = clf_func(Xc, yc, X_test)
                results.append({
                    'seed': seed, 'subject': ho, 'n_cal': n_cal,
                    'sampling': 'Random', 'classifier': clf_name,
                    'accuracy': accuracy_score(y_test, preds),
                    'pseudo_balance': 0.5,
                    'conf_mean': 0.5
                })

        print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv('results/final/gacs_results.csv', index=False)

print('')
print('\n' + '='*60)
print('SUMMARY')
print('='*60)

df = pd.DataFrame(results)

print('\nGACS vs Random (SVM classifier):')
for n in shot_settings:
    print(f'\n{n}-shot:')
    for s in ['Random', 'GACS_anchor', 'GACS_boundary', 'GACS_balanced']:
        d = df[(df.sampling == s) & (df.n_cal == n) & (df.classifier == 'SVM')]
        if len(d) > 0:
            print(f'  {s}: {d.accuracy.mean():.4f}')

print('\nGACS vs Random (SRGC classifier):')
for n in shot_settings:
    print(f'\n{n}-shot:')
    for s in ['Random', 'GACS_anchor', 'GACS_boundary', 'GACS_balanced']:
        d = df[(df.sampling == s) & (df.n_cal == n) & (df.classifier == 'SRGC')]
        if len(d) > 0:
            print(f'  {s}: {d.accuracy.mean():.4f}')

print('\nDone!')