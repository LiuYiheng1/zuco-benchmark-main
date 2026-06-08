"""SAGE: Shot-Adaptive Gaussian-Discriminative Calibration - Fast Version"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None
    data = np.load(path, allow_pickle=True).item()
    X, y = [], []
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 2 and parts[1] == "NR":
            label = 1
        elif len(parts) >= 2 and parts[1] == "TSR":
            label = 0
        else:
            continue
        features = np.array(values[:-1], dtype=np.float64)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def compute_class_stats(X, y, class_label):
    X_class = X[y == class_label]
    if len(X_class) == 0:
        return None, None
    mu = np.mean(X_class, axis=0)
    sigma = np.std(X_class, axis=0) + 1e-8
    return mu, sigma

def sr_gaussian_predict(X_cal, y_cal, X_test, mu_s0, sigma_s0, mu_s1, sigma_s1, alpha=0.25):
    mu_c0, sigma_c0 = compute_class_stats(X_cal, y_cal, 0)
    mu_c1, sigma_c1 = compute_class_stats(X_cal, y_cal, 1)
    if mu_c0 is None or mu_c1 is None:
        return np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))

    mu_b0 = alpha * mu_c0 + (1 - alpha) * mu_s0
    mu_b1 = alpha * mu_c1 + (1 - alpha) * mu_s1
    sigma_b0 = alpha * sigma_c0 + (1 - alpha) * sigma_s0
    sigma_b1 = alpha * sigma_c1 + (1 - alpha) * sigma_s1

    cov_inv_0 = np.linalg.inv(np.diag(sigma_b0 ** 2 + 1e-8))
    cov_inv_1 = np.linalg.inv(np.diag(sigma_b1 ** 2 + 1e-8))

    scores_0 = np.array([np.sqrt(np.dot(np.dot(x - mu_b0, cov_inv_0), x - mu_b0)) for x in X_test])
    scores_1 = np.array([np.sqrt(np.dot(np.dot(x - mu_b1, cov_inv_1), x - mu_b1)) for x in X_test])

    preds = (scores_1 < scores_0).astype(int)
    return preds, scores_0, scores_1

def svm_predict(X_cal, y_cal, X_test, return_scores=False):
    if len(np.unique(y_cal)) < 2:
        return np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)
    preds = clf.predict(X_test_s)
    if return_scores:
        return preds, probs[:, 0], probs[:, 1]
    return preds

def balanced_sampling(y_pool, n_per_class):
    idx0 = np.where(y_pool == 0)[0]
    idx1 = np.where(y_pool == 1)[0]
    np.random.shuffle(idx0)
    np.random.shuffle(idx1)
    n0 = min(n_per_class, len(idx0))
    n1 = min(n_per_class, len(idx1))
    selected = np.concatenate([idx0[:n0], idx1[:n1]])
    np.random.shuffle(selected)
    return selected

print('SAGE Experiments (Fast)')
print('='*60)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]
SAGE_LAMBDA = {3: 0.9, 5: 0.9, 10: 0.7, 20: 0.5, 50: 0.1}

source_cache = {}
for subj in Y_SUBJECTS:
    X, y = load_eeg_data(subj)
    if X is not None:
        source_cache[subj] = (X, y)

for seed in seeds:
    print(f'\nSeed {seed}:')
    np.random.seed(seed)

    for held_out in Y_SUBJECTS:
        X_test_orig, y_test_orig = source_cache.get(held_out, (None, None))
        if X_test_orig is None:
            continue

        X_train_all = []
        y_train_all = []
        for subj in Y_SUBJECTS:
            if subj != held_out and subj in source_cache:
                X, y = source_cache[subj]
                X_train_all.append(X)
                y_train_all.append(y)

        if len(X_train_all) == 0:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        mu_s0, sigma_s0 = compute_class_stats(X_train_all, y_train_all, 0)
        mu_s1, sigma_s1 = compute_class_stats(X_train_all, y_train_all, 1)
        if mu_s0 is None or mu_s1 is None:
            continue

        indices = np.random.permutation(len(y_test_orig))
        test_size = len(y_test_orig) // 3
        cal_pool_idx = indices[test_size:]
        test_idx = indices[:test_size]

        X_test = X_test_orig[test_idx]
        y_test = y_test_orig[test_idx]
        X_cal_pool = X_test_orig[cal_pool_idx]
        y_cal_pool = y_test_orig[cal_pool_idx]

        print(f' {held_out}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(cal_pool_idx):
                continue

            cal_idx = balanced_sampling(y_cal_pool, n_cal)
            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            svm_p, svm_s0, svm_s1 = svm_predict(X_cal, y_cal, X_test, return_scores=True)
            sr_p, sr_s0, sr_s1 = sr_gaussian_predict(X_cal, y_cal, X_test, mu_s0, sigma_s0, mu_s1, sigma_s1, alpha=0.25)

            for m, p in [('EEG_SVM', svm_p), ('SR-GC', sr_p)]:
                acc = accuracy_score(y_test, p)
                f1 = f1_score(y_test, p, average='macro')
                bacc = balanced_accuracy_score(y_test, p)
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': m, 'lambda': 1.0 if m == 'SR-GC' else 0.0,
                    'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc
                })

            lam = SAGE_LAMBDA[n_cal]
            fused = lam * (sr_s1 - sr_s0) + (1 - lam) * (svm_s1 - svm_s0)
            p_rule = (fused > 0).astype(int)
            acc_r = accuracy_score(y_test, p_rule)
            f1_r = f1_score(y_test, p_rule, average='macro')
            bacc_r = balanced_accuracy_score(y_test, p_rule)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SAGE_rule', 'lambda': lam,
                'accuracy': acc_r, 'macro_f1': f1_r, 'balanced_accuracy': bacc_r
            })

            n = len(y_cal)
            if n >= 4:
                cv_err_sr = []
                cv_err_svm = []
                for i in range(n):
                    tr = np.array([j for j in range(n) if j != i])
                    vl = np.array([i])
                    _, sr0, sr1 = sr_gaussian_predict(X_cal[tr], y_cal[tr], X_cal[vl], mu_s0, sigma_s0, mu_s1, sigma_s1, alpha=0.25)
                    _, sv0, sv1 = svm_predict(X_cal[tr], y_cal[tr], X_cal[vl], return_scores=True)
                    cv_err_sr.append(1 - accuracy_score(y_cal[vl], (sr1 < sr0).astype(int)))
                    cv_err_svm.append(1 - accuracy_score(y_cal[vl], (sv1 > sv0).astype(int)))
                R_g, R_d = np.mean(cv_err_sr), np.mean(cv_err_svm)
                lam_cv = 1.0 / (1.0 + np.exp(-5.0 * (R_d - R_g)))
            else:
                lam_cv = 0.5

            fused_cv = lam_cv * (sr_s1 - sr_s0) + (1 - lam_cv) * (svm_s1 - svm_s0)
            p_cv = (fused_cv > 0).astype(int)
            acc_cv = accuracy_score(y_test, p_cv)
            f1_cv = f1_score(y_test, p_cv, average='macro')
            bacc_cv = balanced_accuracy_score(y_test, p_cv)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SAGE_cv', 'lambda': lam_cv,
                'accuracy': acc_cv, 'macro_f1': f1_cv, 'balanced_accuracy': bacc_cv
            })

        print('.', end='', flush=True)

df = pd.DataFrame(results)
df.to_csv(RESULTS_DIR + '/sage_results.csv', index=False)

print('\n\n' + '='*60)
print('SAGE Results Summary')
print('='*60)

for n_cal in shot_settings:
    print(f'\n{n_cal}-shot:')
    svm_a = df[(df['method'] == 'EEG_SVM') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    sr_a = df[(df['method'] == 'SR-GC') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    rule_a = df[(df['method'] == 'SAGE_rule') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    cv_a = df[(df['method'] == 'SAGE_cv') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    print(f'  EEG_SVM:   {svm_a:.4f}')
    print(f'  SR-GC:     {sr_a:.4f} ({sr_a-svm_a:+.4f})')
    print(f'  SAGE_rule: {rule_a:.4f} ({rule_a-svm_a:+.4f}, vs SR-GC: {rule_a-sr_a:+.4f})')
    print(f'  SAGE_cv:   {cv_a:.4f} ({cv_a-svm_a:+.4f}, vs SR-GC: {cv_a-sr_a:+.4f})')

print('\nOverall:')
for m in ['EEG_SVM', 'SR-GC', 'SAGE_rule', 'SAGE_cv']:
    if m in df['method'].values:
        print(f'  {m}: {df[df["method"]==m]["accuracy"].mean():.4f}')

print('\n3/5/10 avg:')
for m in ['EEG_SVM', 'SR-GC', 'SAGE_rule', 'SAGE_cv']:
    if m in df['method'].values:
        avg = df[(df['method']==m) & (df['n_cal'].isin([3,5,10]))]['accuracy'].mean()
        print(f'  {m}: {avg:.4f}')

print('\nDone!')