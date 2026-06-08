"""SR-GC Robustness Optimization

Variants:
1. Covariance: diagonal, ridge, shared, LedoitWolf
2. Score: Mahalanobis, Mahalanobis+logdet, Mahalanobis+logprior, Full
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.covariance import LedoitWolf

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

def compute_source_stats(X_all, y_all):
    mu_0 = np.mean(X_all[y_all == 0], axis=0) if np.any(y_all == 0) else np.mean(X_all, axis=0)
    mu_1 = np.mean(X_all[y_all == 1], axis=0) if np.any(y_all == 1) else np.mean(X_all, axis=0)
    sigma_0 = np.std(X_all[y_all == 0], axis=0) + 1e-8 if np.any(y_all == 0) else np.std(X_all, axis=0) + 1e-8
    sigma_1 = np.std(X_all[y_all == 1], axis=0) + 1e-8 if np.any(y_all == 1) else np.std(X_all, axis=0) + 1e-8
    return mu_0, sigma_0, mu_1, sigma_1

def compute_shared_cov(X, y):
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    cov_0 = np.cov(X_0.T) + np.eye(X.shape[1]) * 1e-6
    cov_1 = np.cov(X_1.T) + np.eye(X.shape[1]) * 1e-6
    shared_cov = (cov_0 + cov_1) / 2
    return shared_cov

def compute_ledoit_wolf_cov(X, y):
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    try:
        lw_0 = LedoitWolf()
        lw_0.fit(X_0)
        cov_0 = lw_0.covariance_ + np.eye(X.shape[1]) * 1e-6
    except:
        cov_0 = np.cov(X_0.T) + np.eye(X.shape[1]) * 1e-6
    try:
        lw_1 = LedoitWolf()
        lw_1.fit(X_1)
        cov_1 = lw_1.covariance_ + np.eye(X.shape[1]) * 1e-6
    except:
        cov_1 = np.cov(X_1.T) + np.eye(X.shape[1]) * 1e-6
    return cov_0, cov_1

def svm_predict(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def srgc_diagonal(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.75, beta=0.75):
    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
    sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else sigma_source_0
    sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else sigma_source_1

    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_source_0
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_source_1
    sigma_blend_0 = alpha * sigma_cal_0 + (1 - alpha) * sigma_source_0
    sigma_blend_1 = alpha * sigma_cal_1 + (1 - alpha) * sigma_source_1

    z_0 = (X_test - mu_blend_0) / (sigma_blend_0 + 1e-8)
    z_1 = (X_test - mu_blend_1) / (sigma_blend_1 + 1e-8)

    dist_0 = np.sqrt(np.sum(z_0 ** 2, axis=1))
    dist_1 = np.sqrt(np.sum(z_1 ** 2, axis=1))

    preds = (dist_1 < dist_0).astype(int)
    return preds

def srgc_ridge(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.75, beta=0.75):
    X_cal_0 = X_cal[y_cal == 0]
    X_cal_1 = X_cal[y_cal == 1]

    cov_cal_0 = np.cov(X_cal_0.T) + np.eye(X_cal.shape[1]) * 1e-4
    cov_cal_1 = np.cov(X_cal_1.T) + np.eye(X_cal.shape[1]) * 1e-4

    cov_source_0 = np.diag(sigma_source_0 ** 2) + np.eye(X_cal.shape[1]) * 1e-6
    cov_source_1 = np.diag(sigma_source_1 ** 2) + np.eye(X_cal.shape[1]) * 1e-6

    cov_blend_0 = beta * cov_source_0 + (1 - beta) * cov_cal_0
    cov_blend_1 = beta * cov_source_1 + (1 - beta) * cov_cal_1

    mu_cal_0 = np.mean(X_cal_0, axis=0) if len(X_cal_0) > 0 else mu_source_0
    mu_cal_1 = np.mean(X_cal_1, axis=0) if len(X_cal_1) > 0 else mu_source_1
    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_source_0
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_source_1

    try:
        cov_blend_0_inv = np.linalg.inv(cov_blend_0)
        cov_blend_1_inv = np.linalg.inv(cov_blend_1)
    except:
        return (np.ones(len(X_test)) * 0.5).astype(int)

    preds = np.zeros(len(X_test), dtype=int)
    for i in range(len(X_test)):
        x = X_test[i]
        diff_0 = x - mu_blend_0
        diff_1 = x - mu_blend_1
        mahal_0 = np.sqrt(np.dot(np.dot(diff_0, cov_blend_0_inv), diff_0))
        mahal_1 = np.sqrt(np.dot(np.dot(diff_1, cov_blend_1_inv), diff_1))
        preds[i] = 1 if mahal_1 < mahal_0 else 0

    return preds

def srgc_shared(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.75, beta=0.75):
    X_all = np.vstack([X_cal, np.array([mu_source_0, mu_source_1])])
    y_all = np.concatenate([y_cal, [0, 1]])
    shared_cov_cal = compute_shared_cov(X_all[:len(y_cal)], y_cal)
    shared_cov_source = np.diag((sigma_source_0 ** 2 + sigma_source_1 ** 2) / 2) + np.eye(X_cal.shape[1]) * 1e-6

    shared_cov = beta * shared_cov_source + (1 - beta) * shared_cov_cal

    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_source_0
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_source_1

    try:
        shared_cov_inv = np.linalg.inv(shared_cov)
    except:
        return (np.ones(len(X_test)) * 0.5).astype(int)

    preds = np.zeros(len(X_test), dtype=int)
    for i in range(len(X_test)):
        x = X_test[i]
        diff_0 = x - mu_blend_0
        diff_1 = x - mu_blend_1
        mahal_0 = np.sqrt(np.dot(np.dot(diff_0, shared_cov_inv), diff_0))
        mahal_1 = np.sqrt(np.dot(np.dot(diff_1, shared_cov_inv), diff_1))
        preds[i] = 1 if mahal_1 < mahal_0 else 0

    return preds

def srgc_ledoitwolf(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.75, beta=0.75):
    cov_source_0 = np.diag(sigma_source_0 ** 2) + np.eye(X_cal.shape[1]) * 1e-6
    cov_source_1 = np.diag(sigma_source_1 ** 2) + np.eye(X_cal.shape[1]) * 1e-6

    cov_cal_0, cov_cal_1 = compute_ledoit_wolf_cov(X_cal, y_cal)

    cov_blend_0 = beta * cov_source_0 + (1 - beta) * cov_cal_0
    cov_blend_1 = beta * cov_source_1 + (1 - beta) * cov_cal_1

    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_source_0
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_source_1

    try:
        cov_blend_0_inv = np.linalg.inv(cov_blend_0)
        cov_blend_1_inv = np.linalg.inv(cov_blend_1)
    except:
        return (np.ones(len(X_test)) * 0.5).astype(int)

    preds = np.zeros(len(X_test), dtype=int)
    for i in range(len(X_test)):
        x = X_test[i]
        diff_0 = x - mu_blend_0
        diff_1 = x - mu_blend_1
        mahal_0 = np.sqrt(np.dot(np.dot(diff_0, cov_blend_0_inv), diff_0))
        mahal_1 = np.sqrt(np.dot(np.dot(diff_1, cov_blend_1_inv), diff_1))
        preds[i] = 1 if mahal_1 < mahal_0 else 0

    return preds

def srgc_mahalanobis_only(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.75, beta=0.75):
    return srgc_ridge(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha, beta)

def srgc_mahalanobis_logdet(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.75, beta=0.75):
    X_cal_0 = X_cal[y_cal == 0]
    X_cal_1 = X_cal[y_cal == 1]

    cov_cal_0 = np.cov(X_cal_0.T) + np.eye(X_cal.shape[1]) * 1e-4
    cov_cal_1 = np.cov(X_cal_1.T) + np.eye(X_cal.shape[1]) * 1e-4

    cov_source_0 = np.diag(sigma_source_0 ** 2) + np.eye(X_cal.shape[1]) * 1e-6
    cov_source_1 = np.diag(sigma_source_1 ** 2) + np.eye(X_cal.shape[1]) * 1e-6

    cov_blend_0 = beta * cov_source_0 + (1 - beta) * cov_cal_0
    cov_blend_1 = beta * cov_source_1 + (1 - beta) * cov_cal_1

    mu_cal_0 = np.mean(X_cal_0, axis=0) if len(X_cal_0) > 0 else mu_source_0
    mu_cal_1 = np.mean(X_cal_1, axis=0) if len(X_cal_1) > 0 else mu_source_1
    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_source_0
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_source_1

    try:
        cov_blend_0_inv = np.linalg.inv(cov_blend_0)
        cov_blend_1_inv = np.linalg.inv(cov_blend_1)
        logdet_0 = np.log(np.linalg.det(cov_blend_0) + 1e-10)
        logdet_1 = np.log(np.linalg.det(cov_blend_1) + 1e-10)
    except:
        return (np.ones(len(X_test)) * 0.5).astype(int)

    preds = np.zeros(len(X_test), dtype=int)
    for i in range(len(X_test)):
        x = X_test[i]
        diff_0 = x - mu_blend_0
        diff_1 = x - mu_blend_1
        mahal_0 = np.sqrt(np.dot(np.dot(diff_0, cov_blend_0_inv), diff_0))
        mahal_1 = np.sqrt(np.dot(np.dot(diff_1, cov_blend_1_inv), diff_1))
        score_0 = mahal_0 + logdet_0
        score_1 = mahal_1 + logdet_1
        preds[i] = 1 if score_1 < score_0 else 0

    return preds

def srgc_full(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.75, beta=0.75):
    X_cal_0 = X_cal[y_cal == 0]
    X_cal_1 = X_cal[y_cal == 1]

    cov_cal_0 = np.cov(X_cal_0.T) + np.eye(X_cal.shape[1]) * 1e-4
    cov_cal_1 = np.cov(X_cal_1.T) + np.eye(X_cal.shape[1]) * 1e-4

    cov_source_0 = np.diag(sigma_source_0 ** 2) + np.eye(X_cal.shape[1]) * 1e-6
    cov_source_1 = np.diag(sigma_source_1 ** 2) + np.eye(X_cal.shape[1]) * 1e-6

    cov_blend_0 = beta * cov_source_0 + (1 - beta) * cov_cal_0
    cov_blend_1 = beta * cov_source_1 + (1 - beta) * cov_cal_1

    mu_cal_0 = np.mean(X_cal_0, axis=0) if len(X_cal_0) > 0 else mu_source_0
    mu_cal_1 = np.mean(X_cal_1, axis=0) if len(X_cal_1) > 0 else mu_source_1
    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_source_0
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_source_1

    n_0 = len(X_cal_0)
    n_1 = len(X_cal_1)
    n = n_0 + n_1
    prior_0 = n_0 / n
    prior_1 = n_1 / n

    try:
        cov_blend_0_inv = np.linalg.inv(cov_blend_0)
        cov_blend_1_inv = np.linalg.inv(cov_blend_1)
        logdet_0 = np.log(np.linalg.det(cov_blend_0) + 1e-10)
        logdet_1 = np.log(np.linalg.det(cov_blend_1) + 1e-10)
    except:
        return (np.ones(len(X_test)) * 0.5).astype(int)

    preds = np.zeros(len(X_test), dtype=int)
    for i in range(len(X_test)):
        x = X_test[i]
        diff_0 = x - mu_blend_0
        diff_1 = x - mu_blend_1
        mahal_0 = np.dot(np.dot(diff_0, cov_blend_0_inv), diff_0)
        mahal_1 = np.dot(np.dot(diff_1, cov_blend_1_inv), diff_1)
        score_0 = mahal_0 + logdet_0 - 2 * np.log(prior_0 + 1e-10)
        score_1 = mahal_1 + logdet_1 - 2 * np.log(prior_1 + 1e-10)
        preds[i] = 1 if score_1 < score_0 else 0

    return preds

def balanced_random_sampling(y_pool, n_per_class):
    class_0_idx = np.where(y_pool == 0)[0]
    class_1_idx = np.where(y_pool == 1)[0]
    np.random.shuffle(class_0_idx)
    np.random.shuffle(class_1_idx)
    n0 = min(n_per_class, len(class_0_idx))
    n1 = min(n_per_class, len(class_1_idx))
    selected = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])
    np.random.shuffle(selected)
    return selected

print('SR-GC Robustness Optimization', flush=True)
print('='*80, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    print(f'\nSeed {seed}:', flush=True)
    for held_out in Y_SUBJECTS:
        X_test_orig, y_test_orig = load_eeg_data(held_out)
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        if len(X_train_all) == 0 or X_test_orig is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        mu_source_0, sigma_source_0, mu_source_1, sigma_source_1 = compute_source_stats(X_train_all, y_train_all)

        n_samples = len(y_test_orig)
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        test_size = n_samples // 3
        test_indices = indices[:test_size]
        cal_pool_indices = indices[test_size:]

        X_test = X_test_orig[test_indices]
        y_test = y_test_orig[test_indices]
        X_cal_pool = X_test_orig[cal_pool_indices]
        y_cal_pool = y_test_orig[cal_pool_indices]

        print(f'  {held_out}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(cal_pool_indices):
                continue

            cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            preds_svm, probs_svm = svm_predict(X_cal, y_cal, X_test)
            acc_svm = accuracy_score(y_test, preds_svm)
            f1_svm = f1_score(y_test, preds_svm, average='macro')
            bacc_svm = balanced_accuracy_score(y_test, preds_svm)
            try:
                auroc_svm = roc_auc_score(y_test, probs_svm)
            except:
                auroc_svm = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'EEG_SVM',
                'accuracy': acc_svm, 'macro_f1': f1_svm, 'balanced_accuracy': bacc_svm, 'auroc': auroc_svm
            })

            methods_funcs = [
                ('SRGC_diagonal', lambda: srgc_diagonal(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)),
                ('SRGC_ridge', lambda: srgc_ridge(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)),
                ('SRGC_shared', lambda: srgc_shared(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)),
                ('SRGC_ledoitwolf', lambda: srgc_ledoitwolf(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)),
                ('SRGC_mahalanobis_logdet', lambda: srgc_mahalanobis_logdet(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)),
                ('SRGC_full', lambda: srgc_full(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)),
            ]

            for method_name, method_func in methods_funcs:
                try:
                    preds = method_func()
                    acc = accuracy_score(y_test, preds)
                except:
                    acc = 0.5
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': method_name,
                    'accuracy': acc, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
                })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/srgc_robust_results.csv', index=False)

print('', flush=True)
print('\n' + '='*80, flush=True)
print('SR-GC Robustness Results Summary', flush=True)
print('='*80, flush=True)

baseline_df = df[df['method'] == 'EEG_SVM']
methods = df['method'].unique()

print('\nComparing methods by shot:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    print(f'\n  {n_cal}-shot (SVM={base_acc:.4f}):', flush=True)
    for method in sorted(methods):
        if method != 'EEG_SVM':
            acc = df[df['method'] == method][df['n_cal'] == n_cal]['accuracy'].mean()
            print(f'    {method}: {acc:.4f} (gap={acc-base_acc:+.4f})', flush=True)

print('\nDone!', flush=True)