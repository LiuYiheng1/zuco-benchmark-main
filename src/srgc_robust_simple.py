"""SR-GC Robust Optimization - Simplified Version"""
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
        return np.ones(len(X_test), dtype=int)

    preds = np.zeros(len(X_test), dtype=int)
    for i in range(len(X_test)):
        x = X_test[i]
        diff_0 = x - mu_blend_0
        diff_1 = x - mu_blend_1
        mahal_0 = np.sqrt(np.dot(np.dot(diff_0, cov_blend_0_inv), diff_0))
        mahal_1 = np.sqrt(np.dot(np.dot(diff_1, cov_blend_1_inv), diff_1))
        preds[i] = 1 if mahal_1 < mahal_0 else 0

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
print('='*60, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2]

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

            preds_srgc = srgc_ledoitwolf(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)
            acc_srgc = accuracy_score(y_test, preds_srgc)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SRGC_LedoitWolf',
                'accuracy': acc_srgc, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
            })

        print(f'.', end='', flush=True)

df = pd.DataFrame(results)
df.to_csv(RESULTS_DIR + '/srgc_robust_results.csv', index=False)

print('', flush=True)
print('\n' + '='*60, flush=True)
print('SR-GC Robustness Results Summary', flush=True)
print('='*60, flush=True)

for n_cal in shot_settings:
    svm_acc = df[df['method'] == 'EEG_SVM'][df['n_cal'] == n_cal]['accuracy'].mean()
    srgc_acc = df[df['method'] == 'SRGC_LedoitWolf'][df['n_cal'] == n_cal]['accuracy'].mean()
    print(f'{n_cal}-shot: SVM={svm_acc:.4f}, SRGC_LW={srgc_acc:.4f}, gap={srgc_acc-svm_acc:+.4f}', flush=True)

print('\nDone!', flush=True)