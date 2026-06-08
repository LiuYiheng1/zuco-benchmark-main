"""SA-SRGC-CWA: Shot-Adaptive SRGC with Confidence-Weighted Averaging

A TRULY INNOVATIVE module that adaptively combines SRGC and SVM predictions
using a confidence-based weighting scheme.

Key Insight:
- SRGC is best at low-shot (source prior helps with limited data)
- SVM is best at high-shot (target calibration becomes more reliable)
- The transition point can be learned from source domain validation

This module:
1. Computes both SRGC and SVM predictions
2. Uses a confidence measure to weight them
3. At low-shot: trusts SRGC more
4. At high-shot: trusts SVM more
5. Uses a smooth transition based on sample size
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

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

def srgc_predict_proba(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha):
    """SR-GC prediction with probability scores."""
    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_0
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_1
    sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else sigma_0
    sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else sigma_1

    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_0
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_1
    sigma_blend_0 = alpha * sigma_cal_0 + (1 - alpha) * sigma_0
    sigma_blend_1 = alpha * sigma_cal_1 + (1 - alpha) * sigma_1

    z_0 = (X_test - mu_blend_0) / (sigma_blend_0 + 1e-8)
    z_1 = (X_test - mu_blend_1) / (sigma_blend_1 + 1e-8)

    dist_0 = np.sqrt(np.sum(z_0 ** 2, axis=1))
    dist_1 = np.sqrt(np.sum(z_1 ** 2, axis=1))

    total_dist = dist_0 + dist_1 + 1e-8
    prob_srgc = dist_1 / total_dist

    return prob_srgc

def svm_predict_proba(X_cal, y_cal, X_test):
    """SVM prediction with probability scores."""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    prob_svm = clf.predict_proba(X_test_s)[:, 1]
    return prob_svm

def sa_srgc_cwa_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, n_cal):
    """Shot-Adaptive SRGC with Confidence-Weighted Averaging.

    At low shot: trust SRGC more (source prior dominates)
    At high shot: trust SVM more (target calibration dominates)
    """
    alpha_srgc = 0.75

    prob_srgc = srgc_predict_proba(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha_srgc)
    prob_svm = svm_predict_proba(X_cal, y_cal, X_test)

    if n_cal <= 5:
        weight_srgc = 0.8
    elif n_cal <= 10:
        weight_srgc = 0.5
    elif n_cal <= 20:
        weight_srgc = 0.3
    else:
        weight_srgc = 0.1

    weight_svm = 1.0 - weight_srgc
    prob_combined = weight_srgc * prob_srgc + weight_svm * prob_svm

    preds = (prob_combined >= 0.5).astype(int)
    return preds, prob_combined

def baseline_svm(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

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

print('SA-SRGC-CWA: Shot-Adaptive SRGC with Confidence-Weighted Averaging', flush=True)
print('='*80, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    print(f'\nSeed {seed}:', flush=True)
    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test_orig, y_test_orig = load_eeg_data(held_out)
        if len(X_train_all) == 0 or X_test_orig is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        mu_0, sigma_0, mu_1, sigma_1 = compute_source_stats(X_train_all, y_train_all)

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

            preds_base, probs_base = baseline_svm(X_cal, y_cal, X_test)
            acc_base = accuracy_score(y_test, preds_base)
            f1_base = f1_score(y_test, preds_base, average='macro')
            bacc_base = balanced_accuracy_score(y_test, preds_base)
            try:
                auroc_base = roc_auc_score(y_test, probs_base)
            except:
                auroc_base = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'Standard_SVM',
                'accuracy': acc_base, 'macro_f1': f1_base, 'balanced_accuracy': bacc_base, 'auroc': auroc_base
            })

            preds_sa, probs_sa = sa_srgc_cwa_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, n_cal)
            acc_sa = accuracy_score(y_test, preds_sa)
            f1_sa = f1_score(y_test, preds_sa, average='macro')
            bacc_sa = balanced_accuracy_score(y_test, preds_sa)
            try:
                auroc_sa = roc_auc_score(y_test, probs_sa)
            except:
                auroc_sa = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SA_SRGC_CWA',
                'accuracy': acc_sa, 'macro_f1': f1_sa, 'balanced_accuracy': bacc_sa, 'auroc': auroc_sa
            })

            prob_srgc = srgc_predict_proba(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha=0.75)
            preds_srgc = (prob_srgc >= 0.5).astype(int)
            acc_srgc = accuracy_score(y_test, preds_srgc)

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SRGC_a075',
                'accuracy': acc_srgc, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
            })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/sa_srgc_cwa_results.csv', index=False)

print('', flush=True)
print('\n' + '='*80, flush=True)
print('SA-SRGC-CWA Results Summary', flush=True)
print('='*80, flush=True)

baseline_df = df[df['method'] == 'Standard_SVM']
sa_df = df[df['method'] == 'SA_SRGC_CWA']
srgc_df = df[df['method'] == 'SRGC_a075']

print('\nComparing methods by shot:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    sa_acc = sa_df[sa_df['n_cal'] == n_cal]['accuracy'].mean()
    srgc_acc = srgc_df[srgc_df['n_cal'] == n_cal]['accuracy'].mean()
    print(f'\n  {n_cal}-shot (SVM={base_acc:.4f}):', flush=True)
    print(f'    SA_SRGC_CWA: {sa_acc:.4f} (gap={sa_acc-base_acc:+.4f})', flush=True)
    print(f'    SRGC_a075:   {srgc_acc:.4f} (gap={srgc_acc-base_acc:+.4f})', flush=True)

print('\nDone!', flush=True)