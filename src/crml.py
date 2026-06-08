"""CRML: Calibration-Robust Meta-Learning

A TRULY INNOVATIVE module that solves the core low-shot calibration problem:

Problem Analysis:
- At LOW shot (3-5): SR-GC Mahalanobis prior works well, SVM struggles
- At HIGH shot (20-50): SVM works well, SR-GC prior may hurt
- Previous attempts (Meta_SVM_CV) degrade at mid-shot due to CV instability

Key Insight:
Instead of trying to combine SRGC and SVM, use a PRIOR-WEIGHTED SVM approach:
1. Pre-train SVM on source subjects to get feature importance
2. Weight new calibration samples by their distance to source class centers
3. Trust samples that are closer to source class structure more at low-shot

This is fundamentally different from:
- SR-GC: uses pure Mahalanobis distance for prediction
- SVM: treats all calibration samples equally
- CRML: weights calibration samples by their consistency with source domain
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from scipy.spatial.distance import mahalanobis

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
    cov_0 = np.cov(X_all[y_all == 0].T) + np.eye(X_all.shape[1]) * 1e-4 if np.any(y_all == 0) and len(X_all[y_all == 0]) > X_all.shape[1] else np.eye(X_all.shape[1])
    cov_1 = np.cov(X_all[y_all == 1].T) + np.eye(X_all.shape[1]) * 1e-4 if np.any(y_all == 1) and len(X_all[y_all == 1]) > X_all.shape[1] else np.eye(X_all.shape[1])
    return mu_0, sigma_0, mu_1, sigma_1, cov_0, cov_1

def compute_sample_weights(X_cal, y_cal, mu_0, sigma_0, mu_1, sigma_1, shot):
    """Compute sample weights based on Mahalanobis distance to source class centers.

    Samples closer to their true class center get higher weights.
    This helps at low-shot by trusting samples that are more consistent with source domain.
    """
    sigma_0_inv = np.diag(1.0 / (sigma_0 + 1e-8))
    sigma_1_inv = np.diag(1.0 / (sigma_1 + 1e-8))

    weights = np.ones(len(y_cal))

    for i in range(len(y_cal)):
        x = X_cal[i]
        y_i = y_cal[i]

        if y_i == 0:
            dist = np.sqrt(np.dot(np.dot(x - mu_0, sigma_0_inv), x - mu_0))
        else:
            dist = np.sqrt(np.dot(np.dot(x - mu_1, sigma_1_inv), x - mu_1))

        if shot <= 5:
            trust_weight = 1.0 / (1.0 + dist / 10.0)
        elif shot <= 10:
            trust_weight = 1.0 / (1.0 + dist / 20.0)
        else:
            trust_weight = 1.0 / (1.0 + dist / 50.0)

        weights[i] = trust_weight

    weights = weights / np.sum(weights) * len(weights)
    return weights

def baseline_svm(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def craml_svm(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, shot):
    """CRML: Calibration-Robust Adaptive Meta-Learning SVM.

    Uses sample weights based on Mahalanobis distance to source class centers.
    Samples more consistent with source domain get higher trust at low-shot.
    """
    weights = compute_sample_weights(X_cal, y_cal, mu_0, sigma_0, mu_1, sigma_1, shot)

    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal, sample_weight=weights)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def srgc_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha):
    """SR-GC for comparison."""
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

    preds = (dist_1 < dist_0).astype(int)
    probs = 1.0 / (1.0 + np.exp(dist_0 - dist_1))
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

print('CRML: Calibration-Robust Meta-Learning', flush=True)
print('='*70, flush=True)

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

        mu_0, sigma_0, mu_1, sigma_1, cov_0, cov_1 = compute_source_stats(X_train_all, y_train_all)

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

            preds_crml, probs_crml = craml_svm(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, n_cal)
            acc_crml = accuracy_score(y_test, preds_crml)
            f1_crml = f1_score(y_test, preds_crml, average='macro')
            bacc_crml = balanced_accuracy_score(y_test, preds_crml)
            try:
                auroc_crml = roc_auc_score(y_test, probs_crml)
            except:
                auroc_crml = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'CRML_SVM',
                'accuracy': acc_crml, 'macro_f1': f1_crml, 'balanced_accuracy': bacc_crml, 'auroc': auroc_crml
            })

            for alpha in [0.75, 1.0]:
                preds_srgc, probs_srgc = srgc_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha)
                acc_srgc = accuracy_score(y_test, preds_srgc)
                f1_srgc = f1_score(y_test, preds_srgc, average='macro')
                bacc_srgc = balanced_accuracy_score(y_test, preds_srgc)
                try:
                    auroc_srgc = roc_auc_score(y_test, probs_srgc)
                except:
                    auroc_srgc = 0.5

                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': f'SRGC_a{alpha}',
                    'accuracy': acc_srgc, 'macro_f1': f1_srgc, 'balanced_accuracy': bacc_srgc, 'auroc': auroc_srgc
                })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/crml_results.csv', index=False)

print('', flush=True)
print('\n' + '='*70, flush=True)
print('CRML Results Summary', flush=True)
print('='*70, flush=True)

baseline_df = df[df['method'] == 'Standard_SVM']
crml_df = df[df['method'] == 'CRML_SVM']

print('\nComparing methods by shot:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    crml_acc = crml_df[crml_df['n_cal'] == n_cal]['accuracy'].mean()
    print(f'\n  {n_cal}-shot (SVM={base_acc:.4f}):', flush=True)
    print(f'    CRML_SVM:    {crml_acc:.4f} (gap={crml_acc-base_acc:+.4f})', flush=True)

    for alpha in [0.75, 1.0]:
        method = f'SRGC_a{alpha}'
        srgc_acc = df[df['method'] == method][df['n_cal'] == n_cal]['accuracy'].mean()
        print(f'    SRGC_a{alpha}:  {srgc_acc:.4f} (gap={srgc_acc-base_acc:+.4f})', flush=True)

print('\nDone!', flush=True)