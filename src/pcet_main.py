"""PCET Main Experiment: Compare with baselines

Compares:
- EEG_SVM: Standard SVM
- SRGC: Source-Regularized Gaussian Calibration
- PCET: Predictive Coding EEG Transfer
- PCET_SRGC: Combination of PCET and SRGC
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.decomposition import PCA

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

def train_pca_models(X_train, y_train, n_components=20):
    pca_models = {}
    for c in [0, 1]:
        X_c = X_train[y_train == c]
        if len(X_c) > n_components:
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X_c)
            pca_models[c] = pca
        else:
            pca_models[c] = None
    return pca_models

def compute_prediction_errors(X, pca_models):
    n_samples = len(X)
    error_features = np.zeros((n_samples, 2))

    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_reconstructed = pca.inverse_transform(pca.transform(X))
            errors = X - X_reconstructed
            error_features[:, i] = np.sqrt(np.sum(errors ** 2, axis=1))

    return error_features

def svm_predict(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def srgc_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.75):
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

def pcet_predict(X_cal, y_cal, X_test, n_components=20, lambda_reg=0.1):
    pca_models = train_pca_models(X_cal, y_cal, n_components)

    error_cal = compute_prediction_errors(X_cal, pca_models)
    error_test = compute_prediction_errors(X_test, pca_models)

    scaler = StandardScaler()
    X_cal_combined = np.hstack([scaler.fit_transform(X_cal), error_cal])
    X_test_combined = np.hstack([scaler.transform(X_test), error_test])

    clf = RidgeClassifier(alpha=lambda_reg)
    clf.fit(X_cal_combined, y_cal)
    preds = clf.predict(X_test_combined)
    return preds

def pcet_srgc_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.75, n_components=20, reg_lambda=0.1):
    pca_models = train_pca_models(X_cal, y_cal, n_components)

    error_cal = compute_prediction_errors(X_cal, pca_models)
    error_test = compute_prediction_errors(X_test, pca_models)

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

    scaler = StandardScaler()
    X_cal_combined = np.hstack([scaler.fit_transform(X_cal), error_cal])
    X_test_combined = np.hstack([scaler.transform(X_test), error_test])

    clf = RidgeClassifier(alpha=reg_lambda)
    clf.fit(X_cal_combined, y_cal)
    preds_pcet = clf.predict(X_test_combined)

    srgc_preds = (dist_1 < dist_0).astype(int)
    final_preds = ((preds_pcet == 1) & (srgc_preds == 1)).astype(int)
    return final_preds

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

print('PCET Main Experiment', flush=True)
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

            preds_srgc = srgc_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)
            acc_srgc = accuracy_score(y_test, preds_srgc)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SRGC',
                'accuracy': acc_srgc, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
            })

            preds_pcet = pcet_predict(X_cal, y_cal, X_test)
            acc_pcet = accuracy_score(y_test, preds_pcet)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'PCET',
                'accuracy': acc_pcet, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
            })

            preds_pcet_srgc = pcet_srgc_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)
            acc_pcet_srgc = accuracy_score(y_test, preds_pcet_srgc)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'PCET_SRGC',
                'accuracy': acc_pcet_srgc, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
            })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/pcet_main_results.csv', index=False)

print('', flush=True)
print('\n' + '='*80, flush=True)
print('PCET Main Results Summary', flush=True)
print('='*80, flush=True)

methods = ['EEG_SVM', 'SRGC', 'PCET', 'PCET_SRGC']
print('\nComparing methods by shot:', flush=True)
for n_cal in shot_settings:
    print(f'\n  {n_cal}-shot:', flush=True)
    for method in methods:
        acc = df[df['method'] == method][df['n_cal'] == n_cal]['accuracy'].mean()
        svm_acc = df[df['method'] == 'EEG_SVM'][df['n_cal'] == n_cal]['accuracy'].mean()
        gap = acc - svm_acc
        print(f'    {method}: {acc:.4f} (gap={gap:+.4f})', flush=True)

print('\nDone!', flush=True)