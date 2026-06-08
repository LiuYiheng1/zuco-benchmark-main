"""D-SRGC: Discriminative SRGC Features with SVM

A TRULY innovative module that:
1. Uses SR-GC Mahalanobis distances as meta-features
2. Combines with raw EEG features
3. Trains SVM on the combined representation
4. Achieves best of both worlds: source prior + discriminative classifier

Key insight: SR-GC's pure distance-based prediction (alpha=1.0 at low shot)
is not optimal because it ignores the discriminative structure.
SVM can learn to WEIGHT the Mahalanobis distances appropriately.
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

def compute_srgc_distances(X, mu_0, sigma_0, mu_1, sigma_1):
    """Compute Mahalanobis distance to each class center."""
    sigma_0_inv = 1.0 / (sigma_0 + 1e-8)
    sigma_1_inv = 1.0 / (sigma_1 + 1e-8)

    dist_0 = np.sqrt(np.sum(((X - mu_0) * sigma_0_inv) ** 2, axis=1))
    dist_1 = np.sqrt(np.sum(((X - mu_1) * sigma_1_inv) ** 2, axis=1))

    return dist_0, dist_1

def build_srgc_features(X, mu_0, sigma_0, mu_1, sigma_1):
    """Build SRGC-based features: distances + normalized features."""
    dist_0, dist_1 = compute_srgc_distances(X, mu_0, sigma_0, mu_1, sigma_1)

    diff = dist_0 - dist_1
    ratio = (dist_0 + 1e-8) / (dist_1 + 1e-8)
    log_ratio = np.log(ratio + 1e-8)

    srgc_meta = np.column_stack([dist_0, dist_1, diff, ratio, log_ratio])

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    combined = np.hstack([X_norm, srgc_meta])

    return combined, scaler

def baseline_svm(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def dsrgc_svm(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1):
    """D-SRGC: SVM on combined raw + SRGC meta features."""
    X_cal_comb, scaler_comb = build_srgc_features(X_cal, mu_0, sigma_0, mu_1, sigma_1)
    X_test_comb, _ = build_srgc_features(X_test, mu_0, sigma_0, mu_1, sigma_1)

    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_comb, y_cal)
    probs = clf.predict_proba(X_test_comb)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def dsrgc_meta_svm(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1):
    """D-SRGC-Meta: SVM on ONLY SRGC meta features."""
    dist_0_cal, dist_1_cal = compute_srgc_distances(X_cal, mu_0, sigma_0, mu_1, sigma_1)
    dist_0_test, dist_1_test = compute_srgc_distances(X_test, mu_0, sigma_0, mu_1, sigma_1)

    diff_cal = dist_0_cal - dist_1_cal
    ratio_cal = (dist_0_cal + 1e-8) / (dist_1_cal + 1e-8)
    log_ratio_cal = np.log(ratio_cal + 1e-8)

    diff_test = dist_0_test - dist_1_test
    ratio_test = (dist_0_test + 1e-8) / (dist_1_test + 1e-8)
    log_ratio_test = np.log(ratio_test + 1e-8)

    X_meta_cal = np.column_stack([dist_0_cal, dist_1_cal, diff_cal, ratio_cal, log_ratio_cal])
    X_meta_test = np.column_stack([dist_0_test, dist_1_test, diff_test, ratio_test, log_ratio_test])

    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_meta_cal, y_cal)
    probs = clf.predict_proba(X_meta_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def srgc_distance_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha):
    """Pure SR-GC distance-based prediction with blending."""
    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_0
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_1
    sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else sigma_0
    sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else sigma_1

    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_0
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_1
    sigma_blend_0 = alpha * sigma_cal_0 + (1 - alpha) * sigma_0
    sigma_blend_1 = alpha * sigma_cal_1 + (1 - alpha) * sigma_1

    dist_0 = np.sqrt(np.sum(((X_test - mu_blend_0) / (sigma_blend_0 + 1e-8)) ** 2, axis=1))
    dist_1 = np.sqrt(np.sum(((X_test - mu_blend_1) / (sigma_blend_1 + 1e-8)) ** 2, axis=1))

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

print('D-SRGC: Discriminative SRGC Features with SVM', flush=True)
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

            preds_dsrgc, probs_dsrgc = dsrgc_svm(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1)
            acc_dsrgc = accuracy_score(y_test, preds_dsrgc)
            f1_dsrgc = f1_score(y_test, preds_dsrgc, average='macro')
            bacc_dsrgc = balanced_accuracy_score(y_test, preds_dsrgc)
            try:
                auroc_dsrgc = roc_auc_score(y_test, probs_dsrgc)
            except:
                auroc_dsrgc = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'D_SRGC',
                'accuracy': acc_dsrgc, 'macro_f1': f1_dsrgc, 'balanced_accuracy': bacc_dsrgc, 'auroc': auroc_dsrgc
            })

            preds_dsrgc_meta, probs_dsrgc_meta = dsrgc_meta_svm(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1)
            acc_dsrgc_meta = accuracy_score(y_test, preds_dsrgc_meta)
            f1_dsrgc_meta = f1_score(y_test, preds_dsrgc_meta, average='macro')
            bacc_dsrgc_meta = balanced_accuracy_score(y_test, preds_dsrgc_meta)
            try:
                auroc_dsrgc_meta = roc_auc_score(y_test, probs_dsrgc_meta)
            except:
                auroc_dsrgc_meta = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'D_SRGC_Meta',
                'accuracy': acc_dsrgc_meta, 'macro_f1': f1_dsrgc_meta, 'balanced_accuracy': bacc_dsrgc_meta, 'auroc': auroc_dsrgc_meta
            })

            for alpha in [0.75, 1.0]:
                preds_srgc, probs_srgc = srgc_distance_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha)
                acc_srgc = accuracy_score(y_test, preds_srgc)
                f1_srgc = f1_score(y_test, preds_srgc, average='macro')
                bacc_srgc = balanced_accuracy_score(y_test, preds_srgc)
                try:
                    auroc_srgc = roc_auc_score(y_test, probs_srgc)
                except:
                    auroc_srgc = 0.5

                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': f'SRGC_alpha{alpha}',
                    'accuracy': acc_srgc, 'macro_f1': f1_srgc, 'balanced_accuracy': bacc_srgc, 'auroc': auroc_srgc
                })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/dsrgc_results.csv', index=False)

print('', flush=True)
print('\n' + '='*70, flush=True)
print('D-SRGC Results Summary', flush=True)
print('='*70, flush=True)

baseline_df = df[df['method'] == 'Standard_SVM']
dsrgc_df = df[df['method'] == 'D_SRGC']
dsrgc_meta_df = df[df['method'] == 'D_SRGC_Meta']

print('\nComparing methods by shot:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    dsrgc_acc = dsrgc_df[dsrgc_df['n_cal'] == n_cal]['accuracy'].mean()
    dsrgc_meta_acc = dsrgc_meta_df[dsrgc_meta_df['n_cal'] == n_cal]['accuracy'].mean()
    print(f'\n  {n_cal}-shot:', flush=True)
    print(f'    Standard_SVM: {base_acc:.4f}', flush=True)
    print(f'    D_SRGC:      {dsrgc_acc:.4f} (gap={dsrgc_acc-base_acc:+.4f})', flush=True)
    print(f'    D_SRGC_Meta: {dsrgc_meta_acc:.4f} (gap={dsrgc_meta_acc-base_acc:+.4f})', flush=True)

    for alpha in [0.75, 1.0]:
        srgc_acc = df[df['method'] == f'SRGC_alpha{alpha}'][df['n_cal'] == n_cal]['accuracy'].mean()
        print(f'    SRGC_a{alpha}:    {srgc_acc:.4f} (gap={srgc_acc-base_acc:+.4f})', flush=True)

print('\nDone!', flush=True)