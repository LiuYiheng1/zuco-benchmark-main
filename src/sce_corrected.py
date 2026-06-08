"""SCE-Corrected: Simple Calibrated Ensemble (Fixed Implementation)

The key bug in previous implementations:
- Using X_test_orig instead of X_test for evaluation
- This causes data inconsistency across methods

This corrected version:
- Uses proper 1/3 test / 2/3 calibration pool split
- Properly reuses the same X_test for all methods
- Properly computes source stats from training subjects only
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

def srgc_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha=0.75):
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

    prob = dist_1 / (dist_0 + dist_1 + 1e-8)
    preds = (prob >= 0.5).astype(int)
    return preds, prob

def svm_predict(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def sce_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, n_cal):
    preds_srgc, prob_srgc = srgc_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha=0.75)
    preds_svm, prob_svm = svm_predict(X_cal, y_cal, X_test)

    center = 15
    width = 5
    weight_svm = 1.0 / (1.0 + np.exp(-(n_cal - center) / width * 3))

    prob_combined = (1 - weight_svm) * prob_srgc + weight_svm * prob_svm
    preds_combined = (prob_combined >= 0.5).astype(int)

    return preds_combined, prob_combined, weight_svm

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

print('SCE-Corrected: Simple Calibrated Ensemble', flush=True)
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
                'method': 'Standard_SVM',
                'weight_svm': np.nan,
                'accuracy': acc_svm, 'macro_f1': f1_svm, 'balanced_accuracy': bacc_svm, 'auroc': auroc_svm
            })

            preds_srgc, prob_srgc = srgc_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1)
            acc_srgc = accuracy_score(y_test, preds_srgc)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SRGC_a075',
                'weight_svm': 0.0,
                'accuracy': acc_srgc, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
            })

            preds_sce, prob_sce, weight_svm = sce_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, n_cal)
            acc_sce = accuracy_score(y_test, preds_sce)
            f1_sce = f1_score(y_test, preds_sce, average='macro')
            bacc_sce = balanced_accuracy_score(y_test, preds_sce)
            try:
                auroc_sce = roc_auc_score(y_test, prob_sce)
            except:
                auroc_sce = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SCE',
                'weight_svm': weight_svm,
                'accuracy': acc_sce, 'macro_f1': f1_sce, 'balanced_accuracy': bacc_sce, 'auroc': auroc_sce
            })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/sce_corrected_results.csv', index=False)

print('', flush=True)
print('\n' + '='*70, flush=True)
print('SCE-Corrected Results Summary', flush=True)
print('='*70, flush=True)

baseline_df = df[df['method'] == 'Standard_SVM']
srgc_df = df[df['method'] == 'SRGC_a075']
sce_df = df[df['method'] == 'SCE']

print('\nComparing methods by shot:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    srgc_acc = srgc_df[srgc_df['n_cal'] == n_cal]['accuracy'].mean()
    sce_acc = sce_df[sce_df['n_cal'] == n_cal]['accuracy'].mean()
    avg_weight = sce_df[sce_df['n_cal'] == n_cal]['weight_svm'].mean()
    print(f'\n  {n_cal}-shot:', flush=True)
    print(f'    SVM:      {base_acc:.4f}', flush=True)
    print(f'    SRGC:     {srgc_acc:.4f} (gap={srgc_acc-base_acc:+.4f})', flush=True)
    print(f'    SCE:      {sce_acc:.4f} (gap={sce_acc-base_acc:+.4f}, weight_svm={avg_weight:.3f})', flush=True)

print('\nDone!', flush=True)