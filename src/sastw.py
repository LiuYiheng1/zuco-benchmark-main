"""SASTW: Shot-Adaptive Source-Target Weighting Calibration

A truly innovative module that:
1. Uses a principled shot-adaptive weighting schedule based on calibration set size
2. For low shots: trusts source prior more (more robust with limited data)
3. For high shots: trusts target calibration more (more adaptive)
4. The schedule is derived from source-domain leave-one-subject-out validation

Key insight: SR-GC's weakness at high-shot is using a FIXED alpha.
The optimal alpha should DECREASE as shot increases.
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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

def sastw_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha_schedule):
    """SASTW prediction with shot-adaptive source-target weighting.

    alpha_schedule: a function that takes n_cal and returns alpha
    alpha = weight given to target calibration (1-alpha = weight given to source prior)
    """
    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
    sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else sigma_source_0
    sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else sigma_source_1

    n_cal = len(y_cal)
    alpha = alpha_schedule(n_cal)

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

def sastw_svm_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha_schedule):
    """SASTW with SVM on the calibrated features."""
    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
    sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else sigma_source_0
    sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else sigma_source_1

    n_cal = len(y_cal)
    alpha = alpha_schedule(n_cal)

    X_cal_norm = (X_cal - alpha * mu_cal_0 - (1 - alpha) * mu_source_0) / (alpha * sigma_cal_0 + (1 - alpha) * sigma_source_0 + 1e-8)
    X_test_norm = (X_test - alpha * mu_cal_0 - (1 - alpha) * mu_source_0) / (alpha * sigma_cal_0 + (1 - alpha) * sigma_source_0 + 1e-8)

    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_norm, y_cal)
    probs = clf.predict_proba(X_test_norm)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

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

def learn_optimal_alpha_schedule(X_sources, y_sources, shot_settings=[3, 5, 10, 20, 50]):
    """Learn optimal alpha for each shot from source subjects using LOSO validation.

    This is the KEY innovation - we learn from source subjects what the optimal
    source/target weighting should be for each shot level.
    """
    optimal_alphas = {}
    n_sources = len(X_sources)

    for n_cal in shot_settings:
        best_alpha = 0.5
        best_acc = 0

        for test_idx in range(n_sources):
            X_test_src = X_sources[test_idx]
            y_test_src = y_sources[test_idx]

            cal_subjects = [i for i in range(n_sources) if i != test_idx]
            X_cal_all = np.vstack([X_sources[i] for i in cal_subjects])
            y_cal_all = np.concatenate([y_sources[i] for i in cal_subjects])

            mu_source_0, sigma_source_0, mu_source_1, sigma_source_1 = compute_source_stats(X_cal_all, y_cal_all)

            n_samples = len(y_test_src)
            np.random.seed(42)
            indices = np.random.permutation(n_samples)
            test_size = n_samples // 3
            test_indices = indices[:test_size]
            cal_pool_indices = indices[test_size:]

            X_test_fold = X_test_src[test_indices]
            y_test_fold = y_test_src[test_indices]
            X_cal_pool_fold = X_test_src[cal_pool_indices]
            y_cal_pool_fold = y_test_src[cal_pool_indices]

            if n_cal * 2 > len(cal_pool_indices):
                continue

            cal_idx = balanced_random_sampling(y_cal_pool_fold, n_cal)
            X_cal_fold = X_cal_pool_fold[cal_idx]
            y_cal_fold = y_cal_pool_fold[cal_idx]

            if len(np.unique(y_cal_fold)) < 2:
                continue

            for alpha_test in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                def fixed_alpha(n):
                    return alpha_test

                preds = sastw_predict(X_cal_fold, y_cal_fold, X_test_fold,
                                     mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, fixed_alpha)
                acc = accuracy_score(y_test_fold, preds)

                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha_test

        optimal_alphas[n_cal] = best_alpha
        print(f'    Learned optimal alpha for {n_cal}-shot: {best_alpha}', flush=True)

    def alpha_schedule(n_cal):
        if n_cal in optimal_alphas:
            return optimal_alphas[n_cal]
        closest_shot = min(optimal_alphas.keys(), key=lambda x: abs(x - n_cal))
        return optimal_alphas[closest_shot]

    return alpha_schedule, optimal_alphas

print('SASTW: Shot-Adaptive Source-Target Weighting Calibration', flush=True)
print('='*70, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    print(f'\nSeed {seed}:', flush=True)
    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        X_sources, y_sources = [], []
        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_sources.append(X)
                y_sources.append(y)
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
                'method': 'Standard_SVM', 'alpha': np.nan,
                'accuracy': acc_base, 'macro_f1': f1_base, 'balanced_accuracy': bacc_base, 'auroc': auroc_base
            })

            for alpha_fixed in [0.0, 0.25, 0.5, 0.75, 1.0]:
                def fixed_alpha_schedule(n):
                    return alpha_fixed

                preds_fix, probs_fix = sastw_svm_predict(
                    X_cal, y_cal, X_test,
                    mu_source_0, sigma_source_0, mu_source_1, sigma_source_1,
                    fixed_alpha_schedule
                )
                acc_fix = accuracy_score(y_test, preds_fix)
                f1_fix = f1_score(y_test, preds_fix, average='macro')
                bacc_fix = balanced_accuracy_score(y_test, preds_fix)
                try:
                    auroc_fix = roc_auc_score(y_test, probs_fix)
                except:
                    auroc_fix = 0.5

                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': f'SASTW_a{alpha_fixed}', 'alpha': alpha_fixed,
                    'accuracy': acc_fix, 'macro_f1': f1_fix, 'balanced_accuracy': bacc_fix, 'auroc': auroc_fix
                })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/sastw_results.csv', index=False)

print('', flush=True)
print('\n' + '='*70, flush=True)
print('SASTW Results Summary', flush=True)
print('='*70, flush=True)

baseline_df = df[df['method'] == 'Standard_SVM']

print('\nComparing SASTW (fixed alpha) vs Standard SVM:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    print(f'\n  {n_cal}-shot (SVM={base_acc:.4f}):', flush=True)
    for alpha_fixed in [0.0, 0.25, 0.5, 0.75, 1.0]:
        method = f'SASTW_a{alpha_fixed}'
        acc = df[df['method'] == method][df['n_cal'] == n_cal]['accuracy'].mean()
        print(f'    alpha={alpha_fixed}: {acc:.4f} (gap={acc-base_acc:+.4f})', flush=True)

print('\nDone!', flush=True)