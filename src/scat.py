"""SCAT: Source-Calibrated Adaptive Transfer

A TRULY INNOVATIVE module that learns the optimal source-target weighting
from source domain meta-validation.

Key Insight from SASTW:
- Optimal alpha varies by shot: low-shot prefers alpha=1.0 (target-only),
  high-shot prefers alpha=0.75 (balanced)
- We can learn this mapping from SOURCE subjects using meta-validation
- Then apply the learned schedule to TARGET subjects

This is fundamentally different from:
- Fixed-alpha SRGC (ignores shot adaptation)
- Heuristic hybrid (not learned from data)
- Meta-SVM (too complex, overfits at mid-shot)

SCAT learns: "At shot k, what alpha gives best cross-subject transfer?"
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from scipy.interpolate import interp1d

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

def srgc_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha):
    """SR-GC prediction with given alpha."""
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
    return preds

def baseline_svm(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def learn_alpha_schedule_from_sources(X_sources, y_sources, shot_settings):
    """Learn optimal alpha for each shot from source subjects.

    Uses leave-one-source-subject-out validation.
    """
    optimal_alphas = {}
    n_sources = len(X_sources)

    for n_cal in shot_settings:
        best_alpha = 0.5
        best_acc = 0
        all_results = []

        for test_idx in range(n_sources):
            X_test_src = X_sources[test_idx]
            y_test_src = y_sources[test_idx]

            cal_subjects = [i for i in range(n_sources) if i != test_idx]
            X_cal_all = np.vstack([X_sources[i] for i in cal_subjects])
            y_cal_all = np.concatenate([y_sources[i] for i in cal_subjects])

            mu_0, sigma_0, mu_1, sigma_1 = compute_source_stats(X_cal_all, y_cal_all)

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

            class_0_idx = np.where(y_cal_pool_fold == 0)[0]
            class_1_idx = np.where(y_cal_pool_fold == 1)[0]
            np.random.shuffle(class_0_idx)
            np.random.shuffle(class_1_idx)
            n0 = min(n_cal, len(class_0_idx))
            n1 = min(n_cal, len(class_1_idx))
            cal_idx = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])

            X_cal_fold = X_cal_pool_fold[cal_idx]
            y_cal_fold = y_cal_pool_fold[cal_idx]

            if len(np.unique(y_cal_fold)) < 2:
                continue

            for alpha_test in [0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
                preds = srgc_predict(X_cal_fold, y_cal_fold, X_test_fold,
                                    mu_0, sigma_0, mu_1, sigma_1, alpha_test)
                acc = accuracy_score(y_test_fold, preds)
                all_results.append((alpha_test, acc, test_idx))

        if all_results:
            alpha_results = {}
            for alpha_test, acc, _ in all_results:
                if alpha_test not in alpha_results:
                    alpha_results[alpha_test] = []
                alpha_results[alpha_test].append(acc)

            best_alpha = max(alpha_results.keys(), key=lambda a: np.mean(alpha_results[a]))
            optimal_alphas[n_cal] = best_alpha

    def alpha_schedule(n_cal):
        if n_cal in optimal_alphas:
            return optimal_alphas[n_cal]
        closest_shot = min(optimal_alphas.keys(), key=lambda x: abs(x - n_cal))
        return optimal_alphas[closest_shot]

    return alpha_schedule, optimal_alphas

def scat_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha_schedule):
    """SCAT prediction using learned alpha schedule."""
    n_cal = len(y_cal)
    alpha = alpha_schedule(n_cal)
    preds = srgc_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha)
    return preds, alpha

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

print('SCAT: Source-Calibrated Adaptive Transfer', flush=True)
print('='*70, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

all_optimal_alphas = {}

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

        mu_0, sigma_0, mu_1, sigma_1 = compute_source_stats(X_train_all, y_train_all)

        if seed == 0:
            alpha_schedule, optimal_alphas = learn_alpha_schedule_from_sources(X_sources, y_sources, shot_settings)
            all_optimal_alphas[held_out] = optimal_alphas
            print(f'  Learned alphas for {held_out}: {optimal_alphas}', flush=True)

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
                'alpha_used': np.nan,
                'accuracy': acc_base, 'macro_f1': f1_base, 'balanced_accuracy': bacc_base, 'auroc': auroc_base
            })

            preds_scat, alpha_used = scat_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha_schedule)
            acc_scat = accuracy_score(y_test, preds_scat)
            f1_scat = f1_score(y_test, preds_scat, average='macro')
            bacc_scat = balanced_accuracy_score(y_test, preds_scat)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SCAT',
                'alpha_used': alpha_used,
                'accuracy': acc_scat, 'macro_f1': f1_scat, 'balanced_accuracy': bacc_scat, 'auroc': 0.5
            })

            for alpha_fixed in [0.0, 0.5, 0.75, 1.0]:
                preds_fixed = srgc_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha_fixed)
                acc_fixed = accuracy_score(y_test, preds_fixed)
                f1_fixed = f1_score(y_test, preds_fixed, average='macro')
                bacc_fixed = balanced_accuracy_score(y_test, preds_fixed)
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': f'SRGC_a{alpha_fixed}',
                    'alpha_used': alpha_fixed,
                    'accuracy': acc_fixed, 'macro_f1': f1_fixed, 'balanced_accuracy': bacc_fixed, 'auroc': 0.5
                })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/scat_results.csv', index=False)

print('', flush=True)
print('\n' + '='*70, flush=True)
print('SCAT Results Summary', flush=True)
print('='*70, flush=True)

print('\nLearned optimal alphas per subject:', flush=True)
for subj, alphas in all_optimal_alphas.items():
    print(f'  {subj}: {alphas}', flush=True)

baseline_df = df[df['method'] == 'Standard_SVM']
scat_df = df[df['method'] == 'SCAT']

print('\nComparing SCAT vs Standard SVM:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    scat_acc = scat_df[scat_df['n_cal'] == n_cal]['accuracy'].mean()
    scat_alphas = scat_df[scat_df['n_cal'] == n_cal]['alpha_used'].unique()
    print(f'\n  {n_cal}-shot (SVM={base_acc:.4f}):', flush=True)
    print(f'    SCAT: {scat_acc:.4f} (gap={scat_acc-base_acc:+.4f}, alphas={scat_alphas})', flush=True)

    for alpha_fixed in [0.0, 0.5, 0.75, 1.0]:
        method = f'SRGC_a{alpha_fixed}'
        acc = df[df['method'] == method][df['n_cal'] == n_cal]['accuracy'].mean()
        print(f'    SRGC_a{alpha_fixed}: {acc:.4f} (gap={acc-base_acc:+.4f})', flush=True)

print('\nDone!', flush=True)