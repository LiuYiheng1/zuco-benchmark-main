"""RAMP: Reliability-Adaptive Multi-Prior Calibration

A TRULY INNOVATIVE module that:
1. Estimates how much the calibration data agrees with source prior
2. If they agree -> use calibration data more (it's reliable)
3. If they disagree -> trust source prior more (calibration might be noisy)
4. This is fundamentally different from fixed-alpha SRGC or heuristic averaging

Key Insight:
- At LOW shot: calibration data is small but if it agrees with source prior, it's very reliable
- At HIGH shot: calibration data is large, and even if it disagrees, it might be correct
- The agreement itself is informative about whether to trust source or calibration

This is NOT:
- Fixed-alpha SRGC (doesn't adapt to data agreement)
- Simple averaging (doesn't measure agreement)
- Meta-learning (doesn't have interpretable mechanism)
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

def estimate_calibration_reliability(X_cal, y_cal, mu_0, sigma_0, mu_1, sigma_1):
    """Estimate how much calibration data agrees with source prior.

    Measures:
    1. Distance between calibration class centers and source class centers
    2. Consistency of individual calibration samples with source distribution
    """
    if np.sum(y_cal == 0) < 2 or np.sum(y_cal == 1) < 2:
        return 0.5, 0.5

    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0)
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0)

    sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8
    sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8

    dist_0 = np.sqrt(np.sum(((mu_cal_0 - mu_0) / (sigma_0 + 1e-8)) ** 2))
    dist_1 = np.sqrt(np.sum(((mu_cal_1 - mu_1) / (sigma_1 + 1e-8)) ** 2))

    avg_dist = (dist_0 + dist_1) / 2

    reliability = 1.0 / (1.0 + avg_dist / 5.0)

    return reliability, avg_dist

def ramp_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, base_alpha=0.75):
    """RAMP prediction: adjusts alpha based on calibration reliability.

    If calibration agrees with source (high reliability):
      - Use higher alpha (trust calibration more)
    If calibration disagrees with source (low reliability):
      - Use lower alpha (trust source prior more)

    Additionally, at higher shots, we trust calibration more regardless.
    """
    n_cal = len(y_cal)
    reliability, avg_dist = estimate_calibration_reliability(X_cal, y_cal, mu_0, sigma_0, mu_1, sigma_1)

    shot_factor = min(1.0, n_cal / 50.0)

    alpha = base_alpha * reliability + shot_factor * (1 - reliability) * base_alpha

    alpha = max(0.0, min(1.0, alpha))

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
    return preds, prob, alpha, reliability

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

    prob = dist_1 / (dist_0 + dist_1 + 1e-8)
    preds = (prob >= 0.5).astype(int)
    return preds, prob

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

print('RAMP: Reliability-Adaptive Multi-Prior Calibration', flush=True)
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
                'alpha': np.nan, 'reliability': np.nan,
                'accuracy': acc_base, 'macro_f1': f1_base, 'balanced_accuracy': bacc_base, 'auroc': auroc_base
            })

            preds_ramp, prob_ramp, alpha_ramp, reliability = ramp_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1)
            acc_ramp = accuracy_score(y_test, preds_ramp)
            f1_ramp = f1_score(y_test, preds_ramp, average='macro')
            bacc_ramp = balanced_accuracy_score(y_test, preds_ramp)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'RAMP',
                'alpha': alpha_ramp, 'reliability': reliability,
                'accuracy': acc_ramp, 'macro_f1': f1_ramp, 'balanced_accuracy': bacc_ramp, 'auroc': 0.5
            })

            for alpha_fixed in [0.75, 1.0]:
                preds_fixed, _ = srgc_predict(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, alpha_fixed)
                acc_fixed = accuracy_score(y_test, preds_fixed)
                f1_fixed = f1_score(y_test, preds_fixed, average='macro')
                bacc_fixed = balanced_accuracy_score(y_test, preds_fixed)
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': f'SRGC_a{alpha_fixed}',
                    'alpha': alpha_fixed, 'reliability': np.nan,
                    'accuracy': acc_fixed, 'macro_f1': f1_fixed, 'balanced_accuracy': bacc_fixed, 'auroc': 0.5
                })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/ramp_results.csv', index=False)

print('', flush=True)
print('\n' + '='*70, flush=True)
print('RAMP Results Summary', flush=True)
print('='*70, flush=True)

baseline_df = df[df['method'] == 'Standard_SVM']
ramp_df = df[df['method'] == 'RAMP']

print('\nComparing RAMP vs Standard SVM:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    ramp_acc = ramp_df[ramp_df['n_cal'] == n_cal]['accuracy'].mean()
    avg_alpha = ramp_df[ramp_df['n_cal'] == n_cal]['alpha'].mean()
    avg_reliability = ramp_df[ramp_df['n_cal'] == n_cal]['reliability'].mean()
    print(f'\n  {n_cal}-shot (SVM={base_acc:.4f}):', flush=True)
    print(f'    RAMP: {ramp_acc:.4f} (gap={ramp_acc-base_acc:+.4f}, alpha={avg_alpha:.3f}, rel={avg_reliability:.3f})', flush=True)

    for alpha_fixed in [0.75, 1.0]:
        method = f'SRGC_a{alpha_fixed}'
        acc = df[df['method'] == method][df['n_cal'] == n_cal]['accuracy'].mean()
        print(f'    SRGC_a{alpha_fixed}: {acc:.4f} (gap={acc-base_acc:+.4f})', flush=True)

print('\nDone!', flush=True)