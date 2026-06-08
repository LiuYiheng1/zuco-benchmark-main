"""CRWC: Cross-Validated Feature-Reliability-Weighted Calibration

A truly innovative module that:
1. Estimates feature-level cross-subject reliability from source subjects
2. Applies feature-specific source/target mixing for calibration
3. Combines subject-invariant (reliable) and subject-specific (unreliable) features differently

Key insight: Different EEG features have different cross-subject invariance.
Some features (e.g., alpha power) are more subject-invariant.
Other features (e.g., high-frequency components) are more subject-specific.
CRWC weights the calibration accordingly.
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

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

def compute_feature_reliability_scores(X_sources, y_sources, n_sources_used=5):
    """Compute feature-level cross-subject reliability.

    Reliability = how consistent is the class-discriminative information across subjects.
    High reliability = feature is subject-invariant (good for cross-subject transfer).
    Low reliability = feature is subject-specific (needs target calibration).

    We compute:
    1. Within-subject class separation (how well each subject can discriminate classes)
    2. Across-subject consistency of class means
    """
    n_features = X_sources[0].shape[1]
    reliability_scores = np.zeros(n_features)

    subject_means_0 = []
    subject_means_1 = []

    for i, (X_subj, y_subj) in enumerate(zip(X_sources, y_sources)):
        X_0 = X_subj[y_subj == 0]
        X_1 = X_subj[y_subj == 1]
        if len(X_0) < 2 or len(X_1) < 2:
            continue
        subject_means_0.append(np.mean(X_0, axis=0))
        subject_means_1.append(np.mean(X_1, axis=0))

    if len(subject_means_0) < 2:
        return np.ones(n_features) / n_features

    subject_means_0 = np.array(subject_means_0)
    subject_means_1 = np.array(subject_means_1)

    mean_across_0 = np.mean(subject_means_0, axis=0)
    mean_across_1 = np.mean(subject_means_1, axis=0)
    std_across_0 = np.std(subject_means_0, axis=0) + 1e-8
    std_across_1 = np.std(subject_means_1, axis=0) + 1e-8

    class_separation = np.abs(mean_across_1 - mean_across_0)

    within_var_0 = np.mean([np.var(X_subj[y_subj == 0], axis=0) for X_subj, y_subj in zip(X_sources, y_sources) if len(y_subj[y_subj == 0]) > 1], axis=0)
    within_var_1 = np.mean([np.var(X_subj[y_subj == 1], axis=0) for X_subj, y_subj in zip(X_sources, y_sources) if len(y_subj[y_subj == 1]) > 1], axis=0)

    pooled_std = np.sqrt((within_var_0 + within_var_1) / 2 + 1e-8)

    reliability_scores = class_separation / (pooled_std + 1e-8)

    if np.max(reliability_scores) > 0:
        reliability_scores = reliability_scores / np.max(reliability_scores)

    return reliability_scores

def crwc_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1,
                  reliability_scores, alpha_reliable=0.9, alpha_unreliable=0.3, percentile_threshold=50):
    """CRWC prediction with feature-specific calibration mixing.

    For reliable features (high reliability): use stronger source prior (alpha_reliable)
    For unreliable features (low reliability): use stronger target calibration (alpha_unreliable)
    """
    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
    sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else sigma_source_0
    sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else sigma_source_1

    threshold = np.percentile(reliability_scores, percentile_threshold)
    feature_weights = np.where(reliability_scores >= threshold, alpha_reliable, alpha_unreliable)

    mu_blend_0 = feature_weights * mu_cal_0 + (1 - feature_weights) * mu_source_0
    mu_blend_1 = feature_weights * mu_cal_1 + (1 - feature_weights) * mu_source_1
    sigma_blend_0 = feature_weights * sigma_cal_0 + (1 - feature_weights) * sigma_source_0
    sigma_blend_1 = feature_weights * sigma_cal_1 + (1 - feature_weights) * sigma_source_1

    z_0 = (X_test - mu_blend_0) / (sigma_blend_0 + 1e-8)
    z_1 = (X_test - mu_blend_1) / (sigma_blend_1 + 1e-8)

    dist_0 = np.sqrt(np.sum(z_0 ** 2, axis=1))
    dist_1 = np.sqrt(np.sum(z_1 ** 2, axis=1))

    preds = (dist_1 < dist_0).astype(int)
    return preds

def crwc_svm_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1,
                      reliability_scores, alpha_reliable=0.9, alpha_unreliable=0.3, percentile_threshold=50):
    """CRWC with SVM on the reliability-weighted calibrated features."""
    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
    sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else sigma_source_0
    sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else sigma_source_1

    threshold = np.percentile(reliability_scores, percentile_threshold)
    feature_weights = np.where(reliability_scores >= threshold, alpha_reliable, alpha_unreliable)

    X_cal_norm = (X_cal - feature_weights * mu_cal_0 - (1 - feature_weights) * mu_source_0) / \
                 (feature_weights * sigma_cal_0 + (1 - feature_weights) * sigma_source_0 + 1e-8)
    X_test_norm = (X_test - feature_weights * mu_cal_0 - (1 - feature_weights) * mu_source_0) / \
                  (feature_weights * sigma_cal_0 + (1 - feature_weights) * sigma_source_0 + 1e-8)

    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_norm, y_cal)
    probs = clf.predict_proba(X_test_norm)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def baseline_svm(X_cal, y_cal, X_test):
    """Standard SVM baseline."""
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

print('CRWC: Cross-Validated Feature-Reliability-Weighted Calibration', flush=True)
print('='*70, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
alpha_reliable_values = [0.7, 0.85, 0.95]
alpha_unreliable_values = [0.1, 0.3, 0.5]
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

        mu_source_0 = np.mean(X_train_all[y_train_all == 0], axis=0) if np.any(y_train_all == 0) else np.mean(X_train_all, axis=0)
        mu_source_1 = np.mean(X_train_all[y_train_all == 1], axis=0) if np.any(y_train_all == 1) else np.mean(X_train_all, axis=0)
        sigma_source_0 = np.std(X_train_all[y_train_all == 0], axis=0) + 1e-8 if np.any(y_train_all == 0) else np.std(X_train_all, axis=0) + 1e-8
        sigma_source_1 = np.std(X_train_all[y_train_all == 1], axis=0) + 1e-8 if np.any(y_train_all == 1) else np.std(X_train_all, axis=0) + 1e-8

        reliability_scores = compute_feature_reliability_scores(X_sources, y_sources)

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
                'method': 'Standard_SVM', 'alpha_r': np.nan, 'alpha_u': np.nan,
                'accuracy': acc_base, 'macro_f1': f1_base, 'balanced_accuracy': bacc_base, 'auroc': auroc_base
            })

            preds_crwc, probs_crwc = crwc_svm_predict(
                X_cal, y_cal, X_test,
                mu_source_0, sigma_source_0, mu_source_1, sigma_source_1,
                reliability_scores,
                alpha_reliable=0.9, alpha_unreliable=0.3, percentile_threshold=50
            )
            acc_crwc = accuracy_score(y_test, preds_crwc)
            f1_crwc = f1_score(y_test, preds_crwc, average='macro')
            bacc_crwc = balanced_accuracy_score(y_test, preds_crwc)
            try:
                auroc_crwc = roc_auc_score(y_test, probs_crwc)
            except:
                auroc_crwc = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'CRWC_SVM', 'alpha_r': 0.9, 'alpha_u': 0.3,
                'accuracy': acc_crwc, 'macro_f1': f1_crwc, 'balanced_accuracy': bacc_crwc, 'auroc': auroc_crwc
            })

            for alpha_r in [0.7, 0.85, 0.95]:
                for alpha_u in [0.1, 0.3, 0.5]:
                    if alpha_r <= alpha_u:
                        continue
                    preds_exp, probs_exp = crwc_svm_predict(
                        X_cal, y_cal, X_test,
                        mu_source_0, sigma_source_0, mu_source_1, sigma_source_1,
                        reliability_scores,
                        alpha_reliable=alpha_r, alpha_unreliable=alpha_u, percentile_threshold=50
                    )
                    acc_exp = accuracy_score(y_test, preds_exp)
                    f1_exp = f1_score(y_test, preds_exp, average='macro')
                    bacc_exp = balanced_accuracy_score(y_test, preds_exp)
                    try:
                        auroc_exp = roc_auc_score(y_test, probs_exp)
                    except:
                        auroc_exp = 0.5

                    results.append({
                        'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                        'method': f'CRWC_a{alpha_r}_u{alpha_u}', 'alpha_r': alpha_r, 'alpha_u': alpha_u,
                        'accuracy': acc_exp, 'macro_f1': f1_exp, 'balanced_accuracy': bacc_exp, 'auroc': auroc_exp
                    })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/crwc_results.csv', index=False)

print('', flush=True)
print('\n' + '='*70, flush=True)
print('CRWC Results Summary', flush=True)
print('='*70, flush=True)

baseline_df = df[df['method'] == 'Standard_SVM']
crwc_df = df[df['method'] == 'CRWC_SVM']

print('\nComparing CRWC vs Standard SVM:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    crwc_acc = crwc_df[crwc_df['n_cal'] == n_cal]['accuracy'].mean()
    print(f'  {n_cal}-shot: SVM={base_acc:.4f}, CRWC={crwc_acc:.4f} (gap={crwc_acc-base_acc:+.4f})', flush=True)

print('\nComparing different alpha settings (CRWC):', flush=True)
for n_cal in shot_settings:
    print(f'  {n_cal}-shot:', flush=True)
    for alpha_r in [0.7, 0.85, 0.95]:
        for alpha_u in [0.1, 0.3, 0.5]:
            if alpha_r <= alpha_u:
                continue
            method = f'CRWC_a{alpha_r}_u{alpha_u}'
            acc = df[df['method'] == method][df['n_cal'] == n_cal]['accuracy'].mean()
            if not np.isnan(acc):
                print(f'    alpha_r={alpha_r}, alpha_u={alpha_u}: {acc:.4f}', flush=True)

print('\nDone!', flush=True)