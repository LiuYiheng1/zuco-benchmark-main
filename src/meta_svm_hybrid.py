"""Meta-SVM Hybrid: Learning to Combine SRGC Prior with Discriminative Power

A TRULY innovative module that:
1. Uses SRGC Mahalanobis distances as meta-features
2. Trains SVM on raw features for comparison
3. Uses a meta-learner to adaptively weight between them based on sample availability

The key insight from D-SRGC experiments:
- D_SRGC_Meta (using only SRGC meta features) works well at low-shot
- Standard SVM works well at high-shot
- Neither works well across ALL shots

We need a SHOT-ADAPTIVE meta-classifier that:
- Learns from source subjects what combination is optimal at each shot level
- Uses this to adapt on the target subject
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

def compute_srgc_meta_features(X, mu_0, sigma_0, mu_1, sigma_1):
    """Compute SRGC meta features: Mahalanobis distances and related statistics."""
    sigma_0_inv = 1.0 / (sigma_0 + 1e-8)
    sigma_1_inv = 1.0 / (sigma_1 + 1e-8)

    dist_0 = np.sqrt(np.sum(((X - mu_0) * sigma_0_inv) ** 2, axis=1))
    dist_1 = np.sqrt(np.sum(((X - mu_1) * sigma_1_inv) ** 2, axis=1))

    diff = dist_0 - dist_1
    ratio = (dist_0 + 1e-8) / (dist_1 + 1e-8)
    log_ratio = np.log(ratio + 1e-8)
    sum_dist = dist_0 + dist_1

    meta_features = np.column_stack([dist_0, dist_1, diff, ratio, log_ratio, sum_dist])
    return meta_features

def baseline_svm(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def meta_svm_hybrid(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, shot):
    """Meta-SVM Hybrid: Adaptively combines SRGC meta features with raw features.

    Strategy:
    - For LOW shot: trust SRGC meta features more
    - For HIGH shot: trust SVM on raw features more

    The combination is done via feature concatenation and SVM training.
    """
    meta_cal = compute_srgc_meta_features(X_cal, mu_0, sigma_0, mu_1, sigma_1)
    meta_test = compute_srgc_meta_features(X_test, mu_0, sigma_0, mu_1, sigma_1)

    if shot <= 5:
        alpha = 0.7
    elif shot <= 10:
        alpha = 0.5
    elif shot <= 20:
        alpha = 0.3
    else:
        alpha = 0.0

    scaler_raw = StandardScaler()
    X_cal_raw_s = scaler_raw.fit_transform(X_cal)
    X_test_raw_s = scaler_raw.transform(X_test)

    if alpha > 0:
        scaler_meta = StandardScaler()
        meta_cal_s = scaler_meta.fit_transform(meta_cal)
        meta_test_s = scaler_meta.transform(meta_test)

        X_cal_combined = np.hstack([alpha * meta_cal_s, (1 - alpha) * X_cal_raw_s])
        X_test_combined = np.hstack([alpha * meta_test_s, (1 - alpha) * X_test_raw_s])
    else:
        X_cal_combined = X_cal_raw_s
        X_test_combined = X_test_raw_s

    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_combined, y_cal)
    probs = clf.predict_proba(X_test_combined)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def meta_svm_learned(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1):
    """Meta-SVM with learned combination weights using Logistic Regression meta-learner.

    This is the TRULY innovative part: we use a meta-learner to find the optimal
    combination of SRGC meta features and raw SVM probabilities.
    """
    meta_cal = compute_srgc_meta_features(X_cal, mu_0, sigma_0, mu_1, sigma_1)
    meta_test = compute_srgc_meta_features(X_test, mu_0, sigma_0, mu_1, sigma_1)

    scaler_meta = StandardScaler()
    meta_cal_s = scaler_meta.fit_transform(meta_cal)
    meta_test_s = scaler_meta.transform(meta_test)

    scaler_raw = StandardScaler()
    X_cal_raw_s = scaler_raw.fit_transform(X_cal)
    X_test_raw_s = scaler_raw.transform(X_test)

    clf_raw = SVC(kernel='rbf', probability=True, random_state=42)
    clf_raw.fit(X_cal_raw_s, y_cal)
    probs_raw = clf_raw.predict_proba(X_test_raw_s)[:, 1]

    clf_meta = SVC(kernel='rbf', probability=True, random_state=42)
    clf_meta.fit(meta_cal_s, y_cal)
    probs_meta = clf_meta.predict_proba(meta_test_s)[:, 1]

    meta_features = np.column_stack([probs_raw, probs_meta, probs_raw * probs_meta,
                                       np.abs(probs_raw - probs_meta),
                                       (probs_raw + probs_meta) / 2])

    meta_clf = LogisticRegression(random_state=42, max_iter=1000)
    meta_clf.fit(meta_features, y_cal)
    final_probs = meta_clf.predict_proba(meta_features)[:, 1]
    final_preds = (final_probs >= 0.5).astype(int)

    out_of_sample_probs = meta_clf.predict_proba(meta_features)[:, 1]
    out_of_sample_preds = (out_of_sample_probs >= 0.5).astype(int)

    return out_of_sample_preds, out_of_sample_probs

def meta_svm_cv(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, n_folds=5):
    """Meta-SVM with proper cross-validation within calibration set.

    This version uses CV within calibration to avoid overfitting the meta-learner.
    """
    meta_cal = compute_srgc_meta_features(X_cal, mu_0, sigma_0, mu_1, sigma_1)
    meta_test = compute_srgc_meta_features(X_test, mu_0, sigma_0, mu_1, sigma_1)

    scaler_meta = StandardScaler()
    meta_cal_s = scaler_meta.fit_transform(meta_cal)
    meta_test_s = scaler_meta.transform(meta_test)

    scaler_raw = StandardScaler()
    X_cal_raw_s = scaler_raw.fit_transform(X_cal)
    X_test_raw_s = scaler_raw.transform(X_test)

    n_cal = len(y_cal)
    fold_size = n_cal // n_folds

    oof_probs_raw = np.zeros(n_cal)
    oof_probs_meta = np.zeros(n_cal)

    indices = np.random.permutation(n_cal)
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_cal

        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        clf_raw = SVC(kernel='rbf', probability=True, random_state=42)
        clf_raw.fit(X_cal_raw_s[train_idx], y_cal[train_idx])
        oof_probs_raw[val_idx] = clf_raw.predict_proba(X_cal_raw_s[val_idx])[:, 1]

        clf_meta = SVC(kernel='rbf', probability=True, random_state=42)
        clf_meta.fit(meta_cal_s[train_idx], y_cal[train_idx])
        oof_probs_meta[val_idx] = clf_meta.predict_proba(meta_cal_s[val_idx])[:, 1]

    meta_features_train = np.column_stack([
        oof_probs_raw, oof_probs_meta,
        oof_probs_raw * oof_probs_meta,
        np.abs(oof_probs_raw - oof_probs_meta),
        (oof_probs_raw + oof_probs_meta) / 2
    ])

    probs_raw_test = clf_raw.fit(X_cal_raw_s, y_cal).predict_proba(X_test_raw_s)[:, 1]
    probs_meta_test = clf_meta.fit(meta_cal_s, y_cal).predict_proba(meta_test_s)[:, 1]

    meta_features_test = np.column_stack([
        probs_raw_test, probs_meta_test,
        probs_raw_test * probs_meta_test,
        np.abs(probs_raw_test - probs_meta_test),
        (probs_raw_test + probs_meta_test) / 2
    ])

    meta_clf = LogisticRegression(random_state=42, max_iter=1000)
    meta_clf.fit(meta_features_train, y_cal)
    final_probs = meta_clf.predict_proba(meta_features_test)[:, 1]
    final_preds = (final_probs >= 0.5).astype(int)

    return final_preds, final_probs

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

print('Meta-SVM Hybrid: Learning to Combine SRGC Prior with Discriminative Power', flush=True)
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

            preds_hybrid, probs_hybrid = meta_svm_hybrid(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1, n_cal)
            acc_hybrid = accuracy_score(y_test, preds_hybrid)
            f1_hybrid = f1_score(y_test, preds_hybrid, average='macro')
            bacc_hybrid = balanced_accuracy_score(y_test, preds_hybrid)
            try:
                auroc_hybrid = roc_auc_score(y_test, probs_hybrid)
            except:
                auroc_hybrid = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'Meta_SVM_Hybrid',
                'accuracy': acc_hybrid, 'macro_f1': f1_hybrid, 'balanced_accuracy': bacc_hybrid, 'auroc': auroc_hybrid
            })

            try:
                preds_cv, probs_cv = meta_svm_cv(X_cal, y_cal, X_test, mu_0, sigma_0, mu_1, sigma_1)
                acc_cv = accuracy_score(y_test, preds_cv)
                f1_cv = f1_score(y_test, preds_cv, average='macro')
                bacc_cv = balanced_accuracy_score(y_test, preds_cv)
                try:
                    auroc_cv = roc_auc_score(y_test, probs_cv)
                except:
                    auroc_cv = 0.5
            except:
                acc_cv, f1_cv, bacc_cv, auroc_cv = 0, 0, 0, 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'Meta_SVM_CV',
                'accuracy': acc_cv, 'macro_f1': f1_cv, 'balanced_accuracy': bacc_cv, 'auroc': auroc_cv
            })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/meta_svm_hybrid_results.csv', index=False)

print('', flush=True)
print('\n' + '='*80, flush=True)
print('Meta-SVM Hybrid Results Summary', flush=True)
print('='*80, flush=True)

baseline_df = df[df['method'] == 'Standard_SVM']
hybrid_df = df[df['method'] == 'Meta_SVM_Hybrid']
cv_df = df[df['method'] == 'Meta_SVM_CV']

print('\nComparing methods by shot:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    hybrid_acc = hybrid_df[hybrid_df['n_cal'] == n_cal]['accuracy'].mean()
    cv_acc = cv_df[cv_df['n_cal'] == n_cal]['accuracy'].mean()
    print(f'\n  {n_cal}-shot (SVM={base_acc:.4f}):', flush=True)
    print(f'    Meta_SVM_Hybrid: {hybrid_acc:.4f} (gap={hybrid_acc-base_acc:+.4f})', flush=True)
    print(f'    Meta_SVM_CV:    {cv_acc:.4f} (gap={cv_acc-base_acc:+.4f})', flush=True)

print('\nDone!', flush=True)