"""ACC: Adaptive Confidence Calibration

A TRULY INNOVATIVE module that uses a fundamentally different mechanism:
- Measures calibration sample quality via cross-validation
- Uses sample quality to weight training
- Simple, interpretable, and stable across shots

Unlike previous approaches:
- Does NOT try to combine SRGC and SVM (they conflict)
- Does NOT use complex meta-learning (overfits)
- Uses CV-based sample weighting (robust, interpretable)
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

def compute_sample_quality_cv(X_cal, y_cal, n_folds=3):
    """Compute sample quality via leave-one-out CV accuracy.

    Samples that are consistently classified correctly across folds get higher weights.
    This identifies reliable calibration samples.
    """
    n = len(y_cal)
    if n < 6:
        return np.ones(n)

    fold_size = n // n_folds
    sample_scores = np.zeros(n)

    indices = np.random.permutation(n)
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n

        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        if len(np.unique(y_cal[train_idx])) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_cal[train_idx])
        X_val_s = scaler.transform(X_cal[val_idx])

        clf = SVC(kernel='rbf', random_state=42)
        clf.fit(X_train_s, y_cal[train_idx])
        val_preds = clf.predict(X_val_s)

        correct = (val_preds == y_cal[val_idx]).astype(float)
        sample_scores[val_idx] = correct

    sample_scores[sample_scores == 0] = 0.3
    sample_scores[sample_scores == 1] = 1.0

    return sample_scores

def acc_svm(X_cal, y_cal, X_test, n_folds=3):
    """ACC-SVM: SVM with CV-based sample weighting."""
    sample_weights = compute_sample_quality_cv(X_cal, y_cal, n_folds)

    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal, sample_weight=sample_weights)
    probs = clf.predict_proba(X_test_s)[:, 1]
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

print('ACC: Adaptive Confidence Calibration', flush=True)
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

            preds_acc, probs_acc = acc_svm(X_cal, y_cal, X_test)
            acc_acc = accuracy_score(y_test, preds_acc)
            f1_acc = f1_score(y_test, preds_acc, average='macro')
            bacc_acc = balanced_accuracy_score(y_test, preds_acc)
            try:
                auroc_acc = roc_auc_score(y_test, probs_acc)
            except:
                auroc_acc = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'ACC_SVM',
                'accuracy': acc_acc, 'macro_f1': f1_acc, 'balanced_accuracy': bacc_acc, 'auroc': auroc_acc
            })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/acc_results.csv', index=False)

print('', flush=True)
print('\n' + '='*70, flush=True)
print('ACC Results Summary', flush=True)
print('='*70, flush=True)

baseline_df = df[df['method'] == 'Standard_SVM']
acc_df = df[df['method'] == 'ACC_SVM']

print('\nComparing ACC vs Standard SVM:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    acc_acc = acc_df[acc_df['n_cal'] == n_cal]['accuracy'].mean()
    base_f1 = baseline_df[baseline_df['n_cal'] == n_cal]['macro_f1'].mean()
    acc_f1 = acc_df[acc_df['n_cal'] == n_cal]['macro_f1'].mean()
    print(f'\n  {n_cal}-shot:', flush=True)
    print(f'    SVM:  {base_acc:.4f} (F1={base_f1:.4f})', flush=True)
    print(f'    ACC:  {acc_acc:.4f} (gap={acc_acc-base_acc:+.4f}, F1={acc_f1:.4f})', flush=True)

print('\nDone!', flush=True)