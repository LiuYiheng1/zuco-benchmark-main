"""SAN: Source-Anchored Normalization - Label-Free Version (No Leakage)"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.cluster import KMeans

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

def kmeans_sampling_label_free(X_pool, n_select):
    n_pool = len(X_pool)
    k = min(n_select, n_pool)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_pool)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    kmeans.fit(X_s)
    selected = []
    for c in range(k):
        idx = np.where(kmeans.labels_ == c)[0]
        if len(idx) > 0:
            centroid = kmeans.cluster_centers_[c]
            dists = np.linalg.norm(X_s[idx] - centroid, axis=1)
            selected.append(idx[np.argmin(dists)])
    while len(selected) < n_select and len(selected) < n_pool:
        for c in range(k):
            idx = np.where(kmeans.labels_ == c)[0]
            for i in idx:
                if i not in selected:
                    selected.append(i)
                    if len(selected) >= n_select:
                        break
            if len(selected) >= n_select:
                break
    return np.array(selected[:n_select])

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

def compute_global_stats(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0) + 1e-8
    return mu, sigma

def compute_class_stats(X, y, class_label):
    X_class = X[y == class_label]
    if len(X_class) == 0:
        return None, None
    mu = np.mean(X_class, axis=0)
    sigma = np.std(X_class, axis=0) + 1e-8
    return mu, sigma

def normalize_global(X, mu, sigma):
    return (X - mu) / sigma

def train_and_evaluate(X_cal, y_cal, X_test, y_test):
    if len(np.unique(y_cal)) < 2:
        return 0.0, 0.0, 0.0, 0.5
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5
    return acc, f1, bacc, auroc

print('SAN Label-Free Experiments', flush=True)
print('='*60, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    print(f'\nSeed {seed}:', flush=True)
    for held_out in Y_SUBJECTS:
        X_train_all, y_train_all = [], []
        for subj in Y_SUBJECTS:
            if subj == held_out:
                continue
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test_orig, y_test_orig = load_eeg_data(held_out)
        if len(X_train_all) == 0 or X_test_orig is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        mu_source_global, sigma_source_global = compute_global_stats(X_train_all)

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

        mu_target_global, sigma_target_global = compute_global_stats(X_cal_pool)

        print(f'  {held_out}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(cal_pool_indices):
                continue

            cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            acc, f1, bacc, auroc = train_and_evaluate(X_cal, y_cal, X_test, y_test)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'StandardScaler', 'type': 'baseline',
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

            X_cal_global_san = normalize_global(X_cal, mu_source_global, sigma_source_global)
            X_test_global_san = normalize_global(X_test, mu_source_global, sigma_source_global)
            acc, f1, bacc, auroc = train_and_evaluate(X_cal_global_san, y_cal, X_test_global_san, y_test)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SAN_global', 'type': 'main',
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

            X_cal_target_global = normalize_global(X_cal, mu_target_global, sigma_target_global)
            X_test_target_global = normalize_global(X_test, mu_target_global, sigma_target_global)
            acc, f1, bacc, auroc = train_and_evaluate(X_cal_target_global, y_cal, X_test_target_global, y_test)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'TargetNorm_global', 'type': 'comparison',
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

            cal_idx_accs = kmeans_sampling_label_free(X_cal_pool, n_cal * 2)
            X_cal_accs = X_cal_pool[cal_idx_accs]
            y_cal_accs = y_cal_pool[cal_idx_accs]
            if len(np.unique(y_cal_accs)) < 2:
                continue
            acc, f1, bacc, auroc = train_and_evaluate(X_cal_accs, y_cal_accs, X_test, y_test)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'ACCS', 'type': 'comparison',
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

            X_cal_accs_global_san = normalize_global(X_cal_accs, mu_source_global, sigma_source_global)
            X_test_global_san2 = normalize_global(X_test, mu_source_global, sigma_source_global)
            acc, f1, bacc, auroc = train_and_evaluate(X_cal_accs_global_san, y_cal_accs, X_test_global_san2, y_test)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SAN_global_ACCS', 'type': 'combination',
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

            mu_cal_0, sigma_cal_0 = compute_class_stats(X_cal_pool, y_cal_pool, 0)
            mu_cal_1, sigma_cal_1 = compute_class_stats(X_cal_pool, y_cal_pool, 1)
            if mu_cal_0 is not None and mu_cal_1 is not None:
                X_cal_labeled_norm = np.zeros_like(X_cal)
                for i in range(len(X_cal)):
                    if y_cal[i] == 0:
                        X_cal_labeled_norm[i] = (X_cal[i] - mu_cal_0) / sigma_cal_0
                    else:
                        X_cal_labeled_norm[i] = (X_cal[i] - mu_cal_1) / sigma_cal_1
                X_test_labeled_norm = np.zeros_like(X_test)
                for i in range(len(X_test)):
                    if y_test[i] == 0:
                        X_test_labeled_norm[i] = (X_test[i] - mu_cal_0) / sigma_cal_0
                    else:
                        X_test_labeled_norm[i] = (X_test[i] - mu_cal_1) / sigma_cal_1
                acc, f1, bacc, auroc = train_and_evaluate(X_cal_labeled_norm, y_cal, X_test_labeled_norm, y_test)
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': 'TargetNorm_labeled_calibration', 'type': 'comparison',
                    'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
                })

            mu_san_oracle_0, sigma_san_oracle_0 = compute_class_stats(X_train_all, y_train_all, 0)
            mu_san_oracle_1, sigma_san_oracle_1 = compute_class_stats(X_train_all, y_train_all, 1)
            if mu_san_oracle_0 is not None and mu_san_oracle_1 is not None:
                X_cal_oracle = np.zeros_like(X_cal)
                for i in range(len(X_cal)):
                    if y_cal[i] == 0:
                        X_cal_oracle[i] = (X_cal[i] - mu_san_oracle_0) / sigma_san_oracle_0
                    else:
                        X_cal_oracle[i] = (X_cal[i] - mu_san_oracle_1) / sigma_san_oracle_1
                X_test_oracle = np.zeros_like(X_test)
                for i in range(len(X_test)):
                    if y_test[i] == 0:
                        X_test_oracle[i] = (X_test[i] - mu_san_oracle_0) / sigma_san_oracle_0
                    else:
                        X_test_oracle[i] = (X_test[i] - mu_san_oracle_1) / sigma_san_oracle_1
                acc, f1, bacc, auroc = train_and_evaluate(X_cal_oracle, y_cal, X_test_oracle, y_test)
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': 'SAN_oracle_class_conditional', 'type': 'oracle',
                    'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
                })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/san_label_free_results.csv', index=False)

print('', flush=True)
print('\n' + '='*60, flush=True)
print('SAN Label-Free Results Summary', flush=True)
print('='*60, flush=True)

for n_cal in shot_settings:
    print(f'\n{n_cal}-shot per class:', flush=True)
    baseline = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    san_global = df[(df['method'] == 'SAN_global') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    target_global = df[(df['method'] == 'TargetNorm_global') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    accs = df[(df['method'] == 'ACCS') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    san_global_accs = df[(df['method'] == 'SAN_global_ACCS') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    target_labeled = df[(df['method'] == 'TargetNorm_labeled_calibration') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    san_oracle = df[(df['method'] == 'SAN_oracle_class_conditional') & (df['n_cal'] == n_cal)]['accuracy'].mean()

    print(f'  StandardScaler: {baseline:.4f}', flush=True)
    print(f'  SAN_global (MAIN): {san_global:.4f} (gap={san_global-baseline:+.4f})', flush=True)
    print(f'  TargetNorm_global: {target_global:.4f} (gap={target_global-baseline:+.4f})', flush=True)
    print(f'  ACCS: {accs:.4f} (gap={accs-baseline:+.4f})', flush=True)
    print(f'  SAN_global_ACCS: {san_global_accs:.4f} (gap={san_global_accs-accs:+.4f})', flush=True)
    print(f'  TargetNorm_labeled_cal: {target_labeled:.4f} (gap={target_labeled-baseline:+.4f})', flush=True)
    print(f'  SAN_oracle (ORACLE): {san_oracle:.4f} (gap={san_oracle-baseline:+.4f})', flush=True)

print('\nDone!', flush=True)