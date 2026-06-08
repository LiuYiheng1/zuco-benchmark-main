"""Pseudo-label SAN - Using predicted labels for class-conditional normalization"""
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

def normalize_with_pseudo_labels(X, y_pseudo, mu_0, sigma_0, mu_1, sigma_1):
    X_norm = np.zeros_like(X)
    mask_0 = (y_pseudo == 0)
    mask_1 = (y_pseudo == 1)
    X_norm[mask_0] = (X[mask_0] - mu_0) / sigma_0
    X_norm[mask_1] = (X[mask_1] - mu_1) / sigma_1
    return X_norm

def train_and_evaluate(X_cal, y_cal, X_test, y_test):
    if len(np.unique(y_cal)) < 2:
        return 0.0, 0.0, 0.0, 0.5, None, None
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
    pseudo_labels = clf.predict(X_test_s)
    pseudo_acc = accuracy_score(y_test, pseudo_labels)
    return acc, f1, bacc, auroc, pseudo_labels, pseudo_acc

print('Pseudo-label SAN Experiments', flush=True)
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

        mu_source_0, sigma_source_0 = compute_class_stats(X_train_all, y_train_all, 0)
        mu_source_1, sigma_source_1 = compute_class_stats(X_train_all, y_train_all, 1)

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

            acc, f1, bacc, auroc, _, _ = train_and_evaluate(X_cal, y_cal, X_test, y_test)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'StandardScaler', 'pseudo_acc': np.nan,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

            X_cal_norm0 = (X_cal - mu_source_0) / sigma_source_0
            X_test_norm0 = (X_test - mu_source_0) / sigma_source_0
            X_cal_norm1 = (X_cal - mu_source_1) / sigma_source_1
            X_test_norm1 = (X_test - mu_source_1) / sigma_source_1

            X_cal_stacked = np.vstack([X_cal_norm0, X_cal_norm1])
            X_test_stacked = np.vstack([X_test_norm0, X_test_norm1])
            y_cal_stacked = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_cal))])
            y_test_pseudo_for_oracle = np.concatenate([np.zeros(len(X_test)), np.ones(len(X_test))])

            acc, f1, bacc, auroc, _, _ = train_and_evaluate(X_cal_stacked, y_cal_stacked, X_test_stacked, y_test_pseudo_for_oracle)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SAN_oracle_stacked', 'pseudo_acc': np.nan,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

            cal_idx_accs = kmeans_sampling_label_free(X_cal_pool, n_cal * 2)
            X_cal_accs = X_cal_pool[cal_idx_accs]
            y_cal_accs = y_cal_pool[cal_idx_accs]
            if len(np.unique(y_cal_accs)) < 2:
                continue
            acc, f1, bacc, auroc, pseudo_labels, pseudo_acc = train_and_evaluate(X_cal_accs, y_cal_accs, X_test, y_test)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'ACCS', 'pseudo_acc': pseudo_acc,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

            X_cal_pseudo = normalize_with_pseudo_labels(X_cal_accs, y_cal_accs, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)
            X_test_pseudo = normalize_with_pseudo_labels(X_test, pseudo_labels, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)
            acc, f1, bacc, auroc, _, _ = train_and_evaluate(X_cal_pseudo, y_cal_accs, X_test_pseudo, y_test)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SAN_pseudo_label', 'pseudo_acc': pseudo_acc,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/san_pseudo_label_results.csv', index=False)

print('', flush=True)
print('\n' + '='*60, flush=True)
print('Pseudo-label SAN Results Summary', flush=True)
print('='*60, flush=True)

for n_cal in shot_settings:
    print(f'\n{n_cal}-shot per class:', flush=True)
    baseline = df[(df['method'] == 'StandardScaler') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    accs = df[(df['method'] == 'ACCS') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    pseudo_acc_mean = df[(df['method'] == 'ACCS') & (df['n_cal'] == n_cal)]['pseudo_acc'].mean()
    san_pseudo = df[(df['method'] == 'SAN_pseudo_label') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    oracle = df[(df['method'] == 'SAN_oracle_stacked') & (df['n_cal'] == n_cal)]['accuracy'].mean()

    print(f'  StandardScaler: {baseline:.4f}', flush=True)
    print(f'  ACCS: {accs:.4f} (gap={accs-baseline:+.4f}), pseudo_acc={pseudo_acc_mean:.4f}', flush=True)
    print(f'  SAN_pseudo_label: {san_pseudo:.4f} (gap={san_pseudo-baseline:+.4f})', flush=True)
    print(f'  SAN_oracle_stacked: {oracle:.4f} (gap={oracle-baseline:+.4f})', flush=True)

print('\nDone!', flush=True)