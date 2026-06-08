"""SR-GC: Source-Regularized Gaussian Calibration (Fast Version)"""
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

def compute_class_stats_diagonal(X, y, class_label):
    X_class = X[y == class_label]
    if len(X_class) == 0:
        return None, None
    mu = np.mean(X_class, axis=0)
    sigma = np.std(X_class, axis=0) + 1e-8
    return mu, sigma

def srgc_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.5, beta=0.5):
    mu_cal_0, sigma_cal_0 = compute_class_stats_diagonal(X_cal, y_cal, 0)
    mu_cal_1, sigma_cal_1 = compute_class_stats_diagonal(X_cal, y_cal, 1)

    if mu_cal_0 is None or mu_cal_1 is None:
        return np.zeros(len(X_test))

    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_source_0
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_source_1
    sigma_blend_0 = alpha * sigma_cal_0 + (1 - alpha) * sigma_source_0
    sigma_blend_1 = alpha * sigma_cal_1 + (1 - alpha) * sigma_source_1

    z_0 = (X_test - mu_blend_0) / sigma_blend_0
    z_1 = (X_test - mu_blend_1) / sigma_blend_1
    dist_0 = np.sqrt(np.sum(z_0 ** 2, axis=1))
    dist_1 = np.sqrt(np.sum(z_1 ** 2, axis=1))

    preds = (dist_1 < dist_0).astype(int)
    return preds

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

print('SR-GC Experiments (Fast)', flush=True)
print('='*60, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
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

        mu_source_0, sigma_source_0 = compute_class_stats_diagonal(X_train_all, y_train_all, 0)
        mu_source_1, sigma_source_1 = compute_class_stats_diagonal(X_train_all, y_train_all, 1)

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

            acc, f1, bacc, auroc = train_and_evaluate(X_cal, y_cal, X_test, y_test)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'EEG_SVM', 'alpha': np.nan, 'beta': np.nan,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

            preds = srgc_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.0)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            bacc = balanced_accuracy_score(y_test, preds)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SR-GC_source_only', 'alpha': 0.0, 'beta': 0.0,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': 0.5
            })

            for alpha in [0.25, 0.5, 0.75]:
                preds = srgc_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=alpha)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='macro')
                bacc = balanced_accuracy_score(y_test, preds)
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': f'SR-GC_a{alpha}', 'alpha': alpha, 'beta': np.nan,
                    'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': 0.5
                })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/srgc_results.csv', index=False)

print('', flush=True)
print('\n' + '='*60, flush=True)
print('SR-GC Results Summary', flush=True)
print('='*60, flush=True)

for n_cal in shot_settings:
    print(f'\n{n_cal}-shot per class:', flush=True)
    baseline = df[(df['method'] == 'EEG_SVM') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    source_only = df[(df['method'] == 'SR-GC_source_only') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    print(f'  EEG_SVM: {baseline:.4f}', flush=True)
    print(f'  SR-GC_source_only: {source_only:.4f} (gap={source_only-baseline:+.4f})', flush=True)

    for alpha in [0.25, 0.5, 0.75]:
        method = f'SR-GC_a{alpha}'
        acc = df[(df['method'] == method) & (df['n_cal'] == n_cal)]['accuracy'].mean()
        print(f'  SR-GC (alpha={alpha}): {acc:.4f} (gap={acc-baseline:+.4f})', flush=True)

print('\nDone!', flush=True)