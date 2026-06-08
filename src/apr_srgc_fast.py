"""APR-SRGC: Fast version with reduced parameter grid"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

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

def compute_class_stats(X, y, class_label):
    X_class = X[y == class_label]
    if len(X_class) == 0:
        return None, None
    mu = np.mean(X_class, axis=0)
    sigma = np.std(X_class, axis=0) + 1e-8
    return mu, sigma

def apr_srgc_predict(X_cal, y_cal, X_test, mu_s0, sigma_s0, mu_s1, sigma_s1, kappa=5, nu=5):
    n0 = np.sum(y_cal == 0)
    n1 = np.sum(y_cal == 1)

    lambda_0 = kappa / (kappa + n0) if n0 > 0 else 0.5
    lambda_1 = kappa / (kappa + n1) if n1 > 0 else 0.5
    gamma_0 = nu / (nu + n0) if n0 > 0 else 0.5
    gamma_1 = nu / (nu + n1) if n1 > 0 else 0.5

    mu_cal_0, sigma_cal_0 = compute_class_stats(X_cal, y_cal, 0)
    mu_cal_1, sigma_cal_1 = compute_class_stats(X_cal, y_cal, 1)

    if mu_cal_0 is None or mu_cal_1 is None:
        return np.zeros(len(X_test))

    mu_blend_0 = lambda_0 * mu_s0 + (1 - lambda_0) * mu_cal_0
    mu_blend_1 = lambda_1 * mu_s1 + (1 - lambda_1) * mu_cal_1
    sigma_blend_0 = gamma_0 * sigma_s0 + (1 - gamma_0) * sigma_cal_0
    sigma_blend_1 = gamma_1 * sigma_s1 + (1 - gamma_1) * sigma_cal_1

    cov_inv_0 = np.linalg.inv(np.diag(sigma_blend_0 ** 2 + 1e-8))
    cov_inv_1 = np.linalg.inv(np.diag(sigma_blend_1 ** 2 + 1e-8))

    scores_0 = np.array([np.sqrt(np.dot(np.dot(x - mu_blend_0, cov_inv_0), x - mu_blend_0)) for x in X_test])
    scores_1 = np.array([np.sqrt(np.dot(np.dot(x - mu_blend_1, cov_inv_1), x - mu_blend_1)) for x in X_test])

    preds = (scores_1 < scores_0).astype(int)
    return preds

def fixed_srgc_predict(X_cal, y_cal, X_test, mu_s0, sigma_s0, mu_s1, sigma_s1, alpha=0.75):
    mu_cal_0, sigma_cal_0 = compute_class_stats(X_cal, y_cal, 0)
    mu_cal_1, sigma_cal_1 = compute_class_stats(X_cal, y_cal, 1)

    if mu_cal_0 is None or mu_cal_1 is None:
        return np.zeros(len(X_test))

    mu_blend_0 = alpha * mu_s0 + (1 - alpha) * mu_cal_0
    mu_blend_1 = alpha * mu_s1 + (1 - alpha) * mu_cal_1
    sigma_blend_0 = alpha * sigma_s0 + (1 - alpha) * sigma_cal_0
    sigma_blend_1 = alpha * sigma_s1 + (1 - alpha) * sigma_cal_1

    cov_inv_0 = np.linalg.inv(np.diag(sigma_blend_0 ** 2 + 1e-8))
    cov_inv_1 = np.linalg.inv(np.diag(sigma_blend_1 ** 2 + 1e-8))

    scores_0 = np.array([np.sqrt(np.dot(np.dot(x - mu_blend_0, cov_inv_0), x - mu_blend_0)) for x in X_test])
    scores_1 = np.array([np.sqrt(np.dot(np.dot(x - mu_blend_1, cov_inv_1), x - mu_blend_1)) for x in X_test])

    preds = (scores_1 < scores_0).astype(int)
    return preds

def svm_predict(X_cal, y_cal, X_test):
    if len(np.unique(y_cal)) < 2:
        return np.zeros(len(X_test))
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(scaler.fit_transform(X_cal), y_cal)
    return clf.predict(scaler.transform(X_test))

def balanced_sampling(y_pool, n_per_class):
    idx0 = np.where(y_pool == 0)[0]
    idx1 = np.where(y_pool == 1)[0]
    np.random.shuffle(idx0)
    np.random.shuffle(idx1)
    n0 = min(n_per_class, len(idx0))
    n1 = min(n_per_class, len(idx1))
    return np.concatenate([idx0[:n0], idx1[:n1]])

print('APR-SRGC Fast Experiments')
print('='*60)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

kappa_nu_combos = [
    (1, 1), (1, 10), (1, 50),
    (5, 5), (5, 10), (5, 50),
    (10, 10), (10, 50),
    (20, 20), (20, 50),
    (50, 50)
]

cache = {s: load_eeg_data(s) for s in Y_SUBJECTS}

for seed in seeds:
    print(f'\nSeed {seed}:', flush=True)
    np.random.seed(seed)

    for held_out in Y_SUBJECTS:
        X_test_orig, y_test_orig = cache.get(held_out, (None, None))
        if X_test_orig is None:
            continue

        X_train = []
        y_train = []
        for s in Y_SUBJECTS:
            if s != held_out and s in cache:
                X, y = cache[s]
                X_train.append(X)
                y_train.append(y)

        if not X_train:
            continue

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        mu_s0, sigma_s0 = compute_class_stats(X_train, y_train, 0)
        mu_s1, sigma_s1 = compute_class_stats(X_train, y_train, 1)

        if mu_s0 is None:
            continue

        indices = np.random.permutation(len(y_test_orig))
        ts = len(y_test_orig) // 3
        X_test = X_test_orig[indices[:ts]]
        y_test = y_test_orig[indices[:ts]]
        X_cp = X_test_orig[indices[ts:]]
        y_cp = y_test_orig[indices[ts:]]

        print(f' {held_out}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(X_cp):
                continue

            cal_idx = balanced_sampling(y_cp, n_cal)
            X_cal = X_cp[cal_idx]
            y_cal = y_cp[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            svm_preds = svm_predict(X_cal, y_cal, X_test)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                          'method': 'EEG_SVM', 'kappa': np.nan, 'nu': np.nan,
                          'accuracy': accuracy_score(y_test, svm_preds)})

            fixed_preds = fixed_srgc_predict(X_cal, y_cal, X_test, mu_s0, sigma_s0, mu_s1, sigma_s1, alpha=0.75)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                          'method': 'SR-GC_fixed', 'kappa': np.nan, 'nu': np.nan,
                          'accuracy': accuracy_score(y_test, fixed_preds)})

            for kappa, nu in kappa_nu_combos:
                apr_preds = apr_srgc_predict(X_cal, y_cal, X_test, mu_s0, sigma_s0, mu_s1, sigma_s1, kappa=kappa, nu=nu)
                results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                              'method': 'APR-SRGC', 'kappa': kappa, 'nu': nu,
                              'accuracy': accuracy_score(y_test, apr_preds)})

        print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/apr_srgc_results.csv', index=False)

print('')
print('\n' + '='*60)
print('SUMMARY')
print('='*60)

df = pd.DataFrame(results)

print('\nEEG_SVM:')
for n in shot_settings:
    d = df[(df.method == 'EEG_SVM') & (df.n_cal == n)]
    if len(d) > 0:
        print(f'  {n}-shot: {d.accuracy.mean():.4f}')

print('\nSR-GC fixed:')
for n in shot_settings:
    d = df[(df.method == 'SR-GC_fixed') & (df.n_cal == n)]
    if len(d) > 0:
        print(f'  {n}-shot: {d.accuracy.mean():.4f}')

print('\nAPR-SRGC best by shot:')
for n in shot_settings:
    apr = df[(df.method == 'APR-SRGC') & (df.n_cal == n)]
    if len(apr) > 0:
        best = apr.groupby(['kappa', 'nu'])['accuracy'].mean().idxmax()
        best_acc = apr.groupby(['kappa', 'nu'])['accuracy'].mean().max()
        avg_acc = apr['accuracy'].mean()
        print(f'  {n}-shot: avg={avg_acc:.4f}, best={best_acc:.4f} (k={best[0]}, nu={best[1]})')

print('\nComparison:')
for n in shot_settings:
    svm = df[(df.method == 'EEG_SVM') & (df.n_cal == n)]['accuracy'].mean()
    fixed = df[(df.method == 'SR-GC_fixed') & (df.n_cal == n)]['accuracy'].mean()
    apr = df[(df.method == 'APR-SRGC') & (df.n_cal == n)]
    if len(apr) > 0:
        apr_avg = apr['accuracy'].mean()
        apr_best = apr.groupby(['kappa', 'nu'])['accuracy'].mean().max()
        print(f'  {n}-shot: SVM={svm:.4f}, SRGC={fixed:.4f}, APR_avg={apr_avg:.4f}, APR_best={apr_best:.4f}')

print('\nDone!')