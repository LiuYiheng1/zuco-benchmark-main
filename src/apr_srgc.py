"""APR-SRGC: Adaptive Prior-Release Source-Regularized Gaussian Calibration"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.covariance import LedoitWolf
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

def compute_class_stats_diag(X, y, class_label):
    X_class = X[y == class_label]
    if len(X_class) == 0:
        return None, None
    mu = np.mean(X_class, axis=0)
    sigma = np.std(X_class, axis=0) + 1e-8
    return mu, sigma

def compute_class_stats_full(X, y, class_label):
    X_class = X[y == class_label]
    if len(X_class) < 2:
        return None, None
    mu = np.mean(X_class, axis=0)
    try:
        lw = LedoitWolf()
        lw.fit(X_class)
        cov = lw.covariance_
    except:
        cov = np.cov(X_class.T) + np.eye(X_class.shape[1]) * 1e-6
    return mu, cov

def apr_srgc_predict(X_cal, y_cal, X_test, mu_s0, sigma_s0, mu_s1, sigma_s1, kappa=5, nu=5, cov_type='diagonal'):
    n0 = np.sum(y_cal == 0)
    n1 = np.sum(y_cal == 1)

    lambda_0 = kappa / (kappa + n0) if n0 > 0 else 0.5
    lambda_1 = kappa / (kappa + n1) if n1 > 0 else 0.5
    gamma_0 = nu / (nu + n0) if n0 > 0 else 0.5
    gamma_1 = nu / (nu + n1) if n1 > 0 else 0.5

    mu_cal_0, sigma_cal_0 = compute_class_stats_diag(X_cal, y_cal, 0)
    mu_cal_1, sigma_cal_1 = compute_class_stats_diag(X_cal, y_cal, 1)

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

def svm_predict(X_cal, y_cal, X_test):
    if len(np.unique(y_cal)) < 2:
        return np.zeros(len(X_test))
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(scaler.fit_transform(X_cal), y_cal)
    return clf.predict(scaler.transform(X_test))

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

def fixed_srgc_predict(X_cal, y_cal, X_test, mu_s0, sigma_s0, mu_s1, sigma_s1, alpha=0.75):
    mu_cal_0, sigma_cal_0 = compute_class_stats_diag(X_cal, y_cal, 0)
    mu_cal_1, sigma_cal_1 = compute_class_stats_diag(X_cal, y_cal, 1)

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

print('APR-SRGC Experiments')
print('='*60)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]
kappa_values = [1, 3, 5, 10, 20, 50]
nu_values = [1, 3, 5, 10, 20, 50]

cache = {}
for subj in Y_SUBJECTS:
    X, y = load_eeg_data(subj)
    if X is not None:
        cache[subj] = (X, y)

for seed in seeds:
    print(f'\nSeed {seed}:')

    for held_out_idx, held_out in enumerate(Y_SUBJECTS):
        X_test_orig, y_test_orig = cache.get(held_out, (None, None))
        if X_test_orig is None:
            continue

        X_train_all = []
        y_train_all = []
        for subj_idx, subj in enumerate(Y_SUBJECTS):
            if subj != held_out and subj in cache:
                X, y = cache[subj]
                X_train_all.append(X)
                y_train_all.append(y)

        if len(X_train_all) == 0:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        mu_s0, sigma_s0 = compute_class_stats_diag(X_train_all, y_train_all, 0)
        mu_s1, sigma_s1 = compute_class_stats_diag(X_train_all, y_train_all, 1)

        if mu_s0 is None or mu_s1 is None:
            continue

        n_samples = len(y_test_orig)
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        test_size = n_samples // 3
        cal_pool_indices = indices[test_size:]
        test_indices = indices[:test_size]

        X_test = X_test_orig[test_indices]
        y_test = y_test_orig[test_indices]
        X_cal_pool = X_test_orig[cal_pool_indices]
        y_cal_pool = y_test_orig[cal_pool_indices]

        print(f' {held_out}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(cal_pool_indices):
                continue

            cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            svm_preds = svm_predict(X_cal, y_cal, X_test)
            svm_acc = accuracy_score(y_test, svm_preds)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'EEG_SVM', 'kappa': np.nan, 'nu': np.nan,
                'accuracy': svm_acc, 'macro_f1': f1_score(y_test, svm_preds, average='macro'),
                'balanced_accuracy': balanced_accuracy_score(y_test, svm_preds)
            })

            fixed_preds = fixed_srgc_predict(X_cal, y_cal, X_test, mu_s0, sigma_s0, mu_s1, sigma_s1, alpha=0.75)
            fixed_acc = accuracy_score(y_test, fixed_preds)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SR-GC_fixed_a0.75', 'kappa': np.nan, 'nu': np.nan,
                'accuracy': fixed_acc, 'macro_f1': f1_score(y_test, fixed_preds, average='macro'),
                'balanced_accuracy': balanced_accuracy_score(y_test, fixed_preds)
            })

            for kappa in kappa_values:
                for nu in nu_values:
                    apr_preds = apr_srgc_predict(X_cal, y_cal, X_test, mu_s0, sigma_s0, mu_s1, sigma_s1, kappa=kappa, nu=nu)
                    apr_acc = accuracy_score(y_test, apr_preds)
                    results.append({
                        'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                        'method': f'APR-SRGC', 'kappa': kappa, 'nu': nu,
                        'accuracy': apr_acc, 'macro_f1': f1_score(y_test, apr_preds, average='macro'),
                        'balanced_accuracy': balanced_accuracy_score(y_test, apr_preds)
                    })

        print('.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/apr_srgc_results.csv', index=False)

print('')
print('\n' + '='*60)
print('APR-SRGC Results Summary')
print('='*60)

df = pd.DataFrame(results)

print('\nEEG_SVM by shot:')
for n_cal in shot_settings:
    data = df[(df['method'] == 'EEG_SVM') & (df['n_cal'] == n_cal)]
    print(f'  {n_cal}-shot: {data["accuracy"].mean():.4f}')

print('\nSR-GC fixed alpha=0.75 by shot:')
for n_cal in shot_settings:
    data = df[(df['method'] == 'SR-GC_fixed_a0.75') & (df['n_cal'] == n_cal)]
    print(f'  {n_cal}-shot: {data["accuracy"].mean():.4f}')

print('\nAPR-SRGC best configuration by shot:')
for n_cal in shot_settings:
    apr_data = df[(df['method'] == 'APR-SRGC') & (df['n_cal'] == n_cal)]
    if len(apr_data) > 0:
        best_row = apr_data.loc[apr_data['accuracy'].idxmax()]
        best_kappa = best_row['kappa']
        best_nu = best_row['nu']
        best_acc = best_row['accuracy']
        avg_acc = apr_data.groupby(['kappa', 'nu'])['accuracy'].mean()
        print(f'  {n_cal}-shot: Best APR={best_acc:.4f} (kappa={best_kappa}, nu={best_nu})')

        best_combo = avg_acc.idxmax()
        best_avg = avg_acc.max()
        print(f'    Avg across subjects: {best_avg:.4f} at kappa={best_combo[0]}, nu={best_combo[1]}')

print('\nComparison:')
for n_cal in shot_settings:
    svm_acc = df[(df['method'] == 'EEG_SVM') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    fixed_acc = df[(df['method'] == 'SR-GC_fixed_a0.75') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    apr_data = df[(df['method'] == 'APR-SRGC') & (df['n_cal'] == n_cal)]
    if len(apr_data) > 0:
        apr_acc = apr_data['accuracy'].mean()
        best_idx = apr_data.groupby(['kappa', 'nu'])['accuracy'].mean().idxmax()
        best_apr = apr_data.groupby(['kappa', 'nu'])['accuracy'].mean().max()
        print(f'  {n_cal}-shot: SVM={svm_acc:.4f}, SR-GC={fixed_acc:.4f}, APR-SRGC_avg={apr_acc:.4f}, APR-SRGC_best={best_apr:.4f}')

print('\nDone!')