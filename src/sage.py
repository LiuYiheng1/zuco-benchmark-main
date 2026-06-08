"""SAGE: Shot-Adaptive Gaussian-Discriminative Calibration"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

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

def sr_gaussian_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.25):
    mu_cal_0, sigma_cal_0 = compute_class_stats_diagonal(X_cal, y_cal, 0)
    mu_cal_1, sigma_cal_1 = compute_class_stats_diagonal(X_cal, y_cal, 1)

    if mu_cal_0 is None or mu_cal_1 is None:
        return np.zeros(len(X_test))

    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_source_0
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_source_1
    sigma_blend_0 = alpha * sigma_cal_0 + (1 - alpha) * sigma_source_0
    sigma_blend_1 = alpha * sigma_cal_1 + (1 - alpha) * sigma_source_1

    cov_inv_0 = np.linalg.inv(np.diag(sigma_blend_0 ** 2))
    cov_inv_1 = np.linalg.inv(np.diag(sigma_blend_1 ** 2))

    scores_0 = np.array([np.sqrt(np.dot(np.dot(x - mu_blend_0, cov_inv_0), x - mu_blend_0)) for x in X_test])
    scores_1 = np.array([np.sqrt(np.dot(np.dot(x - mu_blend_1, cov_inv_1), x - mu_blend_1)) for x in X_test])

    preds = (scores_1 < scores_0).astype(int)
    return preds, scores_0, scores_1

def discriminative_predict(X_cal, y_cal, X_test, return_scores=False):
    if len(np.unique(y_cal)) < 2:
        if return_scores:
            return np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))
        return np.zeros(len(X_test))
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)
    preds = clf.predict(X_test_s)
    if return_scores:
        return preds, probs[:, 0], probs[:, 1]
    return preds

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

def cv_lambda_estimate(X_cal, y_cal, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.25):
    n = len(y_cal)
    if n < 4:
        return 0.5

    sr_errors = []
    svm_errors = []

    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        val_idx = np.array([i])

        X_train = X_cal[train_idx]
        y_train = y_cal[train_idx]
        X_val = X_cal[val_idx]
        y_val = y_cal[val_idx]

        _, sr_s0, sr_s1 = sr_gaussian_predict(X_train, y_train, X_val, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha)
        _, svm_s0, svm_s1 = discriminative_predict(X_train, y_train, X_val, return_scores=True)

        sr_diff = sr_s1 - sr_s0
        svm_diff = svm_s1 - svm_s0

        sr_preds = (sr_diff > 0).astype(int)
        svm_preds = (svm_diff > 0).astype(int)

        sr_errors.append(1 - accuracy_score(y_val, sr_preds))
        svm_errors.append(1 - accuracy_score(y_val, svm_preds))

    R_g = np.mean(sr_errors)
    R_d = np.mean(svm_errors)

    diff = R_d - R_g
    lambda_cv = 1.0 / (1.0 + np.exp(-5.0 * diff))
    return lambda_cv

print('SAGE Experiments')
print('='*60)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

SAGE_RULE_LAMBDA = {3: 0.9, 5: 0.9, 10: 0.7, 20: 0.5, 50: 0.1}

for seed in seeds:
    print(f'\nSeed {seed}:')

    source_data = []
    for subj in Y_SUBJECTS:
        X, y = load_eeg_data(subj)
        if X is not None:
            source_data.append((X, y))

    for held_out_idx, held_out in enumerate(Y_SUBJECTS):
        X_train_all = []
        y_train_all = []
        for subj_idx, (X, y) in enumerate(source_data):
            if subj_idx != held_out_idx:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test_orig, y_test_orig = load_eeg_data(held_out)
        if len(X_train_all) == 0 or X_test_orig is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        mu_source_0, sigma_source_0 = compute_class_stats_diagonal(X_train_all, y_train_all, 0)
        mu_source_1, sigma_source_1 = compute_class_stats_diagonal(X_train_all, y_train_all, 1)

        if mu_source_0 is None or mu_source_1 is None:
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

        print(f'  {held_out}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(cal_pool_indices):
                continue

            cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            svm_preds, svm_s0, svm_s1 = discriminative_predict(X_cal, y_cal, X_test, return_scores=True)
            sr_preds, sr_s0, sr_s1 = sr_gaussian_predict(X_cal, y_cal, X_test, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.25)

            acc_svm = accuracy_score(y_test, svm_preds)
            acc_sr = accuracy_score(y_test, sr_preds)

            for method_name, preds in [('EEG_SVM', svm_preds), ('SR-GC', sr_preds)]:
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='macro')
                bacc = balanced_accuracy_score(y_test, preds)
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': method_name, 'lambda': 1.0 if method_name == 'SR-GC' else 0.0,
                    'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc
                })

            lambda_rule = SAGE_RULE_LAMBDA[n_cal]
            fused_diff = lambda_rule * (sr_s1 - sr_s0) + (1 - lambda_rule) * (svm_s1 - svm_s0)
            preds_rule = (fused_diff > 0).astype(int)
            acc_rule = accuracy_score(y_test, preds_rule)
            f1_rule = f1_score(y_test, preds_rule, average='macro')
            bacc_rule = balanced_accuracy_score(y_test, preds_rule)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SAGE_rule', 'lambda': lambda_rule,
                'accuracy': acc_rule, 'macro_f1': f1_rule, 'balanced_accuracy': bacc_rule
            })

            lambda_cv = cv_lambda_estimate(X_cal, y_cal, mu_source_0, sigma_source_0, mu_source_1, sigma_source_1, alpha=0.25)
            fused_cv = lambda_cv * (sr_s1 - sr_s0) + (1 - lambda_cv) * (svm_s1 - svm_s0)
            preds_cv = (fused_cv > 0).astype(int)
            acc_cv = accuracy_score(y_test, preds_cv)
            f1_cv = f1_score(y_test, preds_cv, average='macro')
            bacc_cv = balanced_accuracy_score(y_test, preds_cv)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SAGE_cv', 'lambda': lambda_cv,
                'accuracy': acc_cv, 'macro_f1': f1_cv, 'balanced_accuracy': bacc_cv
            })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/sage_results.csv', index=False)

print('')
print('\n' + '='*60)
print('SAGE Results Summary')
print('='*60)

for n_cal in shot_settings:
    print(f'\n{n_cal}-shot per class:')
    baseline = df[(df['method'] == 'EEG_SVM') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    sr_acc = df[(df['method'] == 'SR-GC') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    rule_acc = df[(df['method'] == 'SAGE_rule') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    cv_acc = df[(df['method'] == 'SAGE_cv') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    print(f'  EEG_SVM:   {baseline:.4f}')
    print(f'  SR-GC:     {sr_acc:.4f} (gap={sr_acc-baseline:+.4f})')
    print(f'  SAGE_rule: {rule_acc:.4f} (gap={rule_acc-baseline:+.4f}, vs SR-GC={rule_acc-sr_acc:+.4f})')
    print(f'  SAGE_cv:   {cv_acc:.4f} (gap={cv_acc-baseline:+.4f}, vs SR-GC={cv_acc-sr_acc:+.4f})')

avg_all = df[df['method'].isin(['EEG_SVM', 'SR-GC', 'SAGE_rule', 'SAGE_cv'])].groupby(['method'])['accuracy'].mean()
print('\nOverall average:')
for method in ['EEG_SVM', 'SR-GC', 'SAGE_rule', 'SAGE_cv']:
    if method in avg_all.index:
        print(f'  {method}: {avg_all[method]:.4f}')

avg_3_5_10 = df[(df['n_cal'].isin([3, 5, 10])) & (df['method'].isin(['EEG_SVM', 'SR-GC', 'SAGE_rule', 'SAGE_cv']))].groupby(['method'])['accuracy'].mean()
print('\n3/5/10-shot average:')
for method in ['EEG_SVM', 'SR-GC', 'SAGE_rule', 'SAGE_cv']:
    if method in avg_3_5_10.index:
        print(f'  {method}: {avg_3_5_10[method]:.4f}')

print('\nDone!')