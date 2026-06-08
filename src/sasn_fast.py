"""SASN Fast Test"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

FEATURES_DIR = "features"
RESULTS_DIR = "results/personalized"
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

def kmeans_sampling(X_pool, n_select):
    from sklearn.cluster import KMeans
    k = min(n_select, len(X_pool))
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
    return np.array(selected[:n_select])

def train_and_evaluate(X_cal, y_cal, X_test, y_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
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

def compute_sasn(X_cal_pool, y_cal_pool, X_train_all, y_train_all, kappa):
    class_0 = X_train_all[y_train_all == 0]
    class_1 = X_train_all[y_train_all == 1]
    mu_source_0 = np.mean(class_0, axis=0)
    sigma_source_0 = np.std(class_0, axis=0) + 1e-8
    mu_source_1 = np.mean(class_1, axis=0)
    sigma_source_1 = np.std(class_1, axis=0) + 1e-8

    class_0_pool = X_cal_pool[y_cal_pool == 0]
    class_1_pool = X_cal_pool[y_cal_pool == 1]
    if len(class_0_pool) == 0 or len(class_1_pool) == 0:
        return None
    mu_target_0 = np.mean(class_0_pool, axis=0)
    sigma_target_0 = np.std(class_0_pool, axis=0) + 1e-8
    mu_target_1 = np.mean(class_1_pool, axis=0)
    sigma_target_1 = np.std(class_1_pool, axis=0) + 1e-8

    n_target = min(len(class_0_pool), len(class_1_pool))
    rho = n_target / (n_target + kappa)

    mu_0 = rho * mu_target_0 + (1 - rho) * mu_source_0
    sigma_0 = rho * sigma_target_0 + (1 - rho) * sigma_source_0
    mu_1 = rho * mu_target_1 + (1 - rho) * mu_source_1
    sigma_1 = rho * sigma_target_1 + (1 - rho) * sigma_source_1

    return (mu_0, sigma_0), (mu_1, sigma_1)

print('SASN Fast Test...')

results = []
seed = 0
shot_settings = [5, 10, 20]
kappa_values = [10, 50]

for held_out in Y_SUBJECTS:
    X_train_all, y_train_all = [], []
    for subj in Y_SUBJECTS:
        if subj == held_out:
            continue
        X, y = load_eeg_data(subj)
        if X is not None:
            X_train_all.append(X)
            y_train_all.append(y)

    X_test, y_test = load_eeg_data(held_out)
    if len(X_train_all) == 0 or X_test is None:
        continue

    X_train_all = np.vstack(X_train_all)
    y_train_all = np.concatenate(y_train_all)

    n_samples = len(y_test)
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    test_size = n_samples // 3
    cal_pool_size = n_samples - test_size
    test_indices = indices[:test_size]
    cal_pool_indices = indices[test_size:]

    X_holdout = X_test.copy()
    y_holdout = y_test.copy()

    X_test = X_holdout[test_indices]
    y_test = y_holdout[test_indices]
    X_cal_pool = X_holdout[cal_pool_indices]
    y_cal_pool = y_holdout[cal_pool_indices]

    print(' ' + held_out, end='', flush=True)

    for n_cal in shot_settings:
        if n_cal * 2 > len(cal_pool_indices):
            continue

        class_0_idx = np.where(y_cal_pool == 0)[0]
        class_1_idx = np.where(y_cal_pool == 1)[0]
        np.random.shuffle(class_0_idx)
        np.random.shuffle(class_1_idx)
        cal_idx = np.concatenate([class_0_idx[:n_cal], class_1_idx[:n_cal]])

        X_cal = X_cal_pool[cal_idx]
        y_cal = y_cal_pool[cal_idx]

        if len(np.unique(y_cal)) < 2:
            continue

        acc, f1, bacc, auroc = train_and_evaluate(X_cal, y_cal, X_test, y_test)
        results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'StandardScaler', 'kappa': 0, 'accuracy': acc, 'macro_f1': f1, 'bacc': bacc, 'auroc': auroc})

        cal_idx_accs = kmeans_sampling(X_cal_pool, n_cal * 2)
        X_cal_accs = X_cal_pool[cal_idx_accs]
        y_cal_accs = y_cal_pool[cal_idx_accs]
        acc, f1, bacc, auroc = train_and_evaluate(X_cal_accs, y_cal_accs, X_test, y_test)
        results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'ACCS', 'kappa': 0, 'accuracy': acc, 'macro_f1': f1, 'bacc': bacc, 'auroc': auroc})

        for kappa in kappa_values:
            sasn_params = compute_sasn(X_cal_pool, y_cal_pool, X_train_all, y_train_all, kappa)
            if sasn_params is None:
                continue
            (mu_0, sigma_0), (mu_1, sigma_1) = sasn_params
            X_cal_norm = (X_cal - mu_0) / sigma_0
            X_test_norm = (X_test - mu_0) / sigma_0
            acc, f1, bacc, auroc = train_and_evaluate(X_cal_norm, y_cal, X_test_norm, y_test)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'SASN', 'kappa': kappa, 'accuracy': acc, 'macro_f1': f1, 'bacc': bacc, 'auroc': auroc})

            X_cal_accs_norm_0 = (X_cal_accs - mu_0) / sigma_0
            X_test_norm_0 = (X_test - mu_0) / sigma_0
            acc, f1, bacc, auroc = train_and_evaluate(X_cal_accs_norm_0, y_cal_accs, X_test_norm_0, y_test)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'SASN_ACCS', 'kappa': kappa, 'accuracy': acc, 'macro_f1': f1, 'bacc': bacc, 'auroc': auroc})

print('')

df = pd.DataFrame(results)
df.to_csv(RESULTS_DIR + '/sasn_results.csv', index=False)

print('\nResults:')
for method in ['StandardScaler', 'ACCS', 'SASN', 'SASN_ACCS']:
    for kappa in df[df['method'] == method]['kappa'].unique():
        data = df[(df['method'] == method) & (df['kappa'] == kappa)]
        if len(data) > 0:
            print('  ' + method + ' (kappa=' + str(kappa) + '): acc=' + str(round(data['accuracy'].mean(), 4)))

baseline = df[df['method'] == 'StandardScaler']['accuracy'].mean()
accs = df[df['method'] == 'ACCS']['accuracy'].mean()
print('\nBaseline (StandardScaler): ' + str(round(baseline, 4)))
print('ACCS: ' + str(round(accs, 4)) + ' (gap=' + str(round(accs - baseline, 4)) + ')')

sasn_best = 0
sasn_kappa = None
for kappa in df[df['method'] == 'SASN']['kappa'].unique():
    data = df[(df['method'] == 'SASN') & (df['kappa'] == kappa)]
    acc = data['accuracy'].mean()
    if acc > sasn_best:
        sasn_best = acc
        sasn_kappa = kappa
print('SASN best (kappa=' + str(sasn_kappa) + '): ' + str(round(sasn_best, 4)) + ' (gap=' + str(round(sasn_best - baseline, 4)) + ')')

sasn_accs_best = 0
sasn_accs_kappa = None
for kappa in df[df['method'] == 'SASN_ACCS']['kappa'].unique():
    data = df[(df['method'] == 'SASN_ACCS') & (df['kappa'] == kappa)]
    acc = data['accuracy'].mean()
    if acc > sasn_accs_best:
        sasn_accs_best = acc
        sasn_accs_kappa = kappa
print('SASN_ACCS best (kappa=' + str(sasn_accs_kappa) + '): ' + str(round(sasn_accs_best, 4)) + ' (gap=' + str(round(sasn_accs_best - accs, 4)) + ')')

print('\nSuccess criteria:')
print('  1. SASN vs StandardScaler +2%: ' + ('PASS' if sasn_best - baseline >= 0.02 else 'FAIL'))
print('  2. SASN_ACCS vs ACCS +1%: ' + ('PASS' if sasn_accs_best - accs >= 0.01 else 'FAIL'))

print('\nDone!')