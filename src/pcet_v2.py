"""PCET-v2: PCET Optimization with Error Variants and Scaling

Compares:
1. Error feature variants
2. Predictor variants
3. Feature scaling variants
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.decomposition import PCA

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

def train_pca_predictor(X_train, y_train, n_components=20):
    pca_models = {}
    for c in [0, 1]:
        X_c = X_train[y_train == c]
        if len(X_c) > n_components:
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X_c)
            pca_models[c] = pca
        else:
            pca_models[c] = None
    return pca_models

def train_ridge_autoencoder(X_train, y_train, alpha=1.0):
    ridge_models = {}
    for c in [0, 1]:
        X_c = X_train[y_train == c]
        if len(X_c) > 1:
            ridge = RidgeClassifier(alpha=alpha)
            ridge.fit(X_c, X_c)
            ridge_models[c] = ridge
        else:
            ridge_models[c] = None
    return ridge_models

def compute_errors(x, x_hat):
    e = x - x_hat
    abs_e = np.abs(e)
    energy_e = np.log(1 + e ** 2)
    return e, abs_e, energy_e

def svm_predict(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def pcet_pca_error_only(X_cal, y_cal, X_test, n_components=20):
    pca_models = train_pca_predictor(X_cal, y_cal, n_components)
    error_cal = np.zeros((len(X_cal), 2))
    error_test = np.zeros((len(X_test), 2))
    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_recon_cal = pca.inverse_transform(pca.transform(X_cal))
            X_recon_test = pca.inverse_transform(pca.transform(X_test))
            _, abs_e_cal, _ = compute_errors(X_cal, X_recon_cal)
            _, abs_e_test, _ = compute_errors(X_test, X_recon_test)
            error_cal[:, i] = np.sqrt(np.sum(abs_e_cal ** 2, axis=1))
            error_test[:, i] = np.sqrt(np.sum(abs_e_test ** 2, axis=1))
    scaler = StandardScaler()
    error_cal_s = scaler.fit_transform(error_cal)
    error_test_s = scaler.transform(error_test)
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(error_cal_s, y_cal)
    preds = clf.predict(error_test_s)
    return preds

def pcet_pca_abs_error_only(X_cal, y_cal, X_test, n_components=20):
    pca_models = train_pca_predictor(X_cal, y_cal, n_components)
    error_cal = np.zeros((len(X_cal), 2 * X_cal.shape[1]))
    error_test = np.zeros((len(X_test), 2 * X_test.shape[1]))
    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_recon_cal = pca.inverse_transform(pca.transform(X_cal))
            X_recon_test = pca.inverse_transform(pca.transform(X_test))
            _, abs_e_cal, _ = compute_errors(X_cal, X_recon_cal)
            _, abs_e_test, _ = compute_errors(X_test, X_recon_test)
            error_cal[:, i*X_cal.shape[1]:(i+1)*X_cal.shape[1]] = abs_e_cal
            error_test[:, i*X_test.shape[1]:(i+1)*X_test.shape[1]] = abs_e_test
    scaler = StandardScaler()
    error_cal_s = scaler.fit_transform(error_cal)
    error_test_s = scaler.transform(error_test)
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(error_cal_s, y_cal)
    preds = clf.predict(error_test_s)
    return preds

def pcet_pca_squared_error_only(X_cal, y_cal, X_test, n_components=20):
    pca_models = train_pca_predictor(X_cal, y_cal, n_components)
    error_cal = np.zeros((len(X_cal), 2))
    error_test = np.zeros((len(X_test), 2))
    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_recon_cal = pca.inverse_transform(pca.transform(X_cal))
            X_recon_test = pca.inverse_transform(pca.transform(X_test))
            e_cal = X_cal - X_recon_cal
            e_test = X_test - X_recon_test
            error_cal[:, i] = np.sum(e_cal ** 2, axis=1)
            error_test[:, i] = np.sum(e_test ** 2, axis=1)
    scaler = StandardScaler()
    error_cal_s = scaler.fit_transform(error_cal)
    error_test_s = scaler.transform(error_test)
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(error_cal_s, y_cal)
    preds = clf.predict(error_test_s)
    return preds

def pcet_pca_raw_plus_error(X_cal, y_cal, X_test, n_components=20):
    pca_models = train_pca_predictor(X_cal, y_cal, n_components)
    error_cal = np.zeros((len(X_cal), 2))
    error_test = np.zeros((len(X_test), 2))
    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_recon_cal = pca.inverse_transform(pca.transform(X_cal))
            X_recon_test = pca.inverse_transform(pca.transform(X_test))
            _, abs_e_cal, _ = compute_errors(X_cal, X_recon_cal)
            _, abs_e_test, _ = compute_errors(X_test, X_recon_test)
            error_cal[:, i] = np.sqrt(np.sum(abs_e_cal ** 2, axis=1))
            error_test[:, i] = np.sqrt(np.sum(abs_e_test ** 2, axis=1))
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    combined_cal = np.hstack([X_cal_s, error_cal])
    combined_test = np.hstack([X_test_s, error_test])
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(combined_cal, y_cal)
    preds = clf.predict(combined_test)
    return preds

def pcet_pca_raw_plus_abs_error(X_cal, y_cal, X_test, n_components=20):
    pca_models = train_pca_predictor(X_cal, y_cal, n_components)
    error_cal = np.zeros((len(X_cal), 2 * X_cal.shape[1]))
    error_test = np.zeros((len(X_test), 2 * X_test.shape[1]))
    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_recon_cal = pca.inverse_transform(pca.transform(X_cal))
            X_recon_test = pca.inverse_transform(pca.transform(X_test))
            _, abs_e_cal, _ = compute_errors(X_cal, X_recon_cal)
            _, abs_e_test, _ = compute_errors(X_test, X_recon_test)
            error_cal[:, i*X_cal.shape[1]:(i+1)*X_cal.shape[1]] = abs_e_cal
            error_test[:, i*X_test.shape[1]:(i+1)*X_test.shape[1]] = abs_e_test
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    combined_cal = np.hstack([X_cal_s, error_cal])
    combined_test = np.hstack([X_test_s, error_test])
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(combined_cal, y_cal)
    preds = clf.predict(combined_test)
    return preds

def pcet_pca_raw_plus_error_energy(X_cal, y_cal, X_test, n_components=20):
    pca_models = train_pca_predictor(X_cal, y_cal, n_components)
    error_cal = np.zeros((len(X_cal), 2))
    error_test = np.zeros((len(X_test), 2))
    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_recon_cal = pca.inverse_transform(pca.transform(X_cal))
            X_recon_test = pca.inverse_transform(pca.transform(X_test))
            _, _, energy_cal = compute_errors(X_cal, X_recon_cal)
            _, _, energy_test = compute_errors(X_test, X_recon_test)
            error_cal[:, i] = np.mean(energy_cal, axis=1)
            error_test[:, i] = np.mean(energy_test, axis=1)
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    combined_cal = np.hstack([X_cal_s, error_cal])
    combined_test = np.hstack([X_test_s, error_test])
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(combined_cal, y_cal)
    preds = clf.predict(combined_test)
    return preds

def pcet_pca_raw_plus_full_error(X_cal, y_cal, X_test, n_components=20):
    pca_models = train_pca_predictor(X_cal, y_cal, n_components)
    error_cal = np.zeros((len(X_cal), 2 + 2 * X_cal.shape[1]))
    error_test = np.zeros((len(X_test), 2 + 2 * X_test.shape[1]))
    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_recon_cal = pca.inverse_transform(pca.transform(X_cal))
            X_recon_test = pca.inverse_transform(pca.transform(X_test))
            e_cal, abs_e_cal, energy_cal = compute_errors(X_cal, X_recon_cal)
            e_test, abs_e_test, energy_test = compute_errors(X_test, X_recon_test)
            error_cal[:, i] = np.sqrt(np.sum(abs_e_cal ** 2, axis=1))
            error_cal[:, 2 + i*X_cal.shape[1]:2 + (i+1)*X_cal.shape[1]] = abs_e_cal
            error_test[:, i] = np.sqrt(np.sum(abs_e_test ** 2, axis=1))
            error_test[:, 2 + i*X_test.shape[1]:2 + (i+1)*X_test.shape[1]] = abs_e_test
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    combined_cal = np.hstack([X_cal_s, error_cal])
    combined_test = np.hstack([X_test_s, error_test])
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(combined_cal, y_cal)
    preds = clf.predict(combined_test)
    return preds

def pcet_ridge_raw_plus_error(X_cal, y_cal, X_test, alpha=1.0):
    ridge_models = train_ridge_autoencoder(X_cal, y_cal, alpha)
    error_cal = np.zeros((len(X_cal), 2))
    error_test = np.zeros((len(X_test), 2))
    for i, (c, ridge) in enumerate(ridge_models.items()):
        if ridge is not None:
            X_recon_cal = ridge.predict(X_cal)
            X_recon_test = ridge.predict(X_test)
            _, abs_e_cal, _ = compute_errors(X_cal, X_recon_cal)
            _, abs_e_test, _ = compute_errors(X_test, X_recon_test)
            error_cal[:, i] = np.sqrt(np.sum(abs_e_cal ** 2, axis=1))
            error_test[:, i] = np.sqrt(np.sum(abs_e_test ** 2, axis=1))
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    combined_cal = np.hstack([X_cal_s, error_cal])
    combined_test = np.hstack([X_test_s, error_test])
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(combined_cal, y_cal)
    preds = clf.predict(combined_test)
    return preds

def pcet_joint_scaling(X_cal, y_cal, X_test, n_components=20):
    pca_models = train_pca_predictor(X_cal, y_cal, n_components)
    error_cal = np.zeros((len(X_cal), 2))
    error_test = np.zeros((len(X_test), 2))
    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_recon_cal = pca.inverse_transform(pca.transform(X_cal))
            X_recon_test = pca.inverse_transform(pca.transform(X_test))
            _, abs_e_cal, _ = compute_errors(X_cal, X_recon_cal)
            _, abs_e_test, _ = compute_errors(X_test, X_recon_test)
            error_cal[:, i] = np.sqrt(np.sum(abs_e_cal ** 2, axis=1))
            error_test[:, i] = np.sqrt(np.sum(abs_e_test ** 2, axis=1))
    combined_cal = np.hstack([X_cal, error_cal])
    combined_test = np.hstack([X_test, error_test])
    scaler = StandardScaler()
    combined_cal_s = scaler.fit_transform(combined_cal)
    combined_test_s = scaler.transform(combined_test)
    clf = RidgeClassifier(alpha=0.1)
    clf.fit(combined_cal_s, y_cal)
    preds = clf.predict(combined_test_s)
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

print('PCET-v2 Optimization', flush=True)
print('='*80, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    print(f'\nSeed {seed}:', flush=True)
    for held_out in Y_SUBJECTS:
        X_test_orig, y_test_orig = load_eeg_data(held_out)
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

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

            preds_svm, probs_svm = svm_predict(X_cal, y_cal, X_test)
            acc_svm = accuracy_score(y_test, preds_svm)
            f1_svm = f1_score(y_test, preds_svm, average='macro')
            bacc_svm = balanced_accuracy_score(y_test, preds_svm)
            try:
                auroc_svm = roc_auc_score(y_test, probs_svm)
            except:
                auroc_svm = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'Raw_EEG_SVM',
                'accuracy': acc_svm, 'macro_f1': f1_svm, 'balanced_accuracy': bacc_svm, 'auroc': auroc_svm
            })

            methods_funcs = [
                ('Error_only', lambda: pcet_pca_error_only(X_cal, y_cal, X_test)),
                ('AbsError_only', lambda: pcet_pca_abs_error_only(X_cal, y_cal, X_test)),
                ('SquaredError_only', lambda: pcet_pca_squared_error_only(X_cal, y_cal, X_test)),
                ('Raw_plus_Error', lambda: pcet_pca_raw_plus_error(X_cal, y_cal, X_test)),
                ('Raw_plus_AbsError', lambda: pcet_pca_raw_plus_abs_error(X_cal, y_cal, X_test)),
                ('Raw_plus_ErrorEnergy', lambda: pcet_pca_raw_plus_error_energy(X_cal, y_cal, X_test)),
                ('Raw_plus_FullError', lambda: pcet_pca_raw_plus_full_error(X_cal, y_cal, X_test)),
                ('Ridge_Raw_plus_Error', lambda: pcet_ridge_raw_plus_error(X_cal, y_cal, X_test)),
                ('Joint_Scaling', lambda: pcet_joint_scaling(X_cal, y_cal, X_test)),
            ]

            for method_name, method_func in methods_funcs:
                try:
                    preds = method_func()
                    acc = accuracy_score(y_test, preds)
                except Exception as e:
                    acc = 0.5
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': method_name,
                    'accuracy': acc, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
                })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/pcet_v2_results.csv', index=False)

print('', flush=True)
print('\n' + '='*80, flush=True)
print('PCET-v2 Results Summary', flush=True)
print('='*80, flush=True)

baseline_df = df[df['method'] == 'Raw_EEG_SVM']
methods = df['method'].unique()

print('\nComparing methods by shot:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    print(f'\n  {n_cal}-shot (SVM={base_acc:.4f}):', flush=True)
    for method in sorted(methods):
        if method != 'Raw_EEG_SVM':
            acc = df[df['method'] == method][df['n_cal'] == n_cal]['accuracy'].mean()
            if not np.isnan(acc):
                print(f'    {method}: {acc:.4f} (gap={acc-base_acc:+.4f})', flush=True)

print('\nDone!', flush=True)