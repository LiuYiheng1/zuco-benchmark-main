"""PCET Ablation Study (Simplified)

Comparisons:
1. Raw_EEG: Standard SVM baseline
2. Prediction_error_only: Just the error magnitude features
3. Random_predictor_error: Using random projection instead of PCA
4. Shuffled_predictor_error: Using shuffled labels for PCA training
5. Calibration_trained_PCET: Full PCET (original)
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

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

def compute_prediction_errors_simple(X, pca_models):
    n_samples = len(X)
    error_features = np.zeros((n_samples, 2))
    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_recon = pca.inverse_transform(pca.transform(X))
            errors = X - X_recon
            error_features[:, i] = np.sqrt(np.sum(errors ** 2, axis=1))
    return error_features

def svm_predict(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds

def prediction_error_only(X_cal, y_cal, X_test, n_components=20):
    pca_models = {}
    for c in [0, 1]:
        X_c = X_cal[y_cal == c]
        if len(X_c) > n_components:
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X_c)
            pca_models[c] = pca
        else:
            pca_models[c] = None

    error_cal = compute_prediction_errors_simple(X_cal, pca_models)
    error_test = compute_prediction_errors_simple(X_test, pca_models)

    scaler = StandardScaler()
    error_cal_s = scaler.fit_transform(error_cal)
    error_test_s = scaler.transform(error_test)

    clf = RidgeClassifier(alpha=0.1)
    clf.fit(error_cal_s, y_cal)
    preds = clf.predict(error_test_s)
    return preds

def random_predictor_error(X_cal, y_cal, X_test, n_components=20):
    rp_models = {}
    for c in [0, 1]:
        X_c = X_cal[y_cal == c]
        if len(X_c) > n_components:
            rp = GaussianRandomProjection(n_components=min(n_components, X_c.shape[1]), random_state=42)
            rp.fit(X_c)
            rp_models[c] = rp
        else:
            rp_models[c] = None

    error_cal = np.zeros((len(X_cal), 2))
    error_test = np.zeros((len(X_test), 2))

    for i, (c, rp) in enumerate(rp_models.items()):
        if rp is not None:
            proj_cal = rp.transform(X_cal)
            proj_test = rp.transform(X_test)
            W = rp.components_
            X_recon_cal = np.dot(proj_cal, W)
            X_recon_test = np.dot(proj_test, W)
            error_cal[:, i] = np.sqrt(np.sum((X_cal - X_recon_cal) ** 2, axis=1))
            error_test[:, i] = np.sqrt(np.sum((X_test - X_recon_test) ** 2, axis=1))

    scaler = StandardScaler()
    error_cal_s = scaler.fit_transform(error_cal)
    error_test_s = scaler.transform(error_test)

    clf = RidgeClassifier(alpha=0.1)
    clf.fit(error_cal_s, y_cal)
    preds = clf.predict(error_test_s)
    return preds

def shuffled_predictor_error(X_cal, y_cal, X_test, n_components=20):
    y_shuffled = y_cal.copy()
    np.random.shuffle(y_shuffled)

    pca_models = {}
    for c in [0, 1]:
        X_c = X_cal[y_shuffled == c]
        if len(X_c) > n_components:
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X_c)
            pca_models[c] = pca
        else:
            pca_models[c] = None

    error_cal = compute_prediction_errors_simple(X_cal, pca_models)
    error_test = compute_prediction_errors_simple(X_test, pca_models)

    scaler = StandardScaler()
    error_cal_s = scaler.fit_transform(error_cal)
    error_test_s = scaler.transform(error_test)

    clf = RidgeClassifier(alpha=0.1)
    clf.fit(error_cal_s, y_cal)
    preds = clf.predict(error_test_s)
    return preds

def pcet_predict(X_cal, y_cal, X_test, n_components=20):
    pca_models = {}
    for c in [0, 1]:
        X_c = X_cal[y_cal == c]
        if len(X_c) > n_components:
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X_c)
            pca_models[c] = pca
        else:
            pca_models[c] = None

    error_cal = compute_prediction_errors_simple(X_cal, pca_models)
    error_test = compute_prediction_errors_simple(X_test, pca_models)

    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    combined_cal = np.hstack([X_cal_s, error_cal])
    combined_test = np.hstack([X_test_s, error_test])

    clf = RidgeClassifier(alpha=0.1)
    clf.fit(combined_cal, y_cal)
    preds = clf.predict(combined_test)
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

print('PCET Ablation Study (Simplified)', flush=True)
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

            preds_svm = svm_predict(X_cal, y_cal, X_test)
            acc_svm = accuracy_score(y_test, preds_svm)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'Raw_EEG', 'accuracy': acc_svm})

            preds_error = prediction_error_only(X_cal, y_cal, X_test)
            acc_error = accuracy_score(y_test, preds_error)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'Prediction_error_only', 'accuracy': acc_error})

            preds_random = random_predictor_error(X_cal, y_cal, X_test)
            acc_random = accuracy_score(y_test, preds_random)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'Random_predictor_error', 'accuracy': acc_random})

            preds_shuffled = shuffled_predictor_error(X_cal, y_cal, X_test)
            acc_shuffled = accuracy_score(y_test, preds_shuffled)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'Shuffled_predictor_error', 'accuracy': acc_shuffled})

            preds_pcet = pcet_predict(X_cal, y_cal, X_test)
            acc_pcet = accuracy_score(y_test, preds_pcet)
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'Calibration_trained_PCET', 'accuracy': acc_pcet})

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/pcet_ablation_results.csv', index=False)

print('', flush=True)
print('\n' + '='*80, flush=True)
print('PCET Ablation Results Summary', flush=True)
print('='*80, flush=True)

methods = ['Raw_EEG', 'Prediction_error_only', 'Random_predictor_error', 'Shuffled_predictor_error', 'Calibration_trained_PCET']
print('\nComparing methods by shot:', flush=True)
for n_cal in shot_settings:
    print(f'\n  {n_cal}-shot:', flush=True)
    for method in methods:
        acc = df[df['method'] == method][df['n_cal'] == n_cal]['accuracy'].mean()
        print(f'    {method}: {acc:.4f}', flush=True)

print('\nDone!', flush=True)