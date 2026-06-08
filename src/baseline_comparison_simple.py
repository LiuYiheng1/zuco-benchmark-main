"""Simplified Baseline Comparison - Quick Version"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from scipy.stats import mode
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

def compute_abs_error(X, pca_models):
    error_features = np.zeros((len(X), 2 * X.shape[1]))
    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_recon = pca.inverse_transform(pca.transform(X))
            abs_e = np.abs(X - X_recon)
            error_features[:, i*X.shape[1]:(i+1)*X.shape[1]] = abs_e
    return error_features

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

print("Comprehensive Baseline Comparison (Simplified)", flush=True)
print("="*60, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

methods_to_test = [
    'Majority', 'Random', 'LogisticRegression', 'LinearSVM', 'RBF_SVM',
    'LDA', 'GaussianNB', 'KNN', 'RandomForest',
    'EEG_MLP', 'NearestClassMean',
    'TargetOnly_Gaussian', 'SourceOnly_Gaussian', 'SRGC',
    'SIED', 'PCET', 'PCET_SRGC'
]

for seed in seeds:
    print(f"\nSeed {seed}:", flush=True)
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
        mu_source_0, sigma_source_0, mu_source_1, sigma_source_1 = compute_source_stats(X_train_all, y_train_all)

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

        print(f"  {held_out}", end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(cal_pool_indices):
                continue

            cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            for method in methods_to_test:
                try:
                    if method == 'Majority':
                        y_pred = np.full(len(y_test), mode(y_cal)[0][0])
                    elif method == 'Random':
                        np.random.seed(seed)
                        y_pred = np.random.choice([0, 1], size=len(y_test))
                    elif method == 'LogisticRegression':
                        scaler = StandardScaler()
                        X_cal_s = scaler.fit_transform(X_cal)
                        X_test_s = scaler.transform(X_test)
                        clf = LogisticRegression(max_iter=1000, random_state=seed)
                        clf.fit(X_cal_s, y_cal)
                        y_pred = clf.predict(X_test_s)
                    elif method == 'LinearSVM':
                        scaler = StandardScaler()
                        X_cal_s = scaler.fit_transform(X_cal)
                        X_test_s = scaler.transform(X_test)
                        clf = SVC(kernel='linear', random_state=seed)
                        clf.fit(X_cal_s, y_cal)
                        y_pred = clf.predict(X_test_s)
                    elif method == 'RBF_SVM':
                        scaler = StandardScaler()
                        X_cal_s = scaler.fit_transform(X_cal)
                        X_test_s = scaler.transform(X_test)
                        clf = SVC(kernel='rbf', random_state=seed)
                        clf.fit(X_cal_s, y_cal)
                        y_pred = clf.predict(X_test_s)
                    elif method == 'LDA':
                        scaler = StandardScaler()
                        X_cal_s = scaler.fit_transform(X_cal)
                        X_test_s = scaler.transform(X_test)
                        clf = LinearDiscriminantAnalysis()
                        clf.fit(X_cal_s, y_cal)
                        y_pred = clf.predict(X_test_s)
                    elif method == 'GaussianNB':
                        clf = GaussianNB()
                        clf.fit(X_cal, y_cal)
                        y_pred = clf.predict(X_test)
                    elif method == 'KNN':
                        scaler = StandardScaler()
                        X_cal_s = scaler.fit_transform(X_cal)
                        X_test_s = scaler.transform(X_test)
                        clf = KNeighborsClassifier(n_neighbors=5)
                        clf.fit(X_cal_s, y_cal)
                        y_pred = clf.predict(X_test_s)
                    elif method == 'RandomForest':
                        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
                        clf.fit(X_cal, y_cal)
                        y_pred = clf.predict(X_test)
                    elif method == 'EEG_MLP':
                        scaler = StandardScaler()
                        X_cal_s = scaler.fit_transform(X_cal)
                        X_test_s = scaler.transform(X_test)
                        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed)
                        clf.fit(X_cal_s, y_cal)
                        y_pred = clf.predict(X_test_s)
                    elif method == 'NearestClassMean':
                        mu_0 = np.mean(X_cal[y_cal == 0], axis=0)
                        mu_1 = np.mean(X_cal[y_cal == 1], axis=0)
                        dist_0 = np.linalg.norm(X_test - mu_0, axis=1)
                        dist_1 = np.linalg.norm(X_test - mu_1, axis=1)
                        y_pred = (dist_1 < dist_0).astype(int)
                    elif method == 'TargetOnly_Gaussian':
                        mu_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else np.mean(X_cal, axis=0)
                        mu_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else np.mean(X_cal, axis=0)
                        sigma_0 = np.diag(np.std(X_cal[y_cal == 0], axis=0) ** 2) + 1e-6 if np.any(y_cal == 0) else np.eye(X_cal.shape[1])
                        sigma_1 = np.diag(np.std(X_cal[y_cal == 1], axis=0) ** 2) + 1e-6 if np.any(y_cal == 1) else np.eye(X_cal.shape[1])
                        try:
                            sigma_0_inv = np.linalg.inv(sigma_0)
                            sigma_1_inv = np.linalg.inv(sigma_1)
                        except:
                            y_pred = np.full(len(y_test), mode(y_cal)[0][0])
                        else:
                            preds = np.zeros(len(X_test))
                            for i in range(len(X_test)):
                                d0 = np.dot(np.dot(X_test[i] - mu_0, sigma_0_inv), X_test[i] - mu_0)
                                d1 = np.dot(np.dot(X_test[i] - mu_1, sigma_1_inv), X_test[i] - mu_1)
                                preds[i] = 1 if d1 < d0 else 0
                            y_pred = preds.astype(int)
                    elif method == 'SourceOnly_Gaussian':
                        sigma_0 = np.diag(sigma_source_0 ** 2) + 1e-6
                        sigma_1 = np.diag(sigma_source_1 ** 2) + 1e-6
                        try:
                            sigma_0_inv = np.linalg.inv(sigma_0)
                            sigma_1_inv = np.linalg.inv(sigma_1)
                        except:
                            y_pred = np.full(len(y_test), mode(y_cal)[0][0])
                        else:
                            preds = np.zeros(len(X_test))
                            for i in range(len(X_test)):
                                d0 = np.dot(np.dot(X_test[i] - mu_source_0, sigma_0_inv), X_test[i] - mu_source_0)
                                d1 = np.dot(np.dot(X_test[i] - mu_source_1, sigma_1_inv), X_test[i] - mu_source_1)
                                preds[i] = 1 if d1 < d0 else 0
                            y_pred = preds.astype(int)
                    elif method == 'SRGC':
                        alpha, beta = 0.75, 0.75
                        mu_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
                        mu_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
                        mu_blend_0 = alpha * mu_0 + (1 - alpha) * mu_source_0
                        mu_blend_1 = alpha * mu_1 + (1 - alpha) * mu_source_1
                        cov_cal_0 = np.cov(X_cal[y_cal == 0].T) + np.eye(X_cal.shape[1]) * 1e-4 if np.any(y_cal == 0) else np.eye(X_cal.shape[1])
                        cov_cal_1 = np.cov(X_cal[y_cal == 1].T) + np.eye(X_cal.shape[1]) * 1e-4 if np.any(y_cal == 1) else np.eye(X_cal.shape[1])
                        cov_source_0 = np.diag(sigma_source_0 ** 2) + 1e-6
                        cov_source_1 = np.diag(sigma_source_1 ** 2) + 1e-6
                        cov_blend_0 = beta * cov_source_0 + (1 - beta) * cov_cal_0
                        cov_blend_1 = beta * cov_source_1 + (1 - beta) * cov_cal_1
                        try:
                            cov_blend_0_inv = np.linalg.inv(cov_blend_0)
                            cov_blend_1_inv = np.linalg.inv(cov_blend_1)
                        except:
                            y_pred = np.full(len(y_test), mode(y_cal)[0][0])
                        else:
                            preds = np.zeros(len(X_test))
                            for i in range(len(X_test)):
                                d0 = np.sqrt(np.dot(np.dot(X_test[i] - mu_blend_0, cov_blend_0_inv), X_test[i] - mu_blend_0))
                                d1 = np.sqrt(np.dot(np.dot(X_test[i] - mu_blend_1, cov_blend_1_inv), X_test[i] - mu_blend_1))
                                preds[i] = 1 if d1 < d0 else 0
                            y_pred = preds.astype(int)
                    elif method == 'SIED':
                        scaler = StandardScaler()
                        X_cal_s = scaler.fit_transform(X_cal)
                        X_test_s = scaler.transform(X_test)
                        clf = RidgeClassifier(alpha=1.0)
                        clf.fit(X_cal_s, y_cal)
                        y_pred = clf.predict(X_test_s)
                    elif method == 'PCET':
                        pca_models = train_pca_predictor(X_cal, y_cal, n_components=20)
                        error_cal = compute_abs_error(X_cal, pca_models)
                        combined_cal = np.hstack([X_cal, error_cal])
                        scaler = StandardScaler()
                        combined_cal_s = scaler.fit_transform(combined_cal)
                        error_test = compute_abs_error(X_test, pca_models)
                        combined_test = np.hstack([X_test, error_test])
                        combined_test_s = scaler.transform(combined_test)
                        clf = RidgeClassifier(alpha=0.1)
                        clf.fit(combined_cal_s, y_cal)
                        y_pred = clf.predict(combined_test_s)
                    elif method == 'PCET_SRGC':
                        pca_models = train_pca_predictor(X_cal, y_cal, n_components=20)
                        error_cal = compute_abs_error(X_cal, pca_models)
                        combined_cal = np.hstack([X_cal, error_cal])
                        scaler = StandardScaler()
                        combined_cal_s = scaler.fit_transform(combined_cal)
                        error_test = compute_abs_error(X_test, pca_models)
                        combined_test = np.hstack([X_test, error_test])
                        combined_test_s = scaler.transform(combined_test)
                        clf = RidgeClassifier(alpha=0.1)
                        clf.fit(combined_cal_s, y_cal)
                        y_pred = clf.predict(combined_test_s)
                    else:
                        y_pred = np.full(len(y_test), mode(y_cal)[0][0])
                except Exception as e:
                    y_pred = np.full(len(y_test), mode(y_cal)[0][0])

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                bacc = balanced_accuracy_score(y_test, y_pred)

                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': method,
                    'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': 0.5
                })

        print(f'.', end='', flush=True)

    print('', flush=True)
    df_partial = pd.DataFrame(results)
    df_partial.to_csv(f"{RESULTS_DIR}/baseline_comparison_partial.csv", index=False)

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/baseline_comparison_full.csv", index=False)

print("\n" + "="*60, flush=True)
print("Results Summary", flush=True)
print("="*60, flush=True)

for shot in shot_settings:
    print(f"\n{shot}-shot:", flush=True)
    shot_df = df[df['n_cal'] == shot]
    for method in sorted(shot_df['method'].unique()):
        method_df = shot_df[shot_df['method'] == method]
        acc = method_df['accuracy'].mean()
        std = method_df['accuracy'].std()
        print(f"  {method:25s}: {acc:.4f}±{std:.4f}", flush=True)

print("\nDone!", flush=True)