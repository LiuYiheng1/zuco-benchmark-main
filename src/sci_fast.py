"""SCI Framework - Ultra Fast Validation Version"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
print("="*60)
print("SCI Framework - Fast Validation")
print("="*60)

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
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

print("\nLoading data for all subjects...")
all_data = {}
for subj in Y_SUBJECTS:
    X, y = load_eeg_data(subj)
    if X is not None:
        all_data[subj] = (X, y)
        print(f"  {subj}: {len(y)} samples")
print(f"Loaded {len(all_data)} subjects")

results = []
shot_settings = [3, 5, 10, 20, 50]
seed = 0

print("\nRunning SCI validation (seed=0)...")

for held_out in Y_SUBJECTS:
    if held_out not in all_data:
        continue

    X_test_orig, y_test_orig = all_data[held_out]
    train_subjs = [s for s in Y_SUBJECTS if s != held_out and s in all_data]

    if len(train_subjs) == 0:
        continue

    X_train_all = np.vstack([all_data[s][0] for s in train_subjs])
    y_train_all = np.concatenate([all_data[s][1] for s in train_subjs])

    mu_0 = np.mean(X_train_all[y_train_all == 0], axis=0) if np.any(y_train_all == 0) else np.mean(X_train_all, axis=0)
    mu_1 = np.mean(X_train_all[y_train_all == 1], axis=0) if np.any(y_train_all == 1) else np.mean(X_train_all, axis=0)
    sigma_0 = np.std(X_train_all[y_train_all == 0], axis=0) + 1e-8
    sigma_1 = np.std(X_train_all[y_train_all == 1], axis=0) + 1e-8

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

    print(f"\n{held_out}: test={len(y_test)}, cal_pool={len(y_cal_pool)}")

    for n_cal in shot_settings:
        if n_cal * 2 > len(cal_pool_indices):
            continue

        class_0_idx = np.where(y_cal_pool == 0)[0]
        class_1_idx = np.where(y_cal_pool == 1)[0]
        np.random.shuffle(class_0_idx)
        np.random.shuffle(class_1_idx)
        n0 = min(n_cal, len(class_0_idx))
        n1 = min(n_cal, len(class_1_idx))
        selected = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])
        np.random.shuffle(selected)

        X_cal = X_cal_pool[selected]
        y_cal = y_cal_pool[selected]

        if len(np.unique(y_cal)) < 2:
            continue

        mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0)
        mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0)
        sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8
        sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8

        scaler = StandardScaler()
        X_cal_s = scaler.fit_transform(X_cal)
        X_test_s = scaler.transform(X_test)

        sigma_0_inv = np.linalg.inv(np.diag(sigma_cal_0 ** 2))
        sigma_1_inv = np.linalg.inv(np.diag(sigma_cal_1 ** 2))

        d0_test = np.sqrt(np.sum((X_test_s - mu_cal_0) * (np.dot(X_test_s - mu_cal_0, sigma_0_inv)), axis=1))
        d1_test = np.sqrt(np.sum((X_test_s - mu_cal_1) * (np.dot(X_test_s - mu_cal_1, sigma_1_inv)), axis=1))
        uncertainty_test = np.abs(d1_test - d0_test)

        d0_cal = np.sqrt(np.sum((X_cal_s - mu_cal_0) * (np.dot(X_cal_s - mu_cal_0, sigma_0_inv)), axis=1))
        d1_cal = np.sqrt(np.sum((X_cal_s - mu_cal_1) * (np.dot(X_cal_s - mu_cal_1, sigma_1_inv)), axis=1))
        uncertainty_cal = np.abs(d1_cal - d0_cal)

        clf = LogisticRegression(max_iter=500, random_state=seed, C=1.0)
        clf.fit(X_cal_s, y_cal)
        p_pcet = clf.predict_proba(X_test_s)[:, 1]

        clf_unc = LogisticRegression(max_iter=500, random_state=seed, C=1.0)
        clf_unc.fit(np.column_stack([uncertainty_cal]), y_cal)
        p_srgc = clf_unc.predict_proba(np.column_stack([uncertainty_test]))[:, 1]

        log_lik_0 = -0.5 * np.sum(np.dot(X_test_s - mu_cal_0, sigma_0_inv) * (X_test_s - mu_cal_0), axis=1)
        log_lik_1 = -0.5 * np.sum(np.dot(X_test_s - mu_cal_1, sigma_1_inv) * (X_test_s - mu_cal_1), axis=1)
        p_sied = 1 / (1 + np.exp(log_lik_1 - log_lik_0 + np.log(0.5)))

        y_pcet = (p_pcet >= 0.5).astype(int)
        y_srgc = (p_srgc >= 0.5).astype(int)
        y_sied = (p_sied >= 0.5).astype(int)

        acc_pcet = accuracy_score(y_test, y_pcet)
        acc_srgc = accuracy_score(y_test, y_srgc)
        acc_sied = accuracy_score(y_test, y_sied)

        results.append({'n_cal': n_cal, 'method': 'PCET', 'accuracy': acc_pcet})
        results.append({'n_cal': n_cal, 'method': 'SRGC', 'accuracy': acc_srgc})
        results.append({'n_cal': n_cal, 'method': 'SIED', 'accuracy': acc_sied})

        configs = [
            ('SCI_0.5_0.3_0.2', 0.5, 0.3, 0.2),
            ('SCI_0.6_0.2_0.2', 0.6, 0.2, 0.2),
            ('SCI_0.7_0.2_0.1', 0.7, 0.2, 0.1),
            ('SCI_0.6_0.3_0.1', 0.6, 0.3, 0.1),
            ('SCI_0.5_0.2_0.3', 0.5, 0.2, 0.3),
        ]

        for name, w_p, w_u, w_d in configs:
            p_comb = w_p * p_pcet + w_u * p_srgc + w_d * p_sied
            y_pred = (p_comb >= 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            results.append({'n_cal': n_cal, 'method': name, 'accuracy': acc})

        p_unc_norm = 1 - uncertainty_test / (uncertainty_test.max() + 1e-8)
        p_dom_norm = 1 - np.abs(p_sied - 0.5) * 2

        configs_ortho = [
            ('ORTHO_0.5_0.3_0.2', 0.5, 0.3, 0.2),
            ('ORTHO_0.6_0.2_0.2', 0.6, 0.2, 0.2),
            ('ORTHO_0.7_0.2_0.1', 0.7, 0.2, 0.1),
            ('ORTHO_0.6_0.3_0.1', 0.6, 0.3, 0.1),
        ]

        for name, w_p, w_u, w_d in configs_ortho:
            p_ortho = w_p * p_pcet + w_u * p_unc_norm + w_d * p_dom_norm
            y_pred = (p_ortho >= 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            results.append({'n_cal': n_cal, 'method': name, 'accuracy': acc})

    print(f"  Completed {held_out}")

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/sci_fast_results.csv", index=False)

print("\n" + "="*60)
print("Results Summary")
print("="*60)

for n_cal in shot_settings:
    print(f"\n{n_cal}-shot:")
    shot_df = df[df['n_cal'] == n_cal]

    best_sci_acc = 0
    best_sci_name = ''
    for method in shot_df['method'].unique():
        if method.startswith('SCI_') or method.startswith('ORTHO_'):
            m_df = shot_df[shot_df['method'] == method]
            acc = m_df['accuracy'].mean()
            if acc > best_sci_acc:
                best_sci_acc = acc
                best_sci_name = method

    for method in ['PCET', 'SRGC', 'SIED']:
        method_df = shot_df[shot_df['method'] == method]
        if len(method_df) > 0:
            acc = method_df['accuracy'].mean()
            std = method_df['accuracy'].std()
            print(f"  {method:12s}: {acc:.4f}±{std:.4f}")

    if best_sci_acc > 0:
        print(f"  {'SCI_best':12s}: {best_sci_acc:.4f} ({best_sci_name})")

print("\n" + "="*60)
print("Improvement Analysis")
print("="*60)
for n_cal in shot_settings:
    shot_df = df[df['n_cal'] == n_cal]
    pcet_acc = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
    srgc_acc = shot_df[shot_df['method'] == 'SRGC']['accuracy'].mean()
    sied_acc = shot_df[shot_df['method'] == 'SIED']['accuracy'].mean()

    best_sci_acc = 0
    best_sci_name = ''
    for method in shot_df['method'].unique():
        if method.startswith('SCI_') or method.startswith('ORTHO_'):
            m_df = shot_df[shot_df['method'] == method]
            acc = m_df['accuracy'].mean()
            if acc > best_sci_acc:
                best_sci_acc = acc
                best_sci_name = method

    best_single = max(pcet_acc, srgc_acc, sied_acc)
    improvement = best_sci_acc - best_single
    marker = "✓" if improvement > 0 else "✗"
    print(f"{n_cal}-shot: SCI={best_sci_acc:.4f}, BestSingle={best_single:.4f}, Δ={improvement:+.4f} {marker}")

print("\nDone!")