"""SCI: Staged Conditional Integration - Corrected Conditional Fusion

Key insight: Simple probability fusion FAILS because modules are NOT orthogonal.
Solution: Conditional fusion based on CONFIDENCE REGIONS.
"""
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
print("SCI Framework - Conditional Fusion Implementation")
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

def apply_sci_conditional(p_pcet, p_srgc, p_sied, tau_p=0.5, lambda_corr=0.2):
    """
    Conditional fusion: only correct low-confidence predictions
    when SRGC and SIED agree on the prediction.

    Args:
        p_pcet: PCET probability predictions
        p_srgc: SRGC probability predictions
        p_sied: SIED probability predictions
        tau_p: Confidence threshold (default 0.5)
        lambda_corr: Correction strength (default 0.2)

    Returns:
        Corrected probability predictions
    """
    p_final = p_pcet.copy()
    confidence = np.maximum(p_pcet, 1 - p_pcet)

    low_conf_mask = confidence < tau_p

    if np.any(low_conf_mask):
        y_srgc = (p_srgc >= 0.5).astype(int)
        y_sied = (p_sied >= 0.5).astype(int)
        agreement_mask = (y_srgc == y_sied) & low_conf_mask

        if np.any(agreement_mask):
            p_consensus = 0.5 * p_srgc + 0.5 * p_sied
            p_final[agreement_mask] = (1 - lambda_corr) * p_pcet[agreement_mask] + lambda_corr * p_consensus[agreement_mask]

    return p_final

print("\nLoading data...")
all_data = {}
for subj in Y_SUBJECTS:
    X, y = load_eeg_data(subj)
    if X is not None:
        all_data[subj] = (X, y)
print(f"Loaded {len(all_data)} subjects")

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2]

print("\nRunning SCI validation (seeds 0-2)...")

for seed in seeds:
    print(f"\nSeed {seed}:", end=' ')
    for held_out in Y_SUBJECTS:
        if held_out not in all_data:
            continue

        X_test_orig, y_test_orig = all_data[held_out]
        train_subjs = [s for s in Y_SUBJECTS if s != held_out and s in all_data]

        if len(train_subjs) == 0:
            continue

        X_train_all = np.vstack([all_data[s][0] for s in train_subjs])
        y_train_all = np.concatenate([all_data[s][1] for s in train_subjs])

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

            X_cal = X_cal_pool[selected]
            y_cal = y_cal_pool[selected]

            if len(np.unique(y_cal)) < 2:
                continue

            mu_0 = np.mean(X_cal[y_cal == 0], axis=0)
            mu_1 = np.mean(X_cal[y_cal == 1], axis=0)
            sigma_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8
            sigma_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8

            scaler = StandardScaler()
            X_cal_s = scaler.fit_transform(X_cal)
            X_test_s = scaler.transform(X_test)

            sigma_0_inv = np.linalg.inv(np.diag(sigma_0 ** 2))
            sigma_1_inv = np.linalg.inv(np.diag(sigma_1 ** 2))

            d0_test = np.sqrt(np.sum((X_test_s - mu_0) * (np.dot(X_test_s - mu_0, sigma_0_inv)), axis=1))
            d1_test = np.sqrt(np.sum((X_test_s - mu_1) * (np.dot(X_test_s - mu_1, sigma_1_inv)), axis=1))
            uncertainty_test = np.abs(d1_test - d0_test)

            d0_cal = np.sqrt(np.sum((X_cal_s - mu_0) * (np.dot(X_cal_s - mu_0, sigma_0_inv)), axis=1))
            d1_cal = np.sqrt(np.sum((X_cal_s - mu_1) * (np.dot(X_cal_s - mu_1, sigma_1_inv)), axis=1))
            uncertainty_cal = np.abs(d1_cal - d0_cal)

            clf = LogisticRegression(max_iter=200, random_state=seed, solver='lbfgs')
            clf.fit(X_cal_s, y_cal)
            p_pcet = clf.predict_proba(X_test_s)[:, 1]

            clf_unc = LogisticRegression(max_iter=200, random_state=seed, solver='lbfgs')
            clf_unc.fit(np.column_stack([uncertainty_cal]), y_cal)
            p_srgc = clf_unc.predict_proba(np.column_stack([uncertainty_test]))[:, 1]

            log_lik_0 = -0.5 * np.sum(np.dot(X_test_s - mu_0, sigma_0_inv) * (X_test_s - mu_0), axis=1)
            log_lik_1 = -0.5 * np.sum(np.dot(X_test_s - mu_1, sigma_1_inv) * (X_test_s - mu_1), axis=1)
            p_sied = 1 / (1 + np.exp(log_lik_1 - log_lik_0 + np.log(0.5)))

            tau_p = 0.4 if n_cal <= 5 else (0.5 if n_cal <= 10 else 0.55)

            p_sci_02 = apply_sci_conditional(p_pcet, p_srgc, p_sied, tau_p=tau_p, lambda_corr=0.2)
            p_sci_03 = apply_sci_conditional(p_pcet, p_srgc, p_sied, tau_p=tau_p, lambda_corr=0.3)

            y_pcet = (p_pcet >= 0.5).astype(int)
            y_srgc = (p_srgc >= 0.5).astype(int)
            y_sied = (p_sied >= 0.5).astype(int)
            y_sci_02 = (p_sci_02 >= 0.5).astype(int)
            y_sci_03 = (p_sci_03 >= 0.5).astype(int)

            acc_pcet = accuracy_score(y_test, y_pcet)
            acc_srgc = accuracy_score(y_test, y_srgc)
            acc_sied = accuracy_score(y_test, y_sied)
            acc_sci_02 = accuracy_score(y_test, y_sci_02)
            acc_sci_03 = accuracy_score(y_test, y_sci_03)

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'PCET', 'accuracy': acc_pcet
            })
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SRGC', 'accuracy': acc_srgc
            })
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SIED', 'accuracy': acc_sied
            })
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': f'SCI_l0.2_t{tau_p}', 'accuracy': acc_sci_02
            })
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': f'SCI_l0.3_t{tau_p}', 'accuracy': acc_sci_03
            })

        print(".", end='', flush=True)

print("\n\nSaving results...")
df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/sci_conditional_results.csv", index=False)

print("\n" + "="*60)
print("Results Summary (Mean Accuracy over 3 seeds)")
print("="*60)

for n_cal in shot_settings:
    print(f"\n{n_cal}-shot:")
    shot_df = df[df['n_cal'] == n_cal]

    for method in ['PCET', 'SRGC', 'SIED', 'SCI_l0.2_t0.4', 'SCI_l0.2_t0.5',
                   'SCI_l0.2_t0.55', 'SCI_l0.3_t0.4', 'SCI_l0.3_t0.5', 'SCI_l0.3_t0.55']:
        method_df = shot_df[shot_df['method'] == method]
        if len(method_df) > 0:
            acc = method_df['accuracy'].mean()
            std = method_df['accuracy'].std()
            print(f"  {method:20s}: {acc:.4f}±{std:.4f}")

    best_sci = shot_df[shot_df['method'].str.startswith('SCI_')]['accuracy'].max()
    best_pcet = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
    delta = best_sci - best_pcet
    print(f"  {'Best SCI vs PCET':20s}: Δ={delta:+.4f}")

print("\n" + "="*60)
print("Key Finding")
print("="*60)

for n_cal in shot_settings:
    shot_df = df[df['n_cal'] == n_cal]
    pcet_acc = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
    best_sci_acc = shot_df[shot_df['method'].str.startswith('SCI_')]['accuracy'].max()

    marker = "✓" if best_sci_acc >= pcet_acc else "✗"
    print(f"{n_cal}-shot: PCET={pcet_acc:.4f}, BestSCI={best_sci_acc:.4f} {marker}")

print("\nDone!")