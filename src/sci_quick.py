"""SCI Framework - Quick Test Version"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
print("SCI Quick Test")

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
TEST_SUBJECTS = ['YAC', 'YAG', 'YAK']
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

all_data = {s: load_eeg_data(s) for s in Y_SUBJECTS}
all_data = {k: v for k, v in all_data.items() if v[0] is not None}
print(f"Loaded {len(all_data)} subjects")

results = []
shot_settings = [3, 5, 10]
seed = 0

for held_out in TEST_SUBJECTS:
    if held_out not in all_data:
        continue
    print(f"\nTesting {held_out}...")

    X_test_orig, y_test_orig = all_data[held_out]
    train_subjs = [s for s in Y_SUBJECTS if s != held_out and s in all_data]

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
        print(f"  {n_cal}-shot...", end=' ')

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
            print("skip")
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

        acc_pcet = accuracy_score(y_test, (p_pcet >= 0.5).astype(int))
        acc_srgc = accuracy_score(y_test, (p_srgc >= 0.5).astype(int))
        acc_sied = accuracy_score(y_test, (p_sied >= 0.5).astype(int))

        results.append({'n_cal': n_cal, 'method': 'PCET', 'accuracy': acc_pcet})
        results.append({'n_cal': n_cal, 'method': 'SRGC', 'accuracy': acc_srgc})
        results.append({'n_cal': n_cal, 'method': 'SIED', 'accuracy': acc_sied})

        for name, w_p, w_u, w_d in [('SCI_0.6_0.2_0.2', 0.6, 0.2, 0.2), ('SCI_0.7_0.2_0.1', 0.7, 0.2, 0.1)]:
            p_comb = w_p * p_pcet + w_u * p_srgc + w_d * p_sied
            acc = accuracy_score(y_test, (p_comb >= 0.5).astype(int))
            results.append({'n_cal': n_cal, 'method': name, 'accuracy': acc})

        print(f"PCET={acc_pcet:.3f}, SRGC={acc_srgc:.3f}, SIED={acc_sied:.3f}")

print("\nSaving...")
df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/sci_quick_results.csv", index=False)

print("\n" + "="*50)
print("Summary:")
for n_cal in shot_settings:
    shot_df = df[df['n_cal'] == n_cal]
    pcet = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
    srgc = shot_df[shot_df['method'] == 'SRGC']['accuracy'].mean()
    sied = shot_df[shot_df['method'] == 'SIED']['accuracy'].mean()
    sci = shot_df[shot_df['method'] == 'SCI_0.6_0.2_0.2']['accuracy'].mean()
    print(f"{n_cal}-shot: PCET={pcet:.3f}, SRGC={srgc:.3f}, SIED={sied:.3f}, SCI={sci:.3f}")
print("Done!")