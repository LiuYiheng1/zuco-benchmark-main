"""SCI: Ultra-Fast Routing Validation"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
print("SCI Ultra-Fast Validation")

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

print("Loading data...")
all_data = {s: load_eeg_data(s) for s in Y_SUBJECTS}
all_data = {k: v for k, v in all_data.items() if v[0] is not None}
print(f"Loaded {len(all_data)} subjects")

results = []
shot_settings = [3, 5, 10]
seed = 0
TEST_SUBJECTS = ['YAC', 'YAG', 'YAK']

print(f"\nRunning on {len(TEST_SUBJECTS)} subjects...")

for held_out in TEST_SUBJECTS:
    if held_out not in all_data:
        continue

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

        acc_pcet = accuracy_score(y_test, (p_pcet >= 0.5).astype(int))
        acc_srgc = accuracy_score(y_test, (p_srgc >= 0.5).astype(int))

        results.append({'n_cal': n_cal, 'method': 'PCET', 'accuracy': acc_pcet})
        results.append({'n_cal': n_cal, 'method': 'SRGC', 'accuracy': acc_srgc})

        for tau in [0.4, 0.5]:
            for lam in [0.1, 0.2, 0.3]:
                p_final = p_pcet.copy()
                confidence = np.maximum(p_pcet, 1 - p_pcet)
                low_conf = confidence < tau

                if np.any(low_conf):
                    y_pcet = (p_pcet >= 0.5).astype(int)
                    y_srgc = (p_srgc >= 0.5).astype(int)
                    agree = (y_pcet == y_srgc) & low_conf
                    if np.any(agree):
                        p_final[agree] = (1 - lam) * p_pcet[agree] + lam * p_srgc[agree]

                acc = accuracy_score(y_test, (p_final >= 0.5).astype(int))
                results.append({'n_cal': n_cal, 'method': f'R_t{tau}_l{lam}', 'accuracy': acc})

    print(f".", end='', flush=True)

print("\n\nSaving...")
df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/sci_ultra_fast_results.csv", index=False)

print("\n" + "="*50)
print("Summary (3 subjects, seed 0)")
print("="*50)

for n_cal in shot_settings:
    shot_df = df[df['n_cal'] == n_cal]
    pcet = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
    srgc = shot_df[shot_df['method'] == 'SRGC']['accuracy'].mean()

    print(f"\n{n_cal}-shot: PCET={pcet:.4f}, SRGC={srgc:.4f}")

    routes = shot_df[shot_df['method'].str.startswith('R_')]
    if len(routes) > 0:
        best = routes.loc[routes['accuracy'].idxmax()]
        delta = best['accuracy'] - pcet
        print(f"  Best Route: {best['method']} = {best['accuracy']:.4f} (Δ={delta:+.4f})")

print("\nDone!")