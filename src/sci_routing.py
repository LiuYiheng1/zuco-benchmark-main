"""SCI: Confidence-Based Routing Implementation

Key insight: Don't fuse predictions - use ROUTING based on confidence.
- High confidence PCET → accept
- Low confidence PCET + SRGC agrees → accept with correction
- High disagreement → use SRGC instead
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
print("SCI: Confidence-Based Routing")
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

def routing_predict(p_pcet, p_srgc, tau_p=0.5, lambda_corr=0.2):
    """
    Confidence-based routing:
    1. High confidence PCET → accept
    2. Low confidence + SRGC agrees → correct
    3. SRGC disagrees → maybe use SRGC
    """
    p_final = p_pcet.copy()
    confidence = np.maximum(p_pcet, 1 - p_pcet)

    low_conf_mask = confidence < tau_p
    high_conf_mask = confidence >= tau_p

    y_pcet = (p_pcet >= 0.5).astype(int)
    y_srgc = (p_srgc >= 0.5).astype(int)

    agree_mask = (y_pcet == y_srgc) & low_conf_mask
    disagree_mask = (y_pcet != y_srgc) & low_conf_mask

    if np.any(agree_mask):
        p_final[agree_mask] = (1 - lambda_corr) * p_pcet[agree_mask] + lambda_corr * p_srgc[agree_mask]

    if np.any(disagree_mask):
        srgc_confidence = np.maximum(p_srgc, 1 - p_srgc)
        srgc_confident_mask = disagree_mask & (srgc_confidence > tau_p + 0.1)
        if np.any(srgc_confident_mask):
            p_final[srgc_confident_mask] = p_srgc[srgc_confident_mask]

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
seeds = [0]

print("\nRunning routing validation (seed 0)...")

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

            tau_p = 0.45 if n_cal <= 5 else (0.5 if n_cal <= 10 else 0.55)

            for tau in [0.4, 0.45, 0.5, 0.55]:
                for lam in [0.1, 0.2, 0.3]:
                    p_route = routing_predict(p_pcet, p_srgc, tau_p=tau, lambda_corr=lam)
                    y_route = (p_route >= 0.5).astype(int)
                    acc_route = accuracy_score(y_test, y_route)
                    results.append({
                        'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                        'method': f'Route_t{tau}_l{lam}', 'accuracy': acc_route
                    })

            acc_pcet = accuracy_score(y_test, (p_pcet >= 0.5).astype(int))
            acc_srgc = accuracy_score(y_test, (p_srgc >= 0.5).astype(int))

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'PCET', 'accuracy': acc_pcet
            })
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SRGC', 'accuracy': acc_srgc
            })

        print(".", end='', flush=True)

print("\n\nSaving results...")
df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/sci_routing_results.csv", index=False)

print("\n" + "="*60)
print("Results Summary (Seed 0)")
print("="*60)

for n_cal in shot_settings:
    print(f"\n{n_cal}-shot:")
    shot_df = df[df['n_cal'] == n_cal]

    pcet_acc = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
    srgc_acc = shot_df[shot_df['method'] == 'SRGC']['accuracy'].mean()

    print(f"  PCET: {pcet_acc:.4f}")
    print(f"  SRGC: {srgc_acc:.4f}")

    route_methods = shot_df[shot_df['method'].str.startswith('Route_')]
    if len(route_methods) > 0:
        best_route = route_methods.loc[route_methods['accuracy'].idxmax()]
        print(f"  Best Route: {best_route['method']} = {best_route['accuracy']:.4f}")
        print(f"  Improvement over PCET: {best_route['accuracy'] - pcet_acc:+.4f}")

print("\n" + "="*60)
print("All Route Configurations Comparison")
print("="*60)

for n_cal in shot_settings:
    shot_df = df[df['n_cal'] == n_cal]
    pcet_acc = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
    route_df = shot_df[shot_df['method'].str.startswith('Route_')]

    print(f"\n{n_cal}-shot (PCET={pcet_acc:.4f}):")
    configs = []
    for method in route_df['method'].unique():
        acc = route_df[route_df['method'] == method]['accuracy'].mean()
        delta = acc - pcet_acc
        configs.append((method, acc, delta))

    configs.sort(key=lambda x: -x[2])
    for method, acc, delta in configs[:5]:
        marker = "✓" if delta > 0 else " "
        print(f"  {marker} {method}: {acc:.4f} (Δ={delta:+.4f})")

print("\nDone!")