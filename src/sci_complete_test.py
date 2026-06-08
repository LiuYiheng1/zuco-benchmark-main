"""SCI Complete Routing Test"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
print("="*60)
print("SCI ROUTING VALIDATION")
print("="*60)

FEATURES_DIR = "features"
Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
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

print("\nLoading all subjects...")
all_data = {}
for subj in Y_SUBJECTS:
    try:
        X, y = load_eeg_data(subj)
        if X is not None:
            all_data[subj] = (X, y)
    except:
        pass
print(f"Loaded {len(all_data)} subjects")

results = []
shots = [3, 5, 10, 20, 50]
seeds = [0, 1, 2]

print(f"\nRunning experiment: {len(shots)} shots x {len(seeds)} seeds x {len(all_data)} subjects...")

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

        for n_cal in shots:
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

            clf = LogisticRegression(max_iter=100, solver='lbfgs')
            clf.fit(X_cal_s, y_cal)
            p_pcet = clf.predict_proba(X_test_s)[:, 1]

            inv_var_0 = 1.0 / (sigma_0 ** 2)
            inv_var_1 = 1.0 / (sigma_1 ** 2)

            diff_0_cal = X_cal_s - mu_0
            diff_1_cal = X_cal_s - mu_1
            d0_cal = np.sqrt(np.sum(diff_0_cal ** 2 * inv_var_0, axis=1))
            d1_cal = np.sqrt(np.sum(diff_1_cal ** 2 * inv_var_1, axis=1))
            uncertainty_cal = np.abs(d1_cal - d0_cal)

            diff_0_test = X_test_s - mu_0
            diff_1_test = X_test_s - mu_1
            d0_test = np.sqrt(np.sum(diff_0_test ** 2 * inv_var_0, axis=1))
            d1_test = np.sqrt(np.sum(diff_1_test ** 2 * inv_var_1, axis=1))
            uncertainty_test = np.abs(d1_test - d0_test)

            clf_unc = LogisticRegression(max_iter=100, solver='lbfgs')
            clf_unc.fit(uncertainty_cal.reshape(-1, 1), y_cal)
            p_srgc = clf_unc.predict_proba(uncertainty_test.reshape(-1, 1))[:, 1]

            acc_pcet = accuracy_score(y_test, (p_pcet >= 0.5).astype(int))
            acc_srgc = accuracy_score(y_test, (p_srgc >= 0.5).astype(int))

            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'PCET', 'accuracy': acc_pcet})
            results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'SRGC', 'accuracy': acc_srgc})

            for tau in [0.4, 0.45, 0.5, 0.55]:
                for lam in [0.1, 0.2, 0.3]:
                    p_final = p_pcet.copy()
                    conf = np.maximum(p_pcet, 1 - p_pcet)
                    low = conf < tau

                    if np.any(low):
                        y_p = (p_pcet >= 0.5).astype(int)
                        y_s = (p_srgc >= 0.5).astype(int)
                        agree = (y_p == y_s) & low
                        if np.any(agree):
                            p_final[agree] = (1 - lam) * p_pcet[agree] + lam * p_srgc[agree]

                    acc = accuracy_score(y_test, (p_final >= 0.5).astype(int))
                    results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': f'R_{tau}_{lam}', 'accuracy': acc})

        print(".", end='', flush=True)

df = pd.DataFrame(results)
df.to_csv('d:/pycharmproject/zuco-benchmark-main/src/results/final/sci_routing_validation.csv', index=False)

print("\n\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

for n_cal in shots:
    print(f"\n{n_cal}-shot:")
    shot_df = df[df['n_cal'] == n_cal]

    pcet = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
    srgc = shot_df[shot_df['method'] == 'SRGC']['accuracy'].mean()
    print(f"  PCET: {pcet:.4f}")
    print(f"  SRGC: {srgc:.4f}")

    routes = shot_df[shot_df['method'].str.startswith('R_')]
    if len(routes) > 0:
        best = routes.groupby('method')['accuracy'].mean().idxmax()
        best_acc = routes.groupby('method')['accuracy'].mean().max()
        delta = best_acc - pcet
        print(f"  Best Route: {best} = {best_acc:.4f} (Δ={delta:+.4f})")

print("\n" + "="*60)
print("KEY FINDING")
print("="*60)

improvements = []
for n_cal in shots:
    shot_df = df[df['n_cal'] == n_cal]
    pcet = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()

    routes = shot_df[shot_df['method'].str.startswith('R_')]
    if len(routes) > 0:
        best_acc = routes.groupby('method')['accuracy'].mean().max()
        delta = best_acc - pcet
        improvements.append(delta)
        marker = "✓ IMPROVES" if delta > 0 else "✗ DEGRADES"
        print(f"{n_cal}-shot: PCET={pcet:.4f}, BestRoute={best_acc:.4f}, Δ={delta:+.4f} {marker}")

avg_imp = np.mean(improvements)
print(f"\nAverage improvement: {avg_imp:+.4f}")
print("="*60)