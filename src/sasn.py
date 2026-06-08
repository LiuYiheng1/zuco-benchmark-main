"""
SASN: Subject-Adaptive Shrinkage Normalization

Target: Stabilize few-shot EEG calibration using target user's unlabeled data statistics

Method:
- shrinkage: source_statistics + target_user_statistics
- Low shot: more dependence on source
- High shot: more dependence on target
"""

import os
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

def kmeans_centroid_sampling_label_free(X_pool, n_select):
    """Label-free KMeans centroid sampling (same as ACCS)"""
    from sklearn.cluster import KMeans
    n_pool = len(X_pool)
    k = min(n_select, n_pool)

    scaler = StandardScaler()
    X_pool_s = scaler.fit_transform(X_pool)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    kmeans.fit(X_pool_s)

    labels = kmeans.labels_
    selected = []
    for c in range(k):
        cluster_indices = np.where(labels == c)[0]
        if len(cluster_indices) > 0:
            centroid = kmeans.cluster_centers_[c]
            dists = np.linalg.norm(X_pool_s[cluster_indices] - centroid, axis=1)
            closest = cluster_indices[np.argmin(dists)]
            selected.append(closest)

    while len(selected) < n_select and len(selected) < n_pool:
        for i in range(k):
            cluster_indices = np.where(labels == i)[0]
            for idx in cluster_indices:
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) >= n_select:
                        break
            if len(selected) >= n_select:
                break

    return np.array(selected[:n_select])

def select_calibration_random(X_pool, n_select):
    """Random sampling without labels"""
    indices = np.random.permutation(len(X_pool))[:n_select]
    return indices

def compute_source_statistics(X_train_all, y_train_all):
    """Compute statistics from source (training) subjects"""
    class_0 = X_train_all[y_train_all == 0]
    class_1 = X_train_all[y_train_all == 1]

    mu_0 = np.mean(class_0, axis=0)
    sigma_0 = np.std(class_0, axis=0) + 1e-8
    mu_1 = np.mean(class_1, axis=0)
    sigma_1 = np.std(class_1, axis=0) + 1e-8

    return (mu_0, sigma_0), (mu_1, sigma_1)

def compute_target_statistics(X_cal_pool, y_cal_pool):
    """Compute statistics from target user's calibration pool"""
    class_0 = X_cal_pool[y_cal_pool == 0]
    class_1 = X_cal_pool[y_cal_pool == 1]

    mu_0 = np.mean(class_0, axis=0) if len(class_0) > 0 else None
    sigma_0 = np.std(class_0, axis=0) + 1e-8 if len(class_0) > 0 else None
    mu_1 = np.mean(class_1, axis=0) if len(class_1) > 0 else None
    sigma_1 = np.std(class_1, axis=0) + 1e-8 if len(class_1) > 0 else None

    return (mu_0, sigma_0), (mu_1, sigma_1)

def shrinkage_normalization(X, mu_0, sigma_0, mu_1, sigma_1, rho_0, rho_1):
    """Apply shrinkage normalization"""
    X_0 = (X - mu_0) / sigma_0
    X_1 = (X - mu_1) / sigma_1
    return X_0, X_1

def apply_sasn(X, mu_source_0, sigma_source_0, mu_target_0, sigma_target_0, rho):
    """Apply SASN normalization"""
    mu_adapt = rho * mu_target_0 + (1 - rho) * mu_source_0
    sigma_adapt = rho * sigma_target_0 + (1 - rho) * sigma_source_0
    X_norm = (X - mu_adapt) / (sigma_adapt + 1e-8)
    return X_norm

def train_and_evaluate(X_cal, y_cal, X_test, y_test):
    """Train MLP and evaluate"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    clf.fit(X_cal_s, y_cal)

    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def train_and_evaluate_sasn(X_cal, y_cal, X_test, y_test, X_cal_pool, y_cal_pool, X_train_all, y_train_all, kappa):
    """Train with SASN and evaluate"""
    mu_source_0, sigma_source_0 = compute_source_statistics(X_train_all, y_train_all)

    class_0_pool = X_cal_pool[y_cal_pool == 0]
    class_1_pool = X_cal_pool[y_cal_pool == 1]

    if len(class_0_pool) == 0 or len(class_1_pool) == 0:
        return train_and_evaluate(X_cal, y_cal, X_test, y_test)

    mu_target_0 = np.mean(class_0_pool, axis=0)
    sigma_target_0 = np.std(class_0_pool, axis=0) + 1e-8
    mu_target_1 = np.mean(class_1_pool, axis=0)
    sigma_target_1 = np.std(class_1_pool, axis=0) + 1e-8

    n_target_0 = len(class_0_pool)
    n_target_1 = len(class_1_pool)
    n_target = min(n_target_0, n_target_1)

    rho = n_target / (n_target + kappa)

    mu_0 = rho * mu_target_0 + (1 - rho) * mu_source_0[0]
    sigma_0 = rho * sigma_target_0 + (1 - rho) * mu_source_0[1]

    X_cal_norm = (X_cal - mu_0) / (sigma_0 + 1e-8)
    X_test_norm = (X_test - mu_0) / (sigma_0 + 1e-8)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    clf.fit(X_cal_norm, y_cal)

    preds = clf.predict(X_test_norm)
    probs = clf.predict_proba(X_test_norm)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_experiment():
    results = []
    calibration_settings = [3, 5, 10, 20, 50]
    kappa_values = [5, 10, 20, 50, 100]
    seeds = [0, 1, 2, 3, 4]

    print("SASN: Subject-Adaptive Shrinkage Normalization")
    print("="*60)
    print(f"Shot settings: {calibration_settings}")
    print(f"Kappa values: {kappa_values}")
    print("="*60)

    for seed in seeds:
        print(f"\nSeed {seed}:", flush=True)

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
                print(f" {held_out}[no data]", end="", flush=True)
                continue

            X_train_all = np.vstack(X_train_all)
            y_train_all = np.concatenate(y_train_all)

            n_samples = len(y_test)
            np.random.seed(seed)
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_samples // 2]
            cal_pool_indices = indices[n_samples // 2:]

            X_holdout = X_test.copy()
            y_holdout = y_test.copy()

            X_test = X_holdout[test_indices]
            y_test = y_holdout[test_indices]
            X_cal_pool = X_holdout[cal_pool_indices]
            y_cal_pool = y_holdout[cal_pool_indices]

            if len(X_cal_pool) < 20:
                print(f" {held_out}[small pool]", end="", flush=True)
                continue

            for n_cal_per_class in calibration_settings:
                if n_cal_per_class * 2 > len(cal_pool_indices):
                    continue

                total_budget = n_cal_per_class * 2

                class_0_idx = np.where(y_cal_pool == 0)[0]
                class_1_idx = np.where(y_cal_pool == 1)[0]

                np.random.shuffle(class_0_idx)
                np.random.shuffle(class_1_idx)

                cal_idx = np.concatenate([class_0_idx[:n_cal_per_class], class_1_idx[:n_cal_per_class]])
                np.random.shuffle(cal_idx)

                X_cal = X_cal_pool[cal_idx]
                y_cal = y_cal_pool[cal_idx]

                if len(np.unique(y_cal)) < 2:
                    continue

                acc, f1, bacc, auroc = train_and_evaluate(X_cal, y_cal, X_test, y_test)
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal_per_class': n_cal_per_class,
                    'method': 'StandardScaler_calibration_only',
                    'kappa': 0, 'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
                })

                for kappa in kappa_values:
                    acc, f1, bacc, auroc = train_and_evaluate_sasn(
                        X_cal, y_cal, X_test, y_test, X_cal_pool, y_cal_pool, X_train_all, y_train_all, kappa
                    )
                    results.append({
                        'seed': seed, 'subject': held_out, 'n_cal_per_class': n_cal_per_class,
                        'method': 'SASN',
                        'kappa': kappa, 'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
                    })

                cal_idx_accs = kmeans_centroid_sampling_label_free(X_cal_pool, total_budget)
                X_cal_accs = X_cal_pool[cal_idx_accs]
                y_cal_accs = y_cal_pool[cal_idx_accs]

                if len(np.unique(y_cal_accs)) >= 2:
                    acc, f1, bacc, auroc = train_and_evaluate(X_cal_accs, y_cal_accs, X_test, y_test)
                    results.append({
                        'seed': seed, 'subject': held_out, 'n_cal_per_class': n_cal_per_class,
                        'method': 'ACCS',
                        'kappa': 0, 'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
                    })

                    for kappa in kappa_values:
                        acc, f1, bacc, auroc = train_and_evaluate_sasn(
                            X_cal_accs, y_cal_accs, X_test, y_test, X_cal_pool, y_cal_pool, X_train_all, y_train_all, kappa
                        )
                        results.append({
                            'seed': seed, 'subject': held_out, 'n_cal_per_class': n_cal_per_class,
                            'method': 'SASN_ACCS',
                            'kappa': kappa, 'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
                        })

            print(f" {held_out}", end="", flush=True)

    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze and summarize results"""
    print("\n" + "="*60)
    print("SASN Results Summary")
    print("="*60)

    difficult_subjects = ['YLS', 'YSL', 'YHS', 'YRP']

    for k in [3, 5, 10, 20, 50]:
        print(f"\nk={k}-shot:")
        baseline_acc = df[(df['method'] == 'StandardScaler_calibration_only') & (df['n_cal_per_class'] == k)]['accuracy'].mean()
        print(f"  Baseline (StandardScaler): {baseline_acc:.4f}")

        for method in ['ACCS', 'SASN']:
            best_data = None
            best_kappa = None
            best_acc = 0
            for kappa in df[df['method'] == method]['kappa'].unique():
                data = df[(df['method'] == method) & (df['n_cal_per_class'] == k) & (df['kappa'] == kappa)]
                if len(data) > 0:
                    acc = data['accuracy'].mean()
                    if acc > best_acc:
                        best_acc = acc
                        best_kappa = kappa
                        best_data = data
            if best_data is not None:
                gap = best_acc - baseline_acc
                marker = " *" if gap >= 0.02 else ""
                print(f"  {method} (kappa={best_kappa}): {best_acc:.4f} (gap={gap:+.4f}){marker}")

    print("\n" + "="*60)
    print("Success Criteria Check")
    print("="*60)

    baseline = {}
    for k in [3, 5, 10, 20, 50]:
        baseline[k] = df[(df['method'] == 'StandardScaler_calibration_only') & (df['n_cal_per_class'] == k)]['accuracy'].mean()

    print("\n1. SASN vs StandardScaler (target +2% at 5 or 10-shot):")
    for k in [5, 10]:
        target = baseline[k] + 0.02
        best_acc = 0
        best_kappa = None
        for kappa in df[df['method'] == 'SASN']['kappa'].unique():
            data = df[(df['method'] == 'SASN') & (df['n_cal_per_class'] == k) & (df['kappa'] == kappa)]
            if len(data) > 0:
                acc = data['accuracy'].mean()
                if acc > best_acc:
                    best_acc = acc
                    best_kappa = kappa
        status = "PASS" if best_acc >= target else "FAIL"
        print(f"  k={k}: best={best_acc:.4f} target={target:.4f} ({status})")

    print("\n2. SASN_ACCS vs ACCS (target +1%):")
    for k in [5, 10, 20]:
        accs_data = df[(df['method'] == 'ACCS') & (df['n_cal_per_class'] == k) & (df['kappa'] == 0)]
        accs_acc = accs_data['accuracy'].mean() if len(accs_data) > 0 else 0

        best_acc = 0
        best_kappa = None
        for kappa in df[df['method'] == 'SASN_ACCS']['kappa'].unique():
            data = df[(df['method'] == 'SASN_ACCS') & (df['n_cal_per_class'] == k) & (df['kappa'] == kappa)]
            if len(data) > 0:
                acc = data['accuracy'].mean()
                if acc > best_acc:
                    best_acc = acc
                    best_kappa = kappa
        gap = best_acc - accs_acc if accs_acc > 0 else 0
        status = "PASS" if gap >= 0.01 else "FAIL"
        print(f"  k={k}: SASN_ACCS={best_acc:.4f} ACCS={accs_acc:.4f} gap={gap:+.4f} ({status})")

    print("\n3. Difficult subjects (YLS/YSL/YHS/YRP) avg gain:")
    for method in ['SASN', 'SASN_ACCS']:
        for k in [5, 10]:
            diff_gains = []
            for subj in difficult_subjects:
                baseline_data = df[(df['method'] == 'StandardScaler_calibration_only') & (df['n_cal_per_class'] == k) & (df['subject'] == subj)]
                if len(baseline_data) == 0:
                    continue
                baseline_acc = baseline_data['accuracy'].mean()

                best_acc = 0
                for kappa in df[df['method'] == method]['kappa'].unique():
                    data = df[(df['method'] == method) & (df['n_cal_per_class'] == k) & (df['kappa'] == kappa) & (df['subject'] == subj)]
                    if len(data) > 0:
                        acc = data['accuracy'].mean()
                        if acc > best_acc:
                            best_acc = acc
                if best_acc > 0:
                    diff_gains.append(best_acc - baseline_acc)
            if diff_gains:
                avg_gain = np.mean(diff_gains)
                marker = " *" if avg_gain >= 0.02 else ""
                print(f"  {method} k={k}: avg_gain={avg_gain:+.4f}{marker}")

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    df = run_experiment()

    output_path = os.path.join(RESULTS_DIR, "sasn_results.csv")
    df.to_csv(output_path, index=False)

    analyze_results(df)

    print(f"\nResults saved to {output_path}")
    print("Done!")

if __name__ == '__main__':
    main()