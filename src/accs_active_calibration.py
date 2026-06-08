"""
ACCS: Active Cognitive Calibration Sampling - Label-Free Version

Protocol A: Realistic label-free budget
- Select total budget = 2k samples from calibration pool WITHOUT using labels
- KMeans on entire X_pool (no label information)

Protocol B: Balanced simulation (controlled analysis only)
- Each class k-shot
- For fair comparison with existing few-shot results
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

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
    """LABEL-FREE KMeans centroid sampling"""
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
    """Random sampling (no labels used)"""
    indices = np.random.permutation(len(X_pool))[:n_select]
    return indices

def train_and_evaluate(X_cal, y_cal, X_test, y_test):
    """Train MLP and evaluate"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    clf.fit(X_cal_s, y_cal)
    preds = clf.predict(X_test_s)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_experiment():
    results = []
    calibration_settings = [1, 3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    print("ACCS: Active Cognitive Calibration Sampling (Label-Free)")
    print("="*60)
    print("Protocol A: Label-free total budget selection (MAIN RESULT)")
    print("Protocol B: Balanced simulation (controlled analysis)")
    print("="*60)

    for seed in seeds:
        print(f"\nSeed {seed}:", flush=True)

        for held_out in Y_SUBJECTS:
            X_eeg, y_eeg = load_eeg_data(held_out)
            if X_eeg is None or len(X_eeg) < 50:
                continue

            n_samples = len(y_eeg)
            np.random.seed(seed)
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_samples // 2]
            cal_pool_indices = indices[n_samples // 2:]

            X_test = X_eeg[test_indices]
            y_test = y_eeg[test_indices]
            X_cal_pool = X_eeg[cal_pool_indices]
            y_cal_pool = y_eeg[cal_pool_indices]

            for n_cal_per_class in calibration_settings:
                if n_cal_per_class * 2 > len(cal_pool_indices):
                    continue

                total_budget = n_cal_per_class * 2

                # Protocol A: Label-free (MAIN)
                # Select total_budget samples without using labels
                cal_idx_random_A = select_calibration_random(X_cal_pool, total_budget)
                X_cal_A = X_cal_pool[cal_idx_random_A]
                y_cal_A = y_cal_pool[cal_idx_random_A]

                cal_idx_kmeans_A = kmeans_centroid_sampling_label_free(X_cal_pool, total_budget)
                X_cal_kmeans_A = X_cal_pool[cal_idx_kmeans_A]
                y_cal_kmeans_A = y_cal_pool[cal_idx_kmeans_A]

                # Protocol B: Balanced (controlled)
                # Ensure each class has k samples
                class_0_idx = np.where(y_cal_pool == 0)[0]
                class_1_idx = np.where(y_cal_pool == 1)[0]

                np.random.shuffle(class_0_idx)
                np.random.shuffle(class_1_idx)
                cal_idx_random_B = np.concatenate([class_0_idx[:n_cal_per_class], class_1_idx[:n_cal_per_class]])
                np.random.shuffle(cal_idx_random_B)
                X_cal_B = X_cal_pool[cal_idx_random_B]
                y_cal_B = y_cal_pool[cal_idx_random_B]

                # Protocol B KMeans: per-class KMeans (uses labels for balancing)
                kmeans_0 = kmeans_centroid_sampling_label_free(X_cal_pool[class_0_idx], n_cal_per_class)
                kmeans_1 = kmeans_centroid_sampling_label_free(X_cal_pool[class_1_idx], n_cal_per_class)
                cal_idx_kmeans_B = np.concatenate([class_0_idx[kmeans_0], class_1_idx[kmeans_1]])
                np.random.shuffle(cal_idx_kmeans_B)
                X_cal_kmeans_B = X_cal_pool[cal_idx_kmeans_B]
                y_cal_kmeans_B = y_cal_pool[cal_idx_kmeans_B]

                # Evaluate all
                configs = [
                    ('ProtocolA', 'Random_label_free', X_cal_A, y_cal_A),
                    ('ProtocolA', 'KMeans_label_free', X_cal_kmeans_A, y_cal_kmeans_A),
                    ('ProtocolB', 'Random_balanced', X_cal_B, y_cal_B),
                    ('ProtocolB', 'KMeans_balanced', X_cal_kmeans_B, y_cal_kmeans_B),
                ]

                for protocol, method_name, X_cal, y_cal in configs:
                    acc, f1, bacc, auroc = train_and_evaluate(X_cal, y_cal, X_test, y_test)

                    results.append({
                        'seed': seed,
                        'subject': held_out,
                        'n_cal_per_class': n_cal_per_class,
                        'n_cal_total': total_budget,
                        'protocol': protocol,
                        'method': method_name,
                        'accuracy': acc,
                        'macro_f1': f1,
                        'balanced_accuracy': bacc,
                        'auroc': auroc
                    })

            print(f" {held_out}", end="", flush=True)

    return pd.DataFrame(results)

def run_significance_tests(df):
    """Run paired Wilcoxon tests"""
    print("\n" + "="*60)
    print("Statistical Significance Tests (Paired Wilcoxon)")
    print("="*60)

    sig_results = []

    for k in [3, 5, 10, 20]:
        for protocol in ['A', 'B']:
            random_mask = (df['protocol'] == 'A') & (df['method'].str.contains('Random')) & (df['n_cal_per_class'] == k)
            kmeans_mask = (df['protocol'] == 'A') & (df['method'].str.contains('KMeans')) & (df['n_cal_per_class'] == k)

            random_data = df[random_mask]
            kmeans_data = df[kmeans_mask]

            random_acc = random_data.groupby('subject')['accuracy'].mean()
            kmeans_acc = kmeans_data.groupby('subject')['accuracy'].mean()

            common_subjects = random_acc.index.intersection(kmeans_acc.index)
            if len(common_subjects) >= 5:
                random_vals = random_acc[common_subjects].values.astype(float)
                kmeans_vals = kmeans_acc[common_subjects].values.astype(float)

                diffs = kmeans_vals - random_vals
                if np.all(np.abs(diffs) < 1e-10):
                    print(f"k={k:2d}, Protocol {protocol}: No difference between methods")
                    continue

                try:
                    stat, pval = wilcoxon(kmeans_vals, random_vals, alternative='greater')
                    sig_results.append({
                        'k_shot': k,
                        'protocol': protocol,
                        'random_mean': np.mean(random_vals),
                        'kmeans_mean': np.mean(kmeans_vals),
                        'gap': np.mean(kmeans_vals) - np.mean(random_vals),
                        'p_value': pval,
                        'significant': pval < 0.05
                    })
                    print(f"k={k:2d}, Protocol {protocol}: Random={np.mean(random_vals):.4f}, KMeans={np.mean(kmeans_vals):.4f}, Gap={np.mean(kmeans_vals)-np.mean(random_vals):+.4f}, p={pval:.4f} {'*' if pval < 0.05 else ''}")
                except Exception as e:
                    print(f"k={k:2d}, Protocol {protocol}: Wilcoxon failed: {e}")

    print("\n3/5/10 Average:")
    for protocol in ['A', 'B']:
        all_gaps = []
        for k in [3, 5, 10]:
            random_mask = (df['protocol'] == 'A') & (df['method'].str.contains('Random')) & (df['n_cal_per_class'] == k)
            kmeans_mask = (df['protocol'] == 'A') & (df['method'].str.contains('KMeans')) & (df['n_cal_per_class'] == k)

            random_data = df[random_mask]
            kmeans_data = df[kmeans_mask]

            random_acc = random_data.groupby('subject')['accuracy'].mean()
            kmeans_acc = kmeans_data.groupby('subject')['accuracy'].mean()

            common_subjects = random_acc.index.intersection(kmeans_acc.index)
            gaps = (kmeans_acc[common_subjects] - random_acc[common_subjects]).values.astype(float)
            all_gaps.extend(gaps)

        all_gaps = np.array(all_gaps)
        if len(all_gaps) >= 15:
            try:
                if np.all(np.abs(all_gaps) < 1e-10):
                    print(f"3/5/10_avg, Protocol {protocol}: No difference between methods")
                    continue
                stat, pval = wilcoxon(all_gaps, alternative='greater')
                print(f"3/5/10_avg, Protocol {protocol}: Gap={np.mean(all_gaps):+.4f}, p={pval:.4f} {'*' if pval < 0.05 else ''}")
                sig_results.append({
                    'k_shot': 'avg_3_5_10',
                    'protocol': protocol,
                    'random_mean': np.nan,
                    'kmeans_mean': np.nan,
                    'gap': np.mean(all_gaps),
                    'p_value': pval,
                    'significant': pval < 0.05
                })
            except Exception as e:
                print(f"3/5/10_avg, Protocol {protocol}: Wilcoxon failed: {e}")

    return pd.DataFrame(sig_results)

def analyze_subject_level(df):
    """Analyze subject-level ACCS gains"""
    print("\n" + "="*60)
    print("Subject-Level Analysis (Protocol A - Main Result)")
    print("="*60)

    difficult_subjects = ['YLS', 'YSL', 'YHS', 'YRP', 'YAC']

    for k in [5, 10, 20]:
        print(f"\nk={k}-shot:")
        random_mask = (df['protocol'] == 'A') & (df['method'] == 'Random_label_free') & (df['n_cal_per_class'] == k)
        kmeans_mask = (df['protocol'] == 'A') & (df['method'] == 'KMeans_label_free') & (df['n_cal_per_class'] == k)

        random_acc = df[random_mask].groupby('subject')['accuracy'].mean()
        kmeans_acc = df[kmeans_mask].groupby('subject')['accuracy'].mean()

        gaps = kmeans_acc - random_acc

        for subj in Y_SUBJECTS:
            if subj in gaps.index:
                marker = " *" if subj in difficult_subjects else ""
                print(f"  {subj}: Random={random_acc[subj]:.4f}, KMeans={kmeans_acc[subj]:.4f}, Gap={gaps[subj]:+.4f}{marker}")

    print("\nDifficult Subjects Summary:")
    for k in [5, 10, 20]:
        random_mask = (df['protocol'] == 'A') & (df['method'] == 'Random_label_free') & (df['n_cal_per_class'] == k)
        kmeans_mask = (df['protocol'] == 'A') & (df['method'] == 'KMeans_label_free') & (df['n_cal_per_class'] == k)

        random_acc = df[random_mask].groupby('subject')['accuracy'].mean()
        kmeans_acc = df[kmeans_mask].groupby('subject')['accuracy'].mean()

        diff_subjs = [s for s in difficult_subjects if s in gaps.index]
        avg_gap = np.mean([kmeans_acc[s] - random_acc[s] for s in diff_subjs])
        print(f"  k={k}: Avg gap on difficult subjects = {avg_gap:+.4f}")

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    df = run_experiment()

    output_path = os.path.join(RESULTS_DIR, "accs_active_calibration.csv")
    df.to_csv(output_path, index=False)

    sig_results = run_significance_tests(df)
    sig_path = os.path.join(RESULTS_DIR, "accs_significance_tests.csv")
    sig_results.to_csv(sig_path, index=False)

    analyze_subject_level(df)

    print("\n" + "="*60)
    print("SUMMARY (Protocol A - Label-Free Budget - MAIN RESULT)")
    print("="*60)
    for k in [1, 3, 5, 10, 20, 50]:
        for method in ['Random_label_free', 'KMeans_label_free']:
            data = df[(df['protocol'] == 'A') & (df['method'] == method) & (df['n_cal_per_class'] == k)]
            if len(data) > 0:
                acc = data['accuracy'].mean()
                std = data['accuracy'].std()
                print(f"k={k:2d}, {method}: {acc:.4f}±{std:.4f}")

    print("\n" + "="*60)
    print("SUMMARY (Protocol B - Balanced Simulation)")
    print("="*60)
    for k in [1, 3, 5, 10, 20, 50]:
        for method in ['Random_balanced', 'KMeans_balanced']:
            data = df[(df['protocol'] == 'B') & (df['method'] == method) & (df['n_cal_per_class'] == k)]
            if len(data) > 0:
                acc = data['accuracy'].mean()
                std = data['accuracy'].std()
                print(f"k={k:2d}, {method}: {acc:.4f}±{std:.4f}")

    print("\nDone!")

if __name__ == '__main__':
    main()