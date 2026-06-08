"""
MACS-Fusion: Multimodal Active Calibration Sampling for EEG-Gaze Fusion

Target: Extend ACCS from EEG-only to EEG-Gaze fusion, reducing calibration cost.

Protocol: Label-free total budget
- budget = 2k samples
- k = 5, 10, 20, 50 per class equivalent
- Sampling WITHOUT using NR/TSR labels
- Only after selection, use labels to train EEG/Gaze/fusion classifiers

Methods:
1. Random_Static_Fusion
2. EEG_KMeans_Static_Fusion
3. Gaze_KMeans_Static_Fusion
4. EEG_Gaze_KMeans_Static_Fusion
5. EEG_Gaze_Coreset_Static_Fusion

Success criteria:
1. 10-shot equivalent: ACCS ≥ Random_Static_Fusion + 2%
2. 20-shot equivalent: ACCS ≥ Random_Static_Fusion + 1%
3. Difficult subjects YLS/YSL/YHS avg gain ≥ 2%
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

FEATURES_DIR = "features"
RESULTS_DIR = "results/personalized"
REPORT_DIR = "reports"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    X, y, keys = [], [], []
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
        keys.append(key)
    return np.array(X), np.array(y), keys

def load_gaze_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze.npy")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    X, y, keys = [], [], []
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 2 and parts[1] == "NR":
            label = 1
        elif len(parts) >= 2 and parts[1] == "TSR":
            label = 0
        else:
            continue
        feat_list = list(values)
        features = np.array(feat_list[:-1], dtype=np.float64)
        X.append(features)
        y.append(label)
        keys.append(key)
    return np.array(X), np.array(y), keys

def align_eeg_gaze(eeg_X, eeg_y, eeg_keys, gaze_X, gaze_y, gaze_keys):
    common_len = min(len(eeg_y), len(gaze_y))
    return eeg_X[:common_len], eeg_y[:common_len], gaze_X[:common_len], gaze_y[:common_len], common_len

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

def coreset_sampling_label_free(X_pool, n_select):
    """Coreset sampling using facility location greedy selection"""
    n_pool = len(X_pool)
    if n_select >= n_pool:
        return np.random.permutation(n_pool)[:n_select]

    scaler = StandardScaler()
    X_pool_s = scaler.fit_transform(X_pool)

    selected = []
    remaining = list(range(n_pool))

    first_idx = np.random.choice(remaining)
    selected.append(first_idx)
    remaining.remove(first_idx)

    while len(selected) < n_select and remaining:
        best_idx = None
        best_min_dist = -1
        for idx in remaining:
            dists = np.linalg.norm(X_pool_s[idx] - X_pool_s[selected], axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return np.array(selected)

def select_calibration_random(X_pool, n_select):
    """Random sampling (no labels used)"""
    indices = np.random.permutation(len(X_pool))[:n_select]
    return indices

def train_classifier(X_cal, y_cal):
    """Train MLP classifier"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    clf.fit(X_cal_s, y_cal)
    return clf, scaler

def evaluate_classifier(clf, scaler, X_test, y_test):
    """Evaluate classifier"""
    X_test_s = scaler.transform(X_test)
    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    except:
        auroc = 0.5
    return acc, f1, bacc, auroc

def get_predictions(clf, scaler, X):
    """Get prediction probabilities"""
    X_s = scaler.transform(X)
    return clf.predict_proba(X_s)[:, 1]

def run_experiment():
    results = []
    calibration_settings = [5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    print("MACS-Fusion: Multimodal Active Calibration Sampling")
    print("="*60)
    print("Methods: Random, EEG-KMeans, Gaze-KMeans, EEG-Gaze-KMeans, Coreset")
    print("Protocol: Label-free total budget")
    print("="*60)

    for seed in seeds:
        print(f"\nSeed {seed}:", flush=True)

        for held_out in Y_SUBJECTS:
            eeg_X, eeg_y, eeg_keys = load_eeg_data(held_out)
            gaze_X, gaze_y, gaze_keys = load_gaze_data(held_out)

            if eeg_X is None or gaze_X is None:
                print(f" {held_out}[no data]", end="", flush=True)
                continue

            eeg_X, eeg_y, gaze_X, gaze_y, common_keys = align_eeg_gaze(
                eeg_X, eeg_y, eeg_keys, gaze_X, gaze_y, gaze_keys
            )

            if eeg_X is None or len(eeg_X) < 50:
                print(f" {held_out}[no align:{len(common_keys) if common_keys else 0}]", end="", flush=True)
                continue

            n_samples = len(eeg_y)
            np.random.seed(seed)
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_samples // 2]
            cal_pool_indices = indices[n_samples // 2:]

            eeg_test = eeg_X[test_indices]
            eeg_cal_pool = eeg_X[cal_pool_indices]
            gaze_test = gaze_X[test_indices]
            gaze_cal_pool = gaze_X[cal_pool_indices]
            y_test = eeg_y[test_indices]
            y_cal_pool = eeg_y[cal_pool_indices]

            for n_cal_per_class in calibration_settings:
                if n_cal_per_class * 2 > len(cal_pool_indices):
                    continue

                total_budget = n_cal_per_class * 2

                cal_idx_random = select_calibration_random(eeg_cal_pool, total_budget)

                cal_idx_eeg_kmeans = kmeans_centroid_sampling_label_free(eeg_cal_pool, total_budget)

                cal_idx_gaze_kmeans = kmeans_centroid_sampling_label_free(gaze_cal_pool, total_budget)

                combined_pool = np.hstack([eeg_cal_pool, gaze_cal_pool])
                cal_idx_combined_kmeans = kmeans_centroid_sampling_label_free(combined_pool, total_budget)

                cal_idx_coreset = coreset_sampling_label_free(eeg_cal_pool, total_budget)

                configs = [
                    ('Random', cal_idx_random),
                    ('EEG_KMeans', cal_idx_eeg_kmeans),
                    ('Gaze_KMeans', cal_idx_gaze_kmeans),
                    ('EEG_Gaze_KMeans', cal_idx_combined_kmeans[:len(cal_idx_eeg_kmeans)]),
                    ('Coreset', cal_idx_coreset),
                ]

                for method_name, cal_idx in configs:
                    y_cal = y_cal_pool[cal_idx]

                    if len(np.unique(y_cal)) < 2:
                        continue

                    clf_eeg, scaler_eeg = train_classifier(eeg_cal_pool[cal_idx], y_cal)
                    clf_gaze, scaler_gaze = train_classifier(gaze_cal_pool[cal_idx], y_cal)

                    acc_eeg, f1_eeg, bacc_eeg, auroc_eeg = evaluate_classifier(
                        clf_eeg, scaler_eeg, eeg_test, y_test
                    )
                    acc_gaze, f1_gaze, bacc_gaze, auroc_gaze = evaluate_classifier(
                        clf_gaze, scaler_gaze, gaze_test, y_test
                    )

                    p_eeg = get_predictions(clf_eeg, scaler_eeg, eeg_test)
                    p_gaze = get_predictions(clf_gaze, scaler_gaze, gaze_test)
                    p_fusion = 0.5 * p_eeg + 0.5 * p_gaze
                    fusion_preds = (p_fusion >= 0.5).astype(int)

                    acc_fusion = accuracy_score(y_test, fusion_preds)
                    f1_fusion = f1_score(y_test, fusion_preds, average='macro')
                    bacc_fusion = balanced_accuracy_score(y_test, fusion_preds)
                    try:
                        auroc_fusion = roc_auc_score(y_test, p_fusion)
                    except:
                        auroc_fusion = 0.5

                    results.append({
                        'seed': seed,
                        'subject': held_out,
                        'n_cal_per_class': n_cal_per_class,
                        'n_cal_total': total_budget,
                        'method': method_name,
                        'modality': 'EEG_only',
                        'accuracy': acc_eeg,
                        'macro_f1': f1_eeg,
                        'balanced_accuracy': bacc_eeg,
                        'auroc': auroc_eeg
                    })
                    results.append({
                        'seed': seed,
                        'subject': held_out,
                        'n_cal_per_class': n_cal_per_class,
                        'n_cal_total': total_budget,
                        'method': method_name,
                        'modality': 'Gaze_only',
                        'accuracy': acc_gaze,
                        'macro_f1': f1_gaze,
                        'balanced_accuracy': bacc_gaze,
                        'auroc': auroc_gaze
                    })
                    results.append({
                        'seed': seed,
                        'subject': held_out,
                        'n_cal_per_class': n_cal_per_class,
                        'n_cal_total': total_budget,
                        'method': method_name,
                        'modality': 'Static_Fusion',
                        'accuracy': acc_fusion,
                        'macro_f1': f1_fusion,
                        'balanced_accuracy': bacc_fusion,
                        'auroc': auroc_fusion
                    })

            print(f" {held_out}", end="", flush=True)

    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze and summarize results"""
    print("\n" + "="*60)
    print("MACS-Fusion Results Summary")
    print("="*60)

    difficult_subjects = ['YLS', 'YSL', 'YHS']

    for k in [5, 10, 20, 50]:
        print(f"\nk={k}-shot (total budget={k*2}):")
        for method in ['Random', 'EEG_KMeans', 'Gaze_KMeans', 'EEG_Gaze_KMeans', 'Coreset']:
            fusion_data = df[(df['method'] == method) & (df['modality'] == 'Static_Fusion') & (df['n_cal_per_class'] == k)]
            if len(fusion_data) > 0:
                acc = fusion_data['accuracy'].mean()
                std = fusion_data['accuracy'].std()
                print(f"  {method:20s}: {acc:.4f}±{std:.4f}")

        random_fusion = df[(df['method'] == 'Random') & (df['modality'] == 'Static_Fusion') & (df['n_cal_per_class'] == k)]['accuracy'].mean()
        print(f"\n  vs Random gain:")
        for method in ['EEG_KMeans', 'Gaze_KMeans', 'EEG_Gaze_KMeans', 'Coreset']:
            fusion_data = df[(df['method'] == method) & (df['modality'] == 'Static_Fusion') & (df['n_cal_per_class'] == k)]
            if len(fusion_data) > 0:
                acc = fusion_data['accuracy'].mean()
                gain = acc - random_fusion
                marker = " *" if gain >= 0.02 else ""
                print(f"    {method:20s}: {gain:+.4f}{marker}")

        print(f"\n  Difficult subjects (YLS/YSL/YHS) gain vs Random:")
        for method in ['EEG_KMeans', 'Gaze_KMeans', 'EEG_Gaze_KMeans', 'Coreset']:
            diff_gains = []
            for subj in difficult_subjects:
                random_data = df[(df['method'] == 'Random') & (df['modality'] == 'Static_Fusion') & (df['n_cal_per_class'] == k) & (df['subject'] == subj)]
                method_data = df[(df['method'] == method) & (df['modality'] == 'Static_Fusion') & (df['n_cal_per_class'] == k) & (df['subject'] == subj)]
                if len(random_data) > 0 and len(method_data) > 0:
                    diff_gains.append(method_data['accuracy'].mean() - random_data['accuracy'].mean())
            if diff_gains:
                avg_gain = np.mean(diff_gains)
                marker = " *" if avg_gain >= 0.02 else ""
                print(f"    {method:20s}: {avg_gain:+.4f}{marker}")

    print("\n" + "="*60)
    print("Success Criteria Check")
    print("="*60)

    for k in [10, 20]:
        random_fusion = df[(df['method'] == 'Random') & (df['modality'] == 'Static_Fusion') & (df['n_cal_per_class'] == k)]['accuracy'].mean()
        print(f"\nk={k}: Random={random_fusion:.4f}")
        for method in ['EEG_KMeans', 'Gaze_KMeans', 'EEG_Gaze_KMeans', 'Coreset']:
            fusion_data = df[(df['method'] == method) & (df['modality'] == 'Static_Fusion') & (df['n_cal_per_class'] == k)]
            if len(fusion_data) > 0:
                acc = fusion_data['accuracy'].mean()
                gain = acc - random_fusion
                target = 0.02 if k == 10 else 0.01
                status = "PASS" if gain >= target else "FAIL"
                print(f"  {method}: {acc:.4f} (gain={gain:+.4f}) {status}")

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    df = run_experiment()

    output_path = os.path.join(RESULTS_DIR, "macs_fusion_results.csv")
    df.to_csv(output_path, index=False)

    analyze_results(df)

    print(f"\nResults saved to {output_path}")
    print("Done!")

if __name__ == '__main__':
    main()