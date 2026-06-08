"""
CSA: Contrastive Subject Adaptation

Core idea: Use contrastive learning to learn EEG representations
that preserve subject-specific information while enhancing task-discriminative structure.

Unlike SIED (which removes subject info), CSA keeps subject info but learns
better class-separating representations through contrastive learning.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

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

def compute_contrastive_features(X, y, reference_X=None):
    """Compute contrastive enhanced features

    For each sample, compute:
    - Original features
    - Class centroid distance
    - Class-conditional statistics
    """
    if reference_X is None:
        reference_X = X

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_ref_s = scaler.transform(reference_X)

    class_0_mask = y == 0
    class_1_mask = y == 1

    centroid_0 = np.mean(X_ref_s[class_0_mask], axis=0)
    centroid_1 = np.mean(X_ref_s[class_1_mask], axis=0)

    dist_to_centroid_0 = np.linalg.norm(X_s - centroid_0, axis=1)
    dist_to_centroid_1 = np.linalg.norm(X_s - centroid_1, axis=1)

    within_class_std = np.std(X_ref_s, axis=0)
    within_class_std_expanded = np.tile(within_class_std, (X_s.shape[0], 1))

    contrastive_features = np.column_stack([
        X_s,
        dist_to_centroid_0.reshape(-1, 1),
        dist_to_centroid_1.reshape(-1, 1),
        (dist_to_centroid_0 - dist_to_centroid_1).reshape(-1, 1),
        np.abs(dist_to_centroid_0 - dist_to_centroid_1).reshape(-1, 1),
        within_class_std_expanded
    ])

    return contrastive_features, scaler

def train_contrastive_classifier(X_cal, y_cal, X_test, y_test):
    """Train classifier on contrastive enhanced features"""
    X_contrast_cal, scaler = compute_contrastive_features(X_cal, y_cal)
    X_contrast_test, _ = compute_contrastive_features(X_test, y_cal, reference_X=X_cal)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    clf.fit(X_contrast_cal, y_cal)

    preds = clf.predict(X_contrast_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, clf.predict_proba(X_contrast_test)[:, 1])
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def train_baseline_classifier(X_cal, y_cal, X_test, y_test):
    """Baseline MLP classifier"""
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

    print("CSA: Contrastive Subject Adaptation Experiment")
    print("="*60)

    for seed in seeds:
        print(f"\nSeed {seed}:")

        for held_out in Y_SUBJECTS:
            X_held, y_held = load_eeg_data(held_out)
            if X_held is None or len(X_held) < 50:
                continue

            n_samples = len(y_held)
            np.random.seed(seed)
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_samples // 2]
            cal_pool_indices = indices[n_samples // 2:]

            X_test = X_held[test_indices]
            y_test = y_held[test_indices]
            X_cal_pool = X_held[cal_pool_indices]
            y_cal_pool = y_held[cal_pool_indices]

            for n_cal_per_class in calibration_settings:
                if n_cal_per_class * 2 > len(cal_pool_indices):
                    continue

                cal_idx_0 = np.where(y_cal_pool == 0)[0][:n_cal_per_class]
                cal_idx_1 = np.where(y_cal_pool == 1)[0][:n_cal_per_class]
                cal_idx = np.concatenate([cal_idx_0, cal_idx_1])
                np.random.shuffle(cal_idx)

                X_cal = X_cal_pool[cal_idx]
                y_cal = y_cal_pool[cal_idx]

                baseline_acc, baseline_f1, baseline_bacc, baseline_auroc = train_baseline_classifier(X_cal, y_cal, X_test, y_test)
                csa_acc, csa_f1, csa_bacc, csa_auroc = train_contrastive_classifier(X_cal, y_cal, X_test, y_test)

                results.append({
                    'seed': seed,
                    'subject': held_out,
                    'n_cal_per_class': n_cal_per_class,
                    'n_cal_total': n_cal_per_class * 2,
                    'baseline_acc': baseline_acc,
                    'baseline_f1': baseline_f1,
                    'baseline_bacc': baseline_bacc,
                    'baseline_auroc': baseline_auroc,
                    'CSA_acc': csa_acc,
                    'CSA_f1': csa_f1,
                    'CSA_bacc': csa_bacc,
                    'CSA_auroc': csa_auroc
                })

            print(f" {held_out}", end="", flush=True)

    return pd.DataFrame(results)

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    df = run_experiment()

    output_path = os.path.join(RESULTS_DIR, "csa_contrastive_adaptation.csv")
    df.to_csv(output_path, index=False)

    summary = df.groupby('n_cal_per_class').agg({
        'baseline_acc': ['mean', 'std'],
        'CSA_acc': ['mean', 'std'],
        'baseline_f1': ['mean', 'std'],
        'CSA_f1': ['mean', 'std']
    }).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "csa_contrastive_adaptation_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*60)
    print("SUMMARY (Accuracy)")
    print("="*60)
    for _, row in summary.iterrows():
        k = row['n_cal_per_class']
        bl_acc = row[('baseline_acc', 'mean')]
        bl_std = row[('baseline_acc', 'std')]
        csa_acc = row[('CSA_acc', 'mean')]
        csa_std = row[('CSA_acc', 'std')]
        gap = csa_acc - bl_acc
        print(f"{k:2d}-shot: Baseline={bl_acc:.4f}±{bl_std:.4f}, CSA={csa_acc:.4f}±{csa_std:.4f}, Gap={gap:+.4f}")

    print("\nDone!")

if __name__ == '__main__':
    main()