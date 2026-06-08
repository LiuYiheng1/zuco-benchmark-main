"""
Text-Only Baseline for Material Shortcut Analysis

This experiment evaluates whether NR vs TSR classification can be achieved
using only text/material features extracted from gaze data.

The sent_gaze_sacc features include:
- omr (omission rate)
- weighted_nFix = nFixations / lnorm (where lnorm = sentence length in words)
- weighted_speed = total_fixation_time / lnorm
- Various saccade metrics

Since lnorm (sentence word count) is included, we can use these as text proxies.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import config

FEATURES_DIR = "features"
RESULTS_DIR = "results/personalized"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_gaze_features(subject):
    """Load gaze features and extract text-proxy features"""
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze_sacc.npy")
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

def extract_text_features(gaze_features):
    """Extract text-proxy features from gaze features

    gaze_features structure:
    [omr, weighted_nFix, weighted_speed, smeand, smaxv, smeanv, smaxd, smeana, smaxa]

    weighted_nFix and weighted_speed are normalized by lnorm (sentence length)
    So: lnorm = nFix / weighted_nFix (if weighted_nFix > 0)
        lnorm = total_time / weighted_speed (if weighted_speed > 0)
    """
    text_features = []

    for row in gaze_features:
        omr, weighted_nFix, weighted_speed = row[0], row[1], row[2]

        inferred_lnorm_from_fix = 1.0 / weighted_nFix if weighted_nFix > 0 else 0
        inferred_lnorm_from_speed = 1.0 / weighted_speed if weighted_speed > 0 else 0

        text_features.append([
            inferred_lnorm_from_fix,
            inferred_lnorm_from_speed,
            omr,
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8]
        ])

    return np.array(text_features)

def run_within_subject_text_experiment():
    """Run within-subject text-only experiment using text-proxy features"""
    print("=" * 70)
    print("Within-Subject Text-Proxy Baseline (from Gaze Features)")
    print("=" * 70)

    results = []

    for subject in Y_SUBJECTS:
        print(f"\nProcessing subject: {subject}", flush=True)

        X_gaze, y = load_gaze_features(subject)
        if X_gaze is None or len(X_gaze) < 50:
            print(f"  No data for {subject}, skipping")
            continue

        X = extract_text_features(X_gaze)

        if len(np.unique(y)) < 2:
            continue

        n_class_0 = np.sum(y == 0)
        n_class_1 = np.sum(y == 1)
        min_class = min(n_class_0, n_class_1)

        if min_class < 10:
            print(f"  Insufficient data for {subject}")
            continue

        np.random.seed(42)
        indices = np.random.permutation(len(y))

        class_0_idx = np.where(y == 0)[0]
        class_1_idx = np.where(y == 1)[0]

        n_test_per_class = min(50, min_class // 2)
        n_train_per_class = min_class - n_test_per_class

        test_idx = np.concatenate([
            class_0_idx[:n_test_per_class],
            class_1_idx[:n_test_per_class]
        ])
        train_idx = np.concatenate([
            class_0_idx[n_test_per_class:n_test_per_class + n_train_per_class],
            class_1_idx[n_test_per_class:n_test_per_class + n_train_per_class]
        ])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = SVC(kernel='linear', random_state=42, gamma='scale')
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bacc = balanced_accuracy_score(y_test, y_pred)

        results.append({
            'subject': subject,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'n_train': len(y_train),
            'n_test': len(y_test)
        })

        print(f"  Text-Proxy: Acc={acc:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}", flush=True)

    return pd.DataFrame(results)

def run_personalized_few_shot_text_experiment():
    """Run personalized few-shot text-proxy experiment"""
    print("\n" + "=" * 70)
    print("Personalized Few-Shot Text-Proxy Baseline")
    print("=" * 70)

    all_results = []
    calibration_settings = [1, 3, 5, 10, 20, 50]

    for seed in [0, 1, 2, 3, 4]:
        print(f"\n--- Seed {seed} ---", flush=True)

        for subject in Y_SUBJECTS:
            X_gaze, y = load_gaze_features(subject)
            if X_gaze is None or len(X_gaze) < 50:
                continue

            X = extract_text_features(X_gaze)

            if len(np.unique(y)) < 2:
                continue

            n_class_0 = np.sum(y == 0)
            n_class_1 = np.sum(y == 1)
            min_class_size = min(n_class_0, n_class_1)

            if min_class_size < 50:
                continue

            np.random.seed(seed)
            indices = np.random.permutation(len(y))

            test_indices = indices[:len(indices) // 2]
            cal_pool_indices = indices[len(indices) // 2:]

            X_test, y_test = X[test_indices], y[test_indices]

            for n_cal_per_class in calibration_settings:
                if n_cal_per_class * 2 > len(cal_pool_indices):
                    continue

                cal_class0 = np.where(y[cal_pool_indices] == 0)[0][:n_cal_per_class]
                cal_class1 = np.where(y[cal_pool_indices] == 1)[0][:n_cal_per_class]
                cal_idx = np.concatenate([cal_class0, cal_class1])
                np.random.shuffle(cal_idx)

                X_cal = X[cal_pool_indices][cal_idx]
                y_cal = y[cal_pool_indices][cal_idx]

                scaler = StandardScaler()
                X_cal_s = scaler.fit_transform(X_cal)
                X_test_s = scaler.transform(X_test)

                clf = SVC(kernel='linear', random_state=42, gamma='scale')
                clf.fit(X_cal_s, y_cal)
                y_pred = clf.predict(X_test_s)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                bacc = balanced_accuracy_score(y_test, y_pred)

                all_results.append({
                    'seed': seed,
                    'subject': subject,
                    'n_cal_per_class': n_cal_per_class,
                    'n_cal_total': n_cal_per_class * 2,
                    'n_test': len(y_test),
                    'accuracy': acc,
                    'macro_f1': f1,
                    'balanced_accuracy': bacc
                })

        print(f"  Seed {seed} completed", flush=True)

    return pd.DataFrame(all_results)

def main():
    print("=" * 70)
    print("Text-Proxy Baseline for Material Shortcut Analysis")
    print("Using sentence length inferred from gaze features")
    print("=" * 70)

    print("\n[1/2] Running within-subject text-proxy experiment...")
    within_results = run_within_subject_text_experiment()
    within_results.to_csv(os.path.join(RESULTS_DIR, "text_proxy_within_subject.csv"), index=False)

    print("\n[2/2] Running personalized few-shot text-proxy experiment...")
    fewshot_results = run_personalized_few_shot_text_experiment()
    fewshot_results.to_csv(os.path.join(RESULTS_DIR, "text_proxy_fewshot.csv"), index=False)

    print("\n" + "=" * 70)
    print("WITHIN-SUBJECT TEXT-PROXY RESULTS")
    print("=" * 70)
    if len(within_results) > 0:
        mean_acc = within_results['accuracy'].mean()
        std_acc = within_results['accuracy'].std()
        mean_f1 = within_results['macro_f1'].mean()
        std_f1 = within_results['macro_f1'].std()
        print(f"Text-Proxy: Acc={mean_acc:.4f}±{std_acc:.4f}, F1={mean_f1:.4f}±{std_f1:.4f}")
        print(within_results.to_string())

    print("\n" + "=" * 70)
    print("FEW-SHOT TEXT-PROXY RESULTS")
    print("=" * 70)
    if len(fewshot_results) > 0:
        summary = fewshot_results.groupby('n_cal_per_class').agg({
            'accuracy': ['mean', 'std'],
            'macro_f1': ['mean', 'std'],
            'balanced_accuracy': ['mean', 'std']
        }).reset_index()
        print(summary.to_string())

    print("\nDone!")

if __name__ == '__main__':
    main()