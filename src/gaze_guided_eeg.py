"""
Module B: Gaze-Guided EEG Attention

Core idea: Use gaze's stability to guide EEG attention weighting.

Observation:
- Gaze is relatively stable even at 1-shot (45-55%)
- EEG is unstable at low-shot
- Fusion can be complementary

Method:
- Use gaze features to compute an attention weight
- Weight EEG features by this attention
- Classify on weighted EEG features
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

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

def load_gaze_data(subject):
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

def extract_gaze_attention(gaze_features):
    """Extract attention weights from gaze features"""
    omr, weighted_nFix, weighted_speed = gaze_features[:, 0], gaze_features[:, 1], gaze_features[:, 2]
    attention = (weighted_nFix + weighted_speed) / 2
    return attention.reshape(-1, 1)

def train_gaze_guided_eeg(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test, y_test):
    """Train EEG classifier with gaze-guided attention"""

    scaler_eeg = StandardScaler()
    X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
    X_eeg_test_s = scaler_eeg.transform(X_eeg_test)

    scaler_gaze = StandardScaler()
    X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
    X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

    attention_cal = extract_gaze_attention(X_gaze_cal_s)
    attention_test = extract_gaze_attention(X_gaze_test_s)

    X_eeg_cal_weighted = X_eeg_cal_s * attention_cal
    X_eeg_test_weighted = X_eeg_test_s * attention_test

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    clf.fit(X_eeg_cal_weighted, y_cal)

    preds = clf.predict(X_eeg_test_weighted)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, clf.predict_proba(X_eeg_test_weighted)[:, 1])
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def train_baseline(X_cal, y_cal, X_test, y_test):
    """Baseline EEG classifier"""
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

def train_fusion(X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test, y_test):
    """Simple EEG-Gaze late fusion"""
    scaler_eeg = StandardScaler()
    X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
    X_eeg_test_s = scaler_eeg.transform(X_eeg_test)

    scaler_gaze = StandardScaler()
    X_gaze_cal_s = scaler_gaze.fit_transform(X_gaze_cal)
    X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

    clf_eeg = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    clf_gaze = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    clf_eeg.fit(X_eeg_cal_s, y_cal)
    clf_gaze.fit(X_gaze_cal_s, y_cal)

    prob_eeg = clf_eeg.predict_proba(X_eeg_test_s)[:, 1]
    prob_gaze = clf_gaze.predict_proba(X_gaze_test_s)[:, 1]

    prob_fusion = 0.6 * prob_eeg + 0.4 * prob_gaze
    preds = (prob_fusion > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, prob_fusion)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_experiment():
    results = []
    calibration_settings = [1, 3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    print("Gaze-Guided EEG Attention Experiment")
    print("="*60)

    for seed in seeds:
        print(f"\nSeed {seed}:")

        for held_out in Y_SUBJECTS:
            X_eeg, y_eeg = load_eeg_data(held_out)
            X_gaze, y_gaze = load_gaze_data(held_out)

            if X_eeg is None or X_gaze is None:
                continue

            common_len = min(len(y_eeg), len(y_gaze))
            X_eeg = X_eeg[:common_len]
            y_eeg = y_eeg[:common_len]
            X_gaze = X_gaze[:common_len]

            if len(X_eeg) < 50:
                continue

            n_samples = len(y_eeg)
            np.random.seed(seed)
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_samples // 2]
            cal_pool_indices = indices[n_samples // 2:]

            X_test_eeg = X_eeg[test_indices]
            X_test_gaze = X_gaze[test_indices]
            y_test = y_eeg[test_indices]

            X_cal_pool_eeg = X_eeg[cal_pool_indices]
            X_cal_pool_gaze = X_gaze[cal_pool_indices]
            y_cal_pool = y_eeg[cal_pool_indices]

            for n_cal_per_class in calibration_settings:
                if n_cal_per_class * 2 > len(cal_pool_indices):
                    continue

                cal_idx_0 = np.where(y_cal_pool == 0)[0][:n_cal_per_class]
                cal_idx_1 = np.where(y_cal_pool == 1)[0][:n_cal_per_class]
                cal_idx = np.concatenate([cal_idx_0, cal_idx_1])
                np.random.shuffle(cal_idx)

                X_cal_eeg = X_cal_pool_eeg[cal_idx]
                X_cal_gaze = X_cal_pool_gaze[cal_idx]
                y_cal = y_cal_pool[cal_idx]

                baseline_acc, baseline_f1, baseline_bacc, baseline_auroc = train_baseline(X_cal_eeg, y_cal, X_test_eeg, y_test)
                gaze_guided_acc, gaze_guided_f1, gaze_guided_bacc, gaze_guided_auroc = train_gaze_guided_eeg(
                    X_cal_eeg, y_cal, X_cal_gaze, X_test_eeg, X_test_gaze, y_test)
                fusion_acc, fusion_f1, fusion_bacc, fusion_auroc = train_fusion(
                    X_cal_eeg, y_cal, X_cal_gaze, X_test_eeg, X_test_gaze, y_test)

                results.append({
                    'seed': seed,
                    'subject': held_out,
                    'n_cal_per_class': n_cal_per_class,
                    'n_cal_total': n_cal_per_class * 2,
                    'baseline_acc': baseline_acc,
                    'baseline_f1': baseline_f1,
                    'baseline_bacc': baseline_bacc,
                    'baseline_auroc': baseline_auroc,
                    'gaze_guided_acc': gaze_guided_acc,
                    'gaze_guided_f1': gaze_guided_f1,
                    'gaze_guided_bacc': gaze_guided_bacc,
                    'gaze_guided_auroc': gaze_guided_auroc,
                    'fusion_acc': fusion_acc,
                    'fusion_f1': fusion_f1,
                    'fusion_bacc': fusion_bacc,
                    'fusion_auroc': fusion_auroc
                })

            print(f" {held_out}", end="", flush=True)

    return pd.DataFrame(results)

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    df = run_experiment()

    output_path = os.path.join(RESULTS_DIR, "gaze_guided_eeg_results.csv")
    df.to_csv(output_path, index=False)

    summary = df.groupby('n_cal_per_class').agg({
        'baseline_acc': ['mean', 'std'],
        'gaze_guided_acc': ['mean', 'std'],
        'fusion_acc': ['mean', 'std']
    }).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "gaze_guided_eeg_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*60)
    print("SUMMARY (Accuracy)")
    print("="*60)
    for _, row in summary.iterrows():
        k = row['n_cal_per_class']
        bl_acc = row[('baseline_acc', 'mean')]
        gg_acc = row[('gaze_guided_acc', 'mean')]
        fu_acc = row[('fusion_acc', 'mean')]
        print(f"{k:2d}-shot: Baseline={bl_acc:.4f}, GazeGuided={gg_acc:.4f}, Fusion={fu_acc:.4f}")

    print("\nDone!")

if __name__ == '__main__':
    main()