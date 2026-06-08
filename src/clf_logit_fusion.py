"""
CLF: Calibrated Logit Fusion

This module implements calibrated logit-level fusion of EEG and Gaze predictions.

Methods:
1. Static_EEG_Gaze_average (baseline)
2. Reliability_weighted_EEG_Gaze (baseline)
3. CLF_logistic_stacking: Train a logistic regression on logits
4. CLF_temperature_scaled: Temperature-scaled fusion
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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

def train_eeg_classifier(X_cal, y_cal):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    clf = SVC(kernel='linear', random_state=42, gamma='scale', probability=True)
    clf.fit(X_cal_s, y_cal)
    return clf, scaler

def train_gaze_classifier(X_cal, y_cal):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    clf = SVC(kernel='linear', random_state=42, gamma='scale', probability=True)
    clf.fit(X_cal_s, y_cal)
    return clf, scaler

def get_logits_and_probs(clf, scaler, X):
    """Get logits (decision function) and probabilities"""
    X_s = scaler.transform(X)
    probas = clf.predict_proba(X_s)

    if hasattr(clf, 'decision_function'):
        logits = clf.decision_function(X_s)
    else:
        logits = np.log(probas[:, 1] + 1e-10) - np.log(probas[:, 0] + 1e-10)

    margins = np.abs(logits)

    return logits, probas, margins

def run_experiment(seed, model_type):
    """Run CLF experiment"""
    results = []
    calibration_settings = [1, 3, 5, 10, 20, 50]

    for held_out in Y_SUBJECTS:
        print(f"\n    {model_type} - {held_out}:", flush=True)

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

            clf_eeg, scaler_eeg = train_eeg_classifier(X_cal_eeg, y_cal)
            clf_gaze, scaler_gaze = train_gaze_classifier(X_cal_gaze, y_cal)

            logits_eeg_cal, probas_eeg_cal, margins_eeg_cal = get_logits_and_probs(clf_eeg, scaler_eeg, X_cal_eeg)
            logits_gaze_cal, probas_gaze_cal, margins_gaze_cal = get_logits_and_probs(clf_gaze, scaler_gaze, X_cal_gaze)

            logits_eeg_test, probas_eeg_test, margins_eeg_test = get_logits_and_probs(clf_eeg, scaler_eeg, X_test_eeg)
            logits_gaze_test, probas_gaze_test, margins_gaze_test = get_logits_and_probs(clf_gaze, scaler_gaze, X_test_gaze)

            if model_type == 'EEG_only':
                test_probs = probas_eeg_test[:, 1]

            elif model_type == 'Gaze_only':
                test_probs = probas_gaze_test[:, 1]

            elif model_type == 'Static_EEG_Gaze_average':
                test_probs = 0.5 * probas_eeg_test[:, 1] + 0.5 * probas_gaze_test[:, 1]

            elif model_type == 'Reliability_weighted_EEG_Gaze':
                eeg_reliability = np.mean(margins_eeg_cal) / (np.mean(margins_eeg_cal) + np.mean(margins_gaze_cal) + 1e-6)
                w_eeg = eeg_reliability
                w_gaze = 1 - eeg_reliability
                test_probs = w_eeg * probas_eeg_test[:, 1] + w_gaze * probas_gaze_test[:, 1]

            elif model_type == 'CLF_logistic_stacking':
                cal_features = np.column_stack([
                    logits_eeg_cal,
                    logits_gaze_cal,
                    margins_eeg_cal,
                    margins_gaze_cal,
                    probas_eeg_cal[:, 1],
                    probas_gaze_cal[:, 1]
                ])

                test_features = np.column_stack([
                    logits_eeg_test,
                    logits_gaze_test,
                    margins_eeg_test,
                    margins_gaze_test,
                    probas_eeg_test[:, 1],
                    probas_gaze_test[:, 1]
                ])

                stacker = LogisticRegression(random_state=42, max_iter=1000)
                stacker.fit(cal_features, y_cal)
                test_probs = stacker.predict_proba(test_features)[:, 1]

            elif model_type == 'CLF_temperature_scaled':
                T_eeg = 1.0 + 0.5 * np.std(probas_eeg_cal[:, 1])
                T_gaze = 1.0 + 0.5 * np.std(probas_gaze_cal[:, 1])

                logits_eeg_norm = logits_eeg_test / T_eeg
                logits_gaze_norm = logits_gaze_test / T_gaze

                w_eeg = 0.6
                w_gaze = 0.4

                combined_logits = w_eeg * logits_eeg_norm + w_gaze * logits_gaze_norm
                test_probs = 1 / (1 + np.exp(-combined_logits))

            else:
                continue

            test_preds = (test_probs > 0.5).astype(int)

            acc = accuracy_score(y_test, test_preds)
            f1 = f1_score(y_test, test_preds, average='macro')
            bacc = balanced_accuracy_score(y_test, test_preds)
            try:
                auroc = roc_auc_score(y_test, test_probs)
            except:
                auroc = 0.5

            results.append({
                'model': model_type,
                'seed': seed,
                'subject': held_out,
                'n_cal_per_class': n_cal_per_class,
                'n_cal_total': n_cal_per_class * 2,
                'accuracy': acc,
                'macro_f1': f1,
                'balanced_accuracy': bacc,
                'auroc': auroc
            })

            print(f"      {n_cal_per_class}-shot: Acc={acc:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}", flush=True)

    return results

def main():
    print("="*70)
    print("CLF: Calibrated Logit Fusion Experiment")
    print("="*70)

    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    all_results = []
    model_types = [
        'EEG_only',
        'Gaze_only',
        'Static_EEG_Gaze_average',
        'Reliability_weighted_EEG_Gaze',
        'CLF_logistic_stacking',
        'CLF_temperature_scaled'
    ]

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Running: {model_type}")
        print("="*70)

        for seed in [0, 1, 2, 3, 4]:
            results = run_experiment(seed, model_type)
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "clf_logit_fusion_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n\nSaved to {output_path}")

    summary = df.groupby(['model', 'n_cal_per_class']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'auroc': ['mean', 'std']
    }).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "clf_logit_fusion_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary.to_string())

    print("\nDone!")

if __name__ == '__main__':
    main()