"""
Reliability-Weighted Multi-Modal Calibration

This module implements reliability-aware fusion of EEG, Gaze, and Text-proxy predictions.

Goal: Improve performance on difficult subjects (YLS, YSL, YHS) by dynamically
weighting predictions based on EEG calibration reliability.

Reliability metrics:
1. Calibration classifier margin
2. Cross-validation accuracy on calibration set
3. Prediction entropy
4. Class centroid distance
5. EEG feature variance / SNR proxy

Models:
1. EEG_only
2. Gaze_only
3. Static_EEG_Gaze_average
4. Reliability_weighted_EEG_Gaze
5. Reliability_weighted_EEG_TextProxy
6. Reliability_weighted_EEG_Gaze_TextProxy
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import entropy

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

def extract_text_proxy_features(gaze_features):
    """Extract text-proxy features from gaze (lnorm-based)"""
    text_features = []
    for row in gaze_features:
        omr, weighted_nFix, weighted_speed = row[0], row[1], row[2]
        inferred_lnorm_fix = 1.0 / weighted_nFix if weighted_nFix > 0 else 0
        inferred_lnorm_speed = 1.0 / weighted_speed if weighted_speed > 0 else 0
        text_features.append([
            inferred_lnorm_fix,
            inferred_lnorm_speed,
            omr, row[3], row[4], row[5], row[6], row[7], row[8]
        ])
    return np.array(text_features)

def estimate_calibration_reliability(X_cal, y_cal):
    """Estimate EEG reliability from calibration set

    Returns:
        reliability_score: higher = more reliable
    """
    if len(X_cal) < 4:
        return 0.5

    try:
        scaler = StandardScaler()
        X_cal_s = scaler.fit_transform(X_cal)

        clf = SVC(kernel='linear', random_state=42, gamma='scale', probability=True)
        clf.fit(X_cal_s, y_cal)

        cv_scores = cross_val_score(clf, X_cal_s, y_cal, cv=min(5, len(y_cal)), scoring='accuracy')
        cv_accuracy = np.mean(cv_scores)

        decision_values = clf.decision_function(X_cal_s)
        margins = np.abs(decision_values)
        avg_margin = np.mean(margins)

        probas = clf.predict_proba(X_cal_s)
        pred_entropy = np.mean([entropy(p) for p in probas])

        class_0_mask = y_cal == 0
        class_1_mask = y_cal == 1
        centroid_0 = np.mean(X_cal_s[class_0_mask], axis=0)
        centroid_1 = np.mean(X_cal_s[class_1_mask], axis=0)
        centroid_distance = np.linalg.norm(centroid_0 - centroid_1)

        feature_variance = np.mean(np.var(X_cal_s, axis=0))

        reliability = (
            0.3 * cv_accuracy +
            0.2 * (avg_margin / (avg_margin + 1)) +
            0.2 * (1 - pred_entropy / np.log(2)) +
            0.15 * (centroid_distance / (centroid_distance + 5)) +
            0.15 * min(feature_variance, 1.0)
        )

        return reliability

    except:
        return 0.5

def train_eeg_classifier(X_cal, y_cal):
    """Train EEG SVM classifier"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    clf = SVC(kernel='linear', random_state=42, gamma='scale', probability=True)
    clf.fit(X_cal_s, y_cal)
    return clf, scaler

def train_gaze_classifier(X_cal, y_cal):
    """Train Gaze SVM classifier"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    clf = SVC(kernel='linear', random_state=42, gamma='scale', probability=True)
    clf.fit(X_cal_s, y_cal)
    return clf, scaler

def train_text_proxy_classifier(X_cal, y_cal):
    """Train Text-proxy classifier"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    clf = SVC(kernel='linear', random_state=42, gamma='scale', probability=True)
    clf.fit(X_cal_s, y_cal)
    return clf, scaler

def predict_with_reliability(clf, scaler, X, reliability):
    """Get predictions with reliability weighting"""
    X_s = scaler.transform(X)
    probas = clf.predict_proba(X_s)
    return probas

def run_experiment(seed, model_type):
    """Run reliability weighting experiment"""
    results = []
    calibration_settings = [1, 3, 5, 10, 20, 50]

    for held_out in Y_SUBJECTS:
        print(f"\n  {model_type} - {held_out}:", flush=True)

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

        X_text_proxy = extract_text_proxy_features(X_gaze)

        n_samples = len(y_eeg)
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_samples // 2]
        cal_pool_indices = indices[n_samples // 2:]

        X_test_eeg = X_eeg[test_indices]
        y_test = y_eeg[test_indices]
        X_test_gaze = X_gaze[test_indices]
        X_test_text = X_text_proxy[test_indices]

        X_cal_pool_eeg = X_eeg[cal_pool_indices]
        X_cal_pool_gaze = X_gaze[cal_pool_indices]
        X_cal_pool_text = X_text_proxy[cal_pool_indices]
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
            X_cal_text = X_cal_pool_text[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            eeg_reliability = estimate_calibration_reliability(X_cal_eeg, y_cal)

            if model_type == 'EEG_only':
                clf_eeg, scaler_eeg = train_eeg_classifier(X_cal_eeg, y_cal)
                probas = predict_with_reliability(clf_eeg, scaler_eeg, X_test_eeg, eeg_reliability)
                test_probs = probas[:, 1]

            elif model_type == 'Gaze_only':
                clf_gaze, scaler_gaze = train_gaze_classifier(X_cal_gaze, y_cal)
                probas = predict_with_reliability(clf_gaze, scaler_gaze, X_test_gaze, eeg_reliability)
                test_probs = probas[:, 1]

            elif model_type == 'Static_EEG_Gaze_average':
                clf_eeg, scaler_eeg = train_eeg_classifier(X_cal_eeg, y_cal)
                clf_gaze, scaler_gaze = train_gaze_classifier(X_cal_gaze, y_cal)
                probas_eeg = predict_with_reliability(clf_eeg, scaler_eeg, X_test_eeg, eeg_reliability)
                probas_gaze = predict_with_reliability(clf_gaze, scaler_gaze, X_test_gaze, eeg_reliability)
                test_probs = 0.5 * probas_eeg[:, 1] + 0.5 * probas_gaze[:, 1]

            elif model_type == 'Reliability_weighted_EEG_Gaze':
                clf_eeg, scaler_eeg = train_eeg_classifier(X_cal_eeg, y_cal)
                clf_gaze, scaler_gaze = train_gaze_classifier(X_cal_gaze, y_cal)
                probas_eeg = predict_with_reliability(clf_eeg, scaler_eeg, X_test_eeg, eeg_reliability)
                probas_gaze = predict_with_reliability(clf_gaze, scaler_gaze, X_test_gaze, eeg_reliability)

                w_eeg = eeg_reliability
                w_gaze = 1 - eeg_reliability
                w_sum = w_eeg + w_gaze
                test_probs = (w_eeg * probas_eeg[:, 1] + w_gaze * probas_gaze[:, 1]) / w_sum

            elif model_type == 'Reliability_weighted_EEG_TextProxy':
                clf_eeg, scaler_eeg = train_eeg_classifier(X_cal_eeg, y_cal)
                clf_text, scaler_text = train_text_proxy_classifier(X_cal_text, y_cal)
                probas_eeg = predict_with_reliability(clf_eeg, scaler_eeg, X_test_eeg, eeg_reliability)
                probas_text = predict_with_reliability(clf_text, scaler_text, X_test_text, eeg_reliability)

                w_eeg = eeg_reliability
                w_text = 1 - eeg_reliability
                w_sum = w_eeg + w_text
                test_probs = (w_eeg * probas_eeg[:, 1] + w_text * probas_text[:, 1]) / w_sum

            elif model_type == 'Reliability_weighted_EEG_Gaze_TextProxy':
                clf_eeg, scaler_eeg = train_eeg_classifier(X_cal_eeg, y_cal)
                clf_gaze, scaler_gaze = train_gaze_classifier(X_cal_gaze, y_cal)
                clf_text, scaler_text = train_text_proxy_classifier(X_cal_text, y_cal)
                probas_eeg = predict_with_reliability(clf_eeg, scaler_eeg, X_test_eeg, eeg_reliability)
                probas_gaze = predict_with_reliability(clf_gaze, scaler_gaze, X_test_gaze, eeg_reliability)
                probas_text = predict_with_reliability(clf_text, scaler_text, X_test_text, eeg_reliability)

                w_eeg = eeg_reliability
                w_gaze = (1 - eeg_reliability) * 0.6
                w_text = (1 - eeg_reliability) * 0.4
                w_sum = w_eeg + w_gaze + w_text
                test_probs = (w_eeg * probas_eeg[:, 1] + w_gaze * probas_gaze[:, 1] + w_text * probas_text[:, 1]) / w_sum

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
                'eeg_reliability': eeg_reliability,
                'accuracy': acc,
                'macro_f1': f1,
                'balanced_accuracy': bacc,
                'auroc': auroc
            })

            print(f"    {n_cal_per_class}-shot: Rel={eeg_reliability:.3f}, Acc={acc:.4f}, F1={f1:.4f}", flush=True)

    return results

def main():
    print("="*70)
    print("Reliability-Weighted Multi-Modal Calibration Experiment")
    print("="*70)

    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    all_results = []

    model_types = [
        'EEG_only',
        'Gaze_only',
        'Static_EEG_Gaze_average',
        'Reliability_weighted_EEG_Gaze',
        'Reliability_weighted_EEG_TextProxy',
        'Reliability_weighted_EEG_Gaze_TextProxy'
    ]

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Running: {model_type}")
        print("="*70)

        for seed in [0, 1, 2, 3, 4]:
            print(f"\n--- Seed {seed} ---", flush=True)
            results = run_experiment(seed, model_type)
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "reliability_weighting_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n\nSaved to {output_path}")

    summary = df.groupby(['model', 'n_cal_per_class']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'auroc': ['mean', 'std'],
        'eeg_reliability': ['mean', 'std']
    }).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "reliability_weighting_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary.to_string())

    print("\nDone!")

if __name__ == '__main__':
    main()