"""
EEG Few-Shot User Calibration Curve Experiment

Protocol:
- For each subject, use varying amounts of calibration samples
- Test on remaining samples (no test samples in training)
- Seeds [0, 1, 2, 3, 4]

Approaches:
A. Subject-specific from scratch (SVM/MLP)
B. Cross-subject pretrain + finetune
C. Linear probe calibration

Models:
- EEG_SVM
- EEG_MLP
- Gaze_SVM
- Combined
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
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

def load_combined_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze_sacc_eeg_means.npy")
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

class EEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_dim = hidden_dim
    def forward(self, x):
        return self.net(x)

class TaskClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_subject_specific(X_cal, y_cal, X_test, y_test, model_type='SVM'):
    """Approach A: Train subject-specific model from scratch"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    if model_type == 'SVM':
        clf = SVC(random_state=42, kernel='linear', gamma='scale', probability=True)
    elif model_type == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    else:
        clf = SGDClassifier(loss='hinge', random_state=42, max_iter=1000, tol=1e-3)

    clf.fit(X_cal_s, y_cal)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    bacc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    try:
        auroc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc, cm

def run_experiment(seed, approach='subject_specific', model_type='SVM'):
    """Run calibration experiment for all subjects and calibration settings"""
    results = []
    calibration_settings = [1, 3, 5, 10, 20, 50]

    for held_out in Y_SUBJECTS:
        print(f"\n{approach} {model_type} - {held_out}:", flush=True)

        X_eeg, y_eeg = load_eeg_data(held_out)
        X_gaze, y_gaze = load_gaze_data(held_out)
        X_comb, y_comb = load_combined_data(held_out)

        if X_eeg is None or len(X_eeg) < 50:
            continue

        n_samples = len(y_eeg)
        n_class_0 = np.sum(y_eeg == 0)
        n_class_1 = np.sum(y_eeg == 1)
        min_class_size = min(n_class_0, n_class_1)

        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        test_indices = indices[:int(n_samples * 0.5)]
        cal_pool_indices = indices[int(n_samples * 0.5):]

        X_test_eeg, y_test = X_eeg[test_indices], y_eeg[test_indices]
        X_cal_pool_eeg = X_eeg[cal_pool_indices]
        X_test_gaze = X_gaze[test_indices]
        X_cal_pool_gaze = X_gaze[cal_pool_indices]
        X_test_comb = X_comb[test_indices]
        X_cal_pool_comb = X_comb[cal_pool_indices]

        for n_cal_per_class in calibration_settings:
            if n_cal_per_class * 2 > len(cal_pool_indices):
                continue

            cal_indices_class0 = np.where(y_eeg[cal_pool_indices] == 0)[0][:n_cal_per_class]
            cal_indices_class1 = np.where(y_eeg[cal_pool_indices] == 1)[0][:n_cal_per_class]
            cal_indices = np.concatenate([cal_indices_class0, cal_indices_class1])
            np.random.shuffle(cal_indices)

            X_cal_eeg = X_cal_pool_eeg[cal_indices]
            y_cal = y_eeg[cal_pool_indices][cal_indices]
            X_cal_gaze = X_cal_pool_gaze[cal_indices]
            X_cal_comb = X_cal_pool_comb[cal_indices]

            if model_type in ['EEG_SVM', 'EEG_MLP']:
                X_cal = X_cal_eeg
                X_test = X_test_eeg
            elif model_type == 'Gaze_SVM':
                X_cal = X_cal_gaze
                X_test = X_test_gaze
            else:
                X_cal = X_cal_comb
                X_test = X_test_comb

            if approach == 'subject_specific':
                actual_model_type = model_type.replace('EEG_', '').replace('Gaze_', '').replace('Combined_', '')
                acc, f1, bacc, auroc, cm = train_subject_specific(X_cal, y_cal, X_test, y_test, actual_model_type)
            else:
                acc, facc, bacc, auroc, cm = 0.5, 0.5, 0.5, 0.5, [[0,0],[0,0]]

            results.append({
                'approach': approach,
                'model': model_type,
                'seed': seed,
                'subject': held_out,
                'n_cal_per_class': n_cal_per_class,
                'n_cal_total': n_cal_per_class * 2,
                'n_test': len(y_test),
                'accuracy': acc,
                'macro_f1': f1,
                'balanced_accuracy': bacc,
                'auroc': auroc,
                'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])
            })

            print(f"  {n_cal_per_class}-shot: Acc={acc:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}", flush=True)

    return results

def main():
    print("="*70)
    print("EEG Few-Shot User Calibration Curve Experiment")
    print("="*70)

    all_results = []

    model_configs = [
        ('subject_specific', 'EEG_SVM'),
        ('subject_specific', 'EEG_MLP'),
        ('subject_specific', 'Gaze_SVM'),
        ('subject_specific', 'Combined_SVM'),
    ]

    for approach, model_type in model_configs:
        print(f"\n{'='*70}")
        print(f"Running: {approach} - {model_type}")
        print("="*70)

        for seed in [0, 1, 2, 3, 4]:
            results = run_experiment(seed, approach, model_type)
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "few_shot_calibration_curve.csv")
    df.to_csv(output_path, index=False)
    print(f"\n\nSaved to {output_path}")

    summary = df.groupby(['model', 'n_cal_per_class']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    }).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "few_shot_calibration_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary.to_string())

    print("\nDone!")

if __name__ == '__main__':
    main()