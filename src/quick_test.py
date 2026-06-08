import os
import sys

os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

print('Importing...', flush=True)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

print('Imports done', flush=True)

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
GAZE_FEATURE_FILE = 'sent_gaze_sacc'

def get_trial_id(key):
    parts = key.split('_')
    return f"{parts[0]}_{parts[1]}_{parts[2]}"

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    X, y, trial_ids = [], [], []
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
        trial_ids.append(get_trial_id(key))
    return np.array(X), np.array(y), trial_ids

def load_gaze_features(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_{GAZE_FEATURE_FILE}.npy")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    X, y, trial_ids = [], [], []
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 2 and parts[1] == "NR":
            label = 1
        elif len(parts) >= 2 and parts[1] == "TSR":
            label = 0
        else:
            continue
        numeric_vals = [float(v) for v in values[:-1]]
        features = np.array(numeric_vals, dtype=np.float64)
        X.append(features)
        y.append(label)
        trial_ids.append(get_trial_id(key))
    return np.array(X), np.array(y), trial_ids

def align_eeg_gaze(X_eeg, y_eeg, trial_ids_eeg, X_gaze, y_gaze, trial_ids_gaze):
    gaze_dict = {tid: (X_gaze[i], y_gaze[i]) for i, tid in enumerate(trial_ids_gaze)}
    X_eeg_aligned, y_eeg_aligned, X_gaze_aligned, y_gaze_aligned = [], [], [], []
    for i, tid in enumerate(trial_ids_eeg):
        if tid in gaze_dict:
            X_eeg_aligned.append(X_eeg[i])
            y_eeg_aligned.append(y_eeg[i])
            X_gaze_aligned.append(gaze_dict[tid][0])
            y_gaze_aligned.append(gaze_dict[tid][1])
    return (np.array(X_eeg_aligned), np.array(y_eeg_aligned),
            np.array(X_gaze_aligned), np.array(y_gaze_aligned))

def balanced_random_sampling(y_pool, n_per_class):
    class_0_idx = np.where(y_pool == 0)[0]
    class_1_idx = np.where(y_pool == 1)[0]
    np.random.shuffle(class_0_idx)
    np.random.shuffle(class_1_idx)
    n0 = min(n_per_class, len(class_0_idx))
    n1 = min(n_per_class, len(class_1_idx))
    selected = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])
    np.random.shuffle(selected)
    return selected

print('Functions defined', flush=True)

# Quick test
held_out = 'YAC'
X_eeg_test, y_test, trial_ids_eeg = load_eeg_data(held_out)
X_gaze_test, y_gaze, trial_ids_gaze = load_gaze_features(held_out)
print(f'Loaded {held_out}: EEG {X_eeg_test.shape}, Gaze {X_gaze_test.shape}', flush=True)

X_eeg_test_aligned, y_eeg_test_aligned, X_gaze_test_aligned, y_gaze_test_aligned = align_eeg_gaze(
    X_eeg_test, y_test, trial_ids_eeg, X_gaze_test, y_gaze, trial_ids_gaze)
print(f'Aligned: EEG {X_eeg_test_aligned.shape}, Gaze {X_gaze_test_aligned.shape}', flush=True)

n_samples = len(y_eeg_test_aligned)
np.random.seed(0)
indices = np.random.permutation(n_samples)
test_size = n_samples // 3
test_indices = indices[:test_size]
cal_pool_indices = indices[test_size:]
print(f'Cal pool size: {len(cal_pool_indices)}, Test size: {len(test_indices)}', flush=True)

y_cal_pool = y_eeg_test_aligned[cal_pool_indices]
print(f'Cal pool labels: class0={np.sum(y_cal_pool==0)}, class1={np.sum(y_cal_pool==1)}', flush=True)

n_cal = 5
cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
print(f'Selected cal indices: {len(cal_idx)}', flush=True)
X_eeg_cal = X_eeg_test_aligned[cal_pool_indices][cal_idx]
y_cal = y_cal_pool[cal_idx]
print(f'X_eeg_cal shape: {X_eeg_cal.shape}, y_cal: {y_cal}', flush=True)

print('Testing SVM...', flush=True)
scaler_eeg = StandardScaler()
X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
X_eeg_test_s = scaler_eeg.transform(X_eeg_test_aligned[test_indices])
clf = SVC(kernel='rbf', probability=True, random_state=42)
clf.fit(X_eeg_cal_s, y_cal)
print('SVM fitted!', flush=True)

print('Done!', flush=True)