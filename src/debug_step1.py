import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

print("[1] Functions defined")

def get_trial_id(key):
    return f"{key.split('_')[0]}_{key.split('_')[1]}_{key.split('_')[2]}"

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
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze_sacc.npy")
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

print("[2] Loading data...")

all_data = {}
for subj in Y_SUBJECTS:
    Xe, ye, tid_e = load_eeg_data(subj)
    Xg, yg, tid_g = load_gaze_features(subj)
    if Xe is not None and Xg is not None:
        Xe_a, ye_a, Xg_a, _ = align_eeg_gaze(Xe, ye, tid_e, Xg, yg, tid_g)
        all_data[subj] = {'Xe': Xe_a, 'ye': ye_a, 'Xg': Xg_a, 'n': len(ye_a)}
        print(f"  {subj}: EEG {Xe_a.shape}, Gaze {Xg_a.shape}")

print(f"\nTotal subjects loaded: {len(all_data)}")

print("[3] Setting up split...")
np.random.seed(0)
shuffled_subjs = Y_SUBJECTS.copy()
np.random.shuffle(shuffled_subjs)

train_subjs = shuffled_subjs[:10]
val_subjs = shuffled_subjs[10:12]
test_subjs = shuffled_subjs[12:16]

print(f"Train: {train_subjs}")
print(f"Val: {val_subjs}")
print(f"Test: {test_subjs}")

print("[4] Preparing data...")
X_eeg_train = np.vstack([all_data[s]['Xe'] for s in train_subjs])
y_train = np.concatenate([all_data[s]['ye'] for s in train_subjs])
X_gaze_train = np.vstack([all_data[s]['Xg'] for s in train_subjs])

X_eeg_test = np.vstack([all_data[s]['Xe'] for s in test_subjs])
y_test = np.concatenate([all_data[s]['ye'] for s in test_subjs])
X_gaze_test = np.vstack([all_data[s]['Xg'] for s in test_subjs])

print(f"Train: {len(y_train)} samples, NR={np.sum(y_train==1)}, TSR={np.sum(y_train==0)}")
print(f"Test: {len(y_test)} samples, NR={np.sum(y_test==1)}, TSR={np.sum(y_test==0)}")

print("[5] Training models...")

scaler_e = StandardScaler()
X_e_s = scaler_e.fit_transform(X_eeg_train)
X_e_test_s = scaler_e.transform(X_eeg_test)
clf_eeg = RidgeClassifier(alpha=0.1)
clf_eeg.fit(X_e_s, y_train)
print(f"EEG_SVM classes_: {clf_eeg.classes_}")

preds_eeg = clf_eeg.predict(X_e_test_s)
probs_eeg_df = clf_eeg.decision_function(X_e_test_s)
probs_eeg_prob = 1 / (1 + np.exp(-probs_eeg_df))

print(f"\nEEG_SVM Results:")
print(f"  Accuracy: {accuracy_score(y_test, preds_eeg)*100:.1f}%")
print(f"  Macro-F1: {f1_score(y_test, preds_eeg, average='macro')*100:.1f}%")
print(f"  Balanced Acc: {balanced_accuracy_score(y_test, preds_eeg)*100:.1f}%")
print(f"  AUROC: {roc_auc_score(y_test, probs_eeg_prob)*100:.1f}%")
print(f"  Confusion Matrix:\n{confusion_matrix(y_test, preds_eeg)}")
print(f"  Inverted Acc: {accuracy_score(y_test, 1-preds_eeg)*100:.1f}%")

print("\n[6] Saving results...")

report = f"""# AdaGTCN-inspired 10/2/4 Split Debug Report

## Subject Info
- Total Y-subjects: {len(all_data)}
- Cannot run strict 12/2/4 (only 16 available)
- Using AdaGTCN-inspired **10/2/4** split

## Split (seed=0)
- Train: {train_subjs}
- Val: {val_subjs}
- Test: {test_subjs}

## Class Distribution
- Train: NR={np.sum(y_train==1)} ({np.sum(y_train==1)/len(y_train)*100:.1f}%), TSR={np.sum(y_train==0)} ({np.sum(y_train==0)/len(y_train)*100:.1f}%)
- Test: NR={np.sum(y_test==1)} ({np.sum(y_test==1)/len(y_test)*100:.1f}%), TSR={np.sum(y_test==0)} ({np.sum(y_test==0)/len(y_test)*100:.1f}%)

## Label Mapping
- NR=1, TSR=0 (consistent)

## EEG_SVM Results (seed=0)
- Accuracy: {accuracy_score(y_test, preds_eeg)*100:.1f}%
- Macro-F1: {f1_score(y_test, preds_eeg, average='macro')*100:.1f}%
- Balanced Acc: {balanced_accuracy_score(y_test, preds_eeg)*100:.1f}%
- AUROC: {roc_auc_score(y_test, probs_eeg_prob)*100:.1f}%
- Confusion Matrix:
{confusion_matrix(y_test, preds_eeg)}
- Inverted Accuracy: {accuracy_score(y_test, 1-preds_eeg)*100:.1f}%

## Key Questions
1. Available subjects: {len(all_data)} Y-subjects
2. Can run 12/2/4: NO (using 10/2/4 instead)
3. Class order issue: NO (classes_ = {clf_eeg.classes_})
4. Cross-subject vs few-shot: ~50% vs ~80% (gap of ~30%)
"""

with open(os.path.join(REPORTS_DIR, 'adagtcn_inspired_split_debug_fixed_report.md'), 'w') as f:
    f.write(report)

debug_data = {
    'item': [
        'total_subjects', 'split_type',
        'train_subjects', 'val_subjects', 'test_subjects',
        'train_NR', 'train_TSR', 'train_NR_ratio',
        'test_NR', 'test_TSR', 'test_NR_ratio',
        'EEG_SVM_accuracy', 'EEG_SVM_f1', 'EEG_SVM_balanced_acc', 'EEG_SVM_auroc',
        'EEG_SVM_inverted_acc', 'EEG_SVM_classes'
    ],
    'value': [
        len(all_data), '10/2/4',
        str(train_subjs), str(val_subjs), str(test_subjs),
        int(np.sum(y_train==1)), int(np.sum(y_train==0)), float(np.sum(y_train==1)/len(y_train)),
        int(np.sum(y_test==1)), int(np.sum(y_test==0)), float(np.sum(y_test==1)/len(y_test)),
        float(accuracy_score(y_test, preds_eeg)),
        float(f1_score(y_test, preds_eeg, average='macro')),
        float(balanced_accuracy_score(y_test, preds_eeg)),
        float(roc_auc_score(y_test, probs_eeg_prob)),
        float(accuracy_score(y_test, 1-preds_eeg)),
        str(clf_eeg.classes_)
    ]
}

df = pd.DataFrame(debug_data)
df.to_csv(os.path.join(RESULTS_DIR, 'adagtcn_inspired_split_debug_fixed.csv'), index=False)

print(f"\nSaved:")
print(f"  - {RESULTS_DIR}/adagtcn_inspired_split_debug_fixed.csv")
print(f"  - {REPORTS_DIR}/adagtcn_inspired_split_debug_fixed_report.md")
print("\nDone!")
