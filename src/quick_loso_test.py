import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from datetime import datetime

FEATURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")
Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "loso")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_features(subject, feature_name):
    path = os.path.join(FEATURES_DIR, f"{subject}_{feature_name}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None

def parse_key(key):
    parts = key.split("_")
    subj = parts[0]
    if len(parts) >= 2 and parts[1] == "NR":
        return "NR", True
    elif len(parts) >= 2 and parts[1] == "TSR":
        return "TSR", True
    else:
        return "", False

def load_labeled_data(subjects, feature_name):
    all_X, all_y, all_meta = [], [], []
    for subj in subjects:
        feats = load_features(subj, feature_name)
        if feats is None:
            continue
        for key, values in feats.items():
            label, is_labeled = parse_key(key)
            if not is_labeled:
                continue
            features = np.array(values[:-1], dtype=np.float64)
            label_binary = 1 if label == "NR" else 0
            all_X.append(features)
            all_y.append(label_binary)
            all_meta.append({'subject_id': subj, 'label': label_binary})
    return np.array(all_X), np.array(all_y), all_meta

print("="*60)
print("LOSO-Y SVM EEG-only Experiment (Quick Test)")
print("="*60)

results = []
for held_out in Y_SUBJECTS[:4]:
    train_subjs = [s for s in Y_SUBJECTS if s != held_out]
    X_train, y_train, _ = load_labeled_data(train_subjs, 'electrode_features_all')
    X_test, y_test, _ = load_labeled_data([held_out], 'electrode_features_all')

    print(f"Hold-out: {held_out}, Train: {len(y_train)}, Test: {len(y_test)}, Test NR ratio: {sum(y_test==1)/len(y_test):.4f}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(random_state=0, kernel='linear', gamma='scale')
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    bacc = balanced_accuracy_score(y_test, y_pred)

    results.append({'held_out': held_out, 'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'n_test': len(y_test)})
    print(f"  Acc={acc:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}")

df = np.array([(r['accuracy'], r['macro_f1'], r['balanced_accuracy']) for r in results])
print(f"\nMean: Acc={df[:,0].mean():.4f}±{df[:,0].std():.4f}, F1={df[:,1].mean():.4f}±{df[:,1].std():.4f}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
import pandas as pd
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, f"svm_eeg_loso_quick_{timestamp}.csv"), index=False)
print(f"\nSaved to results/loso/svm_eeg_loso_quick_{timestamp}.csv")