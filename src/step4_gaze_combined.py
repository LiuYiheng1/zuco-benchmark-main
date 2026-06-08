"""
EEG Adaptation - Step 4: Gaze-only and Combined
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

FEATURES_DIR = "features"
RESULTS_DIR = "results/eeg_adaptation"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

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

def run_gaze_only(seed):
    print(f"Gaze-only (seed={seed})...", flush=True)
    results = []

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_gaze_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test, y_test = load_gaze_data(held_out)

        if len(X_train_all) == 0 or X_test is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        np.random.seed(seed)
        indices = np.random.permutation(len(y_train_all))
        val_size = int(len(y_train_all) * 0.1)
        train_idx = indices[val_size:]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_all[train_idx])
        X_test_s = scaler.transform(X_test)

        clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
        clf.fit(X_tr, y_train_all[train_idx])
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bacc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', warn_for=[])
        cm = confusion_matrix(y_test, y_pred)

        try:
            auroc = roc_auc_score(y_test, clf.decision_function(X_test_s))
        except:
            auroc = 0.5

        results.append({
            'model': 'Gaze_only',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'precision_macro': prec,
            'recall_macro': rec,
            'auroc': auroc,
            'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])
        })
        print(f"  {held_out}: {acc:.4f}", flush=True)

    return results

def run_combined(seed):
    print(f"Combined EEG+Gaze (seed={seed})...", flush=True)
    results = []

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_combined_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test, y_test = load_combined_data(held_out)

        if len(X_train_all) == 0 or X_test is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        np.random.seed(seed)
        indices = np.random.permutation(len(y_train_all))
        val_size = int(len(y_train_all) * 0.1)
        train_idx = indices[val_size:]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_all[train_idx])
        X_test_s = scaler.transform(X_test)

        clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
        clf.fit(X_tr, y_train_all[train_idx])
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bacc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', warn_for=[])
        cm = confusion_matrix(y_test, y_pred)

        try:
            auroc = roc_auc_score(y_test, clf.decision_function(X_test_s))
        except:
            auroc = 0.5

        results.append({
            'model': 'Combined',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'precision_macro': prec,
            'recall_macro': rec,
            'auroc': auroc,
            'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])
        })
        print(f"  {held_out}: {acc:.4f}", flush=True)

    return results

if __name__ == '__main__':
    all_results = []
    for seed in [0, 1, 2, 3, 4]:
        all_results.extend(run_gaze_only(seed))
        all_results.extend(run_combined(seed))

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "gaze_combined_5seeds.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(df.groupby('model')['accuracy'].agg(['mean', 'std']))