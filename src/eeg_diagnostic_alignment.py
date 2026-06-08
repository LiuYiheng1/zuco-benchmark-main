"""
EEG Diagnostic 3: Subject Normalization / Alignment Experiment
Tests if different normalization methods can improve EEG LOSO performance
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import SGDClassifier

FEATURES_DIR = "features"
RESULTS_DIR = "results/eeg_diagnostics"
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

def run_alignment_experiment():
    print("="*70)
    print("EEG Diagnostic 4: Subject Normalization / Alignment")
    print("="*70)

    results = []

    for held_out in Y_SUBJECTS:
        print(f"\n--- Held-out: {held_out} ---")

        train_subjs = [s for s in Y_SUBJECTS if s != held_out]

        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test, y_test = load_eeg_data(held_out)

        if len(X_train_all) == 0 or X_test is None or len(X_test) == 0:
            print(f"  Skipping {held_out} - no data")
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        print(f"  Train: {len(y_train_all)}, Test: {len(y_test)}")

        for method in ['global_minmax', 'train_std', 'subject_zscore']:
            np.random.seed(0)
            indices = np.random.permutation(len(y_train_all))
            val_size = int(len(y_train_all) * 0.1)
            train_idx = indices[val_size:]
            val_idx = indices[:val_size]

            X_tr = X_train_all[train_idx]
            y_tr = y_train_all[train_idx]
            X_val = X_train_all[val_idx]
            y_val = y_train_all[val_idx]

            if method == 'global_minmax':
                scaler = MinMaxScaler(feature_range=(0, 1))
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)
                X_test_s = scaler.transform(X_test)
            elif method == 'train_std':
                scaler = StandardScaler(with_mean=True, with_std=True)
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)
                X_test_s = scaler.transform(X_test)
            elif method == 'subject_zscore':
                train_means = X_tr.mean(axis=0)
                train_stds = X_tr.std(axis=0) + 1e-8
                X_tr_s = (X_tr - train_means) / train_stds
                X_val_s = (X_val - train_means) / train_stds
                X_test_s = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0) + 1e-8)

            clf = SGDClassifier(loss='hinge', random_state=0, max_iter=1000, tol=1e-3)
            clf.fit(X_tr_s, y_tr)
            y_pred = clf.predict(X_test_s)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            bacc = balanced_accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', warn_for=[])

            results.append({
                'held_out': held_out,
                'method': method,
                'accuracy': acc,
                'macro_f1': f1,
                'balanced_accuracy': bacc,
                'precision_macro': prec,
                'recall_macro': rec,
                'n_train': len(y_tr),
                'n_test': len(y_test)
            })

            print(f"  {method}: Acc={acc:.4f}, F1={f1:.4f}")

    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, "eeg_alignment_loso.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    summary = df.groupby('method').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    }).reset_index()
    summary.columns = ['method', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std', 'bacc_mean', 'bacc_std']
    summary = summary.sort_values('accuracy_mean', ascending=False)

    summary_path = os.path.join(RESULTS_DIR, "eeg_alignment_loso_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("ALIGNMENT RESULTS SUMMARY")
    print("="*70)
    for _, row in summary.iterrows():
        print(f"{row['method']:20s}: Acc={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}")

    return df, summary

if __name__ == '__main__':
    df, summary = run_alignment_experiment()