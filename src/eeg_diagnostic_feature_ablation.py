"""
EEG Diagnostic 3: EEG Feature Group Ablation
Tests which frequency bands or electrode regions are most informative
NOTE: This is a preliminary analysis since we don't have explicit documentation
about the 420 EEG feature structure. We assume 84 electrodes x 5 frequency bands.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
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

def run_feature_group_ablation():
    print("="*70)
    print("EEG Diagnostic 3: Feature Group Ablation")
    print("="*70)
    print("Note: 420 features = 84 electrodes x 5 frequency bands")
    print("Assuming band order: theta(0-83), alpha(84-167), beta(168-251), gamma(252-335), delta(336-419)")
    print("Or assuming band order by groups of 84: band0(0-83), band1(84-167), band2(168-251), band3(252-335), band4(336-419)")

    EEG_DIM = 420
    N_ELECTRODES = 84
    N_BANDS = 5

    BAND_RANGES = [
        (0, 84, "band_0_83"),
        (84, 168, "band_84_167"),
        (168, 252, "band_168_251"),
        (252, 336, "band_252_335"),
        (336, 420, "band_336_419"),
    ]

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

        np.random.seed(0)
        indices = np.random.permutation(len(y_train_all))
        val_size = int(len(y_train_all) * 0.1)
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]

        X_tr = X_train_all[train_idx]
        y_tr = y_train_all[train_idx]
        X_val = X_train_all[val_idx]
        y_val = y_train_all[val_idx]
        X_test_h = X_test

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test_h)

        for start, end, name in BAND_RANGES:
            X_tr_band = X_tr_s[:, start:end]
            X_val_band = X_val_s[:, start:end]
            X_test_band = X_test_s[:, start:end]

            clf = SGDClassifier(loss='hinge', random_state=0, max_iter=1000, tol=1e-3)
            clf.fit(X_tr_band, y_tr)
            y_pred = clf.predict(X_test_band)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            bacc = balanced_accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', warn_for=[])

            results.append({
                'held_out': held_out,
                'feature_group': name,
                'feature_range': f'{start}-{end}',
                'n_features': end - start,
                'accuracy': acc,
                'macro_f1': f1,
                'balanced_accuracy': bacc,
                'precision_macro': prec,
                'recall_macro': rec,
                'n_train': len(y_tr),
                'n_test': len(y_test)
            })

            print(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}")

    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, "eeg_feature_group_ablation.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    summary = df.groupby('feature_group').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    }).reset_index()
    summary.columns = ['feature_group', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std', 'bacc_mean', 'bacc_std']
    summary = summary.sort_values('accuracy_mean', ascending=False)

    summary_path = os.path.join(RESULTS_DIR, "eeg_feature_group_ablation_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("FEATURE GROUP ABLATION SUMMARY")
    print("="*70)
    for _, row in summary.iterrows():
        print(f"{row['feature_group']:15s}: Acc={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}")

    return df, summary

if __name__ == '__main__':
    df, summary = run_feature_group_ablation()