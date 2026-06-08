"""
Simplified LOSO-Y Experiment
"""
import os
import sys
import numpy as np
import pandas as pd
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
    if len(parts) >= 2 and parts[1] == "NR":
        return "NR", True
    elif len(parts) >= 2 and parts[1] == "TSR":
        return "TSR", True
    return "", False

def load_labeled_data(subjects, feature_name):
    all_X, all_y = [], []
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
    return np.array(all_X), np.array(all_y)

def run_experiment():
    print("="*60)
    print("LOSO-Y Full Experiment")
    print("="*60)

    seeds = [0, 1, 2]
    models = {
        'SVM_EEG': 'electrode_features_all',
        'SVM_GAZE': 'sent_gaze_sacc',
    }

    all_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        for model_name, feature_set in models.items():
            print(f"  Running {model_name}...")
            for held_out in Y_SUBJECTS:
                train_subjs = [s for s in Y_SUBJECTS if s != held_out]

                X_train, y_train = load_labeled_data(train_subjs, feature_set)
                X_test, y_test = load_labeled_data([held_out], feature_set)

                if len(X_train) == 0 or len(X_test) == 0:
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = SVC(random_state=seed, kernel='linear', gamma='scale', probability=True)
                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                bacc = balanced_accuracy_score(y_test, y_pred)

                all_results.append({
                    'model': model_name,
                    'seed': seed,
                    'held_out': held_out,
                    'accuracy': acc,
                    'macro_f1': f1,
                    'balanced_accuracy': bacc,
                    'n_train': len(y_train),
                    'n_test': len(y_test),
                    'test_nr_ratio': sum(y_test==1)/len(y_test) if len(y_test) > 0 else 0
                })

                print(f"    {held_out}: Acc={acc:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}")

    results_df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(RESULTS_DIR, f"loso_all_results_{timestamp}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved to {results_csv}")

    summary = results_df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    }).reset_index()
    summary.columns = ['model', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std', 'balanced_accuracy_mean', 'balanced_accuracy_std']

    summary_csv = os.path.join(RESULTS_DIR, f"summary_mean_std_{timestamp}.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved to {summary_csv}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))

    return results_df, summary

if __name__ == '__main__':
    results_df, summary = run_experiment()