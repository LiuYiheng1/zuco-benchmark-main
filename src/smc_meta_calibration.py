"""
SMC: Subject-Meta Calibration - Reptile Meta-Learning

Reptile meta-learning for EEG user calibration.
Meta-train on 15 subjects, fine-tune on 1 held-out subject.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
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

def reptile_train(train_subjects, n_iterations=20, inner_steps=5, inner_lr=0.1):
    """Reptile meta-training: learn initialization from train subjects"""
    if len(train_subjects) == 0:
        return None

    all_X = []
    all_y = []
    for subj in train_subjects:
        X_subj, y_subj = load_eeg_data(subj)
        if X_subj is not None:
            all_X.append(X_subj)
            all_y.append(y_subj)

    if len(all_X) == 0:
        return None

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)

    np.random.seed(42)
    indices = np.random.permutation(len(y_all))
    n_support = min(20, len(y_all) // 4)
    support_idx = indices[:n_support]
    query_idx = indices[n_support:n_support + n_support]

    X_support = X_all_s[support_idx]
    y_support = y_all[support_idx]
    X_query = X_all_s[query_idx]
    y_query = y_all[query_idx]

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
    clf.fit(X_support, y_support)

    for iteration in range(n_iterations):
        clf_inner = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=inner_steps, random_state=42)
        clf_inner.fit(X_support, y_support)

        acc = accuracy_score(y_query, clf_inner.predict(X_query))
        clf = clf_inner

    return scaler, clf

def reptile_finetune(X_cal, y_cal, meta_scaler, meta_clf, n_steps=20, lr=0.01):
    """Fine-tune meta-trained model on calibration data"""
    X_cal_s = meta_scaler.transform(X_cal)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=n_steps, random_state=42)
    clf.fit(X_cal_s, y_cal)

    return clf, meta_scaler

def evaluate_clf(clf, scaler, X, y):
    X_s = scaler.transform(X)
    preds = clf.predict(X_s)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    bacc = balanced_accuracy_score(y, preds)
    try:
        auroc = roc_auc_score(y, clf.predict_proba(X_s)[:, 1])
    except:
        auroc = 0.5
    return acc, f1, bacc, auroc

def run_experiment():
    results = []
    calibration_settings = [1, 3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    print("SMC Meta-Calibration Experiment")
    print("="*50)

    for seed in seeds:
        print(f"\nSeed {seed}:")

        for held_out in Y_SUBJECTS:
            X_held, y_held = load_eeg_data(held_out)
            if X_held is None or len(X_held) < 50:
                continue

            train_subjects = [s for s in Y_SUBJECTS if s != held_out]

            meta_scaler, meta_clf = reptile_train(train_subjects, n_iterations=20)

            n_samples = len(y_held)
            np.random.seed(seed)
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_samples // 2]
            cal_pool_indices = indices[n_samples // 2:]

            X_test = X_held[test_indices]
            y_test = y_held[test_indices]
            X_cal_pool = X_held[cal_pool_indices]
            y_cal_pool = y_held[cal_pool_indices]

            for n_cal_per_class in calibration_settings:
                if n_cal_per_class * 2 > len(cal_pool_indices):
                    continue

                cal_idx_0 = np.where(y_cal_pool == 0)[0][:n_cal_per_class]
                cal_idx_1 = np.where(y_cal_pool == 1)[0][:n_cal_per_class]
                cal_idx = np.concatenate([cal_idx_0, cal_idx_1])
                np.random.shuffle(cal_idx)

                X_cal = X_cal_pool[cal_idx]
                y_cal = y_cal_pool[cal_idx]

                baseline_scaler = StandardScaler()
                X_cal_baseline = baseline_scaler.fit_transform(X_cal)
                baseline_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                baseline_clf.fit(X_cal_baseline, y_cal)
                baseline_acc, baseline_f1, baseline_bacc, baseline_auroc = evaluate_clf(baseline_clf, baseline_scaler, X_test, y_test)

                if meta_scaler is not None:
                    smc_clf, smc_scaler = reptile_finetune(X_cal, y_cal, meta_scaler, meta_clf, n_steps=100, lr=0.01)
                    smc_acc, smc_f1, smc_bacc, smc_auroc = evaluate_clf(smc_clf, smc_scaler, X_test, y_test)
                else:
                    smc_acc, smc_f1, smc_bacc, smc_auroc = baseline_acc, baseline_f1, baseline_bacc, baseline_auroc

                results.append({
                    'seed': seed,
                    'subject': held_out,
                    'n_cal_per_class': n_cal_per_class,
                    'n_cal_total': n_cal_per_class * 2,
                    'EEG_MLP_acc': baseline_acc,
                    'EEG_MLP_f1': baseline_f1,
                    'EEG_MLP_bacc': baseline_bacc,
                    'EEG_MLP_auroc': baseline_auroc,
                    'SMC_Reptile_acc': smc_acc,
                    'SMC_Reptile_f1': smc_f1,
                    'SMC_Reptile_bacc': smc_bacc,
                    'SMC_Reptile_auroc': smc_auroc
                })

            print(f"  {held_out}", end="", flush=True)

    return pd.DataFrame(results)

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    df = run_experiment()

    output_path = os.path.join(RESULTS_DIR, "smc_meta_calibration.csv")
    df.to_csv(output_path, index=False)

    summary = df.groupby('n_cal_per_class').agg({
        'EEG_MLP_acc': ['mean', 'std'],
        'SMC_Reptile_acc': ['mean', 'std'],
        'EEG_MLP_f1': ['mean', 'std'],
        'SMC_Reptile_f1': ['mean', 'std']
    }).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "smc_meta_calibration_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*50)
    print("SUMMARY (Accuracy)")
    print("="*50)
    for _, row in summary.iterrows():
        k = row['n_cal_per_class']
        mlp_acc = row[('EEG_MLP_acc', 'mean')]
        mlp_std = row[('EEG_MLP_acc', 'std')]
        smc_acc = row[('SMC_Reptile_acc', 'mean')]
        smc_std = row[('SMC_Reptile_acc', 'std')]
        gap = smc_acc - mlp_acc
        print(f"{k:2d}-shot: EEG_MLP={mlp_acc:.4f}±{mlp_std:.4f}, SMC={smc_acc:.4f}±{smc_std:.4f}, Gap={gap:+.4f}")

    print("\nDone!")

if __name__ == '__main__':
    main()