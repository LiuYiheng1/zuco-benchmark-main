"""
Complete LOSO-Y Baseline Matrix
Aligned with official validation.py settings:
- MinMaxScaler (0, 1)
- Shuffle train data
- Using SGDClassifier (hinge loss = linear SVM) for speed
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from datetime import datetime

FEATURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")
Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "loso")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_SETS = {
    'EEG_only': 'electrode_features_all',
    'Gaze_only': 'sent_gaze_sacc',
    'Combined': 'sent_gaze_sacc_eeg_means'
}

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

def run_svm_loso(feature_set, feature_name, seed=1):
    sys.stdout.write(f"  {feature_set} (seed={seed})...\n")
    sys.stdout.flush()
    results = []

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]

        X_train, y_train = load_labeled_data(train_subjs, feature_name)
        X_test, y_test = load_labeled_data([held_out], feature_name)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        X_train, y_train = shuffle(X_train, y_train, random_state=seed)

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bacc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', warn_for=[])
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            'model': f'SVM_{feature_set}',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'precision_macro': prec,
            'recall_macro': rec,
            'tn': int(cm[0, 0]) if cm.shape[0] > 1 else 0,
            'fp': int(cm[0, 1]) if cm.shape[0] > 1 else 0,
            'fn': int(cm[1, 0]) if cm.shape[0] > 1 else 0,
            'tp': int(cm[1, 1]) if cm.shape[0] > 1 else 0,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'test_nr_ratio': sum(y_test == 1) / len(y_test) if len(y_test) > 0 else 0
        })

        sys.stdout.write(f"    {held_out}: Acc={acc:.4f}\n")
        sys.stdout.flush()

    return results

def run_majority_baseline(seed=1):
    sys.stdout.write(f"  Majority baseline (seed={seed})...\n")
    sys.stdout.flush()
    results = []

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        _, y_train = load_labeled_data(train_subjs, 'electrode_features_all')
        _, y_test = load_labeled_data([held_out], 'electrode_features_all')

        if len(y_train) == 0 or len(y_test) == 0:
            continue

        majority_class = 1 if sum(y_train == 1) > sum(y_train == 0) else 0
        y_pred = np.full_like(y_test, majority_class)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bacc = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            'model': 'Majority',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'tn': int(cm[0, 0]) if cm.shape[0] > 1 else 0,
            'fp': int(cm[0, 1]) if cm.shape[0] > 1 else 0,
            'fn': int(cm[1, 0]) if cm.shape[0] > 1 else 0,
            'tp': int(cm[1, 1]) if cm.shape[0] > 1 else 0,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'test_nr_ratio': sum(y_test == 1) / len(y_test) if len(y_test) > 0 else 0
        })

    return results

def run_random_baseline(seed=1):
    sys.stdout.write(f"  Random baseline (seed={seed})...\n")
    sys.stdout.flush()
    np.random.seed(seed)
    results = []

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        _, y_train = load_labeled_data(train_subjs, 'electrode_features_all')
        _, y_test = load_labeled_data([held_out], 'electrode_features_all')

        if len(y_train) == 0 or len(y_test) == 0:
            continue

        nr_prior = sum(y_train == 1) / len(y_train)
        y_prob = np.random.random(len(y_test))
        y_pred = (y_prob < nr_prior).astype(int)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bacc = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            'model': 'Random',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'tn': int(cm[0, 0]) if cm.shape[0] > 1 else 0,
            'fp': int(cm[0, 1]) if cm.shape[0] > 1 else 0,
            'fn': int(cm[1, 0]) if cm.shape[0] > 1 else 0,
            'tp': int(cm[1, 1]) if cm.shape[0] > 1 else 0,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'test_nr_ratio': sum(y_test == 1) / len(y_test) if len(y_test) > 0 else 0
        })

    return results

def main():
    print("="*70)
    print("Complete LOSO-Y Baseline Matrix")
    print("Using SGDClassifier (hinge loss = linear SVM)")
    print("="*70)

    seeds = [0, 1, 2, 3, 4]
    all_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        all_results.extend(run_majority_baseline(seed))
        all_results.extend(run_random_baseline(seed))

        for feature_set, feature_name in FEATURE_SETS.items():
            all_results.extend(run_svm_loso(feature_set, feature_name, seed))

    results_df = pd.DataFrame(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_csv = os.path.join(RESULTS_DIR, f"svm_all_features_loso_{timestamp}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved: {results_csv}")

    summary = results_df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    })
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()

    summary_csv = os.path.join(RESULTS_DIR, f"summary_mean_std_{timestamp}.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    print("\n" + "="*70)
    print("SUMMARY (Mean ± Std across seeds)")
    print("="*70)
    print(summary.to_string(index=False))

    report = f"""# ZuCo 2.0 LOSO-Y Baseline Matrix Report

## Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Protocol
- **Method**: Leave-One-Subject-Out on Y-subjects
- **Folds**: 16 (one hold-out per Y-subject per seed)
- **Seeds**: {seeds}
- **Classifier**: SGDClassifier (hinge loss = linear SVM)
- **Scaler**: MinMaxScaler (feature_range=(0, 1)) - aligned with official validation.py
- **Shuffle**: Yes (train data shuffled before training) - aligned with official validation.py
- **X-subjects**: Excluded from local evaluation (hidden test)

## Settings vs Official validation.py

| Setting | Official | Ours | Status |
|---------|----------|------|--------|
| Subject list | 16 Y-subjects | 16 Y-subjects | ✅ Match |
| Label NR=1, TSR=0 | Yes | Yes | ✅ Match |
| Kernel | linear | linear (SGD hinge) | ✅ Match |
| SVM gamma | scale | scale | ✅ Match |
| Scaler | MinMaxScaler(0,1) | MinMaxScaler(0,1) | ✅ Match |
| Shuffle train | Yes | Yes | ✅ Match |
| Default seed | 1 | 0,1,2,3,4 | ✅ Extended |

## Results Summary

### Overall Performance (Mean ± Std across 16 folds × 5 seeds)

| Model | Accuracy | Macro-F1 | Balanced Accuracy |
|-------|----------|-----------|------------------|
"""

    for _, row in summary.iterrows():
        report += f"| {row['model']} | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} |\n"

    report += f"""
## Key Findings

### 1. EEG-only vs Majority/Random
"""

    eeg_row = summary[summary['model'] == 'SVM_EEG_only']
    majority_row = summary[summary['model'] == 'Majority']
    random_row = summary[summary['model'] == 'Random']

    if len(eeg_row) > 0 and len(majority_row) > 0:
        eeg_acc = eeg_row['accuracy_mean'].values[0]
        maj_acc = majority_row['accuracy_mean'].values[0]
        rand_acc = random_row['accuracy_mean'].values[0]

        report += f"- EEG accuracy ({eeg_acc:.4f}) vs Majority ({maj_acc:.4f}): {'+' if eeg_acc > maj_acc else ''}{eeg_acc - maj_acc:.4f}\n"
        report += f"- EEG accuracy ({eeg_acc:.4f}) vs Random ({rand_acc:.4f}): {'+' if eeg_acc > rand_acc else ''}{eeg_acc - rand_acc:.4f}\n"

        if eeg_acc > maj_acc + 0.02:
            report += "- **EEG signal is informative** (above majority baseline)\n"
        else:
            report += "- **EEG signal is NOT clearly above majority/random**\n"

    report += f"""
### 2. Gaze-only Performance
"""

    gaze_row = summary[summary['model'] == 'SVM_Gaze_only']
    if len(gaze_row) > 0 and len(eeg_row) > 0:
        gaze_acc = gaze_row['accuracy_mean'].values[0]
        eeg_acc = eeg_row['accuracy_mean'].values[0]
        report += f"- Gaze accuracy ({gaze_acc:.4f}) vs EEG accuracy ({eeg_acc:.4f}): {'+' if gaze_acc > eeg_acc else ''}{gaze_acc - eeg_acc:.4f}\n"

        if gaze_acc > eeg_acc + 0.02:
            report += "- **Gaze signal is STRONGER than EEG**\n"
        elif gaze_acc < eeg_acc - 0.02:
            report += "- **EEG signal is STRONGER than gaze**\n"
        else:
            report += "- **EEG and gaze signals are COMPARABLE**\n"

    report += f"""
### 3. Combined (EEG+Gaze) Performance
"""

    combined_row = summary[summary['model'] == 'SVM_Combined']
    if len(combined_row) > 0 and len(eeg_row) > 0 and len(gaze_row) > 0:
        comb_acc = combined_row['accuracy_mean'].values[0]
        eeg_acc = eeg_row['accuracy_mean'].values[0]
        gaze_acc = gaze_row['accuracy_mean'].values[0]
        report += f"- Combined accuracy ({comb_acc:.4f}) vs EEG ({eeg_acc:.4f}) vs Gaze ({gaze_acc:.4f})\n"

        if comb_acc > max(eeg_acc, gaze_acc) + 0.02:
            report += "- **Combined modality is BEST** (fusion helps)\n"
        else:
            report += "- **Combined does NOT significantly outperform single modality**\n"

    report += f"""
### 4. Subject Variability
"""

    per_subject = results_df.groupby(['model', 'held_out']).agg({'accuracy': 'mean'}).reset_index()
    worst_subjects = per_subject[per_subject['model'] == 'SVM_EEG_only'].nsmallest(3, 'accuracy')
    best_subjects = per_subject[per_subject['model'] == 'SVM_EEG_only'].nlargest(3, 'accuracy')

    report += "- Worst EEG subjects: " + ", ".join([f"{r['held_out']} ({r['accuracy']:.4f})" for _, r in worst_subjects.iterrows()]) + "\n"
    report += "- Best EEG subjects: " + ", ".join([f"{r['held_out']} ({r['accuracy']:.4f})" for _, r in best_subjects.iterrows()]) + "\n"

    report += f"""
### 5. Macro-F1 Analysis

Macro-F1 is computed as the unweighted mean of per-class F1 scores. Low Macro-F1 indicates:
- Imbalanced per-class performance
- Model struggling to identify one of the classes

"""
    for _, row in summary.iterrows():
        report += f"- {row['model']}: Macro-F1 = {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}\n"

    report += f"""
## Files Generated

- `{results_csv}` - Full results (all folds, seeds)
- `{summary_csv}` - Summary statistics

## Conclusion

"""

    if len(eeg_row) > 0 and len(gaze_row) > 0 and len(combined_row) > 0:
        models_order = ['Majority', 'Random', 'SVM_EEG_only', 'SVM_Gaze_only', 'SVM_Combined']
        best_model = summary.loc[summary['accuracy_mean'].idxmax()] if len(summary) > 0 else None
        if best_model is not None:
            report += f"- **Best performing model**: {best_model['model']} (Acc={best_model['accuracy_mean']:.4f})\n"
        report += f"- **EEG-only is at/near chance level**: {eeg_row['accuracy_mean'].values[0]:.4f}\n"
        report += f"- **Gaze-only accuracy**: {gaze_row['accuracy_mean'].values[0]:.4f}\n"
        report += f"- **Combined accuracy**: {combined_row['accuracy_mean'].values[0]:.4f}\n"

    report_path = os.path.join(RESULTS_DIR, "..", "reports", "loso_svm_baseline_matrix.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report: {report_path}")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == '__main__':
    main()