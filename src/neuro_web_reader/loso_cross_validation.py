"""
Leave-One-Subject-Out Cross-Validation on Training Data
This provides local evaluation metrics without needing test set labels.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if os.path.basename(PROJECT_ROOT) == 'src':
    SRC_DIR = PROJECT_ROOT
else:
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

FEATURES_DIR = os.path.join(SRC_DIR, "features")
RESULTS_DIR = os.path.join(SRC_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_features(subject, feature_set):
    path = os.path.join(FEATURES_DIR, f"{subject}_{feature_set}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None

def parse_key(key):
    parts = key.split("_")
    return parts[0], parts[1], int(parts[2]), int(parts[3])

def prepare_data(subjects, feature_set, label_col=-1):
    all_X, all_y, all_subj = [], [], []
    for subj in subjects:
        feats = load_features(subj, feature_set)
        if feats is None:
            continue
        for key, values in feats.items():
            _, label, _, _ = parse_key(key)
            features = np.array(values[:label_col], dtype=np.float64)
            label_binary = 1 if label == "NR" else 0
            all_X.append(features)
            all_y.append(label_binary)
            all_subj.append(subj)
    return np.array(all_X), np.array(all_y), all_subj

def loso_cv(feature_set, subjects, seeds=[0, 1, 2]):
    """Leave-One-Subject-Out Cross-Validation"""
    print(f"\n{'='*60}")
    print(f"LOSO CV for: {feature_set}")
    print(f"{'='*60}")

    all_results = []
    all_predictions = []

    for seed in seeds:
        np.random.seed(seed)
        print(f"\nSeed {seed}:")

        for held_out_subj in subjects:
            train_subjects = [s for s in subjects if s != held_out_subj]

            X_train, y_train, _ = prepare_data(train_subjects, feature_set)
            X_test, y_test, test_subjects = prepare_data([held_out_subj], feature_set)

            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = SVC(random_state=seed, kernel='linear', gamma='scale', probability=True)
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            y_prob = clf.predict_proba(X_test_s)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            bacc = balanced_accuracy_score(y_test, y_pred)

            print(f"  Hold-out {held_out_subj}: Acc={acc:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}")

            for i, (subj, true_l, pred_l, prob) in enumerate(zip(test_subjects, y_test, y_pred, y_prob)):
                all_predictions.append({
                    'seed': seed,
                    'held_out_subject': held_out_subj,
                    'subject_id': subj,
                    'true_label': int(true_l),
                    'pred_label': int(pred_l),
                    'pred_prob': float(prob)
                })

            all_results.append({
                'seed': seed,
                'held_out_subject': held_out_subj,
                'accuracy': acc,
                'macro_f1': f1,
                'balanced_accuracy': bacc,
                'n_test_samples': len(y_test)
            })

    return pd.DataFrame(all_results), pd.DataFrame(all_predictions)

def main():
    print("="*70)
    print("Leave-One-Subject-Out Cross-Validation (Training Data Only)")
    print("="*70)

    feature_sets = [
        ('EEG_only', 'electrode_features_all'),
        ('Gaze_only', 'sent_gaze_sacc'),
        ('Combined', 'sent_gaze_sacc_eeg_means')
    ]

    all_results = []
    all_predictions = []

    for name, feat_set in feature_sets:
        results_df, preds_df = loso_cv(feat_set, TRAIN_SUBJECTS, seeds=[0, 1, 2, 3, 4])
        results_df['feature_set'] = name
        preds_df['feature_set'] = name
        all_results.append(results_df)
        all_predictions.append(preds_df)

        print(f"\n{name} Summary (across seeds):")
        seed_summary = results_df.groupby('seed').agg({
            'accuracy': 'mean',
            'macro_f1': 'mean',
            'balanced_accuracy': 'mean'
        })
        print(f"  Accuracy: {seed_summary['accuracy'].mean():.4f} ± {seed_summary['accuracy'].std():.4f}")
        print(f"  Macro-F1: {seed_summary['macro_f1'].mean():.4f} ± {seed_summary['macro_f1'].std():.4f}")
        print(f"  BAcc: {seed_summary['balanced_accuracy'].mean():.4f} ± {seed_summary['balanced_accuracy'].std():.4f}")

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_preds = pd.concat(all_predictions, ignore_index=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_csv = os.path.join(RESULTS_DIR, f"loso_cv_results_{timestamp}.csv")
    combined_results.to_csv(results_csv, index=False)
    print(f"\nSaved results to {results_csv}")

    preds_csv = os.path.join(RESULTS_DIR, f"loso_cv_predictions_{timestamp}.csv")
    combined_preds.to_csv(preds_csv, index=False)
    print(f"Saved predictions to {preds_csv}")

    summary = combined_results.groupby('feature_set').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    })
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()

    summary_csv = os.path.join(RESULTS_DIR, f"loso_cv_summary_{timestamp}.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")

    per_subject_cv = combined_results.groupby(['feature_set', 'held_out_subject']).agg({
        'accuracy': 'mean',
        'macro_f1': 'mean',
        'balanced_accuracy': 'mean'
    }).reset_index()

    per_subject_csv = os.path.join(RESULTS_DIR, f"loso_cv_per_subject_{timestamp}.csv")
    per_subject_cv.to_csv(per_subject_csv, index=False)
    print(f"Saved per-subject CV to {per_subject_csv}")

    report = f"""# Leave-One-Subject-Out Cross-Validation Report

## Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Protocol
- **Method**: Leave-One-Subject-Out (LOSO) Cross-Validation
- **Subjects**: Only Y-* subjects (training set)
- **Seeds**: [0, 1, 2, 3, 4]
- **Folds**: 16 (one hold-out per subject per seed)

## Results Summary

### Overall (Mean ± Std across seeds and subjects)

| Feature Set | Accuracy | Macro-F1 | Balanced Accuracy |
|-------------|----------|----------|------------------|
"""

    for _, row in summary.iterrows():
        report += f"| {row['feature_set']} | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} |\n"

    report += f"""
## Per-Subject Performance

See `results/loso_cv_per_subject_{timestamp}.csv` for detailed per-subject results.

## Key Findings

1. **EEG-only performance**: {"Good" if summary[summary['feature_set']=='EEG_only']['macro_f1_mean'].values[0] > 0.5 else "Limited"} classification ability
2. **Gaze-only performance**: {"Good" if summary[summary['feature_set']=='Gaze_only']['macro_f1_mean'].values[0] > 0.5 else "Limited"} classification ability
3. **Combined performance**: {"Better than single modality" if summary[summary['feature_set']=='Combined']['macro_f1_mean'].values[0] > max(summary[summary['feature_set']=='EEG_only']['macro_f1_mean'].values[0], summary[summary['feature_set']=='Gaze_only']['macro_f1_mean'].values[0]) else "Not better than best single modality"}

## Note

This cross-validation is performed ONLY on training subjects (Y-*) since test set (X-*) labels are missing from the downloaded features.zip.

For official evaluation, submit predictions on X-* subjects to EvalAI.
"""

    report_path = os.path.join(SRC_DIR, "reports", "loso_cv_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")

    print("\n" + "="*70)
    print("LOSO CV COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()