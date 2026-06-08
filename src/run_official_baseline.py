"""
Standalone Baseline Runner for ZuCo 2.0
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(SCRIPT_DIR, "features")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")

TRAIN_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
TEST_SUBJECTS = ["XBB", "XDT", "XLS", "XPB", "XSE", "XTR", "XWS", "XAH", "XBD", "XSS"]

def load_features(subject, feature_set):
    path = os.path.join(FEATURES_DIR, f"{subject}_{feature_set}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None

def parse_feature_key(key):
    parts = key.split("_")
    subject = parts[0]
    label = parts[1]
    sentence_idx = int(parts[2])
    full_idx = int(parts[3])
    return subject, label, sentence_idx, full_idx

def prepare_data(subjects, feature_set, label_col=-1):
    all_X = []
    all_y = []
    all_subj_ids = []
    all_sentence_ids = []

    for subj in subjects:
        feats = load_features(subj, feature_set)
        if feats is None:
            continue
        for key, values in feats.items():
            subj_id, label, sent_idx, full_idx = parse_feature_key(key)
            features = np.array(values[:label_col], dtype=np.float64)
            if label == "NR":
                label_binary = 1
            elif label == "TSR":
                label_binary = 0
            else:
                label_binary = np.nan
            all_X.append(features)
            all_y.append(label_binary)
            all_subj_ids.append(subj_id)
            all_sentence_ids.append(full_idx)

    return np.array(all_X), np.array(all_y), all_subj_ids, all_sentence_ids

def run_svm(X_train, y_train, X_test, y_test, seed=42):
    np.random.seed(seed)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(random_state=seed, kernel='linear', gamma='scale', probability=True)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    return y_pred, y_prob

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data...")
    feature_sets = [
        ('EEG_only', 'electrode_features_all'),
        ('Gaze_only', 'sent_gaze_sacc'),
        ('Combined', 'sent_gaze_sacc_eeg_means')
    ]

    all_summaries = []
    all_detailed = []

    for name, feat_set in feature_sets:
        print(f"\nProcessing {name} ({feat_set})...")

        X_train, y_train, _, _ = prepare_data(TRAIN_SUBJECTS, feat_set)
        X_test, y_test, test_subj_ids, test_sent_ids = prepare_data(TEST_SUBJECTS, feat_set)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  No data found!")
            continue

        train_mask = ~np.isnan(y_train)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask].astype(int)
        has_test_labels = len(y_test) > 0 and not np.isnan(y_test).any()
        y_test_eval = y_test.astype(int) if has_test_labels else None

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Feats: {X_train.shape[1]}")
        if not has_test_labels:
            print("  Test labels unavailable; saving predictions without evaluation metrics.")

        seed_results = []
        for seed in [0, 1, 2, 3, 4]:
            y_pred, y_prob = run_svm(X_train, y_train, X_test, y_test_eval, seed)

            if has_test_labels:
                acc = accuracy_score(y_test_eval, y_pred)
                f1 = f1_score(y_test_eval, y_pred, average='macro')
                bacc = balanced_accuracy_score(y_test_eval, y_pred)
                try:
                    auroc = roc_auc_score(y_test_eval, y_prob)
                except:
                    auroc = None
            else:
                acc = f1 = bacc = auroc = None

            seed_results.append({
                'seed': seed, 'accuracy': acc, 'macro_f1': f1,
                'balanced_accuracy': bacc, 'auroc': auroc
            })

            for i, (subj, sent, true_l, pred_l, prob) in enumerate(zip(test_subj_ids, test_sent_ids, y_test, y_pred, y_prob)):
                all_detailed.append({
                    'seed': seed, 'subject_id': subj, 'sentence_id': sent,
                    'true_label': None if np.isnan(true_l) else int(true_l),
                    'pred_label': pred_l, 'pred_prob': prob,
                    'feature_set': feat_set
                })

            if has_test_labels:
                print(f"  Seed {seed}: Acc={acc:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}")
            else:
                print(f"  Seed {seed}: predictions saved")

        seed_df = pd.DataFrame(seed_results)
        summary = {
            'feature_set': feat_set,
            'accuracy_mean': seed_df['accuracy'].mean(),
            'accuracy_std': seed_df['accuracy'].std(),
            'macro_f1_mean': seed_df['macro_f1'].mean(),
            'macro_f1_std': seed_df['macro_f1'].std(),
            'balanced_accuracy_mean': seed_df['balanced_accuracy'].mean(),
            'balanced_accuracy_std': seed_df['balanced_accuracy'].std()
        }
        all_summaries.append(summary)

        print(f"  Summary: Acc={summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}, "
              f"F1={summary['macro_f1_mean']:.4f}±{summary['macro_f1_std']:.4f}")

    summary_df = pd.DataFrame(all_summaries)
    summary_csv = os.path.join(RESULTS_DIR, f"official_baseline_results_{timestamp}.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved: {summary_csv}")

    detailed_df = pd.DataFrame(all_detailed)
    detailed_csv = os.path.join(RESULTS_DIR, f"official_baseline_detailed_{timestamp}.csv")
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"Saved: {detailed_csv}")

    report = f"""# ZuCo 2.0 Baseline Reproduction Report

## Experiment Setup
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Split Protocol**: Cross-subject (train on Y-subjects, test on X-subjects)
- **Seeds**: [0, 1, 2, 3, 4]

## Feature Sets
1. **EEG-only**: electrode_features_all (420 features)
2. **Gaze-only**: sent_gaze_sacc (10 features)
3. **Combined**: sent_gaze_sacc_eeg_means (14 features)

## Train Subjects ({len(TRAIN_SUBJECTS)})
{', '.join(TRAIN_SUBJECTS)}

## Test Subjects ({len(TEST_SUBJECTS)})
{', '.join(TEST_SUBJECTS)}

## Results

| Feature Set | Accuracy | Macro-F1 | Balanced Accuracy |
|-------------|----------|----------|------------------|
"""
    for _, row in summary_df.iterrows():
        report += f"| {row['feature_set']} | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} |\n"

    report += """
## Notes
- Labels: NR (Normal Reading) = 1, TSR (Task-Specific Reading) = 0
- Cross-subject protocol: No subject overlap between train and test
- Evaluation metrics are only meaningful when heldout labels are present.
- The bundled X-subject feature files have blank labels, so they support prediction/submission generation but not local accuracy reporting.

## Command
```bash
cd src
python run_official_baseline.py
```
"""

    if summary_df['accuracy_mean'].isna().all():
        report = report.replace("nan", "N/A")

    report_path = os.path.join(REPORTS_DIR, "baseline_reproduction.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved: {report_path}")

    print("\nDone!")

if __name__ == '__main__':
    main()
