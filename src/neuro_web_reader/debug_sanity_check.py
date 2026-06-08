"""
Comprehensive Sanity Check for ZuCo 2.0 Baseline
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
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if os.path.basename(PROJECT_ROOT) == 'src':
    SRC_DIR = PROJECT_ROOT
else:
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

FEATURES_DIR = os.path.join(SRC_DIR, "features")
RESULTS_DIR = os.path.join(SRC_DIR, "results")
DEBUG_DIR = os.path.join(RESULTS_DIR, "debug")
CONFUSION_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)

TRAIN_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
TEST_SUBJECTS = ["XBB", "XDT", "XLS", "XPB", "XSE", "XTR", "XWS", "XAH", "XBD", "XSS"]
ALL_SUBJECTS = TRAIN_SUBJECTS + TEST_SUBJECTS

def load_features(subject, feature_set):
    path = os.path.join(FEATURES_DIR, f"{subject}_{feature_set}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None

def parse_key(key):
    parts = key.split("_")
    return parts[0], parts[1], int(parts[2]), int(parts[3])

def prepare_data(subjects, feature_set, label_col=-1):
    all_X, all_y, all_subj, all_sent, all_full, all_keys = [], [], [], [], [], []

    for subj in subjects:
        feats = load_features(subj, feature_set)
        if feats is None:
            continue
        for key, values in feats.items():
            subj_id, label, sent_idx, full_idx = parse_key(key)
            features = np.array(values[:label_col], dtype=np.float64)
            label_binary = 1 if label == "NR" else 0
            all_X.append(features)
            all_y.append(label_binary)
            all_subj.append(subj_id)
            all_sent.append(sent_idx)
            all_full.append(full_idx)
            all_keys.append(key)

    return np.array(all_X), np.array(all_y), all_subj, all_sent, all_full, all_keys

def check_subject_leakage():
    print("\n" + "="*60)
    print("CHECK 1: Subject Leakage")
    print("="*60)

    train_set, test_set = set(TRAIN_SUBJECTS), set(TEST_SUBJECTS)
    is_disjoint = train_set.isdisjoint(test_set)

    print(f"Train: {len(train_set)}, Test: {len(test_set)}, Intersection: {train_set & test_set}")
    print(f"Is disjoint: {is_disjoint}")

    if not is_disjoint:
        raise ValueError("SUBJECT LEAKAGE DETECTED!")
    return True

def check_label_distribution():
    print("\n" + "="*60)
    print("CHECK 2: Global Label Distribution")
    print("="*60)

    eeg_X, y, _, _, _, _ = prepare_data(ALL_SUBJECTS, 'electrode_features_all')

    total = len(y)
    num_normal = int(np.sum(y == 1))
    num_task = int(np.sum(y == 0))
    normal_ratio = num_normal / total if total > 0 else 0
    task_ratio = num_task / total if total > 0 else 0

    print(f"Total: {total}, NR=1: {num_normal} ({normal_ratio:.4f}), TSR=0: {num_task} ({task_ratio:.4f})")
    print(f"Majority class: {'NR' if num_normal > num_task else 'TSR'}")

    return {'total_samples': total, 'num_normal': num_normal, 'num_task_specific': num_task,
            'normal_ratio': normal_ratio, 'task_ratio': task_ratio,
            'majority_class': 'NR' if num_normal > num_task else 'TSR'}, eeg_X, y

def check_per_subject_distribution():
    print("\n" + "="*60)
    print("CHECK 3: Per-Subject Label Distribution")
    print("="*60)

    stats = []
    for subj in ALL_SUBJECTS:
        _, y, _, _, _, _ = prepare_data([subj], 'electrode_features_all')
        n = len(y)
        n_nr = int(np.sum(y == 1))
        n_tsr = int(np.sum(y == 0))
        stats.append({'subject_id': subj, 'num_samples': n, 'num_normal': n_nr,
                      'num_task_specific': n_tsr, 'normal_ratio': n_nr/n if n > 0 else 0,
                      'is_train': subj in TRAIN_SUBJECTS})
        print(f"{subj}: Total={n}, NR={n_nr}, TSR={n_tsr}, NR_ratio={n_nr/n if n > 0 else 0:.4f}")

    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(DEBUG_DIR, "label_distribution_by_subject.csv"), index=False)
    print(f"\nSaved to {DEBUG_DIR}/label_distribution_by_subject.csv")
    return df

def check_fold_distribution():
    print("\n" + "="*60)
    print("CHECK 4: Per-Fold Train/Test Distribution")
    print("="*60)

    _, y_train, _, _, _, _ = prepare_data(TRAIN_SUBJECTS, 'electrode_features_all')
    _, y_test, test_subj, _, _, _ = prepare_data(TEST_SUBJECTS, 'electrode_features_all')

    train_nr = int(np.sum(y_train == 1))
    train_tsr = int(np.sum(y_train == 0))

    print(f"\nTrain fold: Total={len(y_train)}, NR={train_nr}, TSR={train_tsr}")
    if len(y_train) > 0:
        print(f"Train NR ratio: {train_nr/len(y_train):.4f}")

    fold_stats = []
    for subj in TEST_SUBJECTS:
        mask = [s == subj for s in test_subj]
        y_sub = y_test[mask]
        n_nr = int(np.sum(y_sub == 1))
        n_tsr = int(np.sum(y_sub == 0))
        fold_stats.append({
            'test_subject': subj, 'train_num_normal': train_nr, 'train_num_task': train_tsr,
            'test_num_normal': n_nr, 'test_num_task': n_tsr
        })
        print(f"\nTest fold ({subj}): Total={len(y_sub)}, NR={n_nr}, TSR={n_tsr}")
        if len(y_sub) > 0:
            print(f"Test NR ratio: {n_nr/len(y_sub):.4f}")

    df = pd.DataFrame(fold_stats)
    df.to_csv(os.path.join(DEBUG_DIR, "fold_label_distribution.csv"), index=False)
    print(f"\nSaved to {DEBUG_DIR}/fold_label_distribution.csv")

    train_ratio = train_nr / len(y_train) if len(y_train) > 0 else 0.5
    return df, train_nr, train_tsr, len(y_train), train_ratio

def check_feature_quality():
    print("\n" + "="*60)
    print("CHECK 5: Feature Quality (NaN/Inf/Constant)")
    print("="*60)

    eeg_X, _, _, _, _, _ = prepare_data(ALL_SUBJECTS, 'electrode_features_all')
    gaze_X, _, _, _, _, _ = prepare_data(ALL_SUBJECTS, 'sent_gaze_sacc')

    report = []
    for name, X in [('eeg', eeg_X), ('gaze', gaze_X)]:
        n_nan = int(np.sum(np.isnan(X)))
        n_inf = int(np.sum(np.isinf(X)))
        X_valid = X[~np.isnan(X) & ~np.isinf(X)]
        mean_val = float(np.mean(X_valid)) if len(X_valid) > 0 else 0
        std_val = float(np.std(X_valid)) if len(X_valid) > 0 else 0
        min_val = float(np.min(X_valid)) if len(X_valid) > 0 else 0
        max_val = float(np.max(X_valid)) if len(X_valid) > 0 else 0
        X_var = np.var(X, axis=0)
        n_const = int(np.sum(X_var < 1e-10))

        print(f"\n{name.upper()}: Shape={X.shape}, NaN={n_nan}, Inf={n_inf}, Constant={n_const}/{X.shape[1]}")
        print(f"  Range: [{min_val:.4f}, {max_val:.4f}], mean={mean_val:.4f}, std={std_val:.4f}")

        report.append({'feature_type': name, 'num_nan': n_nan, 'num_inf': n_inf,
                       'num_constant_features': n_const, 'total_features': X.shape[1],
                       'mean': mean_val, 'std': std_val, 'min': min_val, 'max': max_val})

    pd.DataFrame(report).to_csv(os.path.join(DEBUG_DIR, "feature_quality_report.csv"), index=False)
    print(f"\nSaved to {DEBUG_DIR}/feature_quality_report.csv")
    return pd.DataFrame(report)

def check_feature_label_alignment():
    print("\n" + "="*60)
    print("CHECK 6: Feature-Label Alignment Preview")
    print("="*60)

    eeg_X, y, subj, sent, full, keys = prepare_data(['YAC'], 'electrode_features_all')
    gaze_X, _, _, _, _, _ = prepare_data(['YAC'], 'sent_gaze_sacc')

    print(f"EEG samples: {len(eeg_X)}, Gaze samples: {len(gaze_X)}")

    preview = []
    for i in range(min(20, len(eeg_X))):
        eeg_nan = int(np.sum(np.isnan(eeg_X[i]))) if i < len(eeg_X) else 0
        gaze_nan = int(np.sum(np.isnan(gaze_X[i]))) if i < len(gaze_X) else 0

        preview.append({
            'index': i,
            'subject_id': subj[i] if i < len(subj) else 'N/A',
            'sample_id': full[i] if i < len(full) else 'N/A',
            'label': int(y[i]) if i < len(y) else 'N/A',
            'eeg_shape': eeg_X[i].shape if i < len(eeg_X) else 'N/A',
            'gaze_shape': gaze_X[i].shape if i < len(gaze_X) else 'N/A',
            'eeg_nan_count': eeg_nan,
            'gaze_nan_count': gaze_nan
        })

    df = pd.DataFrame(preview)
    df.to_csv(os.path.join(DEBUG_DIR, "sample_alignment_preview.csv"), index=False)
    print(f"\nSaved first {len(preview)} samples to {DEBUG_DIR}/sample_alignment_preview.csv")

    for _, row in df.head(5).iterrows():
        print(f"  {row['subject_id']}, sample={row['sample_id']}, label={row['label']}, "
              f"eeg={row['eeg_shape']}, gaze={row['gaze_shape']}")

    return df

def compute_baselines(y_train, y_test, train_nr_ratio):
    print("\n" + "="*60)
    print("CHECK 7: Majority and Random Baselines")
    print("="*60)

    majority_class = 1 if train_nr_ratio > 0.5 else 0
    maj_pred = np.full_like(y_test, majority_class)

    maj_acc = accuracy_score(y_test, maj_pred)
    maj_f1 = f1_score(y_test, maj_pred, average='macro')
    maj_bacc = balanced_accuracy_score(y_test, maj_pred)

    np.random.seed(42)
    rand_pred = (np.random.random(len(y_test)) < train_nr_ratio).astype(int)

    rand_acc = accuracy_score(y_test, rand_pred)
    rand_f1 = f1_score(y_test, rand_pred, average='macro')
    rand_bacc = balanced_accuracy_score(y_test, rand_pred)

    print(f"\nTrain majority class: {'NR' if majority_class == 1 else 'TSR'} (NR ratio={train_nr_ratio:.4f})")
    print(f"Majority: Acc={maj_acc:.4f}, F1={maj_f1:.4f}, BAcc={maj_bacc:.4f}")
    print(f"Random: Acc={rand_acc:.4f}, F1={rand_f1:.4f}, BAcc={rand_bacc:.4f}")

    return {'majority_accuracy': maj_acc, 'majority_macro_f1': maj_f1, 'majority_balanced_accuracy': maj_bacc,
            'random_accuracy': rand_acc, 'random_macro_f1': rand_f1, 'random_balanced_accuracy': rand_bacc,
            'train_nr_ratio': train_nr_ratio}

def check_label_inversion():
    print("\n" + "="*60)
    print("CHECK 8: Label Inversion Check")
    print("="*60)

    X_train, y_train, _, _, _, _ = prepare_data(TRAIN_SUBJECTS, 'electrode_features_all')
    X_test, y_test, _, _, _, _ = prepare_data(TEST_SUBJECTS, 'electrode_features_all')

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    from sklearn.svm import SVC
    clf = SVC(random_state=0, kernel='linear', gamma='scale', probability=True)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc_orig = accuracy_score(y_test, y_pred)
    acc_inv = accuracy_score(y_test, 1 - y_pred)

    print(f"\nOriginal accuracy: {acc_orig:.4f}")
    print(f"Inverted accuracy: {acc_inv:.4f}")
    print(f"Difference: {acc_inv - acc_orig:.4f}")

    if acc_inv > acc_orig + 0.1:
        print("WARNING: Inverted accuracy is significantly higher!")

    result = {'original_accuracy': acc_orig, 'inverted_accuracy': acc_inv,
              'difference': acc_inv - acc_orig, 'likely_inverted': acc_inv > acc_orig + 0.1}

    pd.DataFrame([result]).to_csv(os.path.join(DEBUG_DIR, "label_inversion_check.csv"), index=False)
    print(f"\nSaved to {DEBUG_DIR}/label_inversion_check.csv")
    return result

def run_full_evaluation():
    print("\n" + "="*60)
    print("CHECK 9: Full Evaluation with Confusion Matrix")
    print("="*60)

    feature_sets = [
        ('EEG_only', 'electrode_features_all'),
        ('Gaze_only', 'sent_gaze_sacc'),
        ('Combined', 'sent_gaze_sacc_eeg_means')
    ]

    results = []

    for name, feat_set in feature_sets:
        print(f"\n{name} ({feat_set}):")

        X_train, y_train, _, _, _, _ = prepare_data(TRAIN_SUBJECTS, feat_set)
        X_test, y_test, _, _, _, _ = prepare_data(TEST_SUBJECTS, feat_set)

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        from sklearn.svm import SVC
        clf = SVC(random_state=0, kernel='linear', gamma='scale', probability=True)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bacc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', warn_for=[])
        cm = confusion_matrix(y_test, y_pred)

        print(f"  Acc={acc:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}")
        print(f"  Precision={prec:.4f}, Recall={rec:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        cm_dict = {
            'model': name, 'seed': 0, 'accuracy': acc, 'macro_f1': f1,
            'balanced_accuracy': bacc, 'precision_macro': prec, 'recall_macro': rec,
            'confusion_matrix': cm.tolist(),
            'tn': int(cm[0, 0]) if cm.shape[0] > 1 else 0,
            'fp': int(cm[0, 1]) if cm.shape[0] > 1 else 0,
            'fn': int(cm[1, 0]) if cm.shape[0] > 1 else 0,
            'tp': int(cm[1, 1]) if cm.shape[0] > 1 else 0
        }
        results.append(cm_dict)

        with open(os.path.join(CONFUSION_DIR, f"{name}_seed0.json"), 'w') as f:
            json.dump(cm_dict, f, indent=2)

    return pd.DataFrame(results)

def generate_report(label_stats, baselines, label_inv, eval_df):
    print("\n" + "="*60)
    print("GENERATING SANITY CHECK REPORT")
    print("="*60)

    is_balanced = 0.4 < label_stats['normal_ratio'] < 0.6
    eeg_f1 = eval_df[eval_df['model'] == 'EEG_only']['macro_f1'].values[0]
    eeg_f1_vs_maj = eeg_f1 - baselines['majority_macro_f1']
    gaze_acc = eval_df[eval_df['model'] == 'Gaze_only']['accuracy'].values[0]

    report = f"""# ZuCo 2.0 Baseline Sanity Check Report

## Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Label Distribution

### Global
- Total: {label_stats['total_samples']}, NR=1: {label_stats['num_normal']} ({label_stats['normal_ratio']:.4f}), TSR=0: {label_stats['num_task_specific']} ({label_stats['task_ratio']:.4f})
- Majority class: {label_stats['majority_class']}
- **Assessment**: {"Balanced" if is_balanced else "IMBALANCED - majority is " + label_stats['majority_class']}

### Per-Subject
- See `results/debug/label_distribution_by_subject.csv`

## 2. Cross-Subject Split

**Subject Leakage**: PASSED (no overlap between train and test)

## 3. Train/Test Distribution

- See `results/debug/fold_label_distribution.csv`

## 4. Feature Quality

- See `results/debug/feature_quality_report.csv`

## 5. Majority/Random Baselines

| Baseline | Accuracy | Macro-F1 | Balanced Accuracy |
|----------|----------|----------|------------------|
| Majority | {baselines['majority_accuracy']:.4f} | {baselines['majority_macro_f1']:.4f} | {baselines['majority_balanced_accuracy']:.4f} |
| Random | {baselines['random_accuracy']:.4f} | {baselines['random_macro_f1']:.4f} | {baselines['random_balanced_accuracy']:.4f} |

Train NR ratio: {baselines['train_nr_ratio']:.4f}

## 6. Label Inversion Check

| Metric | Value |
|--------|-------|
| Original Accuracy | {label_inv['original_accuracy']:.4f} |
| Inverted Accuracy | {label_inv['inverted_accuracy']:.4f} |
| Difference | {label_inv['difference']:.4f} |
| Likely Inverted | {label_inv['likely_inverted']} |

## 7. Model Results

| Model | Accuracy | Macro-F1 | Balanced Accuracy | Precision | Recall |
|-------|----------|----------|------------------|-----------|--------|
"""

    for _, row in eval_df.iterrows():
        report += f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['balanced_accuracy']:.4f} | {row['precision_macro']:.4f} | {row['recall_macro']:.4f} |\n"

    report += f"""
## 8. Confusion Matrices

"""
    for _, row in eval_df.iterrows():
        cm = row['confusion_matrix']
        report += f"### {row['model']}\n- TN: {row['tn']}, FP: {row['fp']}, FN: {row['fn']}, TP: {row['tp']}\n\n"

    report += f"""
## 9. Key Findings

1. **Label distribution**: {"Balanced" if is_balanced else "Imbalanced (majority: " + label_stats['majority_class'] + ")"}
2. **Balanced accuracy**: CORRECTLY computed using sklearn.metrics.balanced_accuracy_score
3. **EEG-only vs Majority**: F1={eeg_f1:.4f} vs {baselines['majority_macro_f1']:.4f}, diff={eeg_f1_vs_maj:.4f}
   - **Assessment**: {"EEG shows classification ability" if eeg_f1_vs_maj > 0.05 else "EEG shows LITTLE ability beyond majority"}
4. **Gaze-only Accuracy**: {gaze_acc:.4f} vs Random={baselines['random_accuracy']:.4f}
   - **Assessment**: {"Above random" if gaze_acc > baselines['random_accuracy'] else "Below/at random level"}
5. **Label inversion**: {"NOT detected" if not label_inv['likely_inverted'] else "DETECTED"}
6. **Cross-subject split**: VERIFIED - no leakage

## 10. Conclusions

"""

    if is_balanced:
        report += "- Label distribution is reasonably balanced\n"
    else:
        report += f"- Label distribution is IMBALANCED (majority: {label_stats['majority_class']})\n"

    if eeg_f1_vs_maj > 0.05:
        report += "- EEG-only SIGNIFICANTLY better than majority baseline - can proceed with TGCR\n"
    else:
        report += "- EEG-only similar to majority baseline - limited classification ability\n"

    if gaze_acc > baselines['random_accuracy'] + 0.05:
        report += "- Gaze-only above random - feature has signal\n"
    else:
        report += "- Gaze-only at or below random - check feature alignment\n"

    if label_inv['likely_inverted']:
        report += "- WARNING: Label inversion detected - verify label mapping\n"

    report += "- Cross-subject split verified - no subject leakage\n"

    report_path = os.path.join(SRC_DIR, "reports", "baseline_sanity_check_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")

    return report

def main():
    print("="*70)
    print("ZuCo 2.0 Comprehensive Sanity Check")
    print("="*70)

    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(CONFUSION_DIR, exist_ok=True)

    check_subject_leakage()
    label_stats, _, _ = check_label_distribution()
    check_per_subject_distribution()
    fold_df, train_nr, train_tsr, train_total, train_ratio = check_fold_distribution()
    check_feature_quality()
    check_feature_label_alignment()
    baselines = compute_baselines(train_nr * [1] + train_tsr * [0],
                                   np.concatenate([np.ones(train_nr), np.zeros(train_tsr)]),
                                   train_ratio)
    label_inv = check_label_inversion()
    eval_df = run_full_evaluation()

    generate_report(label_stats, baselines, label_inv, eval_df)

    print("\n" + "="*70)
    print("SANITY CHECK COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()