#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZuCo Benchmark LOSO Baseline with LinearSVC (Fast Version)
"""

import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

FEATURES_DIR = 'src/features'
OUTPUT_DIR = 'audit_results'
DATA_DIR = 'data'

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
HIGH_DROP_SUBJECTS = ['YRK', 'YRP', 'YFR']

def load_feature_file(subject, feature_set):
    filepath = os.path.join(FEATURES_DIR, f'{subject}_{feature_set}.npy')
    if not os.path.exists(filepath):
        return None
    try:
        return np.load(filepath, allow_pickle=True).item()
    except Exception as e:
        return None

def run_loso(X, y, subjects, model_name, model_type='linearsvc', pca_components=None):
    """Run LOSO with LinearSVC or LogisticRegression."""
    loo = LeaveOneOut()
    subject_list = sorted(set(subjects))
    fold_results = []
    
    for train_idx, test_idx in loo.split(subject_list):
        train_subjects = [subject_list[i] for i in train_idx]
        test_subject = subject_list[test_idx[0]]
        
        train_mask = np.isin(subjects, train_subjects)
        test_mask = subjects == test_subject
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Ensure 2D arrays
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        elif X_test.ndim > 2:
            X_test = X_test.reshape(len(X_test), -1)
        if X_train.ndim == 1:
            X_train = X_train.reshape(1, -1)
        elif X_train.ndim > 2:
            X_train = X_train.reshape(len(X_train), -1)
        
        if pca_components:
            pca = PCA(n_components=pca_components, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'logistic':
            clf = LogisticRegression(random_state=42, max_iter=1000)
        else:
            clf = LinearSVC(random_state=42, max_iter=1000, dual=False)
        
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = clf.decision_function(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            roc_auc = np.nan
        
        fold_results.append({
            'model': model_name,
            'subject': test_subject,
            'accuracy': acc,
            'balanced_accuracy': bacc,
            'macro_f1': f1,
            'auroc': roc_auc,
            'test_samples': len(y_test)
        })
    
    return fold_results

def main():
    print("=" * 70)
    print("ZuCo Benchmark LOSO Baseline with LinearSVC")
    print("=" * 70)
    
    # Load aligned data
    data = np.load(os.path.join(DATA_DIR, 'aligned_multimodal_y.npz'))
    eeg = data['eeg']
    gaze = data['gaze']
    y = data['y']
    
    df_meta = pd.read_csv(os.path.join(DATA_DIR, 'aligned_multimodal_y_metadata.csv'))
    subjects = df_meta['subject'].values
    
    print(f"Loaded {len(y)} aligned samples")
    print(f"EEG shape: {eeg.shape}")
    print(f"Gaze shape: {gaze.shape}")
    
    # Create combined features
    eeg_gaze_concat = np.concatenate([eeg, gaze], axis=1)
    
    # Load official combined feature
    official_data = []
    official_subjects = []
    official_labels = []
    
    for subj in Y_SUBJECTS:
        combined = load_feature_file(subj, 'sent_gaze_sacc_eeg_means')
        if combined:
            for key, value in combined.items():
                feat = np.array(value[:-1])
                if feat.ndim > 1:
                    feat = feat.flatten()
                official_data.append(feat)
                official_subjects.append(subj)
                official_labels.append(1 if 'NR' in key else 0)
    
    official_X = np.array(official_data)
    official_y = np.array(official_labels)
    print(f"Official feature shape: {official_X.shape}")
    
    results = []
    
    # 1. EEG-only LinearSVC
    print("\n1. EEG-only LinearSVC (420-D)")
    results.extend(run_loso(eeg, y, subjects, 'EEG-only'))
    
    # 2. Gaze-only LinearSVC
    print("\n2. Gaze-only LinearSVC (9-D)")
    results.extend(run_loso(gaze, y, subjects, 'Gaze-only'))
    
    # 3. EEG+Gaze concat LinearSVC
    print("\n3. EEG+Gaze concat LinearSVC (429-D)")
    results.extend(run_loso(eeg_gaze_concat, y, subjects, 'EEG+Gaze-concat'))
    
    # 4. EEG PCA + Gaze LinearSVC
    print("\n4. EEG PCA + Gaze LinearSVC")
    results.extend(run_loso(eeg, y, subjects, 'EEG-PCA', pca_components=0.95))
    
    # 5. Official 13-D LinearSVC (skipped due to alignment issues)
    print("\n5. Official sent_gaze_sacc_eeg_means LinearSVC (13-D) - SKIPPED")
    
    # 6. LogisticRegression for AUROC stability
    print("\n6. LogisticRegression (EEG+Gaze)")
    results.extend(run_loso(eeg_gaze_concat, y, subjects, 'Logistic-EEG+Gaze', model_type='logistic'))
    
    # Save detailed results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'baseline_loso_linearsvc_aligned.csv'), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/baseline_loso_linearsvc_aligned.csv")
    
    # Generate summary
    summary = """# LOSO Baseline Results Summary (LinearSVC)

## Results on Aligned Multimodal Data (Y Subjects Only)

### Per-Model Statistics
| Model | Accuracy (mean±std) | Balanced Acc (mean±std) | Macro-F1 (mean±std) | AUROC (mean±std) |
|-------|---------------------|-------------------------|---------------------|------------------|
"""
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        summary += f"| {model} | {model_df['accuracy'].mean():.4f} ± {model_df['accuracy'].std():.4f} | {model_df['balanced_accuracy'].mean():.4f} ± {model_df['balanced_accuracy'].std():.4f} | {model_df['macro_f1'].mean():.4f} ± {model_df['macro_f1'].std():.4f} | {model_df['auroc'].mean():.4f} ± {model_df['auroc'].std():.4f} |\n"
    
    summary += """
### Per-Subject Results
| Subject | EEG-only | Gaze-only | EEG+Gaze | EEG-PCA | Official | Logistic |
|---------|----------|-----------|----------|---------|----------|----------|
"""
    
    for subj in sorted(set(subjects)):
        subj_df = df[df['subject'] == subj]
        row = f"| {subj} |"
        for model in ['EEG-only', 'Gaze-only', 'EEG+Gaze-concat', 'EEG-PCA', 'Official-13D', 'Logistic-EEG+Gaze']:
            acc = subj_df[subj_df['model'] == model]['accuracy'].values
            row += f" {acc[0]:.4f} |" if len(acc) > 0 else " - |"
        summary += row + "\n"
    
    # Sensitivity analysis - exclude high-drop subjects
    print("\nSensitivity Analysis: Excluding high-drop subjects")
    low_drop_subjects = [s for s in Y_SUBJECTS if s not in HIGH_DROP_SUBJECTS]
    low_drop_mask = np.isin(subjects, low_drop_subjects)
    
    sensitivity_results = []
    
    sensitivity_results.extend(run_loso(eeg[low_drop_mask], y[low_drop_mask], subjects[low_drop_mask], 'EEG-only'))
    sensitivity_results.extend(run_loso(gaze[low_drop_mask], y[low_drop_mask], subjects[low_drop_mask], 'Gaze-only'))
    sensitivity_results.extend(run_loso(eeg_gaze_concat[low_drop_mask], y[low_drop_mask], subjects[low_drop_mask], 'EEG+Gaze-concat'))
    
    df_sensitivity = pd.DataFrame(sensitivity_results)
    df_sensitivity.to_csv(os.path.join(OUTPUT_DIR, 'baseline_loso_sensitivity_highdrop.csv'), index=False)
    print(f"Sensitivity results saved to {OUTPUT_DIR}/baseline_loso_sensitivity_highdrop.csv")
    
    # Add sensitivity analysis to summary
    summary += f"""
---

## Sensitivity Analysis (Excluding High-Drop Subjects: {', '.join(HIGH_DROP_SUBJECTS)})

| Model | Accuracy (mean±std) | Balanced Acc (mean±std) | Macro-F1 (mean±std) |
|-------|---------------------|-------------------------|---------------------|
"""
    
    for model in ['EEG-only', 'Gaze-only', 'EEG+Gaze-concat']:
        model_df = df_sensitivity[df_sensitivity['model'] == model]
        summary += f"| {model} | {model_df['accuracy'].mean():.4f} ± {model_df['accuracy'].std():.4f} | {model_df['balanced_accuracy'].mean():.4f} ± {model_df['balanced_accuracy'].std():.4f} | {model_df['macro_f1'].mean():.4f} ± {model_df['macro_f1'].std():.4f} |\n"
    
    summary += f"""
---

## Dataset Statistics
- Total aligned samples: {len(y)}
- Subjects: {len(set(subjects))}
- NR samples: {sum(y == 1)}
- TSR samples: {sum(y == 0)}

---

*Generated by ZuCo Benchmark LinearSVC LOSO Audit*
"""
    
    with open(os.path.join(OUTPUT_DIR, 'baseline_loso_linearsvc_summary.md'), 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\nSummary saved to {OUTPUT_DIR}/baseline_loso_linearsvc_summary.md")
    print("\n" + "=" * 70)
    print("LOSO baseline complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()