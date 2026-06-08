#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZuCo Benchmark Alignment Diagnosis - Round 3 (Part 2)
Task 5: LOSO Baselines
"""

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score

FEATURES_DIR = 'src/features'
OUTPUT_DIR = 'audit_results'
DATA_DIR = 'data'

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_feature_file(subject, feature_set):
    filepath = os.path.join(FEATURES_DIR, f'{subject}_{feature_set}.npy')
    if not os.path.exists(filepath):
        return None
    try:
        return np.load(filepath, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def run_loso(X, y, subjects, model_name):
    """Run LOSO for a single model."""
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
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = SVC(kernel='linear', random_state=42, probability=True)
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        
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

def run_loso_with_pca(X, y, subjects, model_name):
    """Run LOSO with PCA preprocessing."""
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
        
        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)
        
        clf = SVC(kernel='linear', random_state=42, probability=True)
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        
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
            'test_samples': len(y_test),
            'pca_components': pca.n_components_
        })
    
    return fold_results

def task5_loso_baselines():
    print("\n=== Task 5: Running LOSO Baselines ===")
    
    # Load aligned data
    data = np.load(os.path.join(DATA_DIR, 'aligned_multimodal_y.npz'))
    eeg = data['eeg']
    gaze = data['gaze']
    y = data['y']
    
    df_meta = pd.read_csv(os.path.join(DATA_DIR, 'aligned_multimodal_y_metadata.csv'))
    subjects = df_meta['subject'].values
    
    print(f"Loaded {len(y)} aligned samples")
    
    # Create combined features
    eeg_gaze_concat = np.concatenate([eeg, gaze], axis=1)
    
    # Load official combined feature
    official_data = []
    official_subjects = []
    
    for subj in Y_SUBJECTS:
        combined = load_feature_file(subj, 'sent_gaze_sacc_eeg_means')
        if combined:
            for key, value in combined.items():
                official_data.append(value[:-1])
                official_subjects.append(subj)
    
    official_X = np.array(official_data)
    official_y = np.array([1 if 'NR' in k else 0 for k in official_subjects])
    
    results = []
    
    # 1. EEG-only SVM
    print("\n1. EEG-only SVM (420-D)")
    results.extend(run_loso(eeg, y, subjects, 'EEG-only'))
    
    # 2. Gaze-only SVM
    print("\n2. Gaze-only SVM (9-D)")
    results.extend(run_loso(gaze, y, subjects, 'Gaze-only'))
    
    # 3. EEG+Gaze clean concat SVM
    print("\n3. EEG+Gaze clean concat SVM (429-D)")
    results.extend(run_loso(eeg_gaze_concat, y, subjects, 'EEG+Gaze-concat'))
    
    # 4. EEG PCA SVM
    print("\n4. EEG PCA SVM")
    results.extend(run_loso_with_pca(eeg, y, subjects, 'EEG-PCA'))
    
    # 5. Official 13-D SVM
    print("\n5. Official sent_gaze_sacc_eeg_means SVM (13-D)")
    if len(official_X) > 0:
        results.extend(run_loso(official_X, official_y, official_subjects, 'Official-13D'))
    
    # Save detailed results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'baseline_loso_clean_multimodal.csv'), index=False)
    print(f"\nLOSO results saved to {OUTPUT_DIR}/baseline_loso_clean_multimodal.csv")
    
    # Generate summary
    summary = """# LOSO Baseline Results Summary

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
| Subject | EEG-only Acc | Gaze-only Acc | EEG+Gaze Acc | EEG-PCA Acc | Official Acc |
|---------|--------------|---------------|--------------|-------------|--------------|
"""
    
    for subj in sorted(set(subjects)):
        subj_df = df[df['subject'] == subj]
        row = f"| {subj} |"
        for model in ['EEG-only', 'Gaze-only', 'EEG+Gaze-concat', 'EEG-PCA', 'Official-13D']:
            acc = subj_df[subj_df['model'] == model]['accuracy'].values
            row += f" {acc[0]:.4f} |" if len(acc) > 0 else " - |"
        summary += row + "\n"
    
    summary += f"""
---

## Dataset Statistics
- Total aligned samples: {len(y)}
- Subjects: {len(set(subjects))}
- NR samples: {sum(y == 1)}
- TSR samples: {sum(y == 0)}

---

*Generated by ZuCo Benchmark Clean Baseline Audit*
"""
    
    with open(os.path.join(OUTPUT_DIR, 'baseline_loso_clean_multimodal_summary.md'), 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\nSummary saved to {OUTPUT_DIR}/baseline_loso_clean_multimodal_summary.md")
    
    return results

if __name__ == '__main__':
    print("=" * 70)
    print("ZuCo Benchmark Alignment Diagnosis - Round 3 (Part 2)")
    print("=" * 70)
    
    task5_loso_baselines()
    
    print("\n" + "=" * 70)
    print("Part 2 complete!")
    print("=" * 70)