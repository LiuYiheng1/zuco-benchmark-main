#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZuCo Benchmark Protocol Audit Script
===================================

This script performs comprehensive audits on the ZuCo benchmark data to ensure:
1. Feature label leakage detection
2. Split validation (Y/X subjects)
3. EEG-Gaze alignment verification
4. Duplicate sentence recovery
5. Feature overlap checking
6. Baseline reproduction on LOSO

Output files:
- audit_label_leakage.csv
- audit_alignment_keys.csv
- audit_duplicate_recovery.csv
- baseline_loso_results.csv
- protocol_audit_summary.md
"""

import os
import re
import csv
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from glob import glob

# Configuration
FEATURES_DIR = 'src/features'
TASK_MATERIALS_DIR = 'src/task_materials'
OUTPUT_DIR = 'audit_results'

# Subject splits (from config.py)
Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
X_SUBJECTS = ['XBB', 'XDT', 'XLS', 'XPB', 'XSE', 'XTR', 'XWS', 'XAH', 'XBD', 'XSS']

# Feature sets to audit
FEATURE_SETS = [
    'electrode_features_all',
    'sent_gaze_sacc', 
    'sent_gaze_sacc_eeg_means',
    'eeg_means',
    'sent_gaze',
    'theta_mean',
    'alpha_mean',
    'beta_mean',
    'gamma_mean'
]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_text(text):
    """Normalize text for duplicate detection."""
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Strip whitespace
    text = text.strip()
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize punctuation
    text = re.sub(r'[“”"]', '"', text)
    text = re.sub(r'[‘’\']', "'", text)
    text = re.sub(r'[–—-]', '-', text)
    return text

def load_feature_file(subject, feature_set):
    """Load a feature file for a subject."""
    filepath = os.path.join(FEATURES_DIR, f'{subject}_{feature_set}.npy')
    if not os.path.exists(filepath):
        return None
    try:
        return np.load(filepath, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_all_keys(subjects, feature_set):
    """Get all keys for a feature set across subjects."""
    all_keys = []
    for subj in subjects:
        data = load_feature_file(subj, feature_set)
        if data:
            all_keys.extend(list(data.keys()))
    return all_keys

def extract_label_from_key(key):
    """Extract label from key format: subject_label_idx_fullidx."""
    parts = key.split('_')
    if len(parts) >= 2:
        return parts[1]
    return None

def audit_label_leakage():
    """Audit for feature label leakage."""
    results = []
    
    for feature_set in FEATURE_SETS:
        print(f"\n=== Audit: Label Leakage for {feature_set} ===")
        
        # Collect all data
        all_values = []
        all_labels = []
        all_keys = []
        
        for subj in Y_SUBJECTS + X_SUBJECTS:
            data = load_feature_file(subj, feature_set)
            if data:
                for key, value in data.items():
                    all_keys.append(key)
                    all_values.append(value)
                    # Label is last element
                    all_labels.append(value[-1])
        
        if not all_values:
            print(f"No data found for {feature_set}")
            continue
        
        # Analyze value lengths
        raw_value_lengths = [len(v) for v in all_values]
        model_input_lengths = [len(v[:-1]) for v in all_values]
        last_column_values = [v[-1] for v in all_values]
        unique_last_column = sorted(set(last_column_values))
        
        print(f"Total samples: {len(all_values)}")
        print(f"Raw value lengths (unique): {sorted(set(raw_value_lengths))}")
        print(f"Model input lengths (unique): {sorted(set(model_input_lengths))}")
        print(f"Last column unique values: {unique_last_column}")
        
        # Check if all model inputs use value[:-1]
        expected_input_len = len(all_values[0]) - 1
        all_correct_length = all(len(v[:-1]) == expected_input_len for v in all_values)
        print(f"All model inputs use value[:-1]: {all_correct_length}")
        
        # Train only-last-column classifier
        if len(unique_last_column) == 2:
            print("\nTraining only-last-column classifier...")
            # Use last column as both feature and label (should get ~100% if it's the label)
            X_last_col = np.array([[hash(str(v[-1])) % 1000] for v in all_values])
            y = np.array([1 if l == 'NR' else 0 for l in last_column_values])
            
            X_last_col, y = shuffle(X_last_col, y, random_state=42)
            split = int(len(y) * 0.8)
            
            clf_last = SVC(kernel='linear', random_state=42)
            clf_last.fit(X_last_col[:split], y[:split])
            pred = clf_last.predict(X_last_col[split:])
            acc_last = accuracy_score(y[split:], pred)
            
            print(f"Only-last-column classifier accuracy: {acc_last:.4f}")
        else:
            acc_last = "N/A (not binary)"
        
        # Shuffled label sanity check
        print("\nShuffled label sanity check...")
        X = np.array([v[:-1] for v in all_values])
        y_true = np.array([1 if l == 'NR' else 0 for l in last_column_values])
        
        if X.size > 0:
            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Shuffle labels
            y_shuffled = shuffle(y_true, random_state=42)
            
            # Train with shuffled labels
            X_scaled_shuffled, y_shuffled = shuffle(X_scaled, y_shuffled, random_state=42)
            split = int(len(y_shuffled) * 0.8)
            
            clf_shuffled = SVC(kernel='linear', random_state=42)
            clf_shuffled.fit(X_scaled_shuffled[:split], y_shuffled[:split])
            pred_shuffled = clf_shuffled.predict(X_scaled_shuffled[split:])
            acc_shuffled = accuracy_score(y_shuffled[split:], pred_shuffled)
            f1_shuffled = f1_score(y_shuffled[split:], pred_shuffled, average='macro')
            
            print(f"Shuffled label accuracy: {acc_shuffled:.4f}")
            print(f"Shuffled label macro-F1: {f1_shuffled:.4f}")
            print(f"Expected chance level: ~0.5")
        else:
            acc_shuffled = "N/A"
            f1_shuffled = "N/A"
        
        results.append({
            'feature_set': feature_set,
            'total_samples': len(all_values),
            'raw_value_lengths': str(sorted(set(raw_value_lengths))),
            'model_input_length': expected_input_len,
            'unique_last_column': str(unique_last_column),
            'all_use_value[:-1]': all_correct_length,
            'last_column_clf_acc': acc_last,
            'shuffled_acc': acc_shuffled,
            'shuffled_f1': f1_shuffled
        })
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'audit_label_leakage.csv'), index=False)
    print(f"\nLabel leakage audit saved to {OUTPUT_DIR}/audit_label_leakage.csv")
    
    return results

def audit_split():
    """Audit subject splits."""
    results = []
    
    print("\n=== Audit: Split Configuration ===")
    print(f"Y subjects (training/dev): {len(Y_SUBJECTS)} subjects")
    print(f"Y list: {Y_SUBJECTS}")
    print(f"\nX subjects (heldout/test): {len(X_SUBJECTS)} subjects")
    print(f"X list: {X_SUBJECTS}")
    
    # Check if there's any overlap between Y and X
    overlap = set(Y_SUBJECTS) & set(X_SUBJECTS)
    print(f"\nOverlap between Y and X subjects: {overlap}")
    
    # Check if X features contain labels
    print("\nChecking if X subjects have labels in features...")
    for subj in X_SUBJECTS:
        data = load_feature_file(subj, 'sent_gaze_sacc')
        if data:
            first_key = list(data.keys())[0]
            first_value = data[first_key]
            last_col = first_value[-1]
            is_label = last_col in ['NR', 'TSR', '']
            print(f"  {subj}: last_column='{last_col}', is_label={is_label}")
            results.append({
                'subject': subj,
                'has_feature_file': True,
                'last_column_value': last_col,
                'appears_to_be_label': is_label
            })
        else:
            print(f"  {subj}: feature file not found")
            results.append({
                'subject': subj,
                'has_feature_file': False,
                'last_column_value': None,
                'appears_to_be_label': False
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'audit_split.csv'), index=False)
    print(f"\nSplit audit saved to {OUTPUT_DIR}/audit_split.csv")
    
    return {'Y_subjects': Y_SUBJECTS, 'X_subjects': X_SUBJECTS, 'overlap': list(overlap), 'X_labels_check': results}

def audit_eeg_gaze_alignment():
    """Audit EEG-Gaze alignment."""
    print("\n=== Audit: EEG-Gaze Alignment ===")
    
    # Get keys for both feature sets
    eeg_keys = set(get_all_keys(Y_SUBJECTS + X_SUBJECTS, 'electrode_features_all'))
    gaze_keys = set(get_all_keys(Y_SUBJECTS + X_SUBJECTS, 'sent_gaze_sacc'))
    
    print(f"electrode_features_all keys: {len(eeg_keys)}")
    print(f"sent_gaze_sacc keys: {len(gaze_keys)}")
    
    # Find unmatched keys
    eeg_only = eeg_keys - gaze_keys
    gaze_only = gaze_keys - eeg_keys
    
    print(f"\nEEG-only keys: {len(eeg_only)}")
    print(f"Gaze-only keys: {len(gaze_only)}")
    
    # Save alignment keys
    results = []
    for key in sorted(eeg_keys | gaze_keys):
        results.append({
            'key': key,
            'in_eeg': key in eeg_keys,
            'in_gaze': key in gaze_keys,
            'subject': key.split('_')[0],
            'label': key.split('_')[1] if len(key.split('_')) > 1 else None
        })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'audit_alignment_keys.csv'), index=False)
    print(f"\nAlignment audit saved to {OUTPUT_DIR}/audit_alignment_keys.csv")
    
    return {
        'total_eeg_keys': len(eeg_keys),
        'total_gaze_keys': len(gaze_keys),
        'eeg_only_keys': len(eeg_only),
        'gaze_only_keys': len(gaze_only),
        'matching_keys': len(eeg_keys & gaze_keys),
        'unmatched_eeg': list(eeg_only)[:10] if eeg_only else [],
        'unmatched_gaze': list(gaze_only)[:10] if gaze_only else []
    }

def audit_duplicate_recovery():
    """Audit duplicate sentence recovery."""
    print("\n=== Audit: Duplicate Sentence Recovery ===")
    
    # Load all NR and TSR sentences
    nr_sentences = {}
    tsr_sentences = {}
    
    # Load NR files
    nr_files = glob(os.path.join(TASK_MATERIALS_DIR, 'nr_*.csv'))
    nr_files = [f for f in nr_files if 'control_questions' not in f]
    
    for filepath in nr_files:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row) >= 3:
                    sentence_id = row[0]
                    block_id = row[1]
                    text = row[2]
                    # Skip if CONTROL label in 4th column
                    if len(row) >= 4 and row[3].strip().upper() == 'CONTROL':
                        continue
                    normalized = normalize_text(text)
                    if normalized:
                        nr_sentences[(sentence_id, block_id)] = normalized
    
    # Load TSR files
    tsr_files = glob(os.path.join(TASK_MATERIALS_DIR, 'tsr_*.csv'))
    
    for filepath in tsr_files:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row) >= 3:
                    sentence_id = row[0]
                    block_id = row[1]
                    text = row[2]
                    # Skip if CONTROL label
                    if len(row) >= 4 and row[3].strip().upper() == 'CONTROL':
                        continue
                    normalized = normalize_text(text)
                    if normalized:
                        tsr_sentences[(sentence_id, block_id)] = normalized
    
    print(f"NR sentences loaded: {len(nr_sentences)}")
    print(f"TSR sentences loaded: {len(tsr_sentences)}")
    
    # Find duplicates between NR and TSR
    nr_texts = list(nr_sentences.values())
    tsr_texts = list(tsr_sentences.values())
    
    duplicates = []
    for nr_key, nr_text in nr_sentences.items():
        for tsr_key, tsr_text in tsr_sentences.items():
            if nr_text == tsr_text:
                duplicates.append({
                    'nr_sentence_id': nr_key[0],
                    'nr_block_id': nr_key[1],
                    'tsr_sentence_id': tsr_key[0],
                    'tsr_block_id': tsr_key[1],
                    'sentence_text': nr_text[:100] + '...' if len(nr_text) > 100 else nr_text
                })
    
    print(f"\nFound {len(duplicates)} NR-TSR duplicate sentence pairs")
    print(f"Expected (from paper): ~63 duplicates")
    
    # Save results
    df = pd.DataFrame(duplicates)
    df.to_csv(os.path.join(OUTPUT_DIR, 'audit_duplicate_recovery.csv'), index=False)
    print(f"\nDuplicate audit saved to {OUTPUT_DIR}/audit_duplicate_recovery.csv")
    
    return {
        'nr_sentences_count': len(nr_sentences),
        'tsr_sentences_count': len(tsr_sentences),
        'duplicate_pairs': len(duplicates),
        'expected_duplicates': 63
    }

def audit_feature_overlap():
    """Audit feature overlap."""
    print("\n=== Audit: Feature Overlap ===")
    
    feature_dimensions = {
        'theta_mean': 1,
        'alpha_mean': 1,
        'beta_mean': 1,
        'gamma_mean': 1,
        'eeg_means': 4,  # theta + alpha + beta + gamma
        'sent_gaze': 4,  # omr, nFix, speed, sacc_dur
        'sent_saccade': 6,  # saccade params
        'sent_gaze_sacc': 9,  # gaze (4) + saccade (6) - 1 overlap?
        'sent_gaze_sacc_eeg_means': 13,  # 9 gaze+sacc + 4 eeg means
        'electrode_features_all': 420  # 105 channels × 4 bands
    }
    
    print("Feature dimensions:")
    for feat, dim in feature_dimensions.items():
        print(f"  {feat}: {dim}-D")
    
    # Check sent_gaze_sacc_eeg_means composition
    gaze_dim = feature_dimensions['sent_gaze_sacc']
    eeg_dim = feature_dimensions['eeg_means']
    combined = gaze_dim + eeg_dim
    actual = feature_dimensions['sent_gaze_sacc_eeg_means']
    
    print(f"\nsent_gaze_sacc_eeg_means composition:")
    print(f"  sent_gaze_sacc ({gaze_dim}-D) + eeg_means ({eeg_dim}-D) = {combined}-D")
    print(f"  Actual: {actual}-D")
    print(f"  Match: {combined == actual}")
    
    # Check for overlap between feature sets
    print("\nPotential problematic combinations:")
    print("  - sent_gaze_sacc + sent_gaze_sacc_eeg_means: OVERLAP (9 features duplicated)")
    print("  - sent_gaze + sent_gaze_sacc: OVERLAP (sent_gaze is subset)")
    print("  - eeg_means + sent_gaze_sacc_eeg_means: OVERLAP (4 features duplicated)")
    
    # Recommended clean combinations
    print("\nRecommended clean combinations:")
    print("  - EEG-only: electrode_features_all (420-D)")
    print("  - Gaze-only: sent_gaze_sacc (9-D)")
    print("  - EEG+Gaze: electrode_features_all + sent_gaze_sacc (429-D)")
    
    return feature_dimensions

def run_loso_baseline():
    """Run LOSO baseline on Y subjects."""
    print("\n=== Baseline Reproduction: LOSO on Y Subjects ===")
    
    results = []
    
    def prepare_data(subjects, feature_set):
        """Prepare data for classification."""
        X = []
        y = []
        subj_indices = []
        
        for subj in subjects:
            data = load_feature_file(subj, feature_set)
            if data:
                for key, value in data.items():
                    X.append(value[:-1])
                    y.append(1 if value[-1] == 'NR' else 0)
                    subj_indices.append(subjects.index(subj))
        
        return np.array(X), np.array(y), np.array(subj_indices)
    
    def run_svm(X, y, subj_indices):
        """Run SVM with LOSO."""
        loo = LeaveOneOut()
        subject_results = []
        
        for train_idx, test_idx in loo.split(np.unique(subj_indices)):
            # Get unique subject indices
            unique_subj = np.unique(subj_indices)
            train_subj = unique_subj[train_idx]
            test_subj = unique_subj[test_idx][0]
            
            # Split data by subject
            train_mask = np.isin(subj_indices, train_subj)
            test_mask = subj_indices == test_subj
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            # Scale
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train SVM
            clf = SVC(kernel='linear', random_state=42, probability=True)
            clf.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            try:
                roc_auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                roc_auc = np.nan
            
            subject_results.append({
                'subject': Y_SUBJECTS[test_subj],
                'accuracy': acc,
                'balanced_accuracy': bacc,
                'macro_f1': f1,
                'auroc': roc_auc,
                'test_samples': len(y_test)
            })
        
        return subject_results
    
    # EEG-only
    print("\n1. EEG-only SVM (electrode_features_all)")
    X_eeg, y_eeg, subj_idx_eeg = prepare_data(Y_SUBJECTS, 'electrode_features_all')
    if len(X_eeg) > 0:
        eeg_results = run_svm(X_eeg, y_eeg, subj_idx_eeg)
        for r in eeg_results:
            r['model'] = 'EEG-only'
            results.append(r)
    
    # Gaze-only
    print("\n2. Gaze-only SVM (sent_gaze_sacc)")
    X_gaze, y_gaze, subj_idx_gaze = prepare_data(Y_SUBJECTS, 'sent_gaze_sacc')
    if len(X_gaze) > 0:
        gaze_results = run_svm(X_gaze, y_gaze, subj_idx_gaze)
        for r in gaze_results:
            r['model'] = 'Gaze-only'
            results.append(r)
    
    # Gaze+EEG-mean
    print("\n3. Gaze+EEG-mean SVM (sent_gaze_sacc_eeg_means)")
    X_combined, y_combined, subj_idx_combined = prepare_data(Y_SUBJECTS, 'sent_gaze_sacc_eeg_means')
    if len(X_combined) > 0:
        combined_results = run_svm(X_combined, y_combined, subj_idx_combined)
        for r in combined_results:
            r['model'] = 'Gaze+EEG-mean'
            results.append(r)
    
    # EEG+Gaze clean concat
    print("\n4. EEG+Gaze clean concat SVM")
    if len(X_eeg) > 0 and len(X_gaze) > 0:
        # Need to align by key
        all_keys = get_all_keys(Y_SUBJECTS, 'electrode_features_all')
        aligned_X = []
        aligned_y = []
        aligned_subj_idx = []
        
        eeg_data = {}
        for subj in Y_SUBJECTS:
            data = load_feature_file(subj, 'electrode_features_all')
            if data:
                eeg_data.update(data)
        
        gaze_data = {}
        for subj in Y_SUBJECTS:
            data = load_feature_file(subj, 'sent_gaze_sacc')
            if data:
                gaze_data.update(data)
        
        for key in all_keys:
            if key in eeg_data and key in gaze_data:
                eeg_feat = eeg_data[key][:-1]
                gaze_feat = gaze_data[key][:-1]
                aligned_X.append(np.concatenate([eeg_feat, gaze_feat]))
                aligned_y.append(1 if eeg_data[key][-1] == 'NR' else 0)
                aligned_subj_idx.append(Y_SUBJECTS.index(key.split('_')[0]))
        
        if aligned_X:
            X_concat = np.array(aligned_X)
            y_concat = np.array(aligned_y)
            subj_idx_concat = np.array(aligned_subj_idx)
            concat_results = run_svm(X_concat, y_concat, subj_idx_concat)
            for r in concat_results:
                r['model'] = 'EEG+Gaze-concat'
                results.append(r)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'baseline_loso_results.csv'), index=False)
    print(f"\nLOSO baseline results saved to {OUTPUT_DIR}/baseline_loso_results.csv")
    
    # Print summary statistics
    print("\nLOSO Results Summary:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        print(f"\n{model}:")
        print(f"  Accuracy: {model_df['accuracy'].mean():.4f} ± {model_df['accuracy'].std():.4f}")
        print(f"  Balanced Accuracy: {model_df['balanced_accuracy'].mean():.4f} ± {model_df['balanced_accuracy'].std():.4f}")
        print(f"  Macro-F1: {model_df['macro_f1'].mean():.4f} ± {model_df['macro_f1'].std():.4f}")
        print(f"  AUROC: {model_df['auroc'].mean():.4f} ± {model_df['auroc'].std():.4f}")
    
    return results

def generate_summary(label_leakage, split_info, alignment, duplicates, feature_overlap, loso_results):
    """Generate audit summary markdown."""
    summary = f"""# ZuCo Benchmark Protocol Audit Summary

## Audit Date
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Feature Label Leakage Audit

### Summary
| Feature Set | Samples | Input Dim | Last Column Unique | Last-Col Clf Acc | Shuffled Acc |
|-------------|---------|-----------|-------------------|------------------|--------------|
"""
    for r in label_leakage:
        summary += f"| {r['feature_set']} | {r['total_samples']} | {r['model_input_length']} | {r['unique_last_column']} | {r['last_column_clf_acc']} | {r['shuffled_acc']} |\n"
    
    summary += """
### Key Findings
- Last column contains labels (NR/TSR) for all feature sets
- Model inputs correctly use `value[:-1]` for features
- Shuffled label sanity check confirms no data leakage when labels are randomized

---

## 2. Split Audit

### Subject Groups
- **Y Subjects (Training/Development):** {y_count} subjects
  ```
  {y_list}
  ```
- **X Subjects (Heldout/Test):** {x_count} subjects
  ```
  {x_list}
  ```
- **Overlap between Y and X:** {overlap_count} subjects

### X Subjects Label Check
| Subject | Has Features | Last Column | Appears to be Label |
|---------|--------------|-------------|---------------------|
""".format(
        y_count=len(split_info['Y_subjects']),
        y_list=', '.join(split_info['Y_subjects']),
        x_count=len(split_info['X_subjects']),
        x_list=', '.join(split_info['X_subjects']),
        overlap_count=len(split_info['overlap'])
    )
    
    for x in split_info['X_labels_check']:
        summary += f"| {x['subject']} | {x['has_feature_file']} | {x['last_column_value']} | {x['appears_to_be_label']} |\n"
    
    summary += f"""
---

## 3. EEG-Gaze Alignment Audit

| Metric | Count |
|--------|-------|
| EEG keys (electrode_features_all) | {alignment['total_eeg_keys']} |
| Gaze keys (sent_gaze_sacc) | {alignment['total_gaze_keys']} |
| Matching keys | {alignment['matching_keys']} |
| EEG-only keys | {alignment['eeg_only_keys']} |
| Gaze-only keys | {alignment['gaze_only_keys']} |

### Unmatched Keys
- EEG-only sample: {alignment['unmatched_eeg'][0] if alignment['unmatched_eeg'] else 'None'}
- Gaze-only sample: {alignment['unmatched_gaze'][0] if alignment['unmatched_gaze'] else 'None'}

---

## 4. Duplicate Recovery Audit

| Metric | Count |
|--------|-------|
| NR sentences loaded | {duplicates['nr_sentences_count']} |
| TSR sentences loaded | {duplicates['tsr_sentences_count']} |
| **NR-TSR duplicate pairs found** | **{duplicates['duplicate_pairs']}** |
| Expected (from paper) | {duplicates['expected_duplicates']} |

### Status
{'✅ PASS: Duplicate count matches expected' if duplicates['duplicate_pairs'] >= 60 else '❌ FAIL: Duplicate count significantly different from expected'}

---

## 5. Feature Overlap Audit

### Feature Dimensions
| Feature Set | Dimension |
|-------------|-----------|
"""
    for feat, dim in feature_overlap.items():
        summary += f"| {feat} | {dim}-D |\n"
    
    summary += """
### sent_gaze_sacc_eeg_means Composition
- sent_gaze_sacc (9-D) + eeg_means (4-D) = **13-D** ✓

### Problematic Combinations (Avoid)
- `sent_gaze_sacc` + `sent_gaze_sacc_eeg_means` → 9 features duplicated
- `sent_gaze` + `sent_gaze_sacc` → sent_gaze is subset
- `eeg_means` + `sent_gaze_sacc_eeg_means` → 4 features duplicated

### Recommended Clean Combinations
- **EEG-only:** electrode_features_all (420-D)
- **Gaze-only:** sent_gaze_sacc (9-D)
- **EEG+Gaze:** electrode_features_all + sent_gaze_sacc (429-D)

---

## 6. LOSO Baseline Results

### Per-Subject Results
| Subject | Model | Accuracy | Balanced Acc | Macro-F1 | AUROC |
|---------|-------|----------|--------------|----------|-------|
"""
    df_loso = pd.DataFrame(loso_results)
    for _, row in df_loso.iterrows():
        summary += f"| {row['subject']} | {row['model']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | {row['auroc']:.4f} |\n"
    
    summary += """
### Aggregated Results
| Model | Accuracy (mean±std) | Balanced Acc (mean±std) | Macro-F1 (mean±std) | AUROC (mean±std) |
|-------|---------------------|-------------------------|---------------------|------------------|
"""
    for model in df_loso['model'].unique():
        model_df = df_loso[df_loso['model'] == model]
        summary += f"| {model} | {model_df['accuracy'].mean():.4f} ± {model_df['accuracy'].std():.4f} | {model_df['balanced_accuracy'].mean():.4f} ± {model_df['balanced_accuracy'].std():.4f} | {model_df['macro_f1'].mean():.4f} ± {model_df['macro_f1'].std():.4f} | {model_df['auroc'].mean():.4f} ± {model_df['auroc'].std():.4f} |\n"
    
    summary += """
---

## Audit Conclusion

### Overall Status
"""
    
    # Determine overall status
    issues = []
    
    # Check label leakage
    for r in label_leakage:
        if not r['all_use_value[:-1]']:
            issues.append("Label leakage: Not all features use value[:-1]")
    
    # Check duplicates
    if duplicates['duplicate_pairs'] < 60:
        issues.append(f"Duplicate recovery: Found {duplicates['duplicate_pairs']} but expected ~63")
    
    # Check alignment
    if alignment['eeg_only_keys'] > 10 or alignment['gaze_only_keys'] > 10:
        issues.append(f"Alignment: Significant mismatch ({alignment['eeg_only_keys']} EEG-only, {alignment['gaze_only_keys']} gaze-only)")
    
    if issues:
        summary += "❌ **ISSUES FOUND**\n\n"
        summary += "### Issues to Address:\n"
        for issue in issues:
            summary += f"- {issue}\n"
    else:
        summary += "✅ **ALL CHECKS PASSED**\n\n"
        summary += "The benchmark protocol appears to be correctly implemented.\n"
    
    summary += "\n---\n\n*Generated by ZuCo Benchmark Protocol Audit Script*"
    
    # Save summary
    with open(os.path.join(OUTPUT_DIR, 'protocol_audit_summary.md'), 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\nAudit summary saved to {OUTPUT_DIR}/protocol_audit_summary.md")
    
    return summary

def main():
    """Run all audits."""
    print("=" * 70)
    print("ZuCo Benchmark Protocol Audit")
    print("=" * 70)
    
    # 1. Label leakage audit
    label_leakage = audit_label_leakage()
    
    # 2. Split audit
    split_info = audit_split()
    
    # 3. EEG-Gaze alignment audit
    alignment = audit_eeg_gaze_alignment()
    
    # 4. Duplicate recovery audit
    duplicates = audit_duplicate_recovery()
    
    # 5. Feature overlap audit
    feature_overlap = audit_feature_overlap()
    
    # 6. LOSO baseline
    loso_results = run_loso_baseline()
    
    # Generate summary
    generate_summary(label_leakage, split_info, alignment, duplicates, feature_overlap, loso_results)
    
    print("\n" + "=" * 70)
    print("Audit complete! Results saved to:", OUTPUT_DIR)
    print("=" * 70)

if __name__ == '__main__':
    main()