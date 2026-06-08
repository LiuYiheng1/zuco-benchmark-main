#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZuCo Benchmark Alignment Diagnosis - Round 3
Build aligned multimodal table and run clean baselines
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score

# Configuration
FEATURES_DIR = 'src/features'
OUTPUT_DIR = 'audit_results'
DATA_DIR = 'data'

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
X_SUBJECTS = ['XBB', 'XDT', 'XLS', 'XPB', 'XSE', 'XTR', 'XWS', 'XAH', 'XBD', 'XSS']

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

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

def get_projection_key(key):
    """Extract subject+label+idx projection from full key."""
    parts = key.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])
    return key

def task1_key_uniqueness():
    """Task 1: Key uniqueness audit for subject+label+idx projection."""
    print("\n=== Task 1: Key Uniqueness Audit ===")
    
    results = []
    
    for feat_set in ['electrode_features_all', 'sent_gaze_sacc']:
        all_keys = []
        all_proj_keys = []
        
        for subj in Y_SUBJECTS + X_SUBJECTS:
            data = load_feature_file(subj, feat_set)
            if data:
                for key in data.keys():
                    all_keys.append(key)
                    all_proj_keys.append(get_projection_key(key))
        
        total_samples = len(all_keys)
        unique_proj_keys = len(set(all_proj_keys))
        
        # Find duplicates
        key_counts = defaultdict(int)
        for pk in all_proj_keys:
            key_counts[pk] += 1
        
        duplicate_keys = [(k, v) for k, v in key_counts.items() if v > 1]
        duplicate_count = len(duplicate_keys)
        duplicate_samples = sum(v - 1 for _, v in duplicate_keys)
        
        # Top 20 duplicate keys
        top_duplicates = sorted(duplicate_keys, key=lambda x: -x[1])[:20]
        
        print(f"{feat_set}:")
        print(f"  Total samples: {total_samples}")
        print(f"  Unique projection keys: {unique_proj_keys}")
        print(f"  Duplicate keys: {duplicate_count}")
        print(f"  Samples lost to duplicates: {duplicate_samples}")
        
        results.append({
            'feature_set': feat_set,
            'total_samples': total_samples,
            'unique_proj_keys': unique_proj_keys,
            'duplicate_key_count': duplicate_count,
            'duplicate_samples': duplicate_samples,
            'is_one_to_one': duplicate_count == 0
        })
        
        # Save duplicate details
        dup_results = [{
            'feature_set': feat_set,
            'proj_key': k,
            'count': v
        } for k, v in top_duplicates]
        
        df_dup = pd.DataFrame(dup_results)
        df_dup.to_csv(os.path.join(OUTPUT_DIR, f'audit_duplicates_{feat_set}.csv'), index=False)
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'audit_key_uniqueness_subject_label_idx.csv'), index=False)
    print(f"\nKey uniqueness audit saved to {OUTPUT_DIR}/audit_key_uniqueness_subject_label_idx.csv")
    
    return results

def task2_build_multimodal_table():
    """Task 2: Build aligned multimodal table."""
    print("\n=== Task 2: Building Aligned Multimodal Table ===")
    
    # Load all EEG and Gaze data
    eeg_data = {}  # proj_key -> (full_key, value)
    gaze_data = {}  # proj_key -> (full_key, value)
    
    for subj in Y_SUBJECTS:
        eeg = load_feature_file(subj, 'electrode_features_all')
        gaze = load_feature_file(subj, 'sent_gaze_sacc')
        
        if eeg:
            for key, value in eeg.items():
                proj_key = get_projection_key(key)
                eeg_data[proj_key] = (key, value)
        
        if gaze:
            for key, value in gaze.items():
                proj_key = get_projection_key(key)
                gaze_data[proj_key] = (key, value)
    
    # Find common projection keys
    common_keys = set(eeg_data.keys()) & set(gaze_data.keys())
    print(f"Found {len(common_keys)} common projection keys")
    
    # Build aligned table
    aligned_data = []
    metadata = []
    
    for proj_key in sorted(common_keys):
        eeg_full_key, eeg_value = eeg_data[proj_key]
        gaze_full_key, gaze_value = gaze_data[proj_key]
        
        # Parse keys
        eeg_parts = eeg_full_key.split('_')
        gaze_parts = gaze_full_key.split('_')
        
        subject = eeg_parts[0]
        eeg_label = eeg_parts[1]
        idx = eeg_parts[2]
        eeg_fullidx = eeg_parts[3] if len(eeg_parts) > 3 else ''
        gaze_fullidx = gaze_parts[3] if len(gaze_parts) > 3 else ''
        
        # Verify labels match
        gaze_label = gaze_parts[1]
        value_label = eeg_value[-1]
        
        if eeg_label != gaze_label or eeg_label != value_label:
            print(f"Label mismatch at {proj_key}: eeg={eeg_label}, gaze={gaze_label}, value={value_label}")
            continue
        
        # Extract features
        eeg_feat = eeg_value[:-1]  # 420-D
        gaze_feat = gaze_value[:-1]  # 9-D
        y = 1 if eeg_label == 'NR' else 0
        
        # Store aligned data
        aligned_data.append({
            'eeg': eeg_feat,
            'gaze': gaze_feat,
            'y': y
        })
        
        # Metadata
        metadata.append({
            'sample_id': proj_key,
            'subject': subject,
            'label': eeg_label,
            'idx': idx,
            'eeg_fullidx': eeg_fullidx,
            'gaze_fullidx': gaze_fullidx,
            'y': y,
            'split': 'train' if subject in Y_SUBJECTS else 'test'
        })
    
    # Convert to numpy arrays
    if aligned_data:
        eeg_array = np.array([d['eeg'] for d in aligned_data])
        gaze_array = np.array([d['gaze'] for d in aligned_data])
        y_array = np.array([d['y'] for d in aligned_data])
        
        # Save to npz
        np.savez(os.path.join(DATA_DIR, 'aligned_multimodal_y.npz'),
                 eeg=eeg_array, gaze=gaze_array, y=y_array)
        print(f"Aligned multimodal data saved to {DATA_DIR}/aligned_multimodal_y.npz")
        
        # Save metadata
        df_meta = pd.DataFrame(metadata)
        df_meta.to_csv(os.path.join(DATA_DIR, 'aligned_multimodal_y_metadata.csv'), index=False)
        print(f"Metadata saved to {DATA_DIR}/aligned_multimodal_y_metadata.csv")
    else:
        print("No aligned data found!")
    
    return len(common_keys), len(metadata)

def task3_validate_against_official():
    """Task 3: Validate against official combined feature."""
    print("\n=== Task 3: Validate Against Official Combined Feature ===")
    
    results = []
    
    for subj in Y_SUBJECTS:
        # Load features
        gaze = load_feature_file(subj, 'sent_gaze_sacc')
        eeg_mean = load_feature_file(subj, 'eeg_means')
        combined = load_feature_file(subj, 'sent_gaze_sacc_eeg_means')
        
        if not all([gaze, eeg_mean, combined]):
            print(f"Skipping {subj}: missing feature files")
            results.append({
                'subject': subj,
                'has_data': False,
                'gaze_match_ratio': 0,
                'eeg_match_ratio': 0,
                'reason': 'missing feature files'
            })
            continue
        
        # Build projection key maps
        gaze_map = {get_projection_key(k): v[:-1] for k, v in gaze.items()}
        eeg_map = {get_projection_key(k): v[:-1] for k, v in eeg_mean.items()}
        combined_map = {get_projection_key(k): v[:-1] for k, v in combined.items()}
        
        # Find common keys
        common_keys = set(gaze_map.keys()) & set(eeg_map.keys()) & set(combined_map.keys())
        
        if not common_keys:
            print(f"Skipping {subj}: no common keys")
            results.append({
                'subject': subj,
                'has_data': True,
                'gaze_match_ratio': 0,
                'eeg_match_ratio': 0,
                'reason': 'no common projection keys'
            })
            continue
        
        # Validate
        gaze_matches = []
        eeg_matches = []
        
        for key in common_keys:
            gaze_feat = gaze_map[key]
            eeg_feat = eeg_map[key]
            combined_feat = combined_map[key]
            
            # Check gaze part (first 9)
            gaze_part = combined_feat[:9]
            gaze_match = np.allclose(gaze_feat, gaze_part, rtol=1e-5, atol=1e-8)
            gaze_matches.append(gaze_match)
            
            # Check EEG part (last 4)
            eeg_part = combined_feat[9:13]
            eeg_match = np.allclose(eeg_feat, eeg_part, rtol=1e-5, atol=1e-8)
            eeg_matches.append(eeg_match)
        
        gaze_ratio = np.mean(gaze_matches)
        eeg_ratio = np.mean(eeg_matches)
        
        reason = 'pass' if gaze_ratio == 1.0 and eeg_ratio == 1.0 else 'partial mismatch'
        
        print(f"{subj}: Gaze={gaze_ratio:.2%}, EEG={eeg_ratio:.2%}, Reason={reason}")
        
        results.append({
            'subject': subj,
            'has_data': True,
            'common_keys': len(common_keys),
            'gaze_match_ratio': gaze_ratio,
            'eeg_match_ratio': eeg_ratio,
            'reason': reason
        })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'audit_join_against_official_combined.csv'), index=False)
    print(f"\nValidation results saved to {OUTPUT_DIR}/audit_join_against_official_combined.csv")
    
    # Find failed subjects
    failed = [r for r in results if r['gaze_match_ratio'] < 1.0 or r['eeg_match_ratio'] < 1.0]
    print(f"\nFailed subjects: {len(failed)}")
    for f in failed:
        print(f"  {f['subject']}: {f['reason']}")
    
    return results

def task4_sample_distribution():
    """Task 4: Sample distribution audit."""
    print("\n=== Task 4: Sample Distribution Audit ===")
    
    # Load metadata
    meta_path = os.path.join(DATA_DIR, 'aligned_multimodal_y_metadata.csv')
    if not os.path.exists(meta_path):
        print("Metadata file not found!")
        return
    
    df_meta = pd.read_csv(meta_path)
    
    # Load original gaze data to count gaze-only samples
    gaze_only_counts = defaultdict(int)
    eeg_counts = defaultdict(int)
    
    for subj in Y_SUBJECTS:
        gaze = load_feature_file(subj, 'sent_gaze_sacc')
        eeg = load_feature_file(subj, 'electrode_features_all')
        
        if gaze:
            gaze_only_counts[subj] = len(gaze)
        if eeg:
            eeg_counts[subj] = len(eeg)
    
    # Calculate statistics
    results = []
    total_aligned = 0
    total_gaze_only = 0
    
    for subj in Y_SUBJECTS:
        subj_meta = df_meta[df_meta['subject'] == subj]
        nr_count = len(subj_meta[subj_meta['label'] == 'NR'])
        tsr_count = len(subj_meta[subj_meta['label'] == 'TSR'])
        aligned_count = len(subj_meta)
        
        # Gaze-only dropped = total gaze - aligned
        gaze_total = gaze_only_counts.get(subj, 0)
        dropped = gaze_total - aligned_count
        
        total_aligned += aligned_count
        total_gaze_only += dropped
        
        results.append({
            'subject': subj,
            'nr_count': nr_count,
            'tsr_count': tsr_count,
            'aligned_count': aligned_count,
            'gaze_total': gaze_total,
            'gaze_only_dropped': dropped,
            'drop_rate': dropped / gaze_total if gaze_total > 0 else 0
        })
    
    # Overall statistics
    results.append({
        'subject': 'ALL',
        'nr_count': len(df_meta[df_meta['label'] == 'NR']),
        'tsr_count': len(df_meta[df_meta['label'] == 'TSR']),
        'aligned_count': total_aligned,
        'gaze_total': sum(gaze_only_counts.values()),
        'gaze_only_dropped': total_gaze_only,
        'drop_rate': total_gaze_only / sum(gaze_only_counts.values()) if sum(gaze_only_counts.values()) > 0 else 0
    })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'audit_aligned_sample_distribution.csv'), index=False)
    print(f"\nSample distribution saved to {OUTPUT_DIR}/audit_aligned_sample_distribution.csv")
    
    # Print summary
    print("\nSample Distribution Summary:")
    print(f"Total aligned samples: {total_aligned}")
    print(f"Total gaze-only dropped: {total_gaze_only}")
    print(f"Overall drop rate: {total_gaze_only / sum(gaze_only_counts.values()):.2%}")
    
    return results

def task5_loso_baselines():
    """Task 5: Run LOSO baselines on aligned multimodal data."""
    print("\n=== Task 5: Running LOSO Baselines ===")
    
    # Load aligned data
    data = np.load(os.path.join(DATA_DIR, 'aligned_multimodal_y.npz'))
    eeg = data['eeg']
    gaze = data['gaze']
    y = data['y']
    
    # Load metadata to get subject info
    df_meta = pd.read_csv(os.path.join(DATA_DIR, 'aligned_multimodal_y_metadata.csv'))
    subjects = df_meta['subject'].values
    
    # Ensure data is aligned
    assert len(eeg) == len(gaze) == len(y) == len(subjects)
    
    # Create combined features
    eeg_gaze_concat = np.concatenate([eeg, gaze], axis=1)
    
    # Load official combined feature for comparison
    official_data = []
    official_subjects = []
    
    for subj in Y_SUBJECTS:
        combined = load_feature_file(subj, 'sent_gaze_sacc_eeg_means')
        if combined:
            for key, value in combined.items():
                official_data.append(value[:-1])  # 13-D
                official_subjects.append(subj)
    
    official_X = np.array(official_data)
    official_y = np.array([1 if 'NR' in k else 0 for k in official_subjects])
    
    results = []
    
    def run_loso(X, y, subjects, model_name):
        """Run LOSO for a single model."""
        loo = LeaveOneOut()
        subject_list = sorted(set(subjects))
        fold_results = []
        
        for train_idx, test_idx in loo.split(subject_list):
            train_subjects = [subject_list[i] for i in train_idx]
            test_subject = subject_list[test_idx[0]]
            
            # Split data
            train_mask = np.isin(subjects, train_subjects)
            test_mask = subjects == test_subject
            
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
            
            # PCA on EEG (keep 95% variance)
            pca = PCA(n_components=0.95, random_state=42)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            # Scale
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
    
    # 1. EEG-only SVM
    print("\n1. EEG-only SVM (420-D)")
    results.extend(run_loso(eeg, y, subjects, 'EEG-only'))
    
    # 2. Gaze-only SVM
    print("\n2. Gaze-only SVM (9-D)")
    results.extend(run_loso(gaze, y, subjects, 'Gaze-only'))
    
    # 3. EEG+Gaze clean concat SVM
    print("\n3. EEG+Gaze clean concat SVM (429-D)")
    results.extend(run_loso(eeg_gaze_concat, y, subjects, 'EEG+Gaze-concat'))
    
    # 4. EEG PCA + Gaze SVM
    print("\n4. EEG PCA + Gaze SVM")
    results.extend(run_loso_with_pca(eeg, y, subjects, 'EEG-PCA'))
    
    # 5. Official sent_gaze_sacc_eeg_means SVM
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

def main():
    """Run all tasks."""
    print("=" * 70)
    print("ZuCo Benchmark Alignment Diagnosis - Round 3")
    print("=" * 70)
    
    # Task 1: Key uniqueness audit
    task1_key_uniqueness()
    
    # Task 2: Build multimodal table
    task2_build_multimodal_table()
    
    # Task 3: Validate against official combined
    task3_validate_against_official()
    
    # Task 4: Sample distribution
    task4_sample_distribution()
    
    # Task 5: LOSO baselines
    task5_loso_baselines()
    
    print("\n" + "=" * 70)
    print("Round 3 diagnosis complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()