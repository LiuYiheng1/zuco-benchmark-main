#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZuCo Benchmark Alignment Diagnosis - Round 3 (Part 1)
Tasks 1-4: Key uniqueness, Build multimodal table, Validation, Distribution
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

FEATURES_DIR = 'src/features'
OUTPUT_DIR = 'audit_results'
DATA_DIR = 'data'

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def load_feature_file(subject, feature_set):
    filepath = os.path.join(FEATURES_DIR, f'{subject}_{feature_set}.npy')
    if not os.path.exists(filepath):
        return None
    try:
        return np.load(filepath, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_projection_key(key):
    parts = key.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])
    return key

def task1_key_uniqueness():
    print("\n=== Task 1: Key Uniqueness Audit ===")
    results = []
    
    for feat_set in ['electrode_features_all', 'sent_gaze_sacc']:
        all_keys = []
        all_proj_keys = []
        
        for subj in Y_SUBJECTS:
            data = load_feature_file(subj, feat_set)
            if data:
                for key in data.keys():
                    all_keys.append(key)
                    all_proj_keys.append(get_projection_key(key))
        
        total_samples = len(all_keys)
        unique_proj_keys = len(set(all_proj_keys))
        
        key_counts = defaultdict(int)
        for pk in all_proj_keys:
            key_counts[pk] += 1
        
        duplicate_keys = [(k, v) for k, v in key_counts.items() if v > 1]
        duplicate_count = len(duplicate_keys)
        duplicate_samples = sum(v - 1 for _, v in duplicate_keys)
        
        print(f"{feat_set}:")
        print(f"  Total samples: {total_samples}")
        print(f"  Unique projection keys: {unique_proj_keys}")
        print(f"  Duplicate keys: {duplicate_count}")
        
        results.append({
            'feature_set': feat_set,
            'total_samples': total_samples,
            'unique_proj_keys': unique_proj_keys,
            'duplicate_key_count': duplicate_count,
            'duplicate_samples': duplicate_samples,
            'is_one_to_one': duplicate_count == 0
        })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'audit_key_uniqueness_subject_label_idx.csv'), index=False)
    print(f"\nKey uniqueness saved to {OUTPUT_DIR}/audit_key_uniqueness_subject_label_idx.csv")
    return results

def task2_build_multimodal_table():
    print("\n=== Task 2: Building Aligned Multimodal Table ===")
    
    eeg_data = {}
    gaze_data = {}
    
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
    
    common_keys = set(eeg_data.keys()) & set(gaze_data.keys())
    print(f"Found {len(common_keys)} common projection keys")
    
    aligned_data = []
    metadata = []
    
    for proj_key in sorted(common_keys):
        eeg_full_key, eeg_value = eeg_data[proj_key]
        gaze_full_key, gaze_value = gaze_data[proj_key]
        
        eeg_parts = eeg_full_key.split('_')
        gaze_parts = gaze_full_key.split('_')
        
        subject = eeg_parts[0]
        eeg_label = eeg_parts[1]
        idx = eeg_parts[2]
        eeg_fullidx = eeg_parts[3] if len(eeg_parts) > 3 else ''
        gaze_fullidx = gaze_parts[3] if len(gaze_parts) > 3 else ''
        
        gaze_label = gaze_parts[1]
        value_label = eeg_value[-1]
        
        if eeg_label != gaze_label or eeg_label != value_label:
            continue
        
        eeg_feat = eeg_value[:-1]
        gaze_feat = gaze_value[:-1]
        y = 1 if eeg_label == 'NR' else 0
        
        aligned_data.append({
            'eeg': eeg_feat,
            'gaze': gaze_feat,
            'y': y
        })
        
        metadata.append({
            'sample_id': proj_key,
            'subject': subject,
            'label': eeg_label,
            'idx': idx,
            'eeg_fullidx': eeg_fullidx,
            'gaze_fullidx': gaze_fullidx,
            'y': y,
            'split': 'train'
        })
    
    if aligned_data:
        eeg_array = np.array([d['eeg'] for d in aligned_data])
        gaze_array = np.array([d['gaze'] for d in aligned_data])
        y_array = np.array([d['y'] for d in aligned_data])
        
        np.savez(os.path.join(DATA_DIR, 'aligned_multimodal_y.npz'),
                 eeg=eeg_array, gaze=gaze_array, y=y_array)
        print(f"Aligned data saved to {DATA_DIR}/aligned_multimodal_y.npz")
        
        df_meta = pd.DataFrame(metadata)
        df_meta.to_csv(os.path.join(DATA_DIR, 'aligned_multimodal_y_metadata.csv'), index=False)
        print(f"Metadata saved to {DATA_DIR}/aligned_multimodal_y_metadata.csv")
    else:
        print("No aligned data found!")
    
    return len(common_keys), len(metadata)

def task3_validate_against_official():
    print("\n=== Task 3: Validate Against Official Combined Feature ===")
    results = []
    
    for subj in Y_SUBJECTS:
        gaze = load_feature_file(subj, 'sent_gaze_sacc')
        eeg_mean = load_feature_file(subj, 'eeg_means')
        combined = load_feature_file(subj, 'sent_gaze_sacc_eeg_means')
        
        if not all([gaze, eeg_mean, combined]):
            print(f"Skipping {subj}: missing files")
            results.append({
                'subject': subj,
                'has_data': False,
                'reason': 'missing feature files'
            })
            continue
        
        gaze_map = {get_projection_key(k): v[:-1] for k, v in gaze.items()}
        eeg_map = {get_projection_key(k): v[:-1] for k, v in eeg_mean.items()}
        combined_map = {get_projection_key(k): v[:-1] for k, v in combined.items()}
        
        common_keys = set(gaze_map.keys()) & set(eeg_map.keys()) & set(combined_map.keys())
        
        if not common_keys:
            print(f"Skipping {subj}: no common keys")
            results.append({
                'subject': subj,
                'has_data': True,
                'reason': 'no common projection keys'
            })
            continue
        
        gaze_matches = []
        eeg_matches = []
        
        for key in common_keys:
            gaze_feat = gaze_map[key]
            eeg_feat = eeg_map[key]
            combined_feat = combined_map[key]
            
            gaze_part = combined_feat[:9]
            gaze_matches.append(np.allclose(gaze_feat, gaze_part, rtol=1e-5, atol=1e-8))
            
            eeg_part = combined_feat[9:13]
            eeg_matches.append(np.allclose(eeg_feat, eeg_part, rtol=1e-5, atol=1e-8))
        
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
    print(f"\nValidation saved to {OUTPUT_DIR}/audit_join_against_official_combined.csv")
    
    failed = [r for r in results if r.get('gaze_match_ratio', 1) < 1.0 or r.get('eeg_match_ratio', 1) < 1.0]
    print(f"\nFailed subjects: {len(failed)}")
    return results

def task4_sample_distribution():
    print("\n=== Task 4: Sample Distribution Audit ===")
    
    meta_path = os.path.join(DATA_DIR, 'aligned_multimodal_y_metadata.csv')
    if not os.path.exists(meta_path):
        print("Metadata file not found!")
        return
    
    df_meta = pd.read_csv(meta_path)
    
    gaze_only_counts = defaultdict(int)
    eeg_counts = defaultdict(int)
    
    for subj in Y_SUBJECTS:
        gaze = load_feature_file(subj, 'sent_gaze_sacc')
        eeg = load_feature_file(subj, 'electrode_features_all')
        
        if gaze:
            gaze_only_counts[subj] = len(gaze)
        if eeg:
            eeg_counts[subj] = len(eeg)
    
    results = []
    total_aligned = 0
    total_gaze_only = 0
    
    for subj in Y_SUBJECTS:
        subj_meta = df_meta[df_meta['subject'] == subj]
        nr_count = len(subj_meta[subj_meta['label'] == 'NR'])
        tsr_count = len(subj_meta[subj_meta['label'] == 'TSR'])
        aligned_count = len(subj_meta)
        
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
    
    print("\nSample Distribution Summary:")
    print(f"Total aligned samples: {total_aligned}")
    print(f"Total gaze-only dropped: {total_gaze_only}")
    print(f"Overall drop rate: {total_gaze_only / sum(gaze_only_counts.values()):.2%}")
    
    return results

def main():
    print("=" * 70)
    print("ZuCo Benchmark Alignment Diagnosis - Round 3 (Part 1)")
    print("=" * 70)
    
    task1_key_uniqueness()
    task2_build_multimodal_table()
    task3_validate_against_official()
    task4_sample_distribution()
    
    print("\n" + "=" * 70)
    print("Part 1 complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()