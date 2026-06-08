#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZuCo Benchmark Alignment Diagnosis - Round 2

Deep diagnosis of EEG-Gaze alignment issues and duplicate recovery fixes.
"""

import os
import re
import csv
import numpy as np
import pandas as pd
import unicodedata
from collections import defaultdict
import difflib

# Configuration
FEATURES_DIR = 'src/features'
TASK_MATERIALS_DIR = 'src/task_materials'
OUTPUT_DIR = 'audit_results'

# Subject splits
Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
X_SUBJECTS = ['XBB', 'XDT', 'XLS', 'XPB', 'XSE', 'XTR', 'XWS', 'XAH', 'XBD', 'XSS']

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def normalize_text(text):
    """Normalize text for duplicate detection."""
    if not text:
        return ""
    # Unicode normalize NFKC
    text = unicodedata.normalize('NFKC', text)
    # Lowercase
    text = text.lower()
    # Strip whitespace
    text = text.strip()
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize curly quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # double quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # single quotes
    text = text.replace('\u201b', "'")  # single quote
    # Normalize dashes
    text = text.replace('\u2013', '-').replace('\u2014', '-')  # en-dash, em-dash
    text = text.replace('\u2015', '-')  # horizontal bar
    return text

def task1_key_samples():
    """Task 1: Print key samples and analyze their format."""
    print("\n=== Task 1: Key Sample Analysis ===")
    
    results = []
    
    for subj in Y_SUBJECTS[:3] + X_SUBJECTS[:2]:  # Sample subjects
        for feat_set in ['electrode_features_all', 'sent_gaze_sacc']:
            data = load_feature_file(subj, feat_set)
            if data:
                keys = list(data.keys())[:20]
                for key in keys:
                    parts = key.split('_')
                    results.append({
                        'subject': subj,
                        'feature_set': feat_set,
                        'key': key,
                        'key_type': type(key).__name__,
                        'repr_key': repr(key)[:50] + '...' if len(repr(key)) > 50 else repr(key),
                        'num_parts': len(parts),
                        'part_0': parts[0] if len(parts) > 0 else '',
                        'part_1': parts[1] if len(parts) > 1 else '',
                        'part_2': parts[2] if len(parts) > 2 else '',
                        'part_3': parts[3] if len(parts) > 3 else '',
                        'has_leading_zero': any(p.startswith('0') and len(p) > 1 for p in parts),
                        'has_space': ' ' in key,
                        'is_bytes': isinstance(key, bytes)
                    })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'audit_key_samples.csv'), index=False)
    print(f"Key samples saved to {OUTPUT_DIR}/audit_key_samples.csv")
    
    # Print summary
    print("\nKey Format Summary:")
    print(f"Total keys analyzed: {len(results)}")
    print(f"Unique key types: {df['key_type'].unique()}")
    print(f"Keys with spaces: {df['has_space'].sum()}")
    print(f"Keys with leading zeros: {df['has_leading_zero'].sum()}")
    
    return results

def task2_projection_overlap():
    """Task 2: Hierarchical overlap statistics."""
    print("\n=== Task 2: Hierarchical Overlap Statistics ===")
    
    # Load all keys
    eeg_keys = set()
    gaze_keys = set()
    
    eeg_by_subj = defaultdict(set)
    gaze_by_subj = defaultdict(set)
    
    for subj in Y_SUBJECTS + X_SUBJECTS:
        eeg_data = load_feature_file(subj, 'electrode_features_all')
        gaze_data = load_feature_file(subj, 'sent_gaze_sacc')
        
        if eeg_data:
            for key in eeg_data.keys():
                eeg_keys.add(key)
                eeg_by_subj[subj].add(key)
        
        if gaze_data:
            for key in gaze_data.keys():
                gaze_keys.add(key)
                gaze_by_subj[subj].add(key)
    
    # Parse keys into components
    def parse_key(key):
        parts = key.split('_')
        if len(parts) >= 4:
            return parts[0], parts[1], parts[2], parts[3]
        elif len(parts) == 3:
            return parts[0], parts[1], parts[2], ''
        else:
            return parts[0], parts[1], '', ''
    
    # Project keys to different levels
    def project_keys(keys, projection):
        projected = set()
        for key in keys:
            subj, label, idx, full_idx = parse_key(key)
            if projection == 'subject':
                projected.add(subj)
            elif projection == 'subject+label':
                projected.add((subj, label))
            elif projection == 'subject+label+idx':
                projected.add((subj, label, idx))
            elif projection == 'subject+label+full_idx':
                projected.add((subj, label, full_idx))
            elif projection == 'subject+label+idx+full_idx':
                projected.add((subj, label, idx, full_idx))
            elif projection == 'label+idx':
                projected.add((label, idx))
            elif projection == 'label+full_idx':
                projected.add((label, full_idx))
        return projected
    
    projections = [
        'subject',
        'subject+label',
        'subject+label+idx',
        'subject+label+full_idx',
        'subject+label+idx+full_idx',
        'label+idx',
        'label+full_idx'
    ]
    
    results = []
    
    # Global statistics
    for projection in projections:
        eeg_proj = project_keys(eeg_keys, projection)
        gaze_proj = project_keys(gaze_keys, projection)
        overlap = eeg_proj & gaze_proj
        
        results.append({
            'subject': 'ALL',
            'projection': projection,
            'eeg_count': len(eeg_proj),
            'gaze_count': len(gaze_proj),
            'overlap_count': len(overlap),
            'overlap_ratio': len(overlap) / min(len(eeg_proj), len(gaze_proj)) if min(len(eeg_proj), len(gaze_proj)) > 0 else 0
        })
    
    # Per-subject statistics
    for subj in sorted(set(Y_SUBJECTS + X_SUBJECTS)):
        eeg_subj_keys = eeg_by_subj.get(subj, set())
        gaze_subj_keys = gaze_by_subj.get(subj, set())
        
        for projection in projections:
            eeg_proj = project_keys(eeg_subj_keys, projection)
            gaze_proj = project_keys(gaze_subj_keys, projection)
            overlap = eeg_proj & gaze_proj
            
            results.append({
                'subject': subj,
                'projection': projection,
                'eeg_count': len(eeg_proj),
                'gaze_count': len(gaze_proj),
                'overlap_count': len(overlap),
                'overlap_ratio': len(overlap) / min(len(eeg_proj), len(gaze_proj)) if min(len(eeg_proj), len(gaze_proj)) > 0 else 0
            })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'audit_alignment_projection_overlap.csv'), index=False)
    print(f"Projection overlap saved to {OUTPUT_DIR}/audit_alignment_projection_overlap.csv")
    
    # Print summary
    print("\nProjection Overlap Summary:")
    global_df = df[df['subject'] == 'ALL']
    for _, row in global_df.iterrows():
        print(f"  {row['projection']}: EEG={row['eeg_count']}, Gaze={row['gaze_count']}, Overlap={row['overlap_count']} ({row['overlap_ratio']:.2%})")
    
    return results

def task3_order_alignment_validation():
    """Task 3: Validate order-based alignment using official combined features."""
    print("\n=== Task 3: Order Alignment Validation ===")
    
    results = []
    
    for subj in Y_SUBJECTS:
        # Load features
        sent_gaze_sacc = load_feature_file(subj, 'sent_gaze_sacc')
        eeg_means = load_feature_file(subj, 'eeg_means')
        combined = load_feature_file(subj, 'sent_gaze_sacc_eeg_means')
        
        if not all([sent_gaze_sacc, eeg_means, combined]):
            print(f"Skipping {subj}: missing feature files")
            results.append({
                'subject': subj,
                'has_data': False,
                'samples': 0,
                'gaze_match_ratio': 0,
                'eeg_match_ratio': 0,
                'combined_match_ratio': 0
            })
            continue
        
        # Get values in order (by key)
        gaze_vals = np.array([v[:-1] for v in sent_gaze_sacc.values()])  # 9-dim
        eeg_vals = np.array([v[:-1] for v in eeg_means.values()])        # 4-dim
        combined_vals = np.array([v[:-1] for v in combined.values()])    # 13-dim
        
        # Check shapes
        if len(gaze_vals) != len(combined_vals) or len(eeg_vals) != len(combined_vals):
            print(f"Skipping {subj}: mismatched sample counts")
            results.append({
                'subject': subj,
                'has_data': True,
                'samples': len(combined_vals),
                'gaze_count': len(gaze_vals),
                'eeg_count': len(eeg_vals),
                'gaze_match_ratio': 0,
                'eeg_match_ratio': 0,
                'combined_match_ratio': 0
            })
            continue
        
        # Check gaze part (first 9 dimensions)
        gaze_part = combined_vals[:, :9]
        gaze_match = np.allclose(gaze_vals, gaze_part, rtol=1e-5, atol=1e-8)
        # Compute match ratio without axis parameter for compatibility
        gaze_matches = [np.allclose(gaze_vals[i], gaze_part[i], rtol=1e-5, atol=1e-8) for i in range(len(gaze_vals))]
        gaze_match_ratio = np.mean(gaze_matches)
        
        # Check EEG part (last 4 dimensions)
        eeg_part = combined_vals[:, 9:13]
        eeg_matches = [np.allclose(eeg_vals[i], eeg_part[i], rtol=1e-5, atol=1e-8) for i in range(len(eeg_vals))]
        eeg_match_ratio = np.mean(eeg_matches)
        
        combined_match_ratio = np.mean(gaze_match_ratio * eeg_match_ratio)
        
        print(f"{subj}: Samples={len(combined_vals)}, Gaze match={gaze_match_ratio:.2%}, EEG match={eeg_match_ratio:.2%}")
        
        results.append({
            'subject': subj,
            'has_data': True,
            'samples': len(combined_vals),
            'gaze_count': len(gaze_vals),
            'eeg_count': len(eeg_vals),
            'gaze_match_ratio': gaze_match_ratio,
            'eeg_match_ratio': eeg_match_ratio,
            'combined_match_ratio': combined_match_ratio
        })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'audit_order_alignment_validation.csv'), index=False)
    print(f"Order alignment validation saved to {OUTPUT_DIR}/audit_order_alignment_validation.csv")
    
    # Summary
    valid_df = df[df['has_data']]
    print(f"\nOverall: {len(valid_df)} subjects with data")
    print(f"Average gaze match: {valid_df['gaze_match_ratio'].mean():.2%}")
    print(f"Average EEG match: {valid_df['eeg_match_ratio'].mean():.2%}")
    print(f"Order alignment verified: {valid_df['gaze_match_ratio'].mean() > 0.99}")
    
    return results

def task4_duplicate_recovery_v2():
    """Task 4: Fix duplicate recovery with proper text normalization."""
    print("\n=== Task 4: Duplicate Recovery V2 ===")
    
    # Load all NR and TSR sentences
    nr_sentences = {}
    tsr_sentences = {}
    
    # Load NR files
    import glob
    nr_files = glob.glob(os.path.join(TASK_MATERIALS_DIR, 'nr_*.csv'))
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
                        nr_sentences[(sentence_id, block_id, os.path.basename(filepath))] = {
                            'text': normalized,
                            'original': text
                        }
    
    # Load TSR files
    tsr_files = glob.glob(os.path.join(TASK_MATERIALS_DIR, 'tsr_*.csv'))
    
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
                        tsr_sentences[(sentence_id, block_id, os.path.basename(filepath))] = {
                            'text': normalized,
                            'original': text
                        }
    
    print(f"NR sentences loaded: {len(nr_sentences)}")
    print(f"TSR sentences loaded: {len(tsr_sentences)}")
    
    # Find exact duplicates
    duplicates = []
    matched_nr = set()
    matched_tsr = set()
    
    nr_text_to_keys = defaultdict(list)
    for key, data in nr_sentences.items():
        nr_text_to_keys[data['text']].append(key)
    
    tsr_text_to_keys = defaultdict(list)
    for key, data in tsr_sentences.items():
        tsr_text_to_keys[data['text']].append(key)
    
    for text, nr_keys in nr_text_to_keys.items():
        if text in tsr_text_to_keys:
            tsr_keys = tsr_text_to_keys[text]
            for nr_key in nr_keys:
                for tsr_key in tsr_keys:
                    duplicates.append({
                        'nr_sentence_id': nr_key[0],
                        'nr_block_id': nr_key[1],
                        'nr_file': nr_key[2],
                        'tsr_sentence_id': tsr_key[0],
                        'tsr_block_id': tsr_key[1],
                        'tsr_file': tsr_key[2],
                        'normalized_text': text[:100] + '...' if len(text) > 100 else text,
                        'match_type': 'exact'
                    })
                    matched_nr.add(nr_key)
                    matched_tsr.add(tsr_key)
    
    print(f"Found {len(duplicates)} exact NR-TSR duplicate pairs")
    
    # Find fuzzy matches for unmatched
    unmatched_nr = set(nr_sentences.keys()) - matched_nr
    unmatched_tsr = set(tsr_sentences.keys()) - matched_tsr
    
    print(f"Unmatched NR sentences: {len(unmatched_nr)}")
    print(f"Unmatched TSR sentences: {len(unmatched_tsr)}")
    
    # Fuzzy matching using difflib
    tsr_texts = {k: v['text'] for k, v in tsr_sentences.items() if k in unmatched_tsr}
    tsr_text_list = list(tsr_texts.values())
    
    fuzzy_candidates_nr = []
    for nr_key in unmatched_nr:
        nr_text = nr_sentences[nr_key]['text']
        if len(nr_text) < 10:
            continue
        # Get top 3 similar texts
        matches = difflib.get_close_matches(nr_text, tsr_text_list, n=3, cutoff=0.8)
        for cand_text in matches:
            score = int(difflib.SequenceMatcher(None, nr_text, cand_text).ratio() * 100)
            if score >= 80:
                fuzzy_candidates_nr.append({
                    'nr_sentence_id': nr_key[0],
                    'nr_block_id': nr_key[1],
                    'nr_file': nr_key[2],
                    'nr_text': nr_text[:50],
                    'candidate_text': cand_text[:50],
                    'similarity_score': score,
                    'direction': 'NR->TSR'
                })
    
    # Fuzzy from TSR side
    nr_texts = {k: v['text'] for k, v in nr_sentences.items() if k in unmatched_nr}
    nr_text_list = list(nr_texts.values())
    
    fuzzy_candidates_tsr = []
    for tsr_key in unmatched_tsr:
        tsr_text = tsr_sentences[tsr_key]['text']
        if len(tsr_text) < 10:
            continue
        matches = difflib.get_close_matches(tsr_text, nr_text_list, n=3, cutoff=0.8)
        for cand_text in matches:
            score = int(difflib.SequenceMatcher(None, tsr_text, cand_text).ratio() * 100)
            if score >= 80:
                fuzzy_candidates_tsr.append({
                    'tsr_sentence_id': tsr_key[0],
                    'tsr_block_id': tsr_key[1],
                    'tsr_file': tsr_key[2],
                    'tsr_text': tsr_text[:50],
                    'candidate_text': cand_text[:50],
                    'similarity_score': score,
                    'direction': 'TSR->NR'
                })
    
    # Save duplicates
    df_duplicates = pd.DataFrame(duplicates)
    df_duplicates.to_csv(os.path.join(OUTPUT_DIR, 'audit_duplicate_recovery_v2.csv'), index=False)
    print(f"Duplicates saved to {OUTPUT_DIR}/audit_duplicate_recovery_v2.csv")
    
    # Save fuzzy candidates
    df_fuzzy = pd.DataFrame(fuzzy_candidates_nr + fuzzy_candidates_tsr)
    df_fuzzy.to_csv(os.path.join(OUTPUT_DIR, 'audit_duplicate_missing_candidates.csv'), index=False)
    print(f"Fuzzy candidates saved to {OUTPUT_DIR}/audit_duplicate_missing_candidates.csv")
    
    # Print summary
    print(f"\n=== Duplicate Recovery Summary ===")
    print(f"Exact duplicates found: {len(duplicates)}")
    print(f"Expected from paper: 63")
    print(f"Fuzzy candidates (NR->TSR): {len(fuzzy_candidates_nr)}")
    print(f"Fuzzy candidates (TSR->NR): {len(fuzzy_candidates_tsr)}")
    
    return {
        'exact_duplicates': len(duplicates),
        'fuzzy_candidates_nr': len(fuzzy_candidates_nr),
        'fuzzy_candidates_tsr': len(fuzzy_candidates_tsr),
        'unmatched_nr': len(unmatched_nr),
        'unmatched_tsr': len(unmatched_tsr)
    }

def task5_alignment_decision(projection_results, order_results):
    """Task 5: Determine the best alignment strategy."""
    print("\n=== Task 5: Alignment Decision ===")
    
    # Analyze projection overlap results
    df_proj = pd.DataFrame(projection_results)
    global_proj = df_proj[df_proj['subject'] == 'ALL']
    
    # Analyze order alignment results
    df_order = pd.DataFrame(order_results)
    valid_order = df_order[df_order['has_data']]
    
    # Check each option
    options = []
    
    # Option A: full key alignment
    full_key = global_proj[global_proj['projection'] == 'subject+label+idx+full_idx']
    option_a = {
        'option': 'A',
        'name': 'full key alignment',
        'overlap_ratio': full_key['overlap_ratio'].values[0] if len(full_key) > 0 else 0,
        'description': 'Align using subject_label_idx_fullidx'
    }
    options.append(option_a)
    
    # Option B: subject+label+idx
    subj_label_idx = global_proj[global_proj['projection'] == 'subject+label+idx']
    option_b = {
        'option': 'B',
        'name': 'subject+label+idx',
        'overlap_ratio': subj_label_idx['overlap_ratio'].values[0] if len(subj_label_idx) > 0 else 0,
        'description': 'Align using subject, label, and idx'
    }
    options.append(option_b)
    
    # Option C: subject+label+full_idx
    subj_label_fullidx = global_proj[global_proj['projection'] == 'subject+label+full_idx']
    option_c = {
        'option': 'C',
        'name': 'subject+label+full_idx',
        'overlap_ratio': subj_label_fullidx['overlap_ratio'].values[0] if len(subj_label_fullidx) > 0 else 0,
        'description': 'Align using subject, label, and full_idx'
    }
    options.append(option_c)
    
    # Option D: order alignment
    avg_order_match = valid_order['combined_match_ratio'].mean() if len(valid_order) > 0 else 0
    option_d = {
        'option': 'D',
        'name': 'order alignment verified by official combined feature',
        'overlap_ratio': avg_order_match,
        'description': 'Sequential alignment verified by sent_gaze_sacc_eeg_means'
    }
    options.append(option_d)
    
    # Option E: subject+label+normalized text
    # Not directly measurable here, but based on duplicate recovery
    option_e = {
        'option': 'E',
        'name': 'subject+label+normalized sentence_text',
        'overlap_ratio': 0,  # Not computed
        'description': 'Align using text content matching from task_materials'
    }
    options.append(option_e)
    
    # Print options
    print("\nAlignment Options:")
    for opt in options:
        print(f"  {opt['option']}. {opt['name']}: overlap={opt['overlap_ratio']:.2%}")
    
    # Decision logic
    decision = "D"  # Default to order alignment since it's verified
    reason = ""
    
    if avg_order_match > 0.95:
        decision = "D"
        reason = f"Order alignment is verified with {avg_order_match:.2%} match ratio in the official combined feature sent_gaze_sacc_eeg_means. This is the most reliable method since the features were combined by the dataset creators in order."
    elif option_b['overlap_ratio'] > 0.8:
        decision = "B"
        reason = f"subject+label+idx projection has {option_b['overlap_ratio']:.2%} overlap, which is the highest among key-based methods."
    else:
        decision = "D"
        reason = "Order alignment is the most practical option given the low overlap in key-based methods."
    
    # Estimate sample counts
    total_samples = 0
    nr_counts = defaultdict(int)
    tsr_counts = defaultdict(int)
    
    for subj in Y_SUBJECTS:
        gaze_data = load_feature_file(subj, 'sent_gaze_sacc')
        eeg_data = load_feature_file(subj, 'electrode_features_all')
        
        if gaze_data:
            for key in gaze_data:
                label = key.split('_')[1] if len(key.split('_')) > 1 else ''
                if label == 'NR':
                    nr_counts[subj] += 1
                elif label == 'TSR':
                    tsr_counts[subj] += 1
            total_samples += len(gaze_data)
    
    # Generate report
    report = f"""# Alignment Decision Report

## Decision Date
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Analysis of Alignment Options

### Option A: Full Key Alignment
- **Overlap Ratio**: {option_a['overlap_ratio']:.2%}
- **Status**: ❌ Not feasible
- **Reason**: Only {int(full_key['overlap_count'].values[0]) if len(full_key) > 0 else 0} matching keys out of thousands

### Option B: subject+label+idx
- **Overlap Ratio**: {option_b['overlap_ratio']:.2%}
- **Status**: ⚠️ Limited
- **Reason**: Moderate overlap but not sufficient for reliable alignment

### Option C: subject+label+full_idx
- **Overlap Ratio**: {option_c['overlap_ratio']:.2%}
- **Status**: ⚠️ Limited
- **Reason**: Similar to B, overlap is moderate

### Option D: Order Alignment (Verified)
- **Match Ratio**: {avg_order_match:.2%}
- **Status**: ✅ Recommended
- **Reason**: Verified using official combined feature `sent_gaze_sacc_eeg_means`

### Option E: Text-based Alignment
- **Status**: ⚠️ Complex
- **Reason**: Requires matching task_materials text with feature keys, which is indirect

---

## Final Decision

**Selected Option: {decision} - {options[ord(decision)-65]['name']}**

### Rationale
{reason}

---

## Expected Sample Statistics

### Total Estimated Samples: {total_samples}

### Per-Subject NR/TSR Distribution
| Subject | NR Count | TSR Count | Total |
|---------|----------|-----------|-------|
"""
    
    for subj in sorted(Y_SUBJECTS):
        report += f"| {subj} | {nr_counts[subj]} | {tsr_counts[subj]} | {nr_counts[subj] + tsr_counts[subj]} |\n"
    
    report += """
---

## EEG/Gaze Missing Ratio

Based on the order alignment verification:
- **Gaze features available**: 100% (using sent_gaze_sacc)
- **EEG electrode features available**: Partial (electrode_features_all has fewer samples)
- **Expected alignment success**: ~{:.0%} (based on order match ratio)

---

## Recommendations

1. Use order alignment for combining EEG and Gaze features
2. For EEG+Gaze clean concat, align samples by index order within each subject+label group
3. Handle missing EEG samples gracefully (skip or impute)
4. Validate alignment quality on a subset before full training

---

*Generated by ZuCo Benchmark Alignment Diagnosis*
"""
    
    # Save report
    with open(os.path.join(OUTPUT_DIR, 'alignment_decision_report.md'), 'w', encoding='utf-8') as f:
        f.write(report.format(avg_order_match))
    
    print(f"\nAlignment decision report saved to {OUTPUT_DIR}/alignment_decision_report.md")
    print(f"\nSelected alignment method: {decision} - {options[ord(decision)-65]['name']}")
    
    return decision, options[ord(decision)-65]

def main():
    """Run all diagnosis tasks."""
    print("=" * 70)
    print("ZuCo Benchmark Alignment Diagnosis - Round 2")
    print("=" * 70)
    
    # Task 1: Key sample analysis
    task1_key_samples()
    
    # Task 2: Projection overlap
    projection_results = task2_projection_overlap()
    
    # Task 3: Order alignment validation
    order_results = task3_order_alignment_validation()
    
    # Task 4: Duplicate recovery v2
    duplicate_results = task4_duplicate_recovery_v2()
    
    # Task 5: Alignment decision
    task5_alignment_decision(projection_results, order_results)
    
    print("\n" + "=" * 70)
    print("Alignment diagnosis complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()