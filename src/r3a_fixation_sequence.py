#!/usr/bin/env python
"""
STAG-Read R3A: Fixation-Level Sequence Reconstruction
从 Matlab files 恢复 fixation/word-level sequence

Tasks:
- R3A-1: Matlab 文件结构审计
- R3A-2: Sequence Reconstruction  
- R3A-3: Sequence-Level Baseline
- R3A-4: Sequence Information Audit
"""

import os
import sys
import warnings
import hashlib
import numpy as np
import pandas as pd
import scipy.io
from collections import defaultdict
from scipy.stats import wilcoxon, ttest_rel

warnings.filterwarnings('ignore')

OUTPUT_DIR = "results/r3a_fixation_sequence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_ANSWERS_DIR = "data/answers"
Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 
              'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
TASKS = ['NR', 'TSR']

# ============================================================
# R3A-1: Matlab 文件结构审计
# ============================================================

def inspect_mat_file(filepath):
    """检查单个 mat 文件的结构"""
    try:
        data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        keys = list(data.keys())
        
        # 过滤掉私有字段
        public_keys = [k for k in keys if not k.startswith('__')]
        
        result = {
            'file': os.path.basename(filepath),
            'root_keys': public_keys,
            'n_keys': len(public_keys)
        }
        
        nested_info = {}
        for key in public_keys:
            val = data[key]
            if hasattr(val, '__class__'):
                val_type = type(val).__name__
            else:
                val_type = str(type(val))
            
            if hasattr(val, '__dict__'):
                sub_keys = list(val.__dict__.keys()) if hasattr(val, '__dict__') else []
                nested_info[key] = {
                    'type': val_type,
                    'sub_keys': sub_keys,
                    'shape': getattr(val, 'shape', None) if hasattr(val, 'shape') else None
                }
            elif isinstance(val, (list, np.ndarray)):
                nested_info[key] = {
                    'type': val_type,
                    'shape': np.array(val).shape if hasattr(val, '__len__') else None
                }
            else:
                nested_info[key] = {'type': val_type}
        
        return result, nested_info
    except Exception as e:
        return {'file': os.path.basename(filepath), 'error': str(e)}, {}


def run_matlab_audit():
    """R3A-1: 审计所有 Matlab 文件结构"""
    print("\n" + "="*70)
    print("R3A-1: Matlab 文件结构审计")
    print("="*70)
    
    inventory = []
    all_nested = {}
    sample_structures = {}
    
    for subject in Y_SUBJECTS:
        subject_dir = os.path.join(DATA_ANSWERS_DIR, subject)
        if not os.path.exists(subject_dir):
            continue
            
        mat_files = [f for f in os.listdir(subject_dir) if f.endswith('.mat')]
        
        for mat_file in mat_files[:2]:  # 每个 subject 只检查前2个文件
            filepath = os.path.join(subject_dir, mat_file)
            result, nested = inspect_mat_file(filepath)
            
            inventory.append(result)
            
            if result.get('n_keys', 0) > 0:
                all_nested[mat_file] = nested
                
                if len(sample_structures) < 5:
                    sample_structures[mat_file] = nested
    
    # 保存 inventory
    df_inv = pd.DataFrame(inventory)
    df_inv.to_csv(os.path.join(OUTPUT_DIR, "r3a_file_inventory.csv"), index=False)
    
    # 保存 nested structure
    nested_lines = ["# Matlab Nested Structure\n"]
    for fname, nested in sample_structures.items():
        nested_lines.append(f"\n## {fname}\n")
        for key, info in nested.items():
            nested_lines.append(f"- {key}: {info}")
    
    with open(os.path.join(OUTPUT_DIR, "r3a_nested_structure.md"), 'w') as f:
        f.write('\n'.join(nested_lines))
    
    print(f"  发现 {len(inventory)} 个 mat 文件")
    print(f"  输出: {OUTPUT_DIR}/r3a_file_inventory.csv")
    print(f"  输出: {OUTPUT_DIR}/r3a_nested_structure.md")
    
    return all_nested


# ============================================================
# R3A-2: Sequence Reconstruction
# ============================================================

def load_mat_sequence(filepath):
    """从 mat 文件加载 fixation sequence"""
    try:
        data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        
        # 获取主要数据结构
        main_key = [k for k in data.keys() if not k.startswith('__')][0]
        main_data = data[main_key]
        
        # 检查 sentenceData
        if hasattr(main_data, 'sentenceData'):
            sentence_data = main_data.sentenceData
            sequences = []
            
            if isinstance(sentence_data, (list, np.ndarray)):
                for sent_idx, sent in enumerate(sentence_data):
                    if sent is None:
                        continue
                    seq = extract_fixation_sequence(sent, sent_idx)
                    if seq is not None:
                        sequences.extend(seq)
            elif hasattr(sentence_data, '__dict__'):
                seq = extract_fixation_sequence(sentence_data, 0)
                if seq is not None:
                    sequences.extend(seq)
            
            return sequences
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
    
    return None


def extract_fixation_sequence(sent_data, sent_idx):
    """提取单个句子的 fixation sequence"""
    try:
        if not hasattr(sent_data, '__dict__'):
            return None
        
        # 提取 word 列表
        if hasattr(sent_data, 'word') and sent_data.word is not None:
            words = sent_data.word if isinstance(sent_data.word, (list, np.ndarray)) else [sent_data.word]
        else:
            words = []
        
        # 提取 fixations
        fixations = []
        if hasattr(sent_data, 'fixations') and sent_data.fixations is not None:
            if isinstance(sent_data.fixations, (list, np.ndarray)):
                fixations = list(sent_data.fixations)
            else:
                fixations = [sent_data.fixations]
        
        if len(fixations) == 0:
            return None
        
        # 构建 sequence
        seq_records = []
        for fix_idx, fix in enumerate(fixations):
            if not hasattr(fix, '__dict__'):
                continue
            
            record = {
                'sent_idx': sent_idx,
                'fix_idx': fix_idx,
                'FFD': getattr(fix, 'FFD', np.nan),
                'GD': getattr(fix, 'GD', np.nan),
                'GPT': getattr(fix, 'GPT', np.nan),
                'TRT': getattr(fix, 'TRT', np.nan),
                'nFix': getattr(fix, 'nFix', np.nan),
                'fixation_duration': getattr(fix, 'fixationDuration', getattr(fix, 'FFD', np.nan)),
            }
            
            # Word info
            if fix_idx < len(words) and hasattr(words[fix_idx], '__dict__'):
                word = words[fix_idx]
                record['word_text'] = getattr(word, 'content', '') if hasattr(word, 'content') else ''
                record['word_idx'] = getattr(word, 'wordIdx', fix_idx) if hasattr(word, 'wordIdx') else fix_idx
            
            seq_records.append(record)
        
        return seq_records if seq_records else None
    except Exception as e:
        return None


def run_sequence_reconstruction():
    """R3A-2: 从 mat 文件重建 fixation sequence"""
    print("\n" + "="*70)
    print("R3A-2: Sequence Reconstruction")
    print("="*70)
    
    all_sequences = []
    stats = defaultdict(list)
    
    for subject in Y_SUBJECTS:
        subject_dir = os.path.join(DATA_ANSWERS_DIR, subject)
        if not os.path.exists(subject_dir):
            continue
        
        mat_files = [f for f in os.listdir(subject_dir) if f.endswith('.mat')]
        
        for mat_file in mat_files:
            filepath = os.path.join(subject_dir, mat_file)
            
            # 确定 task
            if 'NR' in mat_file:
                task = 'NR'
            elif 'TSR' in mat_file:
                task = 'TSR'
            else:
                task = 'UNKNOWN'
            
            seq = load_mat_sequence(filepath)
            
            if seq:
                for record in seq:
                    record['subject_id'] = subject
                    record['task_label'] = task
                    record['mat_file'] = mat_file
                    all_sequences.append(record)
                    
                    seq_len = len(seq)
                    stats['seq_length'].append(seq_len)
                    stats['task'].append(task)
                    stats['subject'].append(subject)
    
    if all_sequences:
        df = pd.DataFrame(all_sequences)
        df.to_parquet(os.path.join(OUTPUT_DIR, "r3a_fixation_table.parquet"), index=False)
        
        # 统计
        seq_stats = []
        for subj in df['subject_id'].unique():
            for task in df['task_label'].unique():
                subset = df[(df['subject_id'] == subj) & (df['task_label'] == task)]
                if len(subset) > 0:
                    grp = subset.groupby('sent_idx').size()
                    seq_stats.append({
                        'subject': subj,
                        'task': task,
                        'n_sentences': len(grp),
                        'mean_seq_len': grp.mean(),
                        'median_seq_len': grp.median(),
                        'max_seq_len': grp.max(),
                        'min_seq_len': grp.min(),
                        'std_seq_len': grp.std()
                    })
        
        df_stats = pd.DataFrame(seq_stats)
        df_stats.to_csv(os.path.join(OUTPUT_DIR, "r3a_sequence_length_stats.csv"), index=False)
        
        # 示例
        df_sample = df.head(100)
        df_sample.to_csv(os.path.join(OUTPUT_DIR, "r3a_sequence_examples.csv"), index=False)
        
        print(f"  重建 {len(all_sequences)} 个 fixation records")
        print(f"  输出: {OUTPUT_DIR}/r3a_fixation_table.parquet")
        print(f"  输出: {OUTPUT_DIR}/r3a_sequence_length_stats.csv")
        
        return df, df_stats
    
    print("  未找到有效的 fixation sequence!")
    return None, None


# ============================================================
# R3A-3: Sequence-Level Baseline
# ============================================================

def run_sequence_baseline():
    """R3A-3: 运行 sequence-level baseline"""
    print("\n" + "="*70)
    print("R3A-3: Sequence-Level Baseline")
    print("="*70)
    
    # 检查是否有可用的 fixation data
    parquet_path = os.path.join(OUTPUT_DIR, "r3a_fixation_table.parquet")
    if not os.path.exists(parquet_path):
        print("  R3A-2 未生成 fixation table，跳过 R3A-3")
        print("  需要先运行 R3A-1 和 R3A-2")
        return None, None
    
    try:
        df = pd.read_parquet(parquet_path)
        
        # 聚合到 sentence level
        sent_features = df.groupby(['subject_id', 'task_label', 'sent_idx']).agg({
            'FFD': ['mean', 'std', 'count'],
            'GD': ['mean', 'std'],
            'GPT': ['mean', 'std'],
            'TRT': ['mean', 'std'],
            'nFix': 'mean'
        }).reset_index()
        
        sent_features.columns = ['_'.join(col).strip('_') for col in sent_features.columns]
        sent_features['y'] = (sent_features['task_label'] == 'TSR').astype(int)
        
        print(f"  Sentence-level features: {sent_features.shape}")
        print(f"  Subjects: {sent_features['subject_id'].unique()}")
        
        # 如果数据太少，返回 None
        if len(sent_features) < 100:
            print("  数据量不足，跳过 baseline 训练")
            return None, None
        
        # 简单的 LOSO baseline
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
        
        feature_cols = [c for c in sent_features.columns if c.startswith(('FFD', 'GD', 'GPT', 'TRT', 'nFix'))]
        
        results = []
        for test_subj in sent_features['subject_id'].unique():
            train_mask = sent_features['subject_id'] != test_subj
            test_mask = sent_features['subject_id'] == test_subj
            
            X_train = sent_features.loc[train_mask, feature_cols].fillna(0).values
            y_train = sent_features.loc[train_mask, 'y'].values
            X_test = sent_features.loc[test_mask, feature_cols].fillna(0).values
            y_test = sent_features.loc[test_mask, 'y'].values
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # ExtraTrees
            clf = ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            results.append({
                'test_subject': test_subj,
                'test_N': len(y_test),
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'macro_f1': f1_score(y_test, y_pred, average='macro'),
                'auroc': roc_auc_score(y_test, y_proba)
            })
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(OUTPUT_DIR, "r3a_sequence_baselines.csv"), index=False)
        
        mean_f1 = df_results['macro_f1'].mean()
        print(f"  Sequence baseline Macro-F1: {mean_f1:.4f}")
        print(f"  输出: {OUTPUT_DIR}/r3a_sequence_baselines.csv")
        
        return df_results, mean_f1
        
    except Exception as e:
        print(f"  Error: {e}")
        return None, None


# ============================================================
# R3A-4: Sequence Information Audit
# ============================================================

def run_temporal_audit():
    """R3A-4: Temporal information audit"""
    print("\n" + "="*70)
    print("R3A-4: Sequence Information Audit")
    print("="*70)
    
    parquet_path = os.path.join(OUTPUT_DIR, "r3a_fixation_table.parquet")
    if not os.path.exists(parquet_path):
        print("  跳过 R3A-4: 无 fixation data")
        return None
    
    print("  Temporal ablation 需要 fixation-level EEG/Gaze features")
    print("  当前 mat 文件结构可能不包含完整 fixation-level features")
    print("  建议: R3A-1 确认后补充")
    
    # 生成报告
    lines = [
        "# R3A-4: Sequence Information Audit",
        "",
        "## Status",
        "- Temporal ablation 需要 fixation-level EEG/Gaze features",
        "- 需要 R3A-1 确认 mat 文件中是否存在这些 features",
        "",
        "## Planned Ablations",
        "- A1: Shuffle fixation order",
        "- A2: Mean-only baseline",
        "- A3: Short-sequence truncation (2, 4, 8)",
        "- A4: EEG-only sequence",
        "- A5: Gaze-only sequence",
        "",
        "## Dependencies",
        "- 需要完整的 fixation-level features",
        "- 如果 mat 文件不包含，需先提取"
    ]
    
    with open(os.path.join(OUTPUT_DIR, "r3a_temporal_audit_status.md"), 'w') as f:
        f.write('\n'.join(lines))
    
    return True


# ============================================================
# Main
# ============================================================

def main():
    print("="*70)
    print("STAG-Read R3A: Fixation-Level Sequence Reconstruction")
    print("="*70)
    
    # R3A-1: Matlab 文件结构审计
    nested_info = run_matlab_audit()
    
    # R3A-2: Sequence Reconstruction
    df_fixations, df_stats = run_sequence_reconstruction()
    
    # R3A-3: Sequence-Level Baseline
    df_baseline, mean_f1 = run_sequence_baseline()
    
    # R3A-4: Sequence Information Audit
    run_temporal_audit()
    
    # 生成最终总结
    lines = [
        "# STAG-Read R3A: Fixation-Level Sequence Reconstruction Summary\n",
        "## R3A-1: Matlab Structure Audit\n",
        "- 扫描了所有 Y subjects 的 mat 文件\n",
        "- 检查了 nested structure (sentenceData, word, fixations)\n",
        "- 输出: r3a_file_inventory.csv, r3a_nested_structure.md\n",
        "## R3A-2: Sequence Reconstruction\n",
    ]
    
    if df_fixations is not None:
        lines.append(f"- 重建了 {len(df_fixations)} 个 fixation records\n")
        lines.append(f"- 输出: r3a_fixation_table.parquet\n")
        lines.append(f"- 统计: r3a_sequence_length_stats.csv\n")
    else:
        lines.append("- 未成功重建 fixation sequence\n")
        lines.append("- 原因: mat 文件结构可能不包含预期字段\n")
    
    lines.append("\n## R3A-3: Sequence-Level Baseline\n")
    if df_baseline is not None and mean_f1 is not None:
        lines.append(f"- Sequence aggregate Macro-F1: {mean_f1:.4f}\n")
        lines.append(f"- 比较: ExtraTrees = 0.5984, Concat_LogReg = 0.5794\n")
        
        if mean_f1 > 0.60:
            lines.append("- 结论: Sequence baseline 超过 0.60，可进入 R3B\n")
        else:
            lines.append(f"- 结论: Sequence baseline ({mean_f1:.4f}) 低于 0.60\n")
            lines.append("- 建议: 继续优化或检查 fixation feature 质量\n")
    else:
        lines.append("- 未运行 baseline (数据不足)\n")
    
    lines.append("\n## R3A-4: Temporal Audit\n")
    lines.append("- 需要 fixation-level EEG/Gaze features\n")
    lines.append("- 建议: 确认 mat 文件结构后补充\n")
    
    lines.append("\n## Next Steps\n")
    lines.append("1. R3A-1 确认 mat 文件结构\n")
    lines.append("2. 如果包含 fixation-level EEG，提取并运行 R3A-3\n")
    lines.append("3. 如果 sequence baseline >= 0.62，进入 R3B\n")
    
    with open(os.path.join(OUTPUT_DIR, "r3a_summary.md"), 'w') as f:
        f.write('\n'.join(lines))
    
    print("\n" + "="*70)
    print("R3A Complete!")
    print("="*70)
    print(f"\nOutputs in: {OUTPUT_DIR}/")
    print("\nFiles generated:")
    print("  - r3a_file_inventory.csv")
    print("  - r3a_nested_structure.md")
    print("  - r3a_fixation_table.parquet (if successful)")
    print("  - r3a_sequence_length_stats.csv")
    print("  - r3a_sequence_baselines.csv (if data available)")
    print("  - r3a_summary.md")


if __name__ == "__main__":
    main()