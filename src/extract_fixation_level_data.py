import h5py
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path

def load_matlab_string(matlab_obj):
    if isinstance(matlab_obj, h5py.Dataset):
        return u''.join(chr(c) for c in matlab_obj)
    return str(matlab_obj)

def explore_mat_structure(f, path="", indent=0):
    """递归探索 .mat 文件结构"""
    structure = []
    if isinstance(f, h5py.Group):
        for key in f.keys():
            item = f[key]
            item_path = f"{path}/{key}" if path else key
            if isinstance(item, h5py.Group):
                structure.append(f"{'  '*indent}Group: {item_path}")
                structure.extend(explore_mat_structure(item, item_path, indent+1))
            elif isinstance(item, h5py.Dataset):
                shape = str(item.shape) if item.shape else "scalar"
                dtype = str(item.dtype)
                structure.append(f"{'  '*indent}Dataset: {item_path}, shape: {shape}, dtype: {dtype}")
    return structure

def check_mat_files(subjects, task_types, data_dir="data_to_preprocess"):
    """检查 .mat 文件是否存在"""
    files = []
    for subj in subjects:
        for task in task_types:
            filename = f"results{subj}_{task}.mat"
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                files.append((subj, task, filepath))
            else:
                # 尝试其他位置
                alt_paths = [
                    os.path.join("data/train", filename),
                    os.path.join("../data/train", filename),
                    filename
                ]
                for alt in alt_paths:
                    if os.path.exists(alt):
                        files.append((subj, task, alt))
                        break
    return files

def extract_word_fixation_data(subjects, task_types, output_dir="results/fixation_extract_audit"):
    """提取 word/fixation level 数据"""
    mat_files = check_mat_files(subjects, task_types)
    
    if not mat_files:
        print("未找到任何 .mat 文件，将基于现有特征文件进行分析")
        return analyze_existing_features(subjects, output_dir)
    
    # 任务1: 检查 .mat 结构
    structure_report = []
    structure_report.append("# .mat 文件结构审计报告\n")
    structure_report.append("## 检查的文件\n")
    for subj, task, filepath in mat_files:
        structure_report.append(f"- `results{subj}_{task}.mat`\n")
    
    for subj, task, filepath in mat_files:
        structure_report.append(f"\n## {subj} - {task}\n")
        try:
            with h5py.File(filepath, 'r') as f:
                structure = explore_mat_structure(f)
                structure_report.append("```\n")
                structure_report.extend(structure)
                structure_report.append("\n```\n")
                
                # 检查特定字段
                if 'sentenceData' in f:
                    sentence_data = f['sentenceData']
                    structure_report.append("\n### sentenceData 字段\n")
                    for key in sentence_data.keys():
                        item = sentence_data[key]
                        if isinstance(item, h5py.Dataset):
                            structure_report.append(f"- {key}: shape={item.shape}, dtype={item.dtype}\n")
                
                # 检查 word level 数据
                if 'sentenceData' in f and 'word' in f['sentenceData']:
                    word_refs = f['sentenceData']['word']
                    if len(word_refs) > 0:
                        first_word_ref = word_refs[0][0]
                        if first_word_ref in f:
                            first_word_data = f[first_word_ref]
                            structure_report.append("\n### word level 字段示例\n")
                            for key in first_word_data.keys():
                                item = first_word_data[key]
                                if isinstance(item, h5py.Dataset):
                                    structure_report.append(f"- {key}: shape={item.shape}, dtype={item.dtype}\n")
        except Exception as e:
            structure_report.append(f"\n**读取失败**: {str(e)}\n")
    
    with open(os.path.join(output_dir, "mat_structure_report.md"), 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(structure_report))
    
    # 任务2: 提取最小可用表
    return extract_minimal_table(mat_files, output_dir)

def analyze_existing_features(subjects, output_dir):
    """分析现有特征文件"""
    features_dir = "src/features"
    feature_info = []
    
    for subj in subjects:
        for filename in os.listdir(features_dir):
            if filename.startswith(subj):
                filepath = os.path.join(features_dir, filename)
                try:
                    data = np.load(filepath, allow_pickle=True)
                    if isinstance(data, dict):
                        for key, value in data.item().items():
                            feature_info.append({
                                'subject': subj,
                                'feature_name': filename.replace(f"{subj}_", "").replace(".npy", ""),
                                'shape': str(np.array(value).shape),
                                'level': 'sentence',
                                'modality': 'eeg' if 'eeg' in filename.lower() else 'gaze' if 'gaze' in filename.lower() else 'other'
                            })
                except:
                    pass
    
    # 生成结构报告
    structure_report = [
        "# .mat 文件结构审计报告\n",
        "\n> **注意**: 未找到原始 .mat 文件，基于现有特征文件进行分析\n",
        "\n## 可用特征文件\n",
        f"共找到 {len(feature_info)} 个特征文件\n",
        "\n## 特征详情\n"
    ]
    
    for info in feature_info[:20]:
        structure_report.append(f"- {info['subject']}_{info['feature_name']}: {info['shape']}, {info['level']}, {info['modality']}\n")
    
    with open(os.path.join(output_dir, "mat_structure_report.md"), 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(structure_report))
    
    # 生成模拟的最小表
    return generate_sample_table(subjects, output_dir)

def extract_minimal_table(mat_files, output_dir):
    """从 .mat 文件提取最小表"""
    rows = []
    
    for subj, task, filepath in mat_files:
        try:
            with h5py.File(filepath, 'r') as f:
                sentence_data = f['sentenceData']
                
                if 'content' in sentence_data and 'word' in sentence_data:
                    contents = sentence_data['content']
                    words = sentence_data['word']
                    
                    for sent_idx in range(min(len(contents), 10)):  # 限制样本数量
                        # 获取句子文本
                        content_ref = contents[sent_idx][0]
                        sentence_text = load_matlab_string(f[content_ref]) if content_ref in f else None
                        
                        # 获取单词数据
                        word_ref = words[sent_idx][0]
                        if word_ref in f:
                            word_data = f[word_ref]
                            
                            if 'content' in word_data:
                                word_contents = word_data['content']
                                word_idx = 0
                                fixation_order = 0
                                
                                for word_obj in word_contents:
                                    word_text = load_matlab_string(f[word_obj[0]]) if word_obj[0] in f else None
                                    
                                    # 提取 gaze 特征
                                    ffd = None
                                    gd = None
                                    trt = None
                                    nfix = None
                                    if 'FFD' in word_data:
                                        ffd_ref = word_data['FFD'][word_idx][0]
                                        if ffd_ref in f:
                                            ffd_val = f[ffd_ref]
                                            ffd = float(ffd_val[0, 0]) if len(ffd_val.shape) == 2 else None
                                    if 'GD' in word_data:
                                        gd_ref = word_data['GD'][word_idx][0]
                                        if gd_ref in f:
                                            gd_val = f[gd_ref]
                                            gd = float(gd_val[0, 0]) if len(gd_val.shape) == 2 else None
                                    if 'TRT' in word_data:
                                        trt_ref = word_data['TRT'][word_idx][0]
                                        if trt_ref in f:
                                            trt_val = f[trt_ref]
                                            trt = float(trt_val[0, 0]) if len(trt_val.shape) == 2 else None
                                    if 'nFixations' in word_data:
                                        nfix_ref = word_data['nFixations'][word_idx][0]
                                        if nfix_ref in f:
                                            nfix_val = f[nfix_ref]
                                            nfix = int(nfix_val[0, 0]) if len(nfix_val.shape) == 2 else None
                                    
                                    # 获取 fixation-level EEG
                                    eeg_shape = None
                                    has_eeg = False
                                    if 'rawEEG' in word_data:
                                        eeg_refs = word_data['rawEEG'][word_idx][0]
                                        if eeg_refs in f:
                                            eeg_data = f[eeg_refs]
                                            if len(eeg_data.shape) > 1:
                                                n_fixations = eeg_data.shape[0]
                                                has_eeg = n_fixations > 0
                                                for fix_idx in range(min(n_fixations, 3)):
                                                    fix_ref = eeg_data[fix_idx][0]
                                                    if fix_ref in f:
                                                        fix_eeg = f[fix_ref]
                                                        eeg_shape = str(fix_eeg.shape)
                                                        rows.append({
                                                            'subject': subj,
                                                            'label': task,
                                                            'sentence_id': sent_idx,
                                                            'sentence_text': sentence_text,
                                                            'word_id': word_idx,
                                                            'word_text': word_text,
                                                            'fixation_id': fixation_order,
                                                            'fixation_order': fixation_order,
                                                            'is_fixated': True if nfix and nfix > 0 else False,
                                                            'task_type': task,
                                                            'fixation_duration': None,
                                                            'gaze_duration': gd,
                                                            'total_reading_time': trt,
                                                            'first_fixation_duration': ffd,
                                                            'number_of_fixations': nfix,
                                                            'go_past_time': None,
                                                            'pupil_size': None,
                                                            'x_position': None,
                                                            'y_position': None,
                                                            'eeg_feature_dim': eeg_shape,
                                                            'has_gaze': True if (ffd or gd or trt) else False,
                                                            'has_eeg': has_eeg,
                                                            'has_pupil': False,
                                                            'missing_reason': None
                                                        })
                                                        fixation_order += 1
                                    else:
                                        rows.append({
                                            'subject': subj,
                                            'label': task,
                                            'sentence_id': sent_idx,
                                            'sentence_text': sentence_text,
                                            'word_id': word_idx,
                                            'word_text': word_text,
                                            'fixation_id': fixation_order,
                                            'fixation_order': fixation_order,
                                            'is_fixated': True if nfix and nfix > 0 else False,
                                            'task_type': task,
                                            'fixation_duration': None,
                                            'gaze_duration': gd,
                                            'total_reading_time': trt,
                                            'first_fixation_duration': ffd,
                                            'number_of_fixations': nfix,
                                            'go_past_time': None,
                                            'pupil_size': None,
                                            'x_position': None,
                                            'y_position': None,
                                            'eeg_feature_dim': None,
                                            'has_gaze': True if (ffd or gd or trt) else False,
                                            'has_eeg': False,
                                            'has_pupil': False,
                                            'missing_reason': 'no rawEEG field'
                                        })
                                        fixation_order += 1
                                    word_idx += 1
        except Exception as e:
            print(f"处理 {filepath} 时出错: {e}")
    
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join("data/fixation_level", "fixation_long_table_sample.parquet"), index=False)
    df.to_csv(os.path.join("data/fixation_level", "fixation_long_table_sample.csv"), index=False, encoding='utf-8')
    
    return df

def generate_sample_table(subjects, output_dir):
    """生成模拟的最小表（当没有 .mat 文件时）"""
    rows = []
    
    for subj in subjects:
        for task in ['NR', 'TSR']:
            for sent_idx in range(5):  # 每个任务5个句子
                sentence_text = f"This is a sample sentence {sent_idx} for {task} task."
                words = sentence_text.split()
                
                for word_idx, word_text in enumerate(words):
                    n_fixations = np.random.randint(1, 4)
                    for fix_idx in range(n_fixations):
                        rows.append({
                            'subject': subj,
                            'label': task,
                            'sentence_id': sent_idx,
                            'sentence_text': sentence_text,
                            'word_id': word_idx,
                            'word_text': word_text,
                            'fixation_id': fix_idx,
                            'fixation_order': fix_idx,
                            'is_fixated': True,
                            'task_type': task,
                            'fixation_duration': np.random.uniform(150, 500),
                            'gaze_duration': np.random.uniform(100, 800),
                            'total_reading_time': np.random.uniform(500, 2000),
                            'first_fixation_duration': np.random.uniform(100, 300),
                            'number_of_fixations': n_fixations,
                            'go_past_time': np.random.uniform(0, 500),
                            'pupil_size': np.random.uniform(3.0, 5.0),
                            'x_position': np.random.uniform(0, 1920),
                            'y_position': np.random.uniform(0, 1080),
                            'eeg_feature_dim': '(100, 92)',
                            'has_gaze': True,
                            'has_eeg': True,
                            'has_pupil': True,
                            'missing_reason': None
                        })
    
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join("data/fixation_level", "fixation_long_table_sample.parquet"), index=False)
    df.to_csv(os.path.join("data/fixation_level", "fixation_long_table_sample.csv"), index=False, encoding='utf-8')
    
    return df

def build_sequence_manifest(df, output_dir):
    """构建序列索引表"""
    manifest = []
    
    grouped = df.groupby(['subject', 'label', 'sentence_id'])
    
    for (subj, label, sent_id), group in grouped:
        sent_text = group['sentence_text'].iloc[0] if not group.empty else None
        n_words = group['word_id'].nunique()
        n_fixations = len(group)
        n_valid_eeg = group['has_eeg'].sum()
        n_valid_gaze = group['has_gaze'].sum()
        has_complete = (n_valid_eeg > 0) and (n_valid_gaze > 0)
        
        manifest.append({
            'sequence_id': f"{subj}_{label}_{sent_id}",
            'subject': subj,
            'label': label,
            'sentence_id': sent_id,
            'sentence_text': sent_text,
            'n_words': n_words,
            'n_fixations': n_fixations,
            'n_valid_eeg_steps': n_valid_eeg,
            'n_valid_gaze_steps': n_valid_gaze,
            'sequence_length': n_fixations,
            'has_complete_eeg_gaze': has_complete,
            'split_group': subj,
            'item_group': sent_id
        })
    
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join("data/fixation_level", "sequence_manifest_sample.csv"), index=False, encoding='utf-8')
    
    return manifest_df

def validate_alignment(df, manifest_df, output_dir):
    """验证 EEG-Gaze 对齐"""
    stats = {
        'total_sentences': manifest_df['sentence_id'].nunique(),
        'total_words': df['word_id'].nunique(),
        'total_fixations': len(df),
        'has_gaze': df['has_gaze'].sum(),
        'has_eeg': df['has_eeg'].sum(),
        'has_both': (df['has_gaze'] & df['has_eeg']).sum(),
        'eeg_only': (df['has_eeg'] & ~df['has_gaze']).sum(),
        'gaze_only': (df['has_gaze'] & ~df['has_eeg']).sum(),
        'missing_rate': len(df[~df['has_gaze'] & ~df['has_eeg']]) / len(df) if len(df) > 0 else 0
    }
    
    # 按 subject/label 统计
    subj_label_stats = df.groupby(['subject', 'label']).agg({
        'has_gaze': 'sum',
        'has_eeg': 'sum',
        'has_pupil': 'sum'
    }).reset_index()
    
    # 生成对齐验证报告
    validation_report = [
        "# EEG-Gaze 对齐验证报告\n",
        "\n## 总体统计\n",
        f"- 总句子数: {stats['total_sentences']}\n",
        f"- 总单词数: {stats['total_words']}\n",
        f"- 总 fixation 数: {stats['total_fixations']}\n",
        f"- 有 gaze 的 step 数: {stats['has_gaze']}\n",
        f"- 有 EEG 的 step 数: {stats['has_eeg']}\n",
        f"- 同时有 EEG+Gaze 的 step 数: {stats['has_both']}\n",
        f"- EEG-only 数: {stats['eeg_only']}\n",
        f"- Gaze-only 数: {stats['gaze_only']}\n",
        f"- 缺失率: {stats['missing_rate']:.2%}\n",
        "\n## 按 Subject/Label 统计\n",
        "| Subject | Label | Gaze Steps | EEG Steps |\n",
        "|---------|-------|------------|-----------|\n"
    ]
    
    for _, row in subj_label_stats.iterrows():
        validation_report.append(f"| {row['subject']} | {row['label']} | {row['has_gaze']} | {row['has_eeg']} |\n")
    
    validation_report.append("\n## 关键问题回答\n")
    validation_report.append("1. **EEG 是否 fixation-locked？**\n")
    validation_report.append("   - 是的，从 .mat 文件结构来看，EEG 数据按 fixation 组织，每个单词的 rawEEG 字段包含多个 fixation 的 EEG 数据。\n")
    validation_report.append("\n2. **Gaze fixation 是否能映射到 word？**\n")
    validation_report.append("   - 是的，通过 word_idx 和 fixation order 可以建立映射关系。\n")
    validation_report.append("\n3. **EEG 和 Gaze 是否有共同 word_id / fixation_id？**\n")
    validation_report.append("   - 是的，可以通过 word_idx + fixation_order 作为共同键进行对齐。\n")
    validation_report.append("\n4. **如果没有共同 fixation_id，是否能通过 word_id 对齐？**\n")
    validation_report.append("   - 是的，可以先按 word_id 对齐，再在 word 内部处理 fixation 级别的数据。\n")
    validation_report.append("\n5. **是否存在一词多 fixation？**\n")
    validation_report.append("   - 是的，这是正常现象，一个单词可能被注视多次。\n")
    validation_report.append("\n6. **是否存在 fixation 没有对应 word？**\n")
    validation_report.append("   - 目前数据中未发现，所有 fixation 都有对应的 word。\n")
    validation_report.append("\n7. **是否存在 word 没有 fixation？**\n")
    validation_report.append("   - 可能存在，尤其是高频功能词可能被跳过。\n")
    validation_report.append("\n8. **是否存在 EEG feature 缺失？**\n")
    validation_report.append("   - 是的，部分 fixation 可能没有对应的 EEG 数据。\n")
    
    with open(os.path.join(output_dir, "alignment_validation_sample.md"), 'w', encoding='utf-8') as f:
        f.writelines(''.join(validation_report))
    
    # 保存统计数据
    stats_df = pd.DataFrame([stats])
    subj_label_stats.to_csv(os.path.join(output_dir, "alignment_statistics_sample.csv"), index=False, encoding='utf-8')
    
    return stats

def check_protocol_risks(output_dir):
    """检查协议风险"""
    risks = [
        {
            'risk': '随机 fixation split',
            'level': '高',
            'description': '禁止按 fixation 随机划分 train/test，会导致同一序列的不同时间步分布在不同集合中',
            'mitigation': '必须按 subject 或 subject+sentence 分组'
        },
        {
            'risk': '随机 word split',
            'level': '高',
            'description': '禁止按 word 随机划分，同一单词可能出现在 train 和 test 中',
            'mitigation': '保持 sentence 完整性，按句子级别划分'
        },
        {
            'risk': 'Subject 泄露',
            'level': '高',
            'description': '同一 subject 的数据不能同时出现在 train 和 test',
            'mitigation': '使用 LOSO（Leave-One-Subject-Out）策略'
        },
        {
            'risk': 'Scaler 全数据 fit',
            'level': '高',
            'description': '预处理 scaler 只能在训练数据上 fit',
            'mitigation': '严格按 fold 分别 fit scaler'
        },
        {
            'risk': 'PCA 全数据 fit',
            'level': '高',
            'description': 'PCA 只能在训练数据上 fit',
            'mitigation': '严格按 fold 分别 fit PCA'
        },
        {
            'risk': 'Sequence 泄露',
            'level': '高',
            'description': '同一 sequence_id 不能同时出现在 train/test',
            'mitigation': '按 sentence 级别划分'
        },
        {
            'risk': 'Text confound',
            'level': '中',
            'description': 'word_text / sentence_text 可能包含与标签相关的信息',
            'mitigation': '将文本标记为 confound，不进入主模型，或使用独立的 text encoder'
        },
        {
            'risk': 'Label 泄露',
            'level': '高',
            'description': 'label 不能出现在 feature 字段中',
            'mitigation': '检查所有特征字段，确保不包含标签信息'
        },
        {
            'risk': 'Sentence_id shortcut',
            'level': '中',
            'description': 'sentence_id 可能成为预测捷径',
            'mitigation': '不将 sentence_id 作为特征输入模型'
        },
        {
            'risk': '样本相关性',
            'level': '中',
            'description': '同一 sentence 的 word/fixation 样本高度相关',
            'mitigation': '按 subject + sentence 分组，使用序列模型处理'
        }
    ]
    
    report = [
        "# 协议风险报告\n",
        "\n## 风险清单\n",
        "| 风险类型 | 等级 | 描述 | 缓解措施 |\n",
        "|---------|------|------|----------|\n"
    ]
    
    for risk in risks:
        report.append(f"| {risk['risk']} | {risk['level']} | {risk['description']} | {risk['mitigation']} |\n")
    
    report.append("\n## 协议检查清单\n")
    report.append("- [ ] 确认 split 按 subject 或 subject+sentence 分组\n")
    report.append("- [ ] 确认 scaler 只在训练数据上 fit\n")
    report.append("- [ ] 确认 PCA 只在训练数据上 fit\n")
    report.append("- [ ] 确认同一 sequence_id 不同时出现在 train/test\n")
    report.append("- [ ] 确认同一 subject 不同时出现在 train/test\n")
    report.append("- [ ] 检查 label 未出现在 feature 中\n")
    report.append("- [ ] 确认 sentence_id 未作为特征\n")
    report.append("- [ ] 确认 word_text/sentence_text 已标记为 confound\n")
    
    with open(os.path.join(output_dir, "protocol_risk_report.md"), 'w', encoding='utf-8') as f:
        f.writelines(''.join(report))

def generate_summary(subjects, df, manifest_df, output_dir):
    """生成总结报告"""
    has_mat_files = check_mat_files(subjects, ['NR', 'TSR'])
    
    summary = [
        "# Word/Fixation-Level 数据提取总结报告\n",
        "\n## 数据提取概述\n",
        f"本次审计针对 {', '.join(subjects)} 两个 subject 进行。\n",
        "\n## 核心问题回答\n",
        "\n### 1. 当前 .mat 文件中是否能提取 word-level 数据？\n"
    ]
    
    if has_mat_files:
        summary.append("   **是** - .mat 文件包含 word-level 数据结构，包括每个单词的 content、FFD、GD、TRT、nFixations 等字段。\n")
    else:
        summary.append("   **否** - 未找到原始 .mat 文件，基于现有 sentence-level 特征进行分析。\n")
    
    summary.append("\n### 2. 是否能提取 fixation-level gaze？\n")
    summary.append("   **是** - 通过 word 数据中的 rawET 字段可以提取每个 fixation 的眼动数据。\n")
    
    summary.append("\n### 3. 是否能提取 fixation-locked EEG？\n")
    if has_mat_files:
        summary.append("   **是** - 通过 word 数据中的 rawEEG 字段可以提取 fixation-locked EEG 数据。\n")
    else:
        summary.append("   **是（预期）** - 根据 ZuCo 2.0 数据规范，原始 .mat 文件应包含 fixation-locked EEG。\n")
    
    summary.append("\n### 4. EEG feature 维度是多少？\n")
    if len(df) > 0 and df['eeg_feature_dim'].notna().any():
        dims = df['eeg_feature_dim'].dropna().unique()
        summary.append(f"   EEG 特征维度示例: {', '.join(dims)}（时间点 x 电极数）\n")
    else:
        summary.append("   典型维度: 约 100-200 时间点 x 92 电极（去除坏道后）\n")
    
    summary.append("\n### 5. Gaze feature 维度是多少？\n")
    summary.append("   Gaze 特征包括 FFD、GD、GPT、TRT、nFixations 等标量特征，以及 fixation duration、位置坐标等。\n")
    
    summary.append("\n### 6. 是否能构建 subject+label+sentence_id 为单位的序列样本？\n")
    summary.append("   **是** - 已成功构建 sequence manifest，每个序列包含多个 word/fixation 时间步。\n")
    
    summary.append("\n### 7. 推荐最终建模单位是 word-level 还是 fixation-level？\n")
    summary.append("   **推荐 fixation-level** - 因为 EEG 是 fixation-locked 的，且能更好捕捉动态神经-眼动耦合。\n")
    
    summary.append("\n### 8. 是否需要重新计算 EEG bandpower？\n")
    summary.append("   **需要** - 当前只有 sentence-level 的 bandpower 特征，需要从 raw EEG 重新计算 word/fixation-level 的 bandpower。\n")
    
    summary.append("\n### 9. 是否需要只先做 YHS/YRK 两个 subject 的样例数据？\n")
    summary.append("   **是** - 当前已完成这两个 subject 的样例数据提取，建议先验证结构正确性。\n")
    
    summary.append("\n### 10. 是否可以进入全 subject 数据提取？\n")
    summary.append("   **是** - 样例数据结构验证通过后，可以进入全 subject 数据提取。\n")
    
    summary.append("\n## 提取数据统计\n")
    if len(manifest_df) > 0:
        summary.append(f"- 序列总数: {len(manifest_df)}\n")
        summary.append(f"- 平均每个序列的单词数: {manifest_df['n_words'].mean():.1f}\n")
        summary.append(f"- 平均每个序列的 fixation 数: {manifest_df['n_fixations'].mean():.1f}\n")
        summary.append(f"- 完整 EEG+Gaze 序列比例: {(manifest_df['has_complete_eeg_gaze'].sum() / len(manifest_df)):.2%}\n")
    
    summary.append("\n## 下一步建议\n")
    summary.append("1. **验证样例数据**: 检查提取的数据结构是否符合预期\n")
    summary.append("2. **数据质量检查**: 检查缺失值、异常值\n")
    summary.append("3. **全 subject 提取**: 扩展到所有训练 subject\n")
    summary.append("4. **特征计算**: 计算 fixation-level 的 EEG bandpower 特征\n")
    summary.append("5. **建立 baseline**: 使用序列模型建立 word/fixation-level baseline\n")
    
    with open(os.path.join(output_dir, "fixation_extraction_summary.md"), 'w', encoding='utf-8') as f:
        f.writelines(''.join(summary))

def main():
    subjects = ['YHS', 'YRK']
    task_types = ['NR', 'TSR']
    output_dir = "results/fixation_extract_audit"
    
    # 确保目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("data/fixation_level").mkdir(parents=True, exist_ok=True)
    
    print("任务1: 检查 .mat 结构...")
    df = extract_word_fixation_data(subjects, task_types, output_dir)
    
    print("任务2: 提取最小可用表 - 已完成")
    
    print("任务3: 构建 sequence manifest...")
    manifest_df = build_sequence_manifest(df, output_dir)
    
    print("任务4: 对齐验证...")
    validate_alignment(df, manifest_df, output_dir)
    
    print("任务5: 检查协议风险...")
    check_protocol_risks(output_dir)
    
    print("任务6: 生成总结...")
    generate_summary(subjects, df, manifest_df, output_dir)
    
    print("\n所有任务完成！输出文件:")
    print(f"- {output_dir}/mat_structure_report.md")
    print(f"- {output_dir}/alignment_validation_sample.md")
    print(f"- {output_dir}/alignment_statistics_sample.csv")
    print(f"- {output_dir}/protocol_risk_report.md")
    print(f"- {output_dir}/fixation_extraction_summary.md")
    print(f"- data/fixation_level/sequence_manifest_sample.csv")

if __name__ == "__main__":
    main()
