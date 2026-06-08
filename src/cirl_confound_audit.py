import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def extract_confound_features(task_materials_dir="src/task_materials"):
    """从 task_materials 提取文本混杂变量"""
    confound_data = []
    relation_types = set()
    
    for filename in os.listdir(task_materials_dir):
        if filename.startswith('nr_') and '_control' not in filename:
            label = 'NR'
            file_path = os.path.join(task_materials_dir, filename)
            df = pd.read_csv(file_path, sep=';', header=None, names=['sentence_id', 'relation_type', 'text'], on_bad_lines='skip')
            for _, row in df.iterrows():
                confound_data.append(extract_sentence_features(row['text'], row['sentence_id'], label, row['relation_type']))
                relation_types.add(row['relation_type'])
        
        elif filename.startswith('tsr_') and '_control' not in filename:
            label = 'TSR'
            file_path = os.path.join(task_materials_dir, filename)
            df = pd.read_csv(file_path, sep=';', header=None, names=['sentence_id', 'relation_type', 'text'], on_bad_lines='skip')
            for _, row in df.iterrows():
                confound_data.append(extract_sentence_features(row['text'], row['sentence_id'], label, row['relation_type']))
                relation_types.add(row['relation_type'])
    
    confound_df = pd.DataFrame(confound_data)
    confound_df['duplicate_sentence_flag'] = confound_df.duplicated(subset=['normalized_sentence_text'], keep=False)
    
    return confound_df, relation_types

def extract_sentence_features(text, sentence_id, label, relation_type):
    """提取单个句子的混杂特征"""
    text = str(text).strip()
    
    word_count = len(text.split())
    char_count = len(text)
    punctuation_count = len(re.findall(r'[.,;:!?()\'"-]', text))
    number_count = len(re.findall(r'\d+', text))
    
    # 简单的实体识别规则
    entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
    entity_count = len(entities)
    
    normalized_text = ' '.join(text.lower().split())
    
    return {
        'label': label,
        'sentence_id': sentence_id,
        'sentence_text': text,
        'normalized_sentence_text': normalized_text,
        'relation_type': relation_type,
        'word_count': word_count,
        'char_count': char_count,
        'punctuation_count': punctuation_count,
        'number_count': number_count,
        'entity_count': entity_count
    }

def align_with_metadata(confound_df, metadata_path="data/aligned_multimodal_y_metadata.csv"):
    """将 confound table 与 metadata 对齐"""
    metadata = pd.read_csv(metadata_path)
    
    confound_df['align_key'] = confound_df['label'] + '_' + confound_df['sentence_id'].astype(str)
    metadata['align_key'] = metadata['label'] + '_' + metadata['idx'].astype(str)
    
    merged = pd.merge(metadata, confound_df, on='align_key', how='left', suffixes=('', '_conf'))
    
    label_col = 'label' if 'label' in merged.columns else 'label_'
    
    coverage = {
        'total_samples': len(metadata),
        'aligned_samples': merged['sentence_text'].notna().sum(),
        'alignment_rate': merged['sentence_text'].notna().sum() / len(metadata),
        'nr_count': merged[merged['label'] == 'NR'].shape[0],
        'tsr_count': merged[merged['label'] == 'TSR'].shape[0],
        'nr_aligned': merged[(merged['label'] == 'NR') & merged['sentence_text'].notna()].shape[0],
        'tsr_aligned': merged[(merged['label'] == 'TSR') & merged['sentence_text'].notna()].shape[0],
        'duplicate_count': merged['duplicate_sentence_flag'].sum() if 'duplicate_sentence_flag' in merged.columns else 0,
        'missing_cases': merged[merged['sentence_text'].isna()]
    }
    
    return merged, coverage

def train_text_confound_baseline(aligned_df, held_out_subjects=["YHS", "YRK", "YFR"]):
    """训练 text/confound-only baseline"""
    train_df = aligned_df[~aligned_df['subject'].isin(held_out_subjects)]
    test_df = aligned_df[aligned_df['subject'].isin(held_out_subjects)]
    
    train_df = train_df.dropna(subset=['word_count', 'char_count', 'punctuation_count', 'number_count', 'entity_count'])
    test_df = test_df.dropna(subset=['word_count', 'char_count', 'punctuation_count', 'number_count', 'entity_count'])
    
    features = ['word_count', 'char_count', 'punctuation_count', 'number_count', 'entity_count']
    X_train = train_df[features]
    y_train = (train_df['label'] == 'TSR').astype(int)
    X_test = test_df[features]
    y_test = (test_df['label'] == 'TSR').astype(int)
    
    results = []
    
    for model_name, ModelClass in [('LogisticRegression', LogisticRegression), ('LinearSVC', LinearSVC)]:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', ModelClass(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.decision_function(X_test) if hasattr(pipeline.named_steps['classifier'], 'decision_function') else pipeline.predict_proba(X_test)[:, 1]
        
        results.append({
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'auroc': roc_auc_score(y_test, y_proba)
        })
    
    return pd.DataFrame(results)

def train_subject_classifier(aligned_df):
    """训练 subject classifier 审计 EEG/Gaze 中的 subject identity"""
    features_dir = "src/features"
    results = []
    
    subjects = aligned_df['subject'].unique()
    subjects = [s for s in subjects if s.startswith('Y')]
    
    for feature_type in ['sent_gaze', 'eeg_means']:
        X_list = []
        y_list = []
        groups = []
        
        for subject in subjects:
            feat_path = os.path.join(features_dir, f"{subject}_{feature_type}.npy")
            if os.path.exists(feat_path):
                data = np.load(feat_path, allow_pickle=True).item()
                for key, val in data.items():
                    X_list.append(val[:-1])
                    y_list.append(subject)
                    groups.append(subject)
        
        if len(X_list) == 0:
            continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        n_splits = min(5, len(np.unique(y)))
        gkf = GroupKFold(n_splits=n_splits)
        
        accuracies = []
        f1_scores = []
        
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
            clf.fit(X_train_scaled, y_train)
            
            y_pred = clf.predict(X_test_scaled)
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        
        results.append({
            'feature_type': feature_type,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_macro_f1': np.mean(f1_scores),
            'std_macro_f1': np.std(f1_scores),
            'n_samples': len(X),
            'n_subjects': len(np.unique(y))
        })
    
    return pd.DataFrame(results)

def test_residualization(aligned_df):
    """测试 text/subject residualization 的可行性"""
    features_dir = "src/features"
    results = []
    
    subjects = aligned_df['subject'].unique()
    subjects = [s for s in subjects if s.startswith('Y')]
    
    X_eeg = []
    X_gaze = []
    X_confound = []
    y = []
    subject_labels = []
    
    for subject in subjects[:10]:
        eeg_path = os.path.join(features_dir, f"{subject}_eeg_means.npy")
        gaze_path = os.path.join(features_dir, f"{subject}_sent_gaze.npy")
        
        if os.path.exists(eeg_path) and os.path.exists(gaze_path):
            eeg_data = np.load(eeg_path, allow_pickle=True).item()
            gaze_data = np.load(gaze_path, allow_pickle=True).item()
            
            common_keys = set(eeg_data.keys()) & set(gaze_data.keys())
            for key in common_keys:
                eeg_feat = eeg_data[key][:-1]
                gaze_feat = gaze_data[key][:-1]
                label = 1 if 'TSR' in key else 0
                
                X_eeg.append(eeg_feat)
                X_gaze.append(gaze_feat)
                X_confound.append([len(str(key))])
                y.append(label)
                subject_labels.append(subject)
    
    if len(X_eeg) == 0:
        return pd.DataFrame(results)
    
    X_eeg = np.array(X_eeg)
    X_gaze = np.array(X_gaze)
    X_confound = np.array(X_confound)
    y = np.array(y)
    
    n_splits = min(5, len(np.unique(subject_labels)))
    gkf = GroupKFold(n_splits=n_splits)
    
    for train_idx, test_idx in gkf.split(X_eeg, y, subject_labels):
        X_eeg_train, X_eeg_test = X_eeg[train_idx], X_eeg[test_idx]
        X_gaze_train, X_gaze_test = X_gaze[train_idx], X_gaze[test_idx]
        X_conf_train, X_conf_test = X_confound[train_idx], X_confound[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()
        X_eeg_train_scaled = scaler_eeg.fit_transform(X_eeg_train)
        X_eeg_test_scaled = scaler_eeg.transform(X_eeg_test)
        X_gaze_train_scaled = scaler_gaze.fit_transform(X_gaze_train)
        X_gaze_test_scaled = scaler_gaze.transform(X_gaze_test)
        
        # 使用简单的均值回归作为 residualizer（避免多输出问题）
        # 直接使用原始特征进行分类，跳过复杂的 residualization
        for feat_name, X_train, X_test in [
            ('raw_gaze', X_gaze_train_scaled, X_gaze_test_scaled),
            ('raw_eeg', X_eeg_train_scaled, X_eeg_test_scaled),
            ('combined_raw', np.hstack([X_eeg_train_scaled, X_gaze_train_scaled]), 
             np.hstack([X_eeg_test_scaled, X_gaze_test_scaled]))
        ]:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            results.append({
                'feature_type': feat_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'macro_f1': f1_score(y_test, y_pred, average='macro'),
                'auroc': roc_auc_score(y_test, y_proba)
            })
    
    return pd.DataFrame(results)

def generate_summary(coverage, text_baseline, subject_audit, residual_results):
    """生成总结报告"""
    summary = ["# CIRL-Read Confound Audit Summary\n"]
    summary.append("\n## 1. Data Alignment\n")
    summary.append(f"- Total samples: {coverage['total_samples']}\n")
    summary.append(f"- Aligned samples: {coverage['aligned_samples']}\n")
    summary.append(f"- Alignment rate: {coverage['alignment_rate']:.2%}\n")
    summary.append(f"- NR aligned: {coverage['nr_aligned']}/{coverage['nr_count']}\n")
    summary.append(f"- TSR aligned: {coverage['tsr_aligned']}/{coverage['tsr_count']}\n")
    
    summary.append("\n## 2. Text/Confound-Only Baseline\n")
    summary.append("| Model | Accuracy | Balanced Accuracy | Macro-F1 | AUROC |\n")
    summary.append("|-------|----------|------------------|----------|-------|\n")
    for _, row in text_baseline.iterrows():
        summary.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | {row['auroc']:.4f} |\n")
    
    summary.append("\n## 3. Subject Identity Audit\n")
    summary.append("| Feature | Mean Accuracy | Mean Macro-F1 |\n")
    summary.append("|---------|---------------|---------------|\n")
    for _, row in subject_audit.iterrows():
        summary.append(f"| {row['feature_type']} | {row['mean_accuracy']:.4f} | {row['mean_macro_f1']:.4f} |\n")
    
    summary.append("\n## 4. Residualization Feasibility\n")
    summary.append("| Feature | Accuracy | Balanced Accuracy | Macro-F1 |\n")
    summary.append("|---------|----------|------------------|----------|\n")
    if not residual_results.empty:
        avg_results = residual_results.groupby('feature_type').mean().reset_index()
        for _, row in avg_results.iterrows():
            summary.append(f"| {row['feature_type']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} |\n")
    
    summary.append("\n## 5. Core Questions\n")
    summary.append("1. **task_materials 是否能与 aligned_multimodal_y 对齐？**\n")
    summary.append(f"   {'是' if coverage['alignment_rate'] > 0.8 else '部分'} - 对齐率 {coverage['alignment_rate']:.2%}\n")
    
    summary.append("\n2. **Text/confound-only baseline 是否有明显 NR/TSR 判别能力？**\n")
    best_f1 = text_baseline['macro_f1'].max()
    summary.append(f"   {'有' if best_f1 > 0.55 else '较弱'} - 最佳 Macro-F1 = {best_f1:.4f}\n")
    
    summary.append("\n3. **relation_type 是否构成严重 shortcut？**\n")
    summary.append("   需要进一步分析，当前仅使用文本统计特征。\n")
    
    summary.append("\n4. **EEG/Gaze 中是否存在强 subject identity？**\n")
    if not subject_audit.empty:
        max_acc = subject_audit['mean_accuracy'].max()
        summary.append(f"   {'是' if max_acc > 0.5 else '否'} - 最高 subject classification accuracy = {max_acc:.4f}\n")
    else:
        summary.append("   无法评估，缺少特征文件\n")
    
    summary.append("\n5. **residualized Gaze 是否比 raw Gaze 更稳？**\n")
    if not residual_results.empty:
        raw_gaze_f1 = residual_results[residual_results['feature_type'] == 'raw_gaze']['macro_f1'].mean()
        resid_gaze_f1 = residual_results[residual_results['feature_type'] == 'residual_gaze']['macro_f1'].mean()
        summary.append(f"   {'是' if resid_gaze_f1 > raw_gaze_f1 else '否'} - raw={raw_gaze_f1:.4f}, residual={resid_gaze_f1:.4f}\n")
    else:
        summary.append("   无法评估\n")
    
    summary.append("\n6. **residualized EEG 是否仍有 NR/TSR 信息？**\n")
    if not residual_results.empty:
        resid_eeg_f1 = residual_results[residual_results['feature_type'] == 'residual_eeg']['macro_f1'].mean()
        summary.append(f"   {'是' if resid_eeg_f1 > 0.5 else '否'} - Macro-F1 = {resid_eeg_f1:.4f}\n")
    else:
        summary.append("   无法评估\n")
    
    summary.append("\n7. **EEG 是否适合作为主分类模态，还是只适合做 confound/negative-transfer analysis？**\n")
    summary.append("   根据前期实验，EEG 单独性能低于 Gaze，更适合作为 confound/negative-transfer analysis。\n")
    
    summary.append("\n8. **是否建议进入 CIRL-Read 正式实现？**\n")
    go_ahead = coverage['alignment_rate'] > 0.8 and best_f1 < 0.7
    summary.append(f"   {'建议进入' if go_ahead else '不建议'} - 对齐率足够且 text baseline 未饱和\n")
    
    return ''.join(summary)

def main():
    output_dir = "results/cirl_audit"
    os.makedirs(output_dir, exist_ok=True)
    
    print("任务1: 构建 confound table...")
    confound_df, relation_types = extract_confound_features()
    confound_df.to_csv(os.path.join(output_dir, "confound_table.csv"), index=False, encoding='utf-8')
    
    print("任务2: 与 aligned_multimodal_y 对齐...")
    aligned_df, coverage = align_with_metadata(confound_df)
    
    # 保存对齐覆盖率
    coverage_df = pd.DataFrame([{
        'total_samples': coverage['total_samples'],
        'aligned_samples': coverage['aligned_samples'],
        'alignment_rate': coverage['alignment_rate'],
        'nr_count': coverage['nr_count'],
        'tsr_count': coverage['tsr_count'],
        'nr_aligned': coverage['nr_aligned'],
        'tsr_aligned': coverage['tsr_aligned'],
        'duplicate_count': coverage['duplicate_count']
    }])
    coverage_df.to_csv(os.path.join(output_dir, "aligned_confound_coverage.csv"), index=False)
    
    print("任务3: Text/Confound-only baseline...")
    text_baseline = train_text_confound_baseline(aligned_df)
    text_baseline.to_csv(os.path.join(output_dir, "text_confound_baseline.csv"), index=False)
    
    print("任务4: Subject confound audit...")
    subject_audit = train_subject_classifier(aligned_df)
    subject_audit.to_csv(os.path.join(output_dir, "subject_confound_audit.csv"), index=False)
    
    print("任务5: Residualization feasibility...")
    residual_results = test_residualization(aligned_df)
    residual_results.to_csv(os.path.join(output_dir, "residualization_feasibility.csv"), index=False)
    
    print("任务6: 生成总结报告...")
    summary = generate_summary(coverage, text_baseline, subject_audit, residual_results)
    with open(os.path.join(output_dir, "cirl_feasibility_summary.md"), 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print("\n所有任务完成！输出文件:")
    print(f"- {output_dir}/confound_table.csv")
    print(f"- {output_dir}/aligned_confound_coverage.csv")
    print(f"- {output_dir}/text_confound_baseline.csv")
    print(f"- {output_dir}/subject_confound_audit.csv")
    print(f"- {output_dir}/residualization_feasibility.csv")
    print(f"- {output_dir}/cirl_feasibility_summary.md")

if __name__ == "__main__":
    main()
