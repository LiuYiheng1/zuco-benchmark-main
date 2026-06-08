import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA

def extract_confound_features(task_materials_dir="src/task_materials"):
    """从 task_materials 提取混杂特征"""
    confound_data = []
    
    for filename in os.listdir(task_materials_dir):
        if filename.startswith('nr_') and '_control' not in filename:
            label = 'NR'
            file_path = os.path.join(task_materials_dir, filename)
            df = pd.read_csv(file_path, sep=';', header=None, names=['sentence_id', 'relation_type', 'text'], on_bad_lines='skip')
            for _, row in df.iterrows():
                confound_data.append(extract_sentence_features(row['text'], row['sentence_id'], label, row['relation_type']))
        
        elif filename.startswith('tsr_') and '_control' not in filename:
            label = 'TSR'
            file_path = os.path.join(task_materials_dir, filename)
            df = pd.read_csv(file_path, sep=';', header=None, names=['sentence_id', 'relation_type', 'text'], on_bad_lines='skip')
            for _, row in df.iterrows():
                confound_data.append(extract_sentence_features(row['text'], row['sentence_id'], label, row['relation_type']))
    
    confound_df = pd.DataFrame(confound_data)
    confound_df['duplicate_sentence_flag'] = confound_df.duplicated(subset=['normalized_sentence_text'], keep=False)
    
    return confound_df

def extract_sentence_features(text, sentence_id, label, relation_type):
    """提取单个句子的混杂特征"""
    text = str(text).strip()
    
    word_count = len(text.split())
    char_count = len(text)
    punctuation_count = len(re.findall(r'[.,;:!?()\'"-]', text))
    number_count = len(re.findall(r'\d+', text))
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
    
    return merged

def load_eeg_gaze_features(subjects, features_dir="src/features"):
    """加载 EEG 和 Gaze 特征"""
    data = []
    
    for subject in subjects:
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
                
                data.append({
                    'subject': subject,
                    'key': key,
                    'eeg': eeg_feat,
                    'gaze': gaze_feat,
                    'label': label
                })
    
    return pd.DataFrame(data)

def compute_residuals(X_confound, X_signal, train_idx, test_idx):
    """计算 residual features"""
    X_train_conf = X_confound[train_idx]
    X_train_signal = X_signal[train_idx]
    X_test_conf = X_confound[test_idx]
    X_test_signal = X_signal[test_idx]
    
    scaler_conf = StandardScaler()
    scaler_signal = StandardScaler()
    
    X_train_conf_scaled = scaler_conf.fit_transform(X_train_conf)
    X_test_conf_scaled = scaler_conf.transform(X_test_conf)
    X_train_signal_scaled = scaler_signal.fit_transform(X_train_signal)
    X_test_signal_scaled = scaler_signal.transform(X_test_signal)
    
    # 使用线性回归预测信号
    from sklearn.linear_model import LinearRegression
    residualizer = LinearRegression()
    residualizer.fit(X_train_conf_scaled, X_train_signal_scaled)
    
    X_train_pred = residualizer.predict(X_train_conf_scaled)
    X_test_pred = residualizer.predict(X_test_conf_scaled)
    
    X_train_residual = X_train_signal_scaled - X_train_pred
    X_test_residual = X_test_signal_scaled - X_test_pred
    
    return X_train_residual, X_test_residual, scaler_signal

def train_expert(X_train, y_train, X_test, y_test, model_type='LogisticRegression'):
    """训练单个 expert"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'LogisticRegression':
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        clf = LinearSVC(max_iter=1000, random_state=42)
    
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X_test_scaled)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'auroc': roc_auc_score(y_test, y_proba),
        'predictions': y_pred,
        'probabilities': y_proba
    }

def train_selector(selector_features_train, selector_labels_train, selector_features_test):
    """训练 no-harm selector"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(selector_features_train)
    X_test_scaled = scaler.transform(selector_features_test)
    
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_train_scaled, selector_labels_train)
    
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    return y_pred, y_proba

def main():
    output_dir = "results/safe_cirl"
    os.makedirs(output_dir, exist_ok=True)
    
    held_out_subjects = ["YHS", "YRK", "YFR"]
    
    print("任务1: 构建 confound features...")
    confound_df = extract_confound_features()
    aligned_df = align_with_metadata(confound_df)
    confound_df.to_csv(os.path.join(output_dir, "confound_features.csv"), index=False, encoding='utf-8')
    
    with open(os.path.join(output_dir, "confound_alignment_report.md"), 'w', encoding='utf-8') as f:
        f.write(f"# Confound Alignment Report\n\n")
        f.write(f"## Alignment Statistics\n\n")
        f.write(f"- Total samples: {len(aligned_df)}\n")
        f.write(f"- Aligned samples: {aligned_df['sentence_text'].notna().sum()}\n")
        f.write(f"- Alignment rate: {aligned_df['sentence_text'].notna().sum() / len(aligned_df):.2%}\n")
    
    print("任务2: 构建 residual features...")
    subjects = [s for s in aligned_df['subject'].unique() if s.startswith('Y')]
    eeg_gaze_df = load_eeg_gaze_features(subjects)
    
    train_data = eeg_gaze_df[~eeg_gaze_df['subject'].isin(held_out_subjects)]
    test_data = eeg_gaze_df[eeg_gaze_df['subject'].isin(held_out_subjects)]
    
    if len(test_data) == 0:
        print("警告: 测试集为空，使用部分训练集作为测试")
        train_data, test_data = train_data[:int(len(train_data)*0.8)], train_data[int(len(train_data)*0.8):]
    
    X_gaze_train = np.array(list(train_data['gaze']))
    X_eeg_train = np.array(list(train_data['eeg']))
    y_train = np.array(train_data['label'])
    
    X_gaze_test = np.array(list(test_data['gaze']))
    X_eeg_test = np.array(list(test_data['eeg']))
    y_test = np.array(test_data['label'])
    
    # 使用词数作为简单的 confound feature
    X_conf_train = np.array([[len(str(k).split())] for k in train_data['key']])
    X_conf_test = np.array([[len(str(k).split())] for k in test_data['key']])
    
    # 计算 residuals
    gaze_resid_train, gaze_resid_test, _ = compute_residuals(X_conf_train, X_gaze_train, np.arange(len(X_conf_train)), np.arange(len(X_conf_test)))
    eeg_resid_train, eeg_resid_test, _ = compute_residuals(X_conf_train, X_eeg_train, np.arange(len(X_conf_train)), np.arange(len(X_conf_test)))
    
    with open(os.path.join(output_dir, "residual_feature_report.md"), 'w', encoding='utf-8') as f:
        f.write(f"# Residual Feature Report\n\n")
        f.write(f"## Feature Shapes\n\n")
        f.write(f"- Gaze raw shape: {X_gaze_train.shape}\n")
        f.write(f"- EEG raw shape: {X_eeg_train.shape}\n")
        f.write(f"- Gaze residual shape: {gaze_resid_train.shape}\n")
        f.write(f"- EEG residual shape: {eeg_resid_train.shape}\n")
        f.write(f"- Train samples: {len(train_data)}\n")
        f.write(f"- Test samples: {len(test_data)}\n")
    
    print("任务3: 训练多个 experts...")
    experts = {}
    
    experts['E1_gaze_raw'] = train_expert(X_gaze_train, y_train, X_gaze_test, y_test)
    experts['E2_gaze_residual'] = train_expert(gaze_resid_train, y_train, gaze_resid_test, y_test)
    experts['E3_eeg_raw'] = train_expert(X_eeg_train, y_train, X_eeg_test, y_test)
    experts['E4_eeg_residual'] = train_expert(eeg_resid_train, y_train, eeg_resid_test, y_test)
    experts['E5_concat_raw'] = train_expert(np.hstack([X_gaze_train, X_eeg_train]), y_train, np.hstack([X_gaze_test, X_eeg_test]), y_test)
    experts['E6_concat_residual'] = train_expert(np.hstack([gaze_resid_train, eeg_resid_train]), y_train, np.hstack([gaze_resid_test, eeg_resid_test]), y_test)
    
    expert_results = []
    for name, result in experts.items():
        expert_results.append({
            'expert': name,
            'accuracy': result['accuracy'],
            'balanced_accuracy': result['balanced_accuracy'],
            'macro_f1': result['macro_f1'],
            'auroc': result['auroc']
        })
    
    pd.DataFrame(expert_results).to_csv(os.path.join(output_dir, "expert_results_stage_a.csv"), index=False)
    
    print("任务4: 训练 no-harm selector...")
    gaze_raw_pred = experts['E1_gaze_raw']['predictions']
    gaze_raw_proba = experts['E1_gaze_raw']['probabilities']
    concat_resid_pred = experts['E6_concat_residual']['predictions']
    concat_resid_proba = experts['E6_concat_residual']['probabilities']
    
    # 构造 selector 训练数据（在训练集上）
    gaze_raw_pred_train = train_expert(X_gaze_train, y_train, X_gaze_train, y_train)['predictions']
    gaze_raw_proba_train = train_expert(X_gaze_train, y_train, X_gaze_train, y_train)['probabilities']
    concat_resid_pred_train = train_expert(np.hstack([gaze_resid_train, eeg_resid_train]), y_train, np.hstack([gaze_resid_train, eeg_resid_train]), y_train)['predictions']
    concat_resid_proba_train = train_expert(np.hstack([gaze_resid_train, eeg_resid_train]), y_train, np.hstack([gaze_resid_train, eeg_resid_train]), y_train)['probabilities']
    
    selector_features_train = np.column_stack([
        gaze_raw_proba_train,
        -np.log(gaze_raw_proba_train.clip(1e-10, 1-1e-10)) * gaze_raw_proba_train - np.log((1-gaze_raw_proba_train).clip(1e-10, 1-1e-10)) * (1-gaze_raw_proba_train),
        concat_resid_proba_train,
        np.abs(gaze_raw_proba_train - concat_resid_proba_train),
        np.linalg.norm(X_eeg_train, axis=1),
        np.linalg.norm(eeg_resid_train, axis=1)
    ])
    
    selector_labels_train = (
        (concat_resid_pred_train == y_train) & (gaze_raw_pred_train != y_train)
    ).astype(int)
    
    selector_features_test = np.column_stack([
        gaze_raw_proba,
        -np.log(gaze_raw_proba.clip(1e-10, 1-1e-10)) * gaze_raw_proba - np.log((1-gaze_raw_proba).clip(1e-10, 1-1e-10)) * (1-gaze_raw_proba),
        concat_resid_proba,
        np.abs(gaze_raw_proba - concat_resid_proba),
        np.linalg.norm(X_eeg_test, axis=1),
        np.linalg.norm(eeg_resid_test, axis=1)
    ])
    
    selector_pred, selector_proba = train_selector(selector_features_train, selector_labels_train, selector_features_test)
    
    selector_results = pd.DataFrame({
        'gaze_raw_pred': gaze_raw_pred,
        'gaze_raw_proba': gaze_raw_proba,
        'concat_resid_pred': concat_resid_pred,
        'concat_resid_proba': concat_resid_proba,
        'selector_pred': selector_pred,
        'selector_proba': selector_proba,
        'true_label': y_test
    })
    selector_results.to_csv(os.path.join(output_dir, "selector_results_stage_a.csv"), index=False)
    
    print("任务5: 比较方法...")
    safe_cirl_pred = np.where(selector_pred == 1, concat_resid_pred, gaze_raw_pred)
    
    oracle_pred = np.where((concat_resid_pred == y_test) & (gaze_raw_pred != y_test), concat_resid_pred, gaze_raw_pred)
    
    methods = {
        'gaze_raw': gaze_raw_pred,
        'best_single_expert': experts['E1_gaze_raw']['predictions'],
        'concat_raw': experts['E5_concat_raw']['predictions'],
        'concat_residual': concat_resid_pred,
        'oracle_selector': oracle_pred,
        'SAFE-CIRL': safe_cirl_pred
    }
    
    comparison_results = []
    for name, pred in methods.items():
        comparison_results.append({
            'method': name,
            'accuracy': accuracy_score(y_test, pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, pred),
            'macro_f1': f1_score(y_test, pred, average='macro'),
            'auroc': roc_auc_score(y_test, experts['E1_gaze_raw']['probabilities']) if name == 'gaze_raw' else roc_auc_score(y_test, concat_resid_proba)
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(output_dir, "safe_cirl_stage_a_results.csv"), index=False)
    
    gaze_raw_f1 = comparison_df[comparison_df['method'] == 'gaze_raw']['macro_f1'].values[0]
    safe_cirl_f1 = comparison_df[comparison_df['method'] == 'SAFE-CIRL']['macro_f1'].values[0]
    no_harm_violations = np.sum((selector_pred == 1) & (concat_resid_pred != y_test) & (gaze_raw_pred == y_test))
    fusion_usage_rate = np.mean(selector_pred == 1)
    
    with open(os.path.join(output_dir, "safe_cirl_stage_a_summary.md"), 'w', encoding='utf-8') as f:
        f.write("# SAFE-CIRL Stage A Summary\n\n")
        f.write("## ⚠️ Key Findings\n\n")
        f.write(f"**gaze_raw Macro-F1**: {gaze_raw_f1:.4f}\n\n")
        f.write(f"**SAFE-CIRL Macro-F1**: {safe_cirl_f1:.4f}\n\n")
        f.write(f"**Difference**: {safe_cirl_f1 - gaze_raw_f1:+.4f}\n\n")
        f.write(f"**No-harm violations**: {no_harm_violations}\n\n")
        f.write(f"**Fusion usage rate**: {fusion_usage_rate:.2%}\n\n")
        
        f.write("## Method Comparison\n\n")
        f.write("| Method | Accuracy | Balanced Accuracy | Macro-F1 |\n")
        f.write("|--------|----------|------------------|----------|\n")
        for _, row in comparison_df.iterrows():
            f.write(f"| {row['method']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} |\n")
        
        f.write("\n## Success Criteria\n\n")
        f.write("1. SAFE-CIRL >= gaze_raw: ")
        f.write(f"{'✅ PASS' if safe_cirl_f1 >= gaze_raw_f1 else '❌ FAIL'}\n\n")
        f.write("2. No-harm violation rate low: ")
        f.write(f"{'✅ PASS' if no_harm_violations < len(y_test) * 0.1 else '⚠️ WARNING'}\n\n")
        f.write("3. > 1% improvement: ")
        f.write(f"{'✅ PASS' if safe_cirl_f1 - gaze_raw_f1 > 0.01 else '❌ FAIL'}\n\n")
        f.write("4. Stage B recommendation: ")
        if safe_cirl_f1 >= gaze_raw_f1:
            f.write("✅ Recommended" if safe_cirl_f1 - gaze_raw_f1 > 0.01 else "⚠️ Conditional")
        else:
            f.write("❌ Not recommended")
        f.write("\n")
    
    selector_diagnostics = pd.DataFrame({
        'fusion_helpful': selector_labels_train.mean(),
        'fusion_usage_test': fusion_usage_rate,
        'no_harm_violations': no_harm_violations,
        'no_harm_rate': no_harm_violations / len(y_test) if len(y_test) > 0 else 0,
        'gaze_raw_f1': gaze_raw_f1,
        'safe_cirl_f1': safe_cirl_f1,
        'improvement': safe_cirl_f1 - gaze_raw_f1
    }, index=[0])
    selector_diagnostics.to_csv(os.path.join(output_dir, "selector_diagnostics.csv"), index=False)
    
    print("\nSAFE-CIRL Stage A 完成！输出文件:")
    print(f"- {output_dir}/confound_features.csv")
    print(f"- {output_dir}/confound_alignment_report.md")
    print(f"- {output_dir}/residual_feature_report.md")
    print(f"- {output_dir}/expert_results_stage_a.csv")
    print(f"- {output_dir}/selector_results_stage_a.csv")
    print(f"- {output_dir}/safe_cirl_stage_a_results.csv")
    print(f"- {output_dir}/safe_cirl_stage_a_summary.md")
    print(f"- {output_dir}/selector_diagnostics.csv")

if __name__ == "__main__":
    main()
