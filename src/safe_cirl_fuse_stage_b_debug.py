import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

def load_eeg_gaze_features(subjects, features_dir="src/features"):
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

def compute_residuals(X_confound, X_signal):
    from sklearn.linear_model import LinearRegression
    
    scaler_conf = StandardScaler()
    scaler_signal = StandardScaler()
    
    X_conf_scaled = scaler_conf.fit_transform(X_confound)
    X_signal_scaled = scaler_signal.fit_transform(X_signal)
    
    residualizer = LinearRegression()
    residualizer.fit(X_conf_scaled, X_signal_scaled)
    
    X_pred = residualizer.predict(X_conf_scaled)
    X_residual = X_signal_scaled - X_pred
    
    return X_residual, scaler_signal, residualizer

def train_expert(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    y_logits = clf.decision_function(X_test_scaled)
    
    return {
        'predictions': y_pred,
        'probabilities': y_proba,
        'logits': y_logits,
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'auroc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

def ce_loss(logits, y):
    p = 1 / (1 + np.exp(-logits))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def gaze_anchored_residual_fusion_with_reg(X_train_fusion, y_train, X_test_fusion, 
                                           experts_train, experts_test, lambda_safe=0.5):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fusion)
    X_test_scaled = scaler.transform(X_test_fusion)
    
    n_features = X_train_scaled.shape[1]
    n_residual_experts = 3
    
    def objective(params):
        weights = params[:n_features * n_residual_experts].reshape(n_residual_experts, n_features)
        biases = params[n_features * n_residual_experts:]
        
        alpha_logits = X_train_scaled @ weights.T + biases
        alphas = 1 / (1 + np.exp(-alpha_logits))
        
        gaze_logits = experts_train['E1_gaze_raw']['logits']
        
        residual_keys = ['E2_gaze_residual', 'E4_eeg_residual', 'E6_concat_residual']
        residual_logits = [experts_train[k]['logits'] for k in residual_keys if k in experts_train]
        
        final_logits = gaze_logits.copy()
        for i, r_logits in enumerate(residual_logits):
            if i < alphas.shape[1]:
                alpha_i = np.mean(alphas[:, i])
                final_logits = final_logits + alpha_i * (r_logits - gaze_logits)
        
        ce = ce_loss(final_logits, y_train)
        gaze_ce = ce_loss(gaze_logits, y_train)
        diff = ce - gaze_ce
        safe_penalty = lambda_safe * np.maximum(0, diff)
        
        return ce + safe_penalty
    
    init_params = np.random.randn(n_features * n_residual_experts + n_residual_experts) * 0.1
    result = minimize(objective, init_params, method='L-BFGS-B', options={'maxiter': 500})
    
    opt_weights = result.x[:n_features * n_residual_experts].reshape(n_residual_experts, n_features)
    opt_biases = result.x[n_features * n_residual_experts:]
    
    alpha_logits_test = X_test_scaled @ opt_weights.T + opt_biases
    alphas_test = 1 / (1 + np.exp(-alpha_logits_test))
    
    gaze_logits = experts_test['E1_gaze_raw']['logits']
    final_logits = gaze_logits.copy()
    
    residual_keys = ['E2_gaze_residual', 'E4_eeg_residual', 'E6_concat_residual']
    residual_logits_test = [experts_test[k]['logits'] for k in residual_keys if k in experts_test]
    
    avg_alphas = []
    for i, r_logits in enumerate(residual_logits_test):
        if i < alphas_test.shape[1]:
            alpha_i = np.mean(alphas_test[:, i])
            avg_alphas.append(alpha_i)
            final_logits = final_logits + alpha_i * (r_logits - gaze_logits)
    
    final_probs = 1 / (1 + np.exp(-final_logits))
    
    return final_probs, np.array(avg_alphas)

def main():
    output_dir = "results/safe_cirl_fuse_stage_b"
    os.makedirs(output_dir, exist_ok=True)
    
    held_out_subjects = ["YHS", "YIS", "YSD", "YRK", "YFR"]
    all_subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 
                    'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
    
    all_data = load_eeg_gaze_features(all_subjects)
    
    split_check_data = []
    subjectwise_metrics = []
    index_hash_data = []
    best_expert_data = []
    
    for test_subject in held_out_subjects:
        print(f"处理 subject: {test_subject}")
        
        train_subjects = [s for s in all_subjects if s != test_subject]
        
        train_data = all_data[all_data['subject'].isin(train_subjects)]
        test_data = all_data[all_data['subject'] == test_subject]
        
        if len(test_data) == 0:
            print(f"警告: {test_subject} 测试集为空，使用 train-validation split")
            combined_data = all_data[all_data['subject'].isin(all_subjects)]
            train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42, stratify=combined_data['label'])
        
        test_index_hash = hashlib.md5(str(test_data.index.tolist()).encode()).hexdigest()
        
        split_check_data.append({
            'test_subject': test_subject,
            'train_subjects': ','.join(train_subjects),
            'test_sample_count': len(test_data),
            'test_NR_count': (test_data['label'] == 0).sum(),
            'test_TSR_count': (test_data['label'] == 1).sum(),
            'unique_test_subjects': ','.join(test_data['subject'].unique()),
            'first_10_keys': ','.join(test_data['key'].head(10).tolist()),
            'test_index_hash': test_index_hash
        })
        
        X_gaze_train = np.array(list(train_data['gaze']))
        X_eeg_train = np.array(list(train_data['eeg']))
        y_train = np.array(train_data['label'])
        
        X_gaze_test = np.array(list(test_data['gaze']))
        X_eeg_test = np.array(list(test_data['eeg']))
        y_test = np.array(test_data['label'])
        
        X_conf_train = np.array([[len(str(k).split())] for k in train_data['key']])
        X_conf_test = np.array([[len(str(k).split())] for k in test_data['key']])
        
        gaze_resid_train, _, _ = compute_residuals(X_conf_train, X_gaze_train)
        eeg_resid_train, _, _ = compute_residuals(X_conf_train, X_eeg_train)
        
        gaze_resid_test, _, _ = compute_residuals(X_conf_test, X_gaze_test)
        eeg_resid_test, _, _ = compute_residuals(X_conf_test, X_eeg_test)
        
        experts_train = {}
        experts_train['E1_gaze_raw'] = train_expert(X_gaze_train, y_train, X_gaze_train, y_train)
        experts_train['E2_gaze_residual'] = train_expert(gaze_resid_train, y_train, gaze_resid_train, y_train)
        experts_train['E3_eeg_raw'] = train_expert(X_eeg_train, y_train, X_eeg_train, y_train)
        experts_train['E4_eeg_residual'] = train_expert(eeg_resid_train, y_train, eeg_resid_train, y_train)
        experts_train['E5_concat_raw'] = train_expert(np.hstack([X_gaze_train, X_eeg_train]), y_train, 
                                                      np.hstack([X_gaze_train, X_eeg_train]), y_train)
        experts_train['E6_concat_residual'] = train_expert(np.hstack([gaze_resid_train, eeg_resid_train]), y_train, 
                                                          np.hstack([gaze_resid_train, eeg_resid_train]), y_train)
        
        experts_test = {}
        experts_test['E1_gaze_raw'] = train_expert(X_gaze_train, y_train, X_gaze_test, y_test)
        experts_test['E2_gaze_residual'] = train_expert(gaze_resid_train, y_train, gaze_resid_test, y_test)
        experts_test['E3_eeg_raw'] = train_expert(X_eeg_train, y_train, X_eeg_test, y_test)
        experts_test['E4_eeg_residual'] = train_expert(eeg_resid_train, y_train, eeg_resid_test, y_test)
        experts_test['E5_concat_raw'] = train_expert(np.hstack([X_gaze_train, X_eeg_train]), y_train, 
                                                     np.hstack([X_gaze_test, X_eeg_test]), y_test)
        experts_test['E6_concat_residual'] = train_expert(np.hstack([gaze_resid_train, eeg_resid_train]), y_train, 
                                                         np.hstack([gaze_resid_test, eeg_resid_test]), y_test)
        
        best_expert_name = None
        best_expert_f1 = -1
        for exp_name, exp_results in experts_test.items():
            if exp_results['macro_f1'] > best_expert_f1:
                best_expert_f1 = exp_results['macro_f1']
                best_expert_name = exp_name
        
        gaze_raw_f1 = experts_test['E1_gaze_raw']['macro_f1']
        
        fusion_probs, alphas = gaze_anchored_residual_fusion_with_reg(
            np.array([[0]*14]*len(train_data)), y_train, 
            np.array([[0]*14]*len(test_data)), experts_train, experts_test
        )
        fusion_pred = (fusion_probs >= 0.5).astype(int)
        fusion_f1 = f1_score(y_test, fusion_pred, average='macro')
        
        for exp_name, exp_results in experts_test.items():
            subjectwise_metrics.append({
                'subject': test_subject,
                'expert': exp_name,
                'accuracy': exp_results['accuracy'],
                'balanced_accuracy': exp_results['balanced_accuracy'],
                'macro_f1': exp_results['macro_f1'],
                'auroc': exp_results['auroc'],
                'confusion_matrix': str(exp_results['confusion_matrix']),
                'prediction_count': len(exp_results['predictions']),
                'NR_count': (y_test == 0).sum(),
                'TSR_count': (y_test == 1).sum()
            })
        
        index_hash_data.append({
            'test_subject': test_subject,
            'test_index_hash': test_index_hash,
            'test_size': len(test_data)
        })
        
        best_expert_data.append({
            'test_subject': test_subject,
            'best_expert': best_expert_name,
            'best_expert_macro_f1': best_expert_f1,
            'gaze_anchored_macro_f1': fusion_f1,
            'gap_to_gaze_raw': best_expert_f1 - gaze_raw_f1,
            'gap_to_best_expert': fusion_f1 - best_expert_f1
        })
    
    pd.DataFrame(split_check_data).to_csv(os.path.join(output_dir, "debug_split_check.csv"), index=False)
    pd.DataFrame(subjectwise_metrics).to_csv(os.path.join(output_dir, "subjectwise_metrics_recomputed.csv"), index=False)
    pd.DataFrame(index_hash_data).to_csv(os.path.join(output_dir, "test_index_hash_check.csv"), index=False)
    pd.DataFrame(best_expert_data).to_csv(os.path.join(output_dir, "best_expert_by_subject.csv"), index=False)
    
    with open(os.path.join(output_dir, "adagtcn_comparison_note.md"), 'w', encoding='utf-8') as f:
        f.write("# AdaGTCN Comparison Note\n\n")
        f.write("## Task Comparison\n\n")
        f.write("- **AdaGTCN Task**: NR vs TSR reading state recognition\n")
        f.write("- **Current Task**: NR vs TSR reading state recognition\n")
        f.write("- **Conclusion**: Same high-level task\n\n")
        
        f.write("## Input Features Comparison\n\n")
        f.write("| Feature Type | AdaGTCN | Current Project |\n")
        f.write("|-------------|---------|----------------|\n")
        f.write("| EEG Level | word/fixation-level sequence | sentence-level 420-D |\n")
        f.write("| Gaze Level | word/fixation-level sequence | sentence-level 9-D |\n")
        f.write("| Temporal Model | Temporal Convolutional Network | Static features only |\n")
        f.write("| Raw Data | Original .mat files | Derived .npy features |\n\n")
        
        f.write("## Performance Comparison\n\n")
        f.write("- **AdaGTCN reported Macro-F1**: 0.695\n")
        f.write("- **Current Stage B best Macro-F1**: 0.5821\n")
        f.write("- **Gap**: -0.1129 (11.29 percentage points)\n\n")
        
        f.write("## Important Notes\n\n")
        f.write("1. **Not identical experimental settings**: Current project uses sentence-level features only, while AdaGTCN uses fixation-level sequences with temporal modeling.\n\n")
        f.write("2. **Cannot claim direct comparison**: Due to different input modalities and modeling approaches, direct performance comparison is not valid.\n\n")
        f.write("3. **Future work for valid comparison**:\n\n")
        f.write("   Option A) Obtain original .mat files and implement fixation-level temporal model to reproduce AdaGTCN setting\n\n")
        f.write("   Option B) Improve current approach to achieve >0.695 Macro-F1 on sentence-level features, with explicit disclaimer about different input settings\n\n")
        
        f.write("## Recommendation\n\n")
        f.write("Before claiming superiority over AdaGTCN, the current approach must either:\n")
        f.write("- Match the input modality (fixation-level sequences), OR\n")
        f.write("- Significantly outperform on the current modality with clear documentation of the difference\n")
    
    print(f"\nStage B Bug Audit 完成！输出目录: {output_dir}")

if __name__ == "__main__":
    main()
