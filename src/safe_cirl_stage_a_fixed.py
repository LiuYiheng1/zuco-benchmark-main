import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

def load_aligned_data(data_dir="data"):
    npz_path = os.path.join(data_dir, "aligned_multimodal_y.npz")
    metadata_path = os.path.join(data_dir, "aligned_multimodal_y_metadata.csv")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"aligned_multimodal_y.npz not found at {npz_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"aligned_multimodal_y_metadata.csv not found at {metadata_path}")
    
    npz_data = np.load(npz_path, allow_pickle=True)
    metadata = pd.read_csv(metadata_path)
    
    metadata['subject'] = metadata['subject'].apply(lambda x: str(x).strip() if pd.notna(x) else x)
    metadata['label_num'] = metadata['label'].map({'NR': 0, 'TSR': 1})
    
    return npz_data, metadata

def get_features_for_sample_indices(npz_data, metadata, indices):
    eeg_data = npz_data['eeg']
    gaze_data = npz_data['gaze']
    
    eeg_features = []
    gaze_features = []
    labels = []
    subjects = []
    keys = []
    
    for idx in indices:
        row = metadata.iloc[idx]
        eeg_idx = int(row['eeg_fullidx'])
        gaze_idx = int(row['gaze_fullidx'])
        
        if eeg_idx < len(eeg_data) and gaze_idx < len(gaze_data):
            eeg_features.append(eeg_data[eeg_idx])
            gaze_features.append(gaze_data[gaze_idx])
            labels.append(row['label_num'])
            subjects.append(row['subject'])
            keys.append(row['sample_id'])
    
    return np.array(eeg_features), np.array(gaze_features), np.array(labels), subjects, keys

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
    
    return X_residual

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
        'auroc': roc_auc_score(y_test, y_proba)
    }

def ce_loss(logits, y):
    p = 1 / (1 + np.exp(-logits))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def gaze_anchored_residual_fusion(X_train_fusion, y_train, X_test_fusion, experts_train, experts_test, lambda_safe=0.5):
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

def build_fusion_features(experts):
    features = []
    for i in range(len(experts[list(experts.keys())[0]]['logits'])):
        probs = np.array([experts[k]['probabilities'][i] for k in experts.keys()])
        confidences = np.array([max(probs[i], 1-probs[i]) for i in range(len(probs))])
        
        features.append({
            'probs': probs,
            'confidences': confidences,
            'prob_variance': np.var(probs),
            'avg_confidence': np.mean(confidences)
        })
    return features

def compute_no_harm_violations(fusion_pred, gaze_pred, y_true):
    violations = (fusion_pred != y_true) & (gaze_pred == y_true)
    helpful = (fusion_pred == y_true) & (gaze_pred != y_true)
    return violations.sum(), violations.mean(), helpful.sum()

def evaluate_method(probs, y_true, gaze_pred):
    predictions = (probs >= 0.5).astype(int)
    violations, violation_rate, helpful = compute_no_harm_violations(predictions, gaze_pred, y_true)
    
    return {
        'accuracy': accuracy_score(y_true, predictions),
        'balanced_accuracy': balanced_accuracy_score(y_true, predictions),
        'macro_f1': f1_score(y_true, predictions, average='macro'),
        'auroc': roc_auc_score(y_true, probs),
        'no_harm_violations': violations,
        'no_harm_rate': violation_rate,
        'fusion_helpful': helpful
    }

def main():
    output_dir = "results/safe_cirl_stage_a_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    held_out_subjects = ["YHS", "YRK", "YFR"]
    
    print("加载 aligned_multimodal_y 数据...")
    npz_data, metadata = load_aligned_data()
    
    all_subjects = sorted(metadata['subject'].unique())
    train_subjects = [s for s in all_subjects if s not in held_out_subjects]
    
    print(f"All subjects: {all_subjects}")
    print(f"Held-out subjects: {held_out_subjects}")
    print(f"Train subjects: {train_subjects}")
    
    train_mask = metadata['subject'].isin(train_subjects)
    test_mask = metadata['subject'].isin(held_out_subjects)
    
    train_indices = metadata[train_mask].index.tolist()
    test_indices_per_subject = {subj: metadata[metadata['subject'] == subj].index.tolist() for subj in held_out_subjects}
    
    print(f"Train indices: {len(train_indices)}")
    for subj, idx in test_indices_per_subject.items():
        print(f"Test indices for {subj}: {len(idx)}")
    
    X_eeg_train, X_gaze_train, y_train, train_subj_list, train_keys = get_features_for_sample_indices(
        npz_data, metadata, train_indices
    )
    
    results_all = []
    split_check_all = []
    subjectwise_all = []
    
    for held_out in held_out_subjects:
        print(f"\n{'='*50}")
        print(f"处理 held_out_subject: {held_out}")
        print(f"{'='*50}")
        
        test_indices = test_indices_per_subject[held_out]
        
        if len(test_indices) == 0:
            raise ValueError(f"CRITICAL: held_out_subject '{held_out}' has NO samples!")
        
        X_eeg_test, X_gaze_test, y_test, test_subj_list, test_keys = get_features_for_sample_indices(
            npz_data, metadata, test_indices
        )
        
        test_hash = hashlib.md5(str(test_indices).encode()).hexdigest()
        print(f"Test hash: {test_hash}")
        print(f"Test samples: {len(test_indices)}, NR: {(y_test==0).sum()}, TSR: {(y_test==1).sum()}")
        
        split_check_all.append({
            'held_out_subject': held_out,
            'train_N': len(train_indices),
            'test_N': len(test_indices),
            'test_NR_count': int((y_test==0).sum()),
            'test_TSR_count': int((y_test==1).sum()),
            'test_index_hash': test_hash
        })
        
        X_conf_train = np.array([[len(k.split('_'))] for k in train_keys])
        X_conf_test = np.array([[len(k.split('_'))] for k in test_keys])
        
        gaze_resid_train = compute_residuals(X_conf_train, X_gaze_train)
        eeg_resid_train = compute_residuals(X_conf_train, X_eeg_train)
        
        gaze_resid_test = compute_residuals(X_conf_test, X_gaze_test)
        eeg_resid_test = compute_residuals(X_conf_test, X_eeg_test)
        
        print("训练 experts...")
        experts_train = {
            'E1_gaze_raw': train_expert(X_gaze_train, y_train, X_gaze_train, y_train),
            'E3_eeg_raw': train_expert(X_eeg_train, y_train, X_eeg_train, y_train),
            'E5_concat_raw': train_expert(np.hstack([X_gaze_train, X_eeg_train]), y_train, 
                                         np.hstack([X_gaze_train, X_eeg_train]), y_train),
        }
        gaze_resid_train_for_train = compute_residuals(X_conf_train, X_gaze_train)
        eeg_resid_train_for_train = compute_residuals(X_conf_train, X_eeg_train)
        experts_train['E2_gaze_residual'] = train_expert(gaze_resid_train_for_train, y_train, gaze_resid_train_for_train, y_train)
        experts_train['E4_eeg_residual'] = train_expert(eeg_resid_train_for_train, y_train, eeg_resid_train_for_train, y_train)
        experts_train['E6_concat_residual'] = train_expert(np.hstack([gaze_resid_train_for_train, eeg_resid_train_for_train]), y_train,
                                                           np.hstack([gaze_resid_train_for_train, eeg_resid_train_for_train]), y_train)
        
        experts_test = {
            'E1_gaze_raw': train_expert(X_gaze_train, y_train, X_gaze_test, y_test),
            'E2_gaze_residual': train_expert(gaze_resid_train, y_train, gaze_resid_test, y_test),
            'E3_eeg_raw': train_expert(X_eeg_train, y_train, X_eeg_test, y_test),
            'E4_eeg_residual': train_expert(eeg_resid_train, y_train, eeg_resid_test, y_test),
            'E5_concat_raw': train_expert(np.hstack([X_gaze_train, X_eeg_train]), y_train,
                                         np.hstack([X_gaze_test, X_eeg_test]), y_test),
            'E6_concat_residual': train_expert(np.hstack([gaze_resid_train, eeg_resid_train]), y_train,
                                              np.hstack([gaze_resid_test, eeg_resid_test]), y_test),
        }
        
        gaze_probs = experts_test['E1_gaze_raw']['probabilities']
        gaze_pred = experts_test['E1_gaze_raw']['predictions']
        
        results = []
        results.append({'subject': held_out, 'method': 'gaze_raw', **evaluate_method(gaze_probs, y_test, gaze_pred)})
        
        best_exp_name = None
        best_exp_f1 = -1
        for name in ['E3_eeg_raw', 'E4_eeg_residual', 'E5_concat_raw', 'E6_concat_residual']:
            if experts_test[name]['macro_f1'] > best_exp_f1:
                best_exp_f1 = experts_test[name]['macro_f1']
                best_exp_name = name
        results.append({'subject': held_out, 'method': 'best_single_expert', **evaluate_method(experts_test[best_exp_name]['probabilities'], y_test, gaze_pred)})
        
        for name in ['E3_eeg_raw', 'E4_eeg_residual', 'E5_concat_raw', 'E6_concat_residual']:
            results.append({'subject': held_out, 'method': name, **evaluate_method(experts_test[name]['probabilities'], y_test, gaze_pred)})
        
        fusion_features_train = build_fusion_features(experts_train)
        fusion_features_test = build_fusion_features(experts_test)
        
        X_train_fusion = np.array([[f['probs'].mean(), f['prob_variance'], f['avg_confidence']] for f in fusion_features_train])
        X_test_fusion = np.array([[f['probs'].mean(), f['prob_variance'], f['avg_confidence']] for f in fusion_features_test])
        
        fusion_probs, alphas = gaze_anchored_residual_fusion(X_train_fusion, y_train, X_test_fusion, experts_train, experts_test, lambda_safe=0.5)
        res = evaluate_method(fusion_probs, y_test, gaze_pred)
        res['subject'] = held_out
        res['method'] = 'gaze_anchored_lambda_0.5'
        res['avg_alpha'] = alphas.tolist()
        results.append(res)
        
        results_all.extend(results)
        
        for name, exp in experts_test.items():
            subjectwise_all.append({
                'subject': held_out,
                'expert': name,
                'macro_f1': exp['macro_f1'],
                'accuracy': exp['accuracy'],
                'auroc': exp['auroc']
            })
        
        print(f"\n结果:")
        print(f"  gaze_raw: {results[0]['macro_f1']:.4f}")
        print(f"  best_single_expert ({best_exp_name}): {best_exp_f1:.4f}")
        print(f"  gaze_anchored: {results[-1]['macro_f1']:.4f}")
    
    results_df = pd.DataFrame(results_all)
    results_df.to_csv(os.path.join(output_dir, "stage_a_fixed_results.csv"), index=False)
    
    split_df = pd.DataFrame(split_check_all)
    split_df.to_csv(os.path.join(output_dir, "stage_a_fixed_split_check.csv"), index=False)
    
    subjectwise_df = pd.DataFrame(subjectwise_all)
    subjectwise_df.to_csv(os.path.join(output_dir, "stage_a_fixed_subjectwise.csv"), index=False)
    
    gaze_raw_f1 = results_df[results_df['method'] == 'gaze_raw']['macro_f1'].mean()
    best_single_f1 = results_df[results_df['method'] == 'best_single_expert']['macro_f1'].mean()
    gaze_anchored_f1 = results_df[results_df['method'] == 'gaze_anchored_lambda_0.5']['macro_f1'].mean()
    
    unique_hashes = split_df['test_index_hash'].unique()
    
    with open(os.path.join(output_dir, "stage_a_fixed_summary.md"), 'w', encoding='utf-8') as f:
        f.write("# SAFE-CIRL-Fuse Stage A Fixed Summary\n\n")
        f.write("## 数据加载修复\n\n")
        f.write("- 现在使用 `aligned_multimodal_y.npz` + `metadata.csv`\n")
        f.write("- 移除了 `load_eeg_gaze_features` 和 fallback random split\n")
        f.write("- 每个 held-out subject 使用真实测试集\n\n")
        
        f.write("## Split 验证\n\n")
        f.write(f"- Held-out subjects: {held_out_subjects}\n")
        f.write(f"- Unique test hashes: {len(unique_hashes)} (应该是 {len(held_out_subjects)})\n")
        f.write(f"- ✅ PASS" if len(unique_hashes) == len(held_out_subjects) else "- ❌ FAIL\n\n")
        
        f.write("## 核心问题回答\n\n")
        f.write("### 1. 修复后 SAFE-CIRL-Fuse 是否仍超过 gaze_raw?\n\n")
        f.write(f"| Method | Macro-F1 (avg) |\n")
        f.write(f"|--------|---------------|\n")
        f.write(f"| gaze_raw | {gaze_raw_f1:.4f} |\n")
        f.write(f"| best_single_expert | {best_single_f1:.4f} |\n")
        f.write(f"| gaze_anchored_lambda_0.5 | {gaze_anchored_f1:.4f} |\n\n")
        
        f.write(f"- Gap (gaze_anchored - gaze_raw): {gaze_anchored_f1 - gaze_raw_f1:.4f}\n\n")
        f.write(f"- {'✅ YES: fusion still beats gaze_raw' if gaze_anchored_f1 > gaze_raw_f1 else '❌ NO: fusion does NOT beat gaze_raw'}\n\n")
        
        f.write("### 2. 每个 subject 的结果是否不同?\n\n")
        subj_f1s = results_df[results_df['method'] == 'gaze_raw'].groupby('subject')['macro_f1'].first()
        unique_f1s = subj_f1s.unique()
        f.write(f"- Unique gaze_raw Macro-F1 values: {len(unique_f1s)}\n")
        f.write(f"- {'✅ YES: results differ by subject' if len(unique_f1s) > 1 else '❌ NO: all subjects have same results'}\n\n")
        
        f.write("### 3. test_index_hash 是否不同?\n\n")
        f.write(f"- Unique hashes: {len(unique_hashes)}\n")
        f.write(f"- {'✅ YES: hashes are unique' if len(unique_hashes) == len(held_out_subjects) else '❌ NO: hashes are NOT unique'}\n\n")
        
        f.write("### 4. 是否值得重新进入 Stage B?\n\n")
        if gaze_anchored_f1 > gaze_raw_f1 + 0.01 and len(unique_hashes) == len(held_out_subjects):
            f.write("- ✅ YES: Recommended for Stage B\n")
        else:
            f.write("- ❌ NO: Not recommended until issues resolved\n")
        
        f.write("\n## Per-Subject Results\n\n")
        f.write("| Subject | gaze_raw | best_single | gaze_anchored | Hash |\n")
        f.write("|---------|----------|-------------|---------------|------|\n")
        for _, row in split_df.iterrows():
            subj = row['held_out_subject']
            gaze_f1 = results_df[(results_df['subject'] == subj) & (results_df['method'] == 'gaze_raw')]['macro_f1'].values[0]
            best_f1 = results_df[(results_df['subject'] == subj) & (results_df['method'] == 'best_single_expert')]['macro_f1'].values[0]
            anch_f1 = results_df[(results_df['subject'] == subj) & (results_df['method'] == 'gaze_anchored_lambda_0.5')]['macro_f1'].values[0]
            f.write(f"| {subj} | {gaze_f1:.4f} | {best_f1:.4f} | {anch_f1:.4f} | {row['test_index_hash'][:8]}... |\n")
    
    print(f"\n{'='*60}")
    print("Stage A Fixed 完成!")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
