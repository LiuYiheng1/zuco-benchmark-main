import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
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
    entropy = -y_proba * np.log(y_proba + 1e-10) - (1 - y_proba) * np.log(1 - y_proba + 1e-10)
    
    return {
        'predictions': y_pred,
        'probabilities': y_proba,
        'logits': y_logits,
        'confidence': np.maximum(y_proba, 1 - y_proba),
        'entropy': entropy,
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'auroc': roc_auc_score(y_test, y_proba),
        'model': clf,
        'scaler': scaler
    }

def build_fusion_features(experts, include_text_features=False, text_features=None):
    features = []
    
    for i in range(len(experts[list(experts.keys())[0]]['logits'])):
        probs = []
        logits = []
        confidences = []
        
        for key in experts.keys():
            probs.append(experts[key]['probabilities'][i])
            logits.append(experts[key]['logits'][i])
            confidences.append(experts[key]['confidence'][i])
        
        probs = np.array(probs)
        logits = np.array(logits)
        confidences = np.array(confidences)
        
        feat = {
            'probs': probs,
            'logits': logits,
            'confidences': confidences,
            'prob_variance': np.var(probs),
            'prob_range': np.max(probs) - np.min(probs),
            'avg_confidence': np.mean(confidences)
        }
        
        if include_text_features and text_features is not None:
            feat['text_features'] = text_features[i]
        
        features.append(feat)
    
    return features

def uniform_average_fusion(experts):
    probs = np.array([experts[key]['probabilities'] for key in experts.keys()])
    avg_probs = np.mean(probs, axis=0)
    return avg_probs

def learned_softmax_fusion(X_train, y_train, X_test, experts_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_experts = len(experts_test.keys())
    weights_models = []
    for i in range(n_experts):
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)
        weights_models.append(clf)
    
    weights = []
    for clf in weights_models:
        w = clf.predict_proba(X_test_scaled)[:, 1]
        weights.append(w)
    weights = np.array(weights).T
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    
    all_logits = np.array([experts_test[key]['logits'] for key in experts_test.keys()])
    final_logits = np.sum(weights.T * all_logits, axis=0)
    final_probs = 1 / (1 + np.exp(-final_logits))
    
    return final_probs, weights

def ce_loss(logits, y):
    p = 1 / (1 + np.exp(-logits))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def gaze_anchored_residual_fusion_with_reg(X_train_fusion, y_train, X_test_fusion, 
                                           experts_train, experts_test, lambda_safe=0.0):
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

def compute_no_harm_violations(fusion_pred, gaze_pred, y_true):
    violations = (fusion_pred != y_true) & (gaze_pred == y_true)
    helpful = (fusion_pred == y_true) & (gaze_pred != y_true)
    return violations.sum(), violations.mean(), helpful.sum()

def evaluate_method(probs, y_true, gaze_pred, gaze_probs):
    predictions = (probs >= 0.5).astype(int)
    
    violations, violation_rate, helpful = compute_no_harm_violations(predictions, gaze_pred, y_true)
    
    return {
        'accuracy': accuracy_score(y_true, predictions),
        'balanced_accuracy': balanced_accuracy_score(y_true, predictions),
        'macro_f1': f1_score(y_true, predictions, average='macro'),
        'auroc': roc_auc_score(y_true, probs),
        'no_harm_violations': violations,
        'no_harm_rate': violation_rate,
        'fusion_helpful': helpful,
        'fusion_helpful_rate': helpful / len(y_true) if len(y_true) > 0 else 0
    }

def run_single_subject_test(test_subject, all_subjects, lambda_safe=0.5, use_text_features=False):
    train_subjects = [s for s in all_subjects if s != test_subject]
    
    eeg_gaze_df = load_eeg_gaze_features(all_subjects)
    train_data = eeg_gaze_df[eeg_gaze_df['subject'].isin(train_subjects)]
    test_data = eeg_gaze_df[eeg_gaze_df['subject'] == test_subject]
    
    if len(test_data) == 0:
        print(f"警告: {test_subject} 测试集为空，使用 train-validation split")
        combined_data = eeg_gaze_df[eeg_gaze_df['subject'].isin(all_subjects)]
        if len(combined_data) == 0:
            print(f"错误: {test_subject} 没有数据")
            return None, None, None
        train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42, stratify=combined_data['label'])
    
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
    
    text_features_train = X_conf_train
    text_features_test = X_conf_test
    
    fusion_features_train = build_fusion_features(experts_train, use_text_features, text_features_train)
    fusion_features_test = build_fusion_features(experts_test, use_text_features, text_features_test)
    
    X_train_fusion = np.array([np.concatenate([f['probs'], f['confidences'], [f['prob_variance'], f['avg_confidence']]]) 
                               for f in fusion_features_train])
    X_test_fusion = np.array([np.concatenate([f['probs'], f['confidences'], [f['prob_variance'], f['avg_confidence']]]) 
                              for f in fusion_features_test])
    
    if use_text_features:
        X_train_fusion = np.hstack([X_train_fusion, text_features_train])
        X_test_fusion = np.hstack([X_test_fusion, text_features_test])
    
    gaze_probs = experts_test['E1_gaze_raw']['probabilities']
    gaze_pred = experts_test['E1_gaze_raw']['predictions']
    
    results = []
    results.append({'subject': test_subject, 'method': 'gaze_raw', **evaluate_method(gaze_probs, y_test, gaze_pred, gaze_probs)})
    
    best_exp = experts_test['E6_concat_residual']
    results.append({'subject': test_subject, 'method': 'best_single_expert', **evaluate_method(best_exp['probabilities'], y_test, gaze_pred, gaze_probs)})
    
    concat_raw_probs = experts_test['E5_concat_raw']['probabilities']
    results.append({'subject': test_subject, 'method': 'concat_raw', **evaluate_method(concat_raw_probs, y_test, gaze_pred, gaze_probs)})
    
    concat_residual_probs = experts_test['E6_concat_residual']['probabilities']
    results.append({'subject': test_subject, 'method': 'concat_residual', **evaluate_method(concat_residual_probs, y_test, gaze_pred, gaze_probs)})
    
    avg_probs = uniform_average_fusion(experts_test)
    results.append({'subject': test_subject, 'method': 'uniform_average', **evaluate_method(avg_probs, y_test, gaze_pred, gaze_probs)})
    
    softmax_probs, softmax_weights = learned_softmax_fusion(X_train_fusion, y_train, X_test_fusion, experts_test)
    res = evaluate_method(softmax_probs, y_test, gaze_pred, gaze_probs)
    res['subject'] = test_subject
    res['method'] = 'learned_softmax_fusion'
    res['avg_weights'] = np.mean(softmax_weights, axis=0).tolist()
    results.append(res)
    
    final_probs, alphas = gaze_anchored_residual_fusion_with_reg(
        X_train_fusion, y_train, X_test_fusion, experts_train, experts_test, lambda_safe
    )
    res = evaluate_method(final_probs, y_test, gaze_pred, gaze_probs)
    res['subject'] = test_subject
    res['method'] = f'gaze_anchored_lambda_{lambda_safe}'
    res['avg_alpha'] = np.mean(alphas, axis=0).tolist() if len(alphas) > 0 else []
    results.append(res)
    
    experts_results = []
    for key in experts_test.keys():
        exp_res = evaluate_method(experts_test[key]['probabilities'], y_test, gaze_pred, gaze_probs)
        exp_res['subject'] = test_subject
        exp_res['expert'] = key
        experts_results.append(exp_res)
    
    return pd.DataFrame(results), pd.DataFrame(experts_results), experts_test

def main():
    output_dir = "results/safe_cirl_fuse_stage_b"
    os.makedirs(output_dir, exist_ok=True)
    
    held_out_subjects = ["YHS", "YIS", "YSD", "YRK", "YFR"]
    
    all_subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 
                    'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
    
    lambda_safe = 0.5
    
    all_results = []
    all_expert_results = []
    
    for test_subject in held_out_subjects:
        print(f"处理 subject: {test_subject}")
        results, expert_results, _ = run_single_subject_test(test_subject, all_subjects, lambda_safe, use_text_features=False)
        
        if results is not None:
            all_results.append(results)
            all_expert_results.append(expert_results)
    
    results_df = pd.concat(all_results, ignore_index=True)
    expert_results_df = pd.concat(all_expert_results, ignore_index=True)
    
    results_df.to_csv(os.path.join(output_dir, "stage_b_results.csv"), index=False)
    expert_results_df.to_csv(os.path.join(output_dir, "expert_results.csv"), index=False)
    
    subjectwise_df = results_df.groupby('subject').apply(
        lambda x: x.set_index('method')['macro_f1'].to_dict()
    ).apply(pd.Series).reset_index()
    subjectwise_df.to_csv(os.path.join(output_dir, "stage_b_subjectwise.csv"), index=False)
    
    no_harm_df = results_df[['subject', 'method', 'no_harm_violations', 'no_harm_rate', 'fusion_helpful']]
    no_harm_df.to_csv(os.path.join(output_dir, "no_harm_metrics.csv"), index=False)
    
    gaze_raw_f1 = results_df[results_df['method'] == 'gaze_raw']['macro_f1'].mean()
    best_single_f1 = results_df[results_df['method'] == 'best_single_expert']['macro_f1'].mean()
    best_fusion_f1 = results_df[results_df['method'].str.contains('gaze_anchored')]['macro_f1'].mean()
    avg_no_harm_rate = results_df[results_df['method'].str.contains('gaze_anchored')]['no_harm_rate'].mean()
    
    with open(os.path.join(output_dir, "stage_b_summary.md"), 'w', encoding='utf-8') as f:
        f.write("# SAFE-CIRL-Fuse Stage B Summary\n\n")
        f.write("## Key Results (Average across 5 subjects)\n\n")
        f.write(f"- gaze_raw Macro-F1: {gaze_raw_f1:.4f}\n")
        f.write(f"- best_single_expert Macro-F1: {best_single_f1:.4f}\n")
        f.write(f"- gaze_anchored_lambda_{lambda_safe} Macro-F1: {best_fusion_f1:.4f}\n")
        f.write(f"- Average no-harm rate: {avg_no_harm_rate:.4f}\n")
        f.write(f"- Improvement over gaze_raw: {best_fusion_f1 - gaze_raw_f1:.4f}\n")
        f.write(f"- Gap to best_single_expert: {best_single_f1 - best_fusion_f1:.4f}\n\n")
        
        f.write("## Subject-wise Results\n\n")
        f.write("| Subject | gaze_raw | best_single | gaze_anchored |\n")
        f.write("|---------|----------|-------------|----------------|\n")
        for subject in held_out_subjects:
            subj_data = results_df[results_df['subject'] == subject]
            gaze = subj_data[subj_data['method'] == 'gaze_raw']['macro_f1'].values[0]
            best = subj_data[subj_data['method'] == 'best_single_expert']['macro_f1'].values[0]
            fusion = subj_data[subj_data['method'] == f'gaze_anchored_lambda_{lambda_safe}']['macro_f1'].values[0]
            f.write(f"| {subject} | {gaze:.4f} | {best:.4f} | {fusion:.4f} |\n")
        
        f.write("\n## Success Criteria for Full LOSO\n\n")
        f.write("1. fair full fusion > gaze_raw + 1%: ")
        f.write(f"{'✅ PASS' if best_fusion_f1 > gaze_raw_f1 + 0.01 else '❌ FAIL'}\n\n")
        f.write("2. fair full fusion >= best_single_expert - 1%: ")
        gap = best_single_f1 - best_fusion_f1
        f.write(f"{'✅ PASS' if gap <= 0.01 else '❌ FAIL'} (gap={gap:.4f})\n\n")
        f.write("3. no-harm rate <= 14.17%: ")
        f.write(f"{'✅ PASS' if avg_no_harm_rate <= 0.1417 else '❌ FAIL'} ({avg_no_harm_rate:.4f})\n\n")
        f.write("4. At least 4/5 subjects >= gaze_raw: ")
        fusion_f1 = results_df[results_df['method'] == f'gaze_anchored_lambda_{lambda_safe}']['macro_f1'].values
        gaze_f1 = results_df[results_df['method'] == 'gaze_raw']['macro_f1'].values
        better_count = sum(fusion_f1 >= gaze_f1)
        f.write(f"{'✅ PASS' if better_count >= 4 else '❌ FAIL'} ({better_count}/5)\n\n")
        f.write("5. Full LOSO recommendation: ")
        if best_fusion_f1 > gaze_raw_f1 + 0.01 and gap <= 0.01 and avg_no_harm_rate <= 0.1417 and better_count >= 4:
            f.write("✅ Recommended")
        else:
            f.write("❌ Not recommended")
        f.write("\n")
    
    print(f"\nSAFE-CIRL-Fuse Stage B 完成！输出目录: {output_dir}")

if __name__ == "__main__":
    main()
