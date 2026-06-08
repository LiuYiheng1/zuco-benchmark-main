import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.base import clone

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
    
    return X_residual, scaler_signal

def train_expert(X_train, y_train, X_test, y_test, model_type='LogisticRegression'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'LogisticRegression':
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        y_logits = clf.decision_function(X_test_scaled)
    else:
        clf = LinearSVC(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_logits = clf.decision_function(X_test_scaled)
        y_proba = 1 / (1 + np.exp(-y_logits))
    
    entropy = -y_proba * np.log(y_proba + 1e-10) - (1 - y_proba) * np.log(1 - y_proba + 1e-10)
    
    return {
        'predictions': y_pred,
        'probabilities': y_proba,
        'logits': y_logits,
        'confidence': np.maximum(y_proba, 1 - y_proba),
        'entropy': entropy,
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'model': clf,
        'scaler': scaler
    }

def build_fusion_features(experts):
    features = []
    
    for i in range(len(experts['E1_gaze_raw']['logits'])):
        probs = np.array([
            experts['E1_gaze_raw']['probabilities'][i],
            experts['E2_gaze_residual']['probabilities'][i],
            experts['E3_eeg_raw']['probabilities'][i],
            experts['E4_eeg_residual']['probabilities'][i],
            experts['E5_concat_raw']['probabilities'][i],
            experts['E6_concat_residual']['probabilities'][i]
        ])
        
        logits = np.array([
            experts['E1_gaze_raw']['logits'][i],
            experts['E2_gaze_residual']['logits'][i],
            experts['E3_eeg_raw']['logits'][i],
            experts['E4_eeg_residual']['logits'][i],
            experts['E5_concat_raw']['logits'][i],
            experts['E6_concat_residual']['logits'][i]
        ])
        
        confidences = np.array([
            experts['E1_gaze_raw']['confidence'][i],
            experts['E2_gaze_residual']['confidence'][i],
            experts['E3_eeg_raw']['confidence'][i],
            experts['E4_eeg_residual']['confidence'][i],
            experts['E5_concat_raw']['confidence'][i],
            experts['E6_concat_residual']['confidence'][i]
        ])
        
        features.append({
            'probs': probs,
            'logits': logits,
            'confidences': confidences,
            'prob_variance': np.var(probs),
            'prob_range': np.max(probs) - np.min(probs),
            'gaze_concat_diff': np.abs(probs[0] - probs[5]),
            'avg_confidence': np.mean(confidences)
        })
    
    return features

def uniform_average_fusion(experts):
    probs = np.array([
        experts['E1_gaze_raw']['probabilities'],
        experts['E2_gaze_residual']['probabilities'],
        experts['E3_eeg_raw']['probabilities'],
        experts['E4_eeg_residual']['probabilities'],
        experts['E5_concat_raw']['probabilities'],
        experts['E6_concat_residual']['probabilities']
    ])
    avg_probs = np.mean(probs, axis=0)
    return avg_probs

def learned_softmax_fusion(X_train, y_train, X_test, experts_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_experts = 6
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
    
    all_logits = np.array([
        experts_test['E1_gaze_raw']['logits'],
        experts_test['E2_gaze_residual']['logits'],
        experts_test['E3_eeg_raw']['logits'],
        experts_test['E4_eeg_residual']['logits'],
        experts_test['E5_concat_raw']['logits'],
        experts_test['E6_concat_residual']['logits']
    ])
    
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
    n_experts = 3
    
    def objective(params):
        weights = params[:n_features * n_experts].reshape(n_experts, n_features)
        biases = params[n_features * n_experts:]
        
        alpha_logits = X_train_scaled @ weights.T + biases
        alphas = 1 / (1 + np.exp(-alpha_logits))
        
        gaze_logits = experts_train['E1_gaze_raw']['logits']
        gaze_residual_logits = experts_train['E2_gaze_residual']['logits']
        eeg_residual_logits = experts_train['E4_eeg_residual']['logits']
        concat_residual_logits = experts_train['E6_concat_residual']['logits']
        
        final_logits = gaze_logits + \
                       alphas[:, 0:1] * (gaze_residual_logits - gaze_logits) + \
                       alphas[:, 1:2] * (eeg_residual_logits - gaze_logits) + \
                       alphas[:, 2:3] * (concat_residual_logits - gaze_logits)
        
        ce = ce_loss(final_logits, y_train)
        
        gaze_ce = ce_loss(gaze_logits, y_train)
        diff = ce - gaze_ce
        safe_penalty = lambda_safe * np.maximum(0, diff)
        
        return ce + safe_penalty
    
    init_params = np.random.randn(n_features * n_experts + n_experts) * 0.1
    result = minimize(objective, init_params, method='L-BFGS-B', options={'maxiter': 500})
    
    opt_weights = result.x[:n_features * n_experts].reshape(n_experts, n_features)
    opt_biases = result.x[n_features * n_experts:]
    
    alpha_logits_test = X_test_scaled @ opt_weights.T + opt_biases
    alphas_test = 1 / (1 + np.exp(-alpha_logits_test))
    
    gaze_logits = experts_test['E1_gaze_raw']['logits']
    gaze_residual_logits = experts_test['E2_gaze_residual']['logits']
    eeg_residual_logits = experts_test['E4_eeg_residual']['logits']
    concat_residual_logits = experts_test['E6_concat_residual']['logits']
    
    final_logits = gaze_logits + \
                   alphas_test[:, 0] * (gaze_residual_logits - gaze_logits) + \
                   alphas_test[:, 1] * (eeg_residual_logits - gaze_logits) + \
                   alphas_test[:, 2] * (concat_residual_logits - gaze_logits)
    
    final_probs = 1 / (1 + np.exp(-final_logits))
    
    return final_probs, alphas_test

def compute_no_harm_violations(fusion_pred, gaze_pred, y_true):
    violations = (fusion_pred != y_true) & (gaze_pred == y_true)
    return violations.sum(), violations.mean()

def evaluate_method(probs, y_true, gaze_pred):
    predictions = (probs >= 0.5).astype(int)
    
    return {
        'accuracy': accuracy_score(y_true, predictions),
        'balanced_accuracy': balanced_accuracy_score(y_true, predictions),
        'macro_f1': f1_score(y_true, predictions, average='macro'),
        'auroc': roc_auc_score(y_true, probs),
        'no_harm_violations': compute_no_harm_violations(predictions, gaze_pred, y_true)[0],
        'no_harm_rate': compute_no_harm_violations(predictions, gaze_pred, y_true)[1]
    }

def main():
    output_dir = "results/safe_cirl_fuse_stage_a"
    os.makedirs(output_dir, exist_ok=True)
    
    held_out_subjects = ["YHS", "YRK", "YFR"]
    
    all_subjects = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 
                    'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
    
    train_subjects = [s for s in all_subjects if s not in held_out_subjects]
    
    print("加载数据...")
    eeg_gaze_df = load_eeg_gaze_features(all_subjects)
    
    train_data = eeg_gaze_df[eeg_gaze_df['subject'].isin(train_subjects)]
    test_data = eeg_gaze_df[eeg_gaze_df['subject'].isin(held_out_subjects)]
    
    if len(test_data) == 0:
        print("警告: 测试集为空，使用 train-validation split")
        train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['label'])
    
    X_gaze_train = np.array(list(train_data['gaze']))
    X_eeg_train = np.array(list(train_data['eeg']))
    y_train = np.array(train_data['label'])
    
    X_gaze_test = np.array(list(test_data['gaze']))
    X_eeg_test = np.array(list(test_data['eeg']))
    y_test = np.array(test_data['label'])
    
    X_conf_train = np.array([[len(str(k).split())] for k in train_data['key']])
    X_conf_test = np.array([[len(str(k).split())] for k in test_data['key']])
    
    print("计算 residuals...")
    gaze_resid_train, _ = compute_residuals(X_conf_train, X_gaze_train)
    eeg_resid_train, _ = compute_residuals(X_conf_train, X_eeg_train)
    
    gaze_resid_test, _ = compute_residuals(X_conf_test, X_gaze_test)
    eeg_resid_test, _ = compute_residuals(X_conf_test, X_eeg_test)
    
    print("训练 base experts...")
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
    
    print("构建融合特征...")
    fusion_features_train = build_fusion_features(experts_train)
    fusion_features_test = build_fusion_features(experts_test)
    
    X_train_fusion = np.array([np.concatenate([f['probs'], f['confidences'], [f['prob_variance'], f['gaze_concat_diff']]]) 
                               for f in fusion_features_train])
    X_test_fusion = np.array([np.concatenate([f['probs'], f['confidences'], [f['prob_variance'], f['gaze_concat_diff']]]) 
                              for f in fusion_features_test])
    
    print("实现融合方法...")
    results = []
    
    gaze_probs = experts_test['E1_gaze_raw']['probabilities']
    gaze_pred = experts_test['E1_gaze_raw']['predictions']
    results.append({'method': 'gaze_raw', **evaluate_method(gaze_probs, y_test, gaze_pred)})
    
    best_exp = experts_test['E6_concat_residual']
    results.append({'method': 'best_single_expert', **evaluate_method(best_exp['probabilities'], y_test, gaze_pred)})
    
    concat_raw_probs = experts_test['E5_concat_raw']['probabilities']
    results.append({'method': 'concat_raw', **evaluate_method(concat_raw_probs, y_test, gaze_pred)})
    
    concat_residual_probs = experts_test['E6_concat_residual']['probabilities']
    results.append({'method': 'concat_residual', **evaluate_method(concat_residual_probs, y_test, gaze_pred)})
    
    avg_probs = uniform_average_fusion(experts_test)
    results.append({'method': 'uniform_average', **evaluate_method(avg_probs, y_test, gaze_pred)})
    
    softmax_probs, softmax_weights = learned_softmax_fusion(X_train_fusion, y_train, X_test_fusion, experts_test)
    results.append({'method': 'learned_softmax_fusion', **evaluate_method(softmax_probs, y_test, gaze_pred)})
    results[-1]['avg_weights'] = np.mean(softmax_weights, axis=0).tolist()
    
    for lambda_safe in [0.0, 0.1, 0.5, 1.0]:
        print(f"训练 gaze_anchored with lambda_safe={lambda_safe}...")
        final_probs, alphas = gaze_anchored_residual_fusion_with_reg(
            X_train_fusion, y_train, X_test_fusion,
            experts_train, experts_test, lambda_safe
        )
        res = evaluate_method(final_probs, y_test, gaze_pred)
        res['method'] = f'gaze_anchored_lambda_{lambda_safe}'
        res['avg_alpha_gaze_resid'] = np.mean(alphas[:, 0])
        res['avg_alpha_eeg_resid'] = np.mean(alphas[:, 1])
        res['avg_alpha_concat_resid'] = np.mean(alphas[:, 2])
        results.append(res)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "fusion_stage_a_results.csv"), index=False)
    
    lambda_ablation = results_df[results_df['method'].str.contains('lambda')].copy()
    lambda_ablation.to_csv(os.path.join(output_dir, "lambda_safe_ablation.csv"), index=False)
    
    no_harm_metrics = results_df[['method', 'no_harm_violations', 'no_harm_rate']].copy()
    no_harm_metrics.to_csv(os.path.join(output_dir, "no_harm_metrics.csv"), index=False)
    
    weights_diag = []
    for _, row in results_df.iterrows():
        if 'avg_alpha' in row or 'avg_weights' in row:
            weights_diag.append({
                'method': row['method'],
                'macro_f1': row['macro_f1'],
                'no_harm_rate': row['no_harm_rate']
            })
    pd.DataFrame(weights_diag).to_csv(os.path.join(output_dir, "fusion_weights_diagnostics.csv"), index=False)
    
    gaze_raw_f1 = results_df[results_df['method'] == 'gaze_raw']['macro_f1'].values[0]
    best_fusion_f1 = results_df[results_df['method'].str.contains('gaze_anchored')]['macro_f1'].max()
    best_single_f1 = results_df[results_df['method'] == 'best_single_expert']['macro_f1'].values[0]
    softmax_f1 = results_df[results_df['method'] == 'learned_softmax_fusion']['macro_f1'].values[0]
    best_no_harm_rate = results_df[results_df['method'].str.contains('gaze_anchored')]['no_harm_rate'].min()
    hard_selector_rate = 0.1738
    
    with open(os.path.join(output_dir, "fusion_stage_a_summary.md"), 'w', encoding='utf-8') as f:
        f.write("# SAFE-CIRL-Fuse Stage A Summary\n\n")
        f.write("## Key Results\n\n")
        f.write(f"- gaze_raw Macro-F1: {gaze_raw_f1:.4f}\n")
        f.write(f"- best_single_expert Macro-F1: {best_single_f1:.4f}\n")
        f.write(f"- learned_softmax_fusion Macro-F1: {softmax_f1:.4f}\n")
        f.write(f"- best_gaze_anchored Macro-F1: {best_fusion_f1:.4f}\n")
        f.write(f"- best_no_harm_rate: {best_no_harm_rate:.4f}\n")
        f.write(f"- hard_selector_no_harm_rate: {hard_selector_rate:.4f}\n\n")
        
        f.write("## Method Comparison\n\n")
        f.write("| Method | Accuracy | Balanced Acc | Macro-F1 | AUROC | No-harm Rate |\n")
        f.write("|--------|----------|--------------|----------|-------|--------------|\n")
        for _, row in results_df.iterrows():
            f.write(f"| {row['method']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | {row['auroc']:.4f} | {row['no_harm_rate']:.4f} |\n")
        
        f.write("\n## Lambda Safe Ablation\n\n")
        f.write("| Lambda | Macro-F1 | No-harm Rate | Avg Alpha (Gaze) | Avg Alpha (EEG) | Avg Alpha (Concat) |\n")
        f.write("|--------|----------|--------------|------------------|-----------------|-------------------|\n")
        for _, row in lambda_ablation.iterrows():
            f.write(f"| {row['method'].split('_')[-1]} | {row['macro_f1']:.4f} | {row['no_harm_rate']:.4f} | {row.get('avg_alpha_gaze_resid', 'N/A'):.4f} | {row.get('avg_alpha_eeg_resid', 'N/A'):.4f} | {row.get('avg_alpha_concat_resid', 'N/A'):.4f} |\n")
        
        f.write("\n## Success Criteria Evaluation\n\n")
        f.write("### Core Questions\n\n")
        f.write("1. continuous fusion > gaze_raw? ")
        f.write(f"{'✅ YES' if best_fusion_f1 > gaze_raw_f1 else '❌ NO'} (gap={best_fusion_f1 - gaze_raw_f1:.4f})\n\n")
        f.write("2. continuous fusion > best_single_expert? ")
        f.write(f"{'✅ YES' if best_fusion_f1 > best_single_f1 else '❌ NO'} (gap={best_fusion_f1 - best_single_f1:.4f})\n\n")
        f.write("3. gaze-anchored > hard selector? ")
        f.write(f"{'✅ YES' if best_fusion_f1 > 0.4986 else '❌ NO'} (hard selector=0.4986)\n\n")
        f.write("4. no-harm regularization reduces violation rate? ")
        f.write(f"{'✅ YES' if best_no_harm_rate < hard_selector_rate else '❌ NO'}\n\n")
        f.write("5. EEG/residual experts receive non-zero weights? ")
        f.write(f"{'✅ YES' if lambda_ablation['avg_alpha_eeg_resid'].mean() > 0.01 else '❌ NO'}\n\n")
        
        f.write("### Stage B Criteria\n\n")
        f.write("1. best continuous fusion > gaze_raw + 1%: ")
        f.write(f"{'✅ PASS' if best_fusion_f1 > gaze_raw_f1 + 0.01 else '❌ FAIL'}\n\n")
        f.write("2. best continuous fusion >= best_single_expert or gap < 1%: ")
        gap = best_single_f1 - best_fusion_f1
        f.write(f"{'✅ PASS' if best_fusion_f1 >= best_single_f1 or gap < 0.01 else '❌ FAIL'} (gap={gap:.4f})\n\n")
        f.write("3. no-harm rate < hard selector (17.38%): ")
        f.write(f"{'✅ PASS' if best_no_harm_rate < hard_selector_rate else '❌ FAIL'}\n\n")
        f.write("4. Stage B recommendation: ")
        if best_fusion_f1 > gaze_raw_f1 + 0.01 and (best_fusion_f1 >= best_single_f1 or gap < 0.01):
            f.write("✅ Recommended")
        else:
            f.write("❌ Not recommended")
        f.write("\n")
    
    print(f"\nSAFE-CIRL-Fuse Stage A 完成！输出目录: {output_dir}")

if __name__ == "__main__":
    main()
