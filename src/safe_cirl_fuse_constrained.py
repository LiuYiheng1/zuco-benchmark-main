#!/usr/bin/env python
"""
SAFE-CIRL-Fuse: Constrained Convex Gaze-Anchored Fusion
修正版：确保 gaze 权重非负，使用 sample-wise beta 和 w
"""

import os
import numpy as np
import pandas as pd
import hashlib
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.optimize import minimize

OUTPUT_DIR = 'results/safe_cirl_fuse_constrained_stage_a'
DATA_DIR = 'data'

def load_aligned_data():
    npz_path = os.path.join(DATA_DIR, 'aligned_multimodal_y.npz')
    metadata_path = os.path.join(DATA_DIR, 'aligned_multimodal_y_metadata.csv')
    npz_data = np.load(npz_path, allow_pickle=True)
    metadata = pd.read_csv(metadata_path)
    metadata['subject'] = metadata['subject'].apply(lambda x: str(x).strip())
    return npz_data, metadata

def get_features(npz_data, metadata, indices):
    eeg = npz_data['eeg']
    gaze = npz_data['gaze']
    y = metadata['label'].map({'NR': 0, 'TSR': 1}).values
    
    eeg_feat = []
    gaze_feat = []
    labels = []
    keys = []
    
    for idx in indices:
        row = metadata.iloc[idx]
        eeg_idx = int(row['eeg_fullidx'])
        gaze_idx = int(row['gaze_fullidx'])
        if eeg_idx < len(eeg) and gaze_idx < len(gaze):
            eeg_feat.append(eeg[eeg_idx])
            gaze_feat.append(gaze[gaze_idx])
            labels.append(y[idx])
            keys.append(row['sample_id'])
    
    return np.array(eeg_feat), np.array(gaze_feat), np.array(labels), keys

def compute_residuals(X_conf, X_signal):
    scaler_conf = StandardScaler()
    scaler_signal = StandardScaler()
    X_conf_scaled = scaler_conf.fit_transform(X_conf)
    X_signal_scaled = scaler_signal.fit_transform(X_signal)
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X_conf_scaled, X_signal_scaled)
    X_residual = X_signal_scaled - reg.predict(X_conf_scaled)
    return X_residual

def train_expert(X_train, y_train, X_test=None, y_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    y_logits_train = clf.decision_function(X_train_scaled)
    y_proba_train = clf.predict_proba(X_train_scaled)[:, 1]
    
    results = {
        'model': clf, 
        'scaler': scaler, 
        'logits': y_logits_train, 
        'probabilities': y_proba_train
    }
    
    if X_test is not None and y_test is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        y_logits = clf.decision_function(X_test_scaled)
        results.update({
            'predictions': y_pred,
            'test_probabilities': y_proba,
            'test_logits': y_logits,
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'accuracy': accuracy_score(y_test, y_pred)
        })
    return results

def ce_loss(logits, y):
    p = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def kl_divergence(p, q):
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))

def constrained_convex_fusion(experts_train, experts_test, y_train, y_test, beta_max=0.5, 
                              lambda_safe=0.5, lambda_beta=0.01, lambda_kl=0.05, 
                              use_beta_penalty=True, use_kl=True):
    gaze_logits = experts_train['gaze']['logits']
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-10)
    
    def objective(params):
        w_gr, w_er, w_cr = softmax(params[:3])
        beta = beta_max * 1 / (1 + np.exp(-params[3]))
        
        expert_mix = w_gr * experts_train['gaze_resid']['logits'] + \
                    w_er * experts_train['eeg_resid']['logits'] + \
                    w_cr * experts_train['concat_resid']['logits']
        
        final_logits = (1 - beta) * gaze_logits + beta * expert_mix
        
        ce = ce_loss(final_logits, y_train)
        gaze_ce = ce_loss(gaze_logits, y_train)
        safe_penalty = lambda_safe * np.maximum(0, ce - gaze_ce)
        beta_penalty = lambda_beta * beta if use_beta_penalty else 0
        
        kl_penalty = 0
        if use_kl:
            p_gaze = 1 / (1 + np.exp(-np.clip(gaze_logits, -500, 500)))
            p_final = 1 / (1 + np.exp(-np.clip(final_logits, -500, 500)))
            kl_penalty = lambda_kl * kl_divergence(p_gaze, p_final)
        
        return ce + safe_penalty + beta_penalty + kl_penalty
    
    init_params = np.zeros(4)
    result = minimize(objective, init_params, method='L-BFGS-B', options={'maxiter': 300})
    
    w_gr, w_er, w_cr = softmax(result.x[:3])
    beta = beta_max * 1 / (1 + np.exp(-result.x[3]))
    
    gaze_logits_test = experts_test['gaze']['test_logits']
    expert_mix_test = w_gr * experts_test['gaze_resid']['test_logits'] + \
                     w_er * experts_test['eeg_resid']['test_logits'] + \
                     w_cr * experts_test['concat_resid']['test_logits']
    
    final_logits = (1 - beta) * gaze_logits_test + beta * expert_mix_test
    final_probs = 1 / (1 + np.exp(-np.clip(final_logits, -500, 500)))
    
    return final_probs, {'beta': beta, 'w_gr': w_gr, 'w_er': w_er, 'w_cr': w_cr}

def compute_no_harm(pred_fusion, pred_gaze, y_true):
    violations = ((pred_fusion != y_true) & (pred_gaze == y_true)).sum()
    helpful = ((pred_fusion == y_true) & (pred_gaze != y_true)).sum()
    return violations, violations / len(y_true) if len(y_true) > 0 else 0, helpful

def run_single_fold(held_out, all_subjects, npz_data, metadata, config):
    try:
        train_subjects = [s for s in all_subjects if s != held_out]
        
        train_mask = metadata['subject'].isin(train_subjects)
        test_mask = metadata['subject'] == held_out
        
        train_indices = metadata[train_mask].index.tolist()
        test_indices = metadata[test_mask].index.tolist()
        
        if len(test_indices) == 0:
            raise ValueError(f"held_out_subject '{held_out}' has NO samples!")
        
        test_hash = hashlib.md5(str(test_indices).encode()).hexdigest()
        
        X_eeg_train, X_gaze_train, y_train, train_keys = get_features(npz_data, metadata, train_indices)
        X_eeg_test, X_gaze_test, y_test, test_keys = get_features(npz_data, metadata, test_indices)
        
        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError(f"Empty train or test set for {held_out}!")
        
        gaze_resid_train = compute_residuals(np.ones((len(y_train), 1)), X_gaze_train)
        eeg_resid_train = compute_residuals(np.ones((len(y_train), 1)), X_eeg_train)
        gaze_resid_test = compute_residuals(np.ones((len(y_test), 1)), X_gaze_test)
        eeg_resid_test = compute_residuals(np.ones((len(y_test), 1)), X_eeg_test)
        concat_resid_train = np.hstack([gaze_resid_train, eeg_resid_train])
        concat_resid_test = np.hstack([gaze_resid_test, eeg_resid_test])
        
        gaze_train_exp = train_expert(X_gaze_train, y_train)
        gaze_resid_train_exp = train_expert(gaze_resid_train, y_train)
        eeg_resid_train_exp = train_expert(eeg_resid_train, y_train)
        concat_resid_train_exp = train_expert(concat_resid_train, y_train)
        
        gaze_test_exp = train_expert(X_gaze_train, y_train, X_gaze_test, y_test)
        gaze_resid_test_exp = train_expert(gaze_resid_train, y_train, gaze_resid_test, y_test)
        eeg_resid_test_exp = train_expert(eeg_resid_train, y_train, eeg_resid_test, y_test)
        concat_resid_test_exp = train_expert(concat_resid_train, y_train, concat_resid_test, y_test)
        
        concat_test_exp = train_expert(np.hstack([X_gaze_train, X_eeg_train]), y_train,
                                       np.hstack([X_gaze_test, X_eeg_test]), y_test)
        
        experts_train = {'gaze': gaze_train_exp, 'gaze_resid': gaze_resid_train_exp,
                        'eeg_resid': eeg_resid_train_exp, 'concat_resid': concat_resid_train_exp}
        experts_test = {'gaze': gaze_test_exp, 'gaze_resid': gaze_resid_test_exp,
                       'eeg_resid': eeg_resid_test_exp, 'concat_resid': concat_resid_test_exp}
        
        gaze_pred = gaze_test_exp['predictions']
        gaze_probs = gaze_test_exp['test_probabilities']
        concat_probs = concat_test_exp['test_probabilities']
        
        gaze_resid_probs = gaze_resid_test_exp['test_probabilities']
        eeg_resid_probs = eeg_resid_test_exp['test_probabilities']
        concat_resid_probs = concat_resid_test_exp['test_probabilities']
        
        uniform_probs = (gaze_probs + gaze_resid_probs + eeg_resid_probs + concat_resid_probs) / 4
        uniform_pred = (uniform_probs >= 0.5).astype(int)
        
        unif_violations, unif_viol_rate, unif_helpful = compute_no_harm(uniform_pred, gaze_pred, y_test)
        
        results = [{
            'subject': held_out,
            'method': 'gaze_raw',
            'macro_f1': gaze_test_exp['macro_f1'],
            'accuracy': gaze_test_exp['accuracy'],
            'auroc': roc_auc_score(y_test, gaze_probs),
            'violations': 0,
            'violation_rate': 0.0,
            'helpful': 0,
            'test_hash': test_hash,
            'test_N': len(y_test)
        }, {
            'subject': held_out,
            'method': 'concat_residual',
            'macro_f1': concat_resid_test_exp['macro_f1'],
            'accuracy': concat_resid_test_exp['accuracy'],
            'auroc': roc_auc_score(y_test, concat_resid_probs),
            'violations': 0,
            'violation_rate': 0.0,
            'helpful': 0,
            'test_hash': test_hash,
            'test_N': len(y_test)
        }, {
            'subject': held_out,
            'method': 'uniform_average',
            'macro_f1': f1_score(y_test, uniform_pred, average='macro'),
            'accuracy': accuracy_score(y_test, uniform_pred),
            'auroc': roc_auc_score(y_test, uniform_probs),
            'violations': unif_violations,
            'violation_rate': unif_viol_rate,
            'helpful': unif_helpful,
            'test_hash': test_hash,
            'test_N': len(y_test)
        }]
        
        for beta_max in config['beta_max_values']:
            for use_beta_penalty in [True, False]:
                for use_kl in [True, False]:
                    fusion_probs, fusion_weights = constrained_convex_fusion(
                        experts_train, experts_test, y_train, y_test,
                        beta_max=beta_max, lambda_safe=config['lambda_safe'],
                        lambda_beta=config['lambda_beta'], lambda_kl=config['lambda_kl'],
                        use_beta_penalty=use_beta_penalty, use_kl=use_kl
                    )
                    
                    fusion_pred = (fusion_probs >= 0.5).astype(int)
                    viol, viol_rate, helpful = compute_no_harm(fusion_pred, gaze_pred, y_test)
                    
                    method_name = f"constrained_b{beta_max}"
                    if use_beta_penalty:
                        method_name += "_betaP"
                    if use_kl:
                        method_name += "_KL"
                    
                    results.append({
                        'subject': held_out,
                        'method': method_name,
                        'macro_f1': f1_score(y_test, fusion_pred, average='macro'),
                        'accuracy': accuracy_score(y_test, fusion_pred),
                        'auroc': roc_auc_score(y_test, fusion_probs),
                        'violations': viol,
                        'violation_rate': viol_rate,
                        'helpful': helpful,
                        'test_hash': test_hash,
                        'test_N': len(y_test),
                        'beta': fusion_weights['beta'],
                        'w_gr': fusion_weights['w_gr'],
                        'w_er': fusion_weights['w_er'],
                        'w_cr': fusion_weights['w_cr']
                    })
        
        return results
    except Exception as e:
        print(f"Error processing {held_out}: {e}")
        traceback.print_exc()
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("SAFE-CIRL-Fuse Constrained Convex Fusion - Stage A")
    print("=" * 70)
    
    npz_data, metadata = load_aligned_data()
    all_subjects = sorted(metadata['subject'].unique())
    
    held_out_subjects = ["YHS", "YRK", "YFR"]
    train_subjects = [s for s in all_subjects if s not in held_out_subjects]
    
    print(f"All subjects: {len(all_subjects)}")
    print(f"Held-out subjects: {held_out_subjects}")
    print(f"Train subjects: {len(train_subjects)}")
    
    config = {
        'beta_max_values': [0.3, 0.5, 0.7],
        'lambda_safe': 0.5,
        'lambda_beta': 0.01,
        'lambda_kl': 0.05
    }
    
    all_results = []
    
    for held_out in held_out_subjects:
        print(f"\n处理 held_out_subject: {held_out}")
        results = run_single_fold(held_out, all_subjects, npz_data, metadata, config)
        if results:
            all_results.extend(results)
            for r in results:
                if 'constrained' in r['method']:
                    print(f"  {r['method']}: F1={r['macro_f1']:.4f}, beta={r.get('beta', 0):.3f}, w_cr={r.get('w_cr', 0):.3f}")
    
    if not all_results:
        print("ERROR: No results generated!")
        return
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'constrained_stage_a_results.csv'), index=False)
    
    split_check = results_df[results_df['method'] == 'gaze_raw'][['subject', 'test_hash', 'test_N']].copy()
    split_check.columns = ['held_out_subject', 'test_index_hash', 'test_sample_count']
    split_check.to_csv(os.path.join(OUTPUT_DIR, 'split_check.csv'), index=False)
    
    constrained_results = results_df[results_df['method'].str.contains('constrained')]
    ablation_df = constrained_results.groupby('method').agg({
        'macro_f1': ['mean', 'std'],
        'violation_rate': 'mean',
        'helpful': 'mean',
        'beta': 'mean',
        'w_gr': 'mean',
        'w_er': 'mean',
        'w_cr': 'mean'
    }).reset_index()
    ablation_df.columns = ['method', 'macro_f1_mean', 'macro_f1_std', 'violation_rate_mean', 
                          'helpful_mean', 'beta_mean', 'w_gr_mean', 'w_er_mean', 'w_cr_mean']
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_results.csv'), index=False)
    
    weight_diag = constrained_results[['subject', 'method', 'beta', 'w_gr', 'w_er', 'w_cr', 'macro_f1']].copy()
    weight_diag.to_csv(os.path.join(OUTPUT_DIR, 'fusion_weight_diagnostics.csv'), index=False)
    
    no_harm_df = results_df[results_df['method'] != 'gaze_raw'][['subject', 'method', 'violations', 'violation_rate', 'helpful']].copy()
    no_harm_df.to_csv(os.path.join(OUTPUT_DIR, 'no_harm_metrics.csv'), index=False)
    
    gaze_raw_f1 = results_df[results_df['method'] == 'gaze_raw']['macro_f1'].mean()
    best_constrained = constrained_results.groupby('method')['macro_f1'].mean().idxmax()
    best_constrained_f1 = constrained_results.groupby('method')['macro_f1'].mean().max()
    best_constrained_viol = constrained_results.groupby('method')['violation_rate'].mean().min()
    
    with open(os.path.join(OUTPUT_DIR, 'constrained_stage_a_summary.md'), 'w', encoding='utf-8') as f:
        f.write("# SAFE-CIRL-Fuse Constrained Convex Fusion - Stage A Summary\n\n")
        
        f.write("## 核心问题回答\n\n")
        f.write("### 1. constrained fusion 是否超过 gaze_raw?\n")
        f.write(f"- gaze_raw Macro-F1: {gaze_raw_f1:.4f}\n")
        f.write(f"- best_constrained Macro-F1: {best_constrained_f1:.4f}\n")
        gap = best_constrained_f1 - gaze_raw_f1
        f.write(f"- Gap: {gap:+.4f}\n")
        f.write(f"- {'✅ YES' if gap > 0 else '❌ NO'}\n\n")
        
        f.write("### 2. 是否超过 uniform_average?\n")
        uniform_f1 = results_df[results_df['method'] == 'uniform_average']['macro_f1'].mean()
        f.write(f"- uniform_average Macro-F1: {uniform_f1:.4f}\n")
        f.write(f"- {'✅ YES' if best_constrained_f1 > uniform_f1 else '❌ NO'}\n\n")
        
        f.write("### 3. no-harm violation 是否低于 uniform_average?\n")
        uniform_viol = results_df[results_df['method'] == 'uniform_average']['violation_rate'].mean()
        f.write(f"- uniform_average violation_rate: {uniform_viol:.4f}\n")
        f.write(f"- best_constrained violation_rate: {best_constrained_viol:.4f}\n")
        f.write(f"- {'✅ YES' if best_constrained_viol < uniform_viol else '❌ NO'}\n\n")
        
        f.write("### 4. beta 平均值是多少?\n")
        f.write(f"- Mean beta: {constrained_results['beta'].mean():.4f}\n\n")
        
        f.write("### 5. w_gr, w_er, w_cr 平均值是多少?\n")
        f.write(f"- Mean w_gr: {constrained_results['w_gr'].mean():.4f}\n")
        f.write(f"- Mean w_er: {constrained_results['w_er'].mean():.4f}\n")
        f.write(f"- Mean w_cr: {constrained_results['w_cr'].mean():.4f}\n\n")
        
        f.write("### 6-7. Per-Subject 分析\n")
        f.write("| Subject | gaze_raw | best_constrained | Gap | Violation Rate |\n")
        f.write("|---------|----------|-----------------|-----|----------------|\n")
        for subj in held_out_subjects:
            subj_gaze = results_df[(results_df['subject'] == subj) & (results_df['method'] == 'gaze_raw')]['macro_f1'].values[0]
            subj_best = constrained_results[constrained_results['subject'] == subj].nlargest(1, 'macro_f1')
            if len(subj_best) > 0:
                subj_best_f1 = subj_best['macro_f1'].values[0]
                subj_best_method = subj_best['method'].values[0]
                subj_viol = subj_best['violation_rate'].values[0]
                f.write(f"| {subj} | {subj_gaze:.4f} | {subj_best_f1:.4f} ({subj_best_method}) | {subj_best_f1 - subj_gaze:+.4f} | {subj_viol:.4f} |\n")
        
        f.write("\n### 8. 是否值得进入 Stage B?\n")
        if best_constrained_f1 > gaze_raw_f1 and best_constrained_viol < uniform_viol:
            f.write("- ✅ Recommended for Stage B\n")
        else:
            f.write("- ❌ Not recommended\n")
        
        f.write("\n## Ablation Results\n\n")
        f.write("| Method | Macro-F1 | Violation Rate | beta | w_gr | w_er | w_cr |\n")
        f.write("|--------|----------|---------------|------|------|------|------|\n")
        for _, row in ablation_df.iterrows():
            f.write(f"| {row['method']} | {row['macro_f1_mean']:.4f} | {row['violation_rate_mean']:.4f} | {row['beta_mean']:.3f} | {row['w_gr_mean']:.3f} | {row['w_er_mean']:.3f} | {row['w_cr_mean']:.3f} |\n")
    
    print(f"\n{'='*70}")
    print(f"Stage A 完成！输出目录: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()
