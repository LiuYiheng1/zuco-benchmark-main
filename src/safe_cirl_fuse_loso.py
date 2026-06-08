#!/usr/bin/env python
"""
SAFE-CIRL-Fuse: Gaze-Anchored Continuous Residual Evidence Fusion
Based on audit_loso_linearsvc.py approach
"""

import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

OUTPUT_DIR = 'results/safe_cirl_fuse_loso'
DATA_DIR = 'data'

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

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

def ce_loss(logits, y):
    p = 1 / (1 + np.exp(-logits))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def train_expert(X_train, y_train, X_test=None, y_test=None, model_type='logistic'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if model_type == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        clf = LinearSVC(max_iter=1000, random_state=42, dual=False)
    
    clf.fit(X_train_scaled, y_train)
    
    results = {'model': clf, 'scaler': scaler}
    
    if X_test is not None and y_test is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)
        if hasattr(clf, 'predict_proba'):
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = 1 / (1 + np.exp(-clf.decision_function(X_test_scaled)))
        y_logits = clf.decision_function(X_test_scaled)
        
        results.update({
            'predictions': y_pred,
            'probabilities': y_proba,
            'logits': y_logits,
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'auroc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        })
    
    return results

def gaze_anchored_residual_fusion(experts_train, experts_test, X_train_conf, X_test_conf, 
                                   X_gaze_train, X_eeg_train, X_gaze_test, X_eeg_test,
                                   y_train, y_test, lambda_safe=0.5):
    gaze_resid_train = compute_residuals(X_train_conf, X_gaze_train)
    eeg_resid_train = compute_residuals(X_train_conf, X_eeg_train)
    gaze_resid_test = compute_residuals(X_test_conf, X_gaze_test)
    eeg_resid_test = compute_residuals(X_test_conf, X_eeg_test)
    
    concat_resid_train = np.hstack([gaze_resid_train, eeg_resid_train])
    concat_resid_test = np.hstack([gaze_resid_test, eeg_resid_test])
    
    gaze_resid_train_exp = train_expert(gaze_resid_train, y_train, gaze_resid_train, y_train)
    eeg_resid_train_exp = train_expert(eeg_resid_train, y_train, eeg_resid_train, y_train)
    concat_resid_train_exp = train_expert(concat_resid_train, y_train, concat_resid_train, y_train)
    
    gaze_resid_test_exp = train_expert(gaze_resid_train, y_train, gaze_resid_test, y_test)
    eeg_resid_test_exp = train_expert(eeg_resid_train, y_train, eeg_resid_test, y_test)
    concat_resid_test_exp = train_expert(concat_resid_train, y_train, concat_resid_test, y_test)
    
    gaze_logits = experts_test['gaze']['logits']
    gaze_resid_logits = gaze_resid_test_exp['logits']
    eeg_resid_logits = eeg_resid_test_exp['logits']
    concat_resid_logits = concat_resid_test_exp['logits']
    
    gaze_resid_train_logits = gaze_resid_train_exp['logits']
    eeg_resid_train_logits = eeg_resid_train_exp['logits']
    concat_resid_train_logits = concat_resid_train_exp['logits']
    
    gaze_train_logits = experts_train['gaze']['logits']
    
    def objective(params):
        alpha_gaze_resid = 1 / (1 + np.exp(-params[0]))
        alpha_eeg_resid = 1 / (1 + np.exp(-params[1]))
        alpha_concat_resid = 1 / (1 + np.exp(-params[2]))
        
        final_logits = gaze_train_logits + \
                      alpha_gaze_resid * (gaze_resid_train_logits - gaze_train_logits) + \
                      alpha_eeg_resid * (eeg_resid_train_logits - gaze_train_logits) + \
                      alpha_concat_resid * (concat_resid_train_logits - gaze_train_logits)
        
        ce = ce_loss(final_logits, y_train)
        gaze_ce = ce_loss(gaze_train_logits, y_train)
        safe_penalty = lambda_safe * np.maximum(0, ce - gaze_ce)
        
        return ce + safe_penalty
    
    result = minimize(objective, [0, 0, 0], method='L-BFGS-B', options={'maxiter': 200})
    
    alpha_gaze_resid = 1 / (1 + np.exp(-result.x[0]))
    alpha_eeg_resid = 1 / (1 + np.exp(-result.x[1]))
    alpha_concat_resid = 1 / (1 + np.exp(-result.x[2]))
    
    final_logits = gaze_logits + \
                  alpha_gaze_resid * (gaze_resid_logits - gaze_logits) + \
                  alpha_eeg_resid * (eeg_resid_logits - gaze_logits) + \
                  alpha_concat_resid * (concat_resid_logits - gaze_logits)
    
    final_probs = 1 / (1 + np.exp(-final_logits))
    
    return final_probs, {
        'alpha_gaze_resid': alpha_gaze_resid,
        'alpha_eeg_resid': alpha_eeg_resid,
        'alpha_concat_resid': alpha_concat_resid
    }

def compute_no_harm(pred_fusion, pred_gaze, y_true):
    violations = ((pred_fusion != y_true) & (pred_gaze == y_true)).sum()
    helpful = ((pred_fusion == y_true) & (pred_gaze != y_true)).sum()
    return violations, violations / len(y_true) if len(y_true) > 0 else 0, helpful

def run_loso_with_fusion(eeg, gaze, y, subjects, lambda_safe=0.5):
    loo = LeaveOneOut()
    subject_list = sorted(set(subjects))
    results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(subject_list)):
        test_subject = subject_list[test_idx[0]]
        train_subjects = [subject_list[i] for i in train_idx]
        
        train_mask = np.isin(subjects, train_subjects)
        test_mask = subjects == test_subject
        
        X_eeg_train, X_eeg_test = eeg[train_mask], eeg[test_mask]
        X_gaze_train, X_gaze_test = gaze[train_mask], gaze[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        X_train_conf = np.ones((len(y_train), 1))
        X_test_conf = np.ones((len(y_test), 1))
        
        gaze_train_exp = train_expert(X_gaze_train, y_train, X_gaze_train, y_train)
        eeg_train_exp = train_expert(X_eeg_train, y_train, X_eeg_train, y_train)
        concat_train_exp = train_expert(np.hstack([X_gaze_train, X_eeg_train]), y_train, 
                                        np.hstack([X_gaze_train, X_eeg_train]), y_train)
        
        gaze_test_exp = train_expert(X_gaze_train, y_train, X_gaze_test, y_test)
        eeg_test_exp = train_expert(X_eeg_train, y_train, X_eeg_test, y_test)
        concat_test_exp = train_expert(np.hstack([X_gaze_train, X_eeg_train]), y_train,
                                       np.hstack([X_gaze_test, X_eeg_test]), y_test)
        
        gaze_resid_test_exp = train_expert(
            compute_residuals(X_train_conf, X_gaze_train), y_train,
            compute_residuals(X_test_conf, X_gaze_test), y_test
        )
        eeg_resid_test_exp = train_expert(
            compute_residuals(X_train_conf, X_eeg_train), y_train,
            compute_residuals(X_test_conf, X_eeg_test), y_test
        )
        concat_resid_test_exp = train_expert(
            np.hstack([compute_residuals(X_train_conf, X_gaze_train), compute_residuals(X_train_conf, X_eeg_train)]), y_train,
            np.hstack([compute_residuals(X_test_conf, X_gaze_test), compute_residuals(X_test_conf, X_eeg_test)]), y_test
        )
        
        experts_train = {'gaze': gaze_train_exp, 'eeg': eeg_train_exp, 'concat': concat_train_exp}
        experts_test = {'gaze': gaze_test_exp, 'eeg': eeg_test_exp, 'concat': concat_test_exp}
        
        fusion_probs, fusion_alphas = gaze_anchored_residual_fusion(
            experts_train, experts_test, X_train_conf, X_test_conf,
            X_gaze_train, X_eeg_train, X_gaze_test, X_eeg_test,
            y_train, y_test, lambda_safe
        )
        
        gaze_pred = gaze_test_exp['predictions']
        fusion_pred = (fusion_probs >= 0.5).astype(int)
        
        violations, violation_rate, helpful = compute_no_harm(fusion_pred, gaze_pred, y_test)
        
        all_probs = {
            'gaze': gaze_test_exp['probabilities'],
            'eeg': eeg_test_exp['probabilities'],
            'concat': concat_test_exp['probabilities'],
            'gaze_resid': gaze_resid_test_exp['probabilities'],
            'eeg_resid': eeg_resid_test_exp['probabilities'],
            'concat_resid': concat_resid_test_exp['probabilities']
        }
        uniform_probs = np.mean([all_probs[k] for k in all_probs], axis=0)
        uniform_pred = (uniform_probs >= 0.5).astype(int)
        uniform_violations, uniform_violation_rate, uniform_helpful = compute_no_harm(uniform_pred, gaze_pred, y_test)
        
        results.append({
            'subject': test_subject,
            'test_samples': len(y_test),
            'gaze_accuracy': gaze_test_exp['accuracy'],
            'gaze_balanced_accuracy': gaze_test_exp['balanced_accuracy'],
            'gaze_macro_f1': gaze_test_exp['macro_f1'],
            'gaze_auroc': gaze_test_exp['auroc'],
            'eeg_macro_f1': eeg_test_exp['macro_f1'],
            'concat_macro_f1': concat_test_exp['macro_f1'],
            'gaze_resid_macro_f1': gaze_resid_test_exp['macro_f1'],
            'eeg_resid_macro_f1': eeg_resid_test_exp['macro_f1'],
            'concat_resid_macro_f1': concat_resid_test_exp['macro_f1'],
            'best_single_expert': max(['gaze', 'eeg', 'concat', 'gaze_resid', 'eeg_resid', 'concat_resid'],
                                     key=lambda k: all_probs[k].mean() * 0 + 
                                     f1_score(y_test, (all_probs[k] >= 0.5).astype(int), average='macro')),
            'uniform_macro_f1': f1_score(y_test, uniform_pred, average='macro'),
            'uniform_auroc': roc_auc_score(y_test, uniform_probs),
            'uniform_violation_rate': uniform_violation_rate,
            'uniform_helpful': uniform_helpful,
            'fusion_macro_f1': f1_score(y_test, fusion_pred, average='macro'),
            'fusion_auroc': roc_auc_score(y_test, fusion_probs),
            'fusion_violation_rate': violation_rate,
            'fusion_helpful': helpful,
            'alpha_gaze_resid': fusion_alphas['alpha_gaze_resid'],
            'alpha_eeg_resid': fusion_alphas['alpha_eeg_resid'],
            'alpha_concat_resid': fusion_alphas['alpha_concat_resid']
        })
        
        if (fold_idx + 1) % 4 == 0:
            print(f"  Processed {fold_idx + 1}/{len(subject_list)} folds...")
    
    return pd.DataFrame(results)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("SAFE-CIRL-Fuse LOSO with Gaze-Anchored Continuous Fusion")
    print("=" * 70)
    
    data = np.load(os.path.join(DATA_DIR, 'aligned_multimodal_y.npz'))
    eeg = data['eeg']
    gaze = data['gaze']
    y = data['y']
    
    df_meta = pd.read_csv(os.path.join(DATA_DIR, 'aligned_multimodal_y_metadata.csv'))
    subjects = df_meta['subject'].values
    
    print(f"Loaded {len(y)} samples")
    print(f"EEG shape: {eeg.shape}")
    print(f"Gaze shape: {gaze.shape}")
    print(f"Subjects: {len(set(subjects))}")
    
    print("\nRunning LOSO with SAFE-CIRL-Fuse (lambda_safe=0.5)...")
    results_df = run_loso_with_fusion(eeg, gaze, y, subjects, lambda_safe=0.5)
    
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'safe_cirl_fuse_loso_results.csv'), index=False)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n### Per-Model Statistics (Mean across subjects) ###\n")
    print(f"{'Model':<25} {'Accuracy':>10} {'Bal Acc':>10} {'Macro-F1':>10} {'AUROC':>10}")
    print("-" * 70)
    print(f"{'Gaze-only':<25} {results_df['gaze_accuracy'].mean():>10.4f} {results_df['gaze_balanced_accuracy'].mean():>10.4f} {results_df['gaze_macro_f1'].mean():>10.4f} {results_df['gaze_auroc'].mean():>10.4f}")
    print(f"{'EEG-only':<25} {'--':>10} {'--':>10} {results_df['eeg_macro_f1'].mean():>10.4f} {'--':>10}")
    print(f"{'Concat':<25} {'--':>10} {'--':>10} {results_df['concat_macro_f1'].mean():>10.4f} {'--':>10}")
    print(f"{'Gaze-Residual':<25} {'--':>10} {'--':>10} {results_df['gaze_resid_macro_f1'].mean():>10.4f} {'--':>10}")
    print(f"{'EEG-Residual':<25} {'--':>10} {'--':>10} {results_df['eeg_resid_macro_f1'].mean():>10.4f} {'--':>10}")
    print(f"{'Concat-Residual':<25} {'--':>10} {'--':>10} {results_df['concat_resid_macro_f1'].mean():>10.4f} {'--':>10}")
    print("-" * 70)
    print(f"{'Uniform Average':<25} {'--':>10} {'--':>10} {results_df['uniform_macro_f1'].mean():>10.4f} {results_df['uniform_auroc'].mean():>10.4f}")
    print(f"{'SAFE-CIRL-Fuse':<25} {'--':>10} {'--':>10} {results_df['fusion_macro_f1'].mean():>10.4f} {results_df['fusion_auroc'].mean():>10.4f}")
    
    print("\n### No-Harm Analysis ###\n")
    print(f"{'Model':<25} {'Violation Rate':>15} {'Helpful':>10}")
    print("-" * 50)
    print(f"{'Uniform Average':<25} {results_df['uniform_violation_rate'].mean():>15.4f} {results_df['uniform_helpful'].mean():>10.1f}")
    print(f"{'SAFE-CIRL-Fuse':<25} {results_df['fusion_violation_rate'].mean():>15.4f} {results_df['fusion_helpful'].mean():>10.1f}")
    
    print("\n### Fusion Weights (Mean across subjects) ###\n")
    print(f"Alpha (Gaze-Residual): {results_df['alpha_gaze_resid'].mean():.4f}")
    print(f"Alpha (EEG-Residual):  {results_df['alpha_eeg_resid'].mean():.4f}")
    print(f"Alpha (Concat-Residual): {results_df['alpha_concat_resid'].mean():.4f}")
    
    summary = f"""# SAFE-CIRL-Fuse LOSO Results Summary

## Dataset Statistics
- Total samples: {len(y)}
- Subjects: {len(set(subjects))}
- EEG features: {eeg.shape[1]}-D
- Gaze features: {gaze.shape[1]}-D

## Per-Model Statistics (Mean across 16 subjects)

| Model | Macro-F1 (mean±std) | AUROC (mean±std) |
|-------|---------------------|------------------|
| Gaze-only | {results_df['gaze_macro_f1'].mean():.4f} ± {results_df['gaze_macro_f1'].std():.4f} | {results_df['gaze_auroc'].mean():.4f} ± {results_df['gaze_auroc'].std():.4f} |
| EEG-only | {results_df['eeg_macro_f1'].mean():.4f} ± {results_df['eeg_macro_f1'].std():.4f} | -- |
| Concat | {results_df['concat_macro_f1'].mean():.4f} ± {results_df['concat_macro_f1'].std():.4f} | -- |
| Gaze-Residual | {results_df['gaze_resid_macro_f1'].mean():.4f} ± {results_df['gaze_resid_macro_f1'].std():.4f} | -- |
| EEG-Residual | {results_df['eeg_resid_macro_f1'].mean():.4f} ± {results_df['eeg_resid_macro_f1'].std():.4f} | -- |
| Concat-Residual | {results_df['concat_resid_macro_f1'].mean():.4f} ± {results_df['concat_resid_macro_f1'].std():.4f} | -- |
| Uniform Average | {results_df['uniform_macro_f1'].mean():.4f} ± {results_df['uniform_macro_f1'].std():.4f} | {results_df['uniform_auroc'].mean():.4f} ± {results_df['uniform_auroc'].std():.4f} |
| **SAFE-CIRL-Fuse** | **{results_df['fusion_macro_f1'].mean():.4f} ± {results_df['fusion_macro_f1'].std():.4f}** | **{results_df['fusion_auroc'].mean():.4f} ± {results_df['fusion_auroc'].std():.4f}** |

## No-Harm Analysis

| Model | Violation Rate (mean) | Helpful Count (mean) |
|-------|----------------------|---------------------|
| Uniform Average | {results_df['uniform_violation_rate'].mean():.4f} | {results_df['uniform_helpful'].mean():.1f} |
| SAFE-CIRL-Fuse | {results_df['fusion_violation_rate'].mean():.4f} | {results_df['fusion_helpful'].mean():.1f} |

## Fusion Weights

| Expert | Alpha (mean±std) |
|--------|-----------------|
| Gaze-Residual | {results_df['alpha_gaze_resid'].mean():.4f} ± {results_df['alpha_gaze_resid'].std():.4f} |
| EEG-Residual | {results_df['alpha_eeg_resid'].mean():.4f} ± {results_df['alpha_eeg_resid'].std():.4f} |
| Concat-Residual | {results_df['alpha_concat_resid'].mean():.4f} ± {results_df['alpha_concat_resid'].std():.4f} |

## Per-Subject Results

| Subject | Gaze | Best Single | Uniform | SAFE-CIRL-Fuse | Improvement |
|---------|------|-------------|---------|----------------|-------------|
"""
    
    for _, row in results_df.sort_values('subject').iterrows():
        improvement = row['fusion_macro_f1'] - row['gaze_macro_f1']
        summary += f"| {row['subject']} | {row['gaze_macro_f1']:.4f} | {row['concat_resid_macro_f1']:.4f} | {row['uniform_macro_f1']:.4f} | {row['fusion_macro_f1']:.4f} | {improvement:+.4f} |\n"
    
    summary += f"""

## Key Findings

1. **SAFE-CIRL-Fuse vs Gaze-only**: {'+' if results_df['fusion_macro_f1'].mean() > results_df['gaze_macro_f1'].mean() else ''}{(results_df['fusion_macro_f1'].mean() - results_df['gaze_macro_f1'].mean())*100:.2f}% improvement
2. **SAFE-CIRL-Fuse vs Uniform Average**: {'+' if results_df['fusion_macro_f1'].mean() > results_df['uniform_macro_f1'].mean() else ''}{(results_df['fusion_macro_f1'].mean() - results_df['uniform_macro_f1'].mean())*100:.2f}% improvement
3. **No-harm violation rate**: {results_df['fusion_violation_rate'].mean()*100:.2f}%
4. **Best performing expert**: Concat-Residual ({results_df['concat_resid_macro_f1'].mean():.4f})

---
*Generated by SAFE-CIRL-Fuse LOSO*
"""
    
    with open(os.path.join(OUTPUT_DIR, 'safe_cirl_fuse_loso_summary.md'), 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("=" * 70)

if __name__ == '__main__':
    main()
