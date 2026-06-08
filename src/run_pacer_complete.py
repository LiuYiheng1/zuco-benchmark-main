#!/usr/bin/env python
"""
PACER-Read Complete Pipeline: Step 0 to Step 2.6
All results will be saved to files for verification.
"""
import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

OUTPUT_DIR = 'results/pacer_full_pipeline'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg)
    sys.stdout.flush()

def load_aligned_data():
    log("=" * 60)
    log("STEP 0: Loading aligned data...")
    log("=" * 60)
    
    npz_path = 'data/aligned_multimodal_y.npz'
    meta_path = 'data/aligned_multimodal_y_metadata.csv'
    
    data = np.load(npz_path, allow_pickle=True)
    meta = pd.read_csv(meta_path)
    
    X_eeg = data['eeg']
    X_gaze = data['gaze']
    y = data['y']
    subjects = meta['subject'].values if 'subject' in meta.columns else np.array(['UNKNOWN'] * len(y))
    
    subjects = np.array([str(s).strip() for s in subjects])
    
    results = {
        'X_eeg_shape': str(X_eeg.shape),
        'X_gaze_shape': str(X_gaze.shape),
        'y_shape': str(y.shape),
        'y_classes': str(np.unique(y).tolist()),
        'n_subjects': len(np.unique(subjects)),
        'subjects': ', '.join(np.unique(subjects).tolist()),
        'has_nan_eeg': bool(np.any(np.isnan(X_eeg))),
        'has_nan_gaze': bool(np.any(np.isnan(X_gaze))),
    }
    
    df = pd.DataFrame([results])
    df.to_csv(f'{OUTPUT_DIR}/step0_data_check.csv', index=False)
    
    inventory = []
    for sub in np.unique(subjects):
        mask = subjects == sub
        inventory.append({
            'subject': sub,
            'total_N': int(mask.sum()),
            'NR_N': int((y[mask] == 0).sum()),
            'TSR_N': int((y[mask] == 1).sum()),
            'eeg_dim': int(X_eeg.shape[1]),
            'gaze_dim': int(X_gaze.shape[1])
        })
    
    pd.DataFrame(inventory).to_csv(f'{OUTPUT_DIR}/step0_data_inventory.csv', index=False)
    
    log(f"Data loaded: X_eeg={X_eeg.shape}, X_gaze={X_gaze.shape}")
    log(f"Subjects: {len(np.unique(subjects))}")
    log(f"Classes: {np.unique(y).tolist()}")
    
    return X_eeg, X_gaze, y, subjects, meta

def train_expert(X_train, y_train, X_test=None, y_test=None):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_train_s, y_train)
    
    logits_train = clf.decision_function(X_train_s)
    prob_train = clf.predict_proba(X_train_s)[:, 1]
    
    result = {'logits': logits_train, 'prob': prob_train, 'scaler': scaler, 'clf': clf}
    
    if X_test is not None and y_test is not None:
        X_test_s = scaler.transform(X_test)
        result['pred'] = clf.predict(X_test_s)
        result['prob_test'] = clf.predict_proba(X_test_s)[:, 1]
        result['logits_test'] = clf.decision_function(X_test_s)
        result['f1'] = f1_score(y_test, result['pred'], average='macro')
        result['acc'] = accuracy_score(y_test, result['pred'])
    
    return result

def run_pacer_lite(X_eeg, X_gaze, y, subjects, target_subjects=['YHS', 'YRK', 'YFR'], k_values=[3, 5], seeds=[1]):
    log("=" * 60)
    log("PACER-Lite Simplified Fusion...")
    log("=" * 60)
    
    all_results = []
    subject_results = []
    gamma_records = []
    
    for target in target_subjects:
        for k in k_values:
            for seed in seeds:
                np.random.seed(seed)
                
                target_mask = subjects == target
                source_mask = subjects != target
                
                target_idx = np.where(target_mask)[0]
                source_idx = np.where(source_mask)[0]
                
                y_target = y[target_mask]
                nr_idx = np.where(y_target == 0)[0]
                tsr_idx = np.where(y_target == 1)[0]
                
                try:
                    calib_nr = np.random.choice(nr_idx, k, replace=False)
                    calib_tsr = np.random.choice(tsr_idx, k, replace=False)
                except:
                    continue
                
                calib_idx = np.concatenate([calib_nr, calib_tsr])
                test_idx = np.array([i for i in range(len(y_target)) if i not in calib_idx])
                
                source_train_idx = source_idx
                
                train_idx = np.concatenate([source_train_idx, target_idx[calib_idx]])
                y_train = y[train_idx]
                
                X_test_eeg = X_eeg[target_idx[test_idx]]
                X_test_gaze = X_gaze[target_idx[test_idx]]
                y_test = y_target[test_idx]
                
                X_train_eeg = X_eeg[train_idx]
                X_train_gaze = X_gaze[train_idx]
                X_train_concat = np.hstack([X_train_gaze, X_train_eeg])
                X_test_concat = np.hstack([X_test_gaze, X_test_eeg])
                
                e1 = train_expert(X_train_gaze, y_train, X_test_gaze, y_test)
                e3 = train_expert(X_train_concat, y_train, X_test_concat, y_test)
                
                anchor_logits = e3['logits_test']
                anchor_prob = e3['prob_test']
                
                result_row = {
                    'target': target, 'k': k, 'seed': seed,
                    'test_N': int(len(test_idx)),
                    'E1_gaze_raw_F1': float(e1['f1']),
                    'E3_concat_raw_F1': float(e3['f1']),
                }
                
                gamma = 0.25
                
                fusion_prob = (1 - gamma) * anchor_prob + gamma * e1['prob_test']
                fusion_pred = (fusion_prob > 0.5).astype(int)
                fusion_f1 = f1_score(y_test, fusion_pred, average='macro')
                
                result_row['E3_E1_fusion_F1'] = float(fusion_f1)
                result_row['gamma'] = float(gamma)
                
                no_harm = ((fusion_pred != y_test) & (e3['pred'] == y_test)).sum()
                no_harm_rate = no_harm / len(y_test) if len(y_test) > 0 else 0
                result_row['no_harm_rate_vs_E3'] = float(no_harm_rate)
                
                gamma_records.append({
                    'target': target, 'k': k, 'seed': seed,
                    'gamma_mean': float(gamma),
                    'gamma_min': float(gamma),
                    'gamma_max': float(gamma)
                })
                
                all_results.append(result_row)
                subject_results.append({
                    'target': target, 'k': k, 'seed': seed,
                    'test_N': int(len(test_idx)),
                    'E3_concat_raw': float(e3['f1']),
                    'PACER_Lite_E3_E1': float(fusion_f1),
                    'gamma': float(gamma),
                    'no_harm_rate': float(no_harm_rate),
                })
    
    pd.DataFrame(all_results).to_csv(f'{OUTPUT_DIR}/step26_all_results.csv', index=False)
    pd.DataFrame(subject_results).to_csv(f'{OUTPUT_DIR}/step26_subjectwise.csv', index=False)
    pd.DataFrame(gamma_records).to_csv(f'{OUTPUT_DIR}/step26_gamma_records.csv', index=False)
    
    df = pd.DataFrame(subject_results)
    log("\nSubject-wise Results:")
    for _, row in df.iterrows():
        log(f"  {row['target']}: E3={row['E3_concat_raw']:.4f}, PACER={row['PACER_Lite_E3_E1']:.4f}, diff={row['PACER_Lite_E3_E1']-row['E3_concat_raw']:+.4f}")
    
    return all_results, subject_results

def main():
    log("Starting PACER-Read Complete Pipeline")
    log("=" * 60)
    
    X_eeg, X_gaze, y, subjects, meta = load_aligned_data()
    
    target_subjects = ['YHS', 'YRK', 'YFR']
    k_values = [3, 5]
    seeds = [1]
    
    step26_results, step26_subject = run_pacer_lite(X_eeg, X_gaze, y, subjects, target_subjects, k_values, seeds)
    
    log("=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    
    df = pd.DataFrame(step26_subject)
    
    for method in ['E3_concat_raw', 'PACER_Lite_E3_E1']:
        subject_vals = df.groupby('target')[method].mean()
        unweighted_mean = subject_vals.mean()
        log(f"{method} - Subject-wise unweighted mean: {unweighted_mean:.4f}")
    
    log(f"\nAll results saved to: {OUTPUT_DIR}/")
    
    df_means = df.groupby('target').mean(numeric_only=True)
    e3_mean = df_means['E3_concat_raw'].mean()
    pacer_mean = df_means['PACER_Lite_E3_E1'].mean()
    
    summary = f"""# PACER-Read Step 2.6 Summary

## Subject-wise Results

| Subject | E3_concat_raw | PACER-Lite E3_E1 | Difference |
|---------|---------------|------------------|------------|
"""
    
    for _, row in df.iterrows():
        diff = row['PACER_Lite_E3_E1'] - row['E3_concat_raw']
        summary += f"| {row['target']} | {row['E3_concat_raw']:.4f} | {row['PACER_Lite_E3_E1']:.4f} | {diff:+.4f} |\n"
    
    summary += f"""
## Subject-wise Mean Macro-F1 (Unweighted)

| Method | Macro-F1 |
|--------|----------|
| E3_concat_raw | {e3_mean:.4f} |
| PACER-Lite E3_E1 | {pacer_mean:.4f} |

**Difference: {pacer_mean - e3_mean:+.4f}**

## Conclusion

Based on subject-wise unweighted mean (equal weight per subject):
- E3_concat_raw: {e3_mean:.4f}
- PACER-Lite: {pacer_mean:.4f}
- PACER-Lite {'>' if pacer_mean > e3_mean else '<'} E3_concat_raw by {abs(pacer_mean - e3_mean):.4f}
"""
    
    with open(f'{OUTPUT_DIR}/step26_summary.md', 'w') as f:
        f.write(summary)
    
    log(summary)

if __name__ == '__main__':
    main()
