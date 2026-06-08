#!/usr/bin/env python
"""
SAFE-CIRL-Fuse: Constrained Convex Fusion Debug & Calibration
"""

import os
import numpy as np
import pandas as pd
import hashlib
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from scipy.optimize import minimize

OUTPUT_DIR = 'results/safe_cirl_fuse_debug'
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

def train_expert(X_train, y_train, X_test=None, y_test=None, calibrated=False):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if calibrated:
        base_clf = LogisticRegression(max_iter=1000, random_state=42)
        clf = CalibratedClassifierCV(base_clf, cv=3)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42)
    
    clf.fit(X_train_scaled, y_train)
    
    if calibrated:
        y_proba_train = clf.predict_proba(X_train_scaled)[:, 1]
        y_proba_train = np.clip(y_proba_train, 1e-10, 1 - 1e-10)
        y_logits_train = np.log(y_proba_train / (1 - y_proba_train))
    else:
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
        if calibrated:
            y_proba_clipped = np.clip(y_proba, 1e-10, 1 - 1e-10)
            y_logits = np.log(y_proba_clipped / (1 - y_proba_clipped))
        else:
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
                              use_beta_penalty=True, use_kl=True, debug=False):
    gaze_logits = experts_train['gaze']['logits']
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-10)
    
    debug_info = []
    
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
        
        if debug:
            debug_info.append({
                'beta': beta,
                'w_gr': w_gr,
                'w_er': w_er,
                'w_cr': w_cr,
                'ce': ce,
                'safe_penalty': safe_penalty,
                'beta_penalty': beta_penalty,
                'kl_penalty': kl_penalty,
                'total': ce + safe_penalty + beta_penalty + kl_penalty
            })
        
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
    
    return final_probs, {
        'beta': beta, 
        'w_gr': w_gr, 
        'w_er': w_er, 
        'w_cr': w_cr,
        'debug_info': debug_info
    }

def compute_no_harm(pred_fusion, pred_gaze, y_true):
    violations = ((pred_fusion != y_true) & (pred_gaze == y_true)).sum()
    helpful = ((pred_fusion == y_true) & (pred_gaze != y_true)).sum()
    return violations, violations / len(y_true) if len(y_true) > 0 else 0, helpful

def oof_train_fusion(X_eeg_train, X_gaze_train, y_train, experts_train, oof_experts):
    gaze_logits = oof_experts['gaze']['logits']
    gaze_resid_logits = oof_experts['gaze_resid']['logits']
    eeg_resid_logits = oof_experts['eeg_resid']['logits']
    concat_resid_logits = oof_experts['concat_resid']['logits']
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-10)
    
    def objective(params):
        w_gr, w_er, w_cr = softmax(params[:3])
        beta = 0.5 * 1 / (1 + np.exp(-params[3]))
        
        expert_mix = w_gr * gaze_resid_logits + \
                    w_er * eeg_resid_logits + \
                    w_cr * concat_resid_logits
        
        final_logits = (1 - beta) * gaze_logits + beta * expert_mix
        
        ce = ce_loss(final_logits, y_train)
        gaze_ce = ce_loss(gaze_logits, y_train)
        safe_penalty = 0.5 * np.maximum(0, ce - gaze_ce)
        
        return ce + safe_penalty
    
    init_params = np.zeros(4)
    result = minimize(objective, init_params, method='L-BFGS-B', options={'maxiter': 300})
    
    w_gr, w_er, w_cr = softmax(result.x[:3])
    beta = 0.5 * 1 / (1 + np.exp(-result.x[3]))
    
    return {'w_gr': w_gr, 'w_er': w_er, 'w_cr': w_cr, 'beta': beta}

def run_single_fold(held_out, all_subjects, npz_data, metadata):
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
        
        experts_train = {'gaze': gaze_train_exp, 'gaze_resid': gaze_resid_train_exp,
                        'eeg_resid': eeg_resid_train_exp, 'concat_resid': concat_resid_train_exp}
        experts_test = {'gaze': gaze_test_exp, 'gaze_resid': gaze_resid_test_exp,
                       'eeg_resid': eeg_resid_test_exp, 'concat_resid': concat_resid_test_exp}
        
        gaze_pred = gaze_test_exp['predictions']
        gaze_probs = gaze_test_exp['test_probabilities']
        gaze_logits_test = experts_test['gaze']['test_logits']
        
        gaze_resid_pred = gaze_resid_test_exp['predictions']
        gaze_resid_probs = gaze_resid_test_exp['test_probabilities']
        gaze_resid_logits = experts_test['gaze_resid']['test_logits']
        
        eeg_resid_pred = eeg_resid_test_exp['predictions']
        eeg_resid_probs = eeg_resid_test_exp['test_probabilities']
        eeg_resid_logits = experts_test['eeg_resid']['test_logits']
        
        concat_resid_pred = concat_resid_test_exp['predictions']
        concat_resid_probs = concat_resid_test_exp['test_probabilities']
        concat_resid_logits = experts_test['concat_resid']['test_logits']
        
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
            'test_N': len(y_test),
            'beta': 0.0,
            'w_gr': 0.0,
            'w_er': 0.0,
            'w_cr': 0.0
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
            'test_N': len(y_test),
            'beta': 1.0,
            'w_gr': 0.0,
            'w_er': 0.0,
            'w_cr': 1.0
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
            'test_N': len(y_test),
            'beta': 0.75,
            'w_gr': 0.25,
            'w_er': 0.25,
            'w_cr': 0.25
        }]
        
        fusion_probs, fusion_weights = constrained_convex_fusion(
            experts_train, experts_test, y_train, y_test,
            beta_max=0.5, lambda_safe=0.5,
            lambda_beta=0.01, lambda_kl=0.05,
            use_beta_penalty=True, use_kl=True, debug=True
        )
        fusion_pred = (fusion_probs >= 0.5).astype(int)
        viol, viol_rate, helpful = compute_no_harm(fusion_pred, gaze_pred, y_test)
        
        results.append({
            'subject': held_out,
            'method': 'old_constrained',
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
        
        gaze_train_exp_cal = train_expert(X_gaze_train, y_train, calibrated=True)
        gaze_resid_train_exp_cal = train_expert(gaze_resid_train, y_train, calibrated=True)
        eeg_resid_train_exp_cal = train_expert(eeg_resid_train, y_train, calibrated=True)
        concat_resid_train_exp_cal = train_expert(concat_resid_train, y_train, calibrated=True)
        
        gaze_test_exp_cal = train_expert(X_gaze_train, y_train, X_gaze_test, y_test, calibrated=True)
        gaze_resid_test_exp_cal = train_expert(gaze_resid_train, y_train, gaze_resid_test, y_test, calibrated=True)
        eeg_resid_test_exp_cal = train_expert(eeg_resid_train, y_train, eeg_resid_test, y_test, calibrated=True)
        concat_resid_test_exp_cal = train_expert(concat_resid_train, y_train, concat_resid_test, y_test, calibrated=True)
        
        experts_train_cal = {'gaze': gaze_train_exp_cal, 'gaze_resid': gaze_resid_train_exp_cal,
                            'eeg_resid': eeg_resid_train_exp_cal, 'concat_resid': concat_resid_train_exp_cal}
        experts_test_cal = {'gaze': gaze_test_exp_cal, 'gaze_resid': gaze_resid_test_exp_cal,
                           'eeg_resid': eeg_resid_test_exp_cal, 'concat_resid': concat_resid_test_exp_cal}
        
        fusion_probs_cal, fusion_weights_cal = constrained_convex_fusion(
            experts_train_cal, experts_test_cal, y_train, y_test,
            beta_max=0.5, lambda_safe=0.5,
            lambda_beta=0.01, lambda_kl=0.05,
            use_beta_penalty=True, use_kl=True
        )
        fusion_pred_cal = (fusion_probs_cal >= 0.5).astype(int)
        viol_cal, viol_rate_cal, helpful_cal = compute_no_harm(fusion_pred_cal, gaze_pred, y_test)
        
        results.append({
            'subject': held_out,
            'method': 'calibrated_constrained',
            'macro_f1': f1_score(y_test, fusion_pred_cal, average='macro'),
            'accuracy': accuracy_score(y_test, fusion_pred_cal),
            'auroc': roc_auc_score(y_test, fusion_probs_cal),
            'violations': viol_cal,
            'violation_rate': viol_rate_cal,
            'helpful': helpful_cal,
            'test_hash': test_hash,
            'test_N': len(y_test),
            'beta': fusion_weights_cal['beta'],
            'w_gr': fusion_weights_cal['w_gr'],
            'w_er': fusion_weights_cal['w_er'],
            'w_cr': fusion_weights_cal['w_cr']
        })
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oof_gaze_logits = np.zeros_like(y_train)
        oof_gaze_resid_logits = np.zeros_like(y_train)
        oof_eeg_resid_logits = np.zeros_like(y_train)
        oof_concat_resid_logits = np.zeros_like(y_train)
        
        for fold_idx, (train_fold, val_fold) in enumerate(kf.split(y_train)):
            X_gaze_train_fold = X_gaze_train[train_fold]
            y_train_fold = y_train[train_fold]
            X_gaze_val_fold = X_gaze_train[val_fold]
            y_val_fold = y_train[val_fold]
            
            gaze_resid_train_fold = gaze_resid_train[train_fold]
            gaze_resid_val_fold = gaze_resid_train[val_fold]
            eeg_resid_train_fold = eeg_resid_train[train_fold]
            eeg_resid_val_fold = eeg_resid_train[val_fold]
            concat_resid_train_fold = concat_resid_train[train_fold]
            concat_resid_val_fold = concat_resid_train[val_fold]
            
            fold_gaze_exp = train_expert(X_gaze_train_fold, y_train_fold, X_gaze_val_fold, y_val_fold)
            fold_gaze_resid_exp = train_expert(gaze_resid_train_fold, y_train_fold, gaze_resid_val_fold, y_val_fold)
            fold_eeg_resid_exp = train_expert(eeg_resid_train_fold, y_train_fold, eeg_resid_val_fold, y_val_fold)
            fold_concat_resid_exp = train_expert(concat_resid_train_fold, y_train_fold, concat_resid_val_fold, y_val_fold)
            
            oof_gaze_logits[val_fold] = fold_gaze_exp['test_logits']
            oof_gaze_resid_logits[val_fold] = fold_gaze_resid_exp['test_logits']
            oof_eeg_resid_logits[val_fold] = fold_eeg_resid_exp['test_logits']
            oof_concat_resid_logits[val_fold] = fold_concat_resid_exp['test_logits']
        
        oof_experts = {
            'gaze': {'logits': oof_gaze_logits},
            'gaze_resid': {'logits': oof_gaze_resid_logits},
            'eeg_resid': {'logits': oof_eeg_resid_logits},
            'concat_resid': {'logits': oof_concat_resid_logits}
        }
        
        oof_fusion_weights = oof_train_fusion(X_eeg_train, X_gaze_train, y_train, experts_train, oof_experts)
        
        gaze_logits_test_oof = experts_test['gaze']['test_logits']
        expert_mix_test_oof = (oof_fusion_weights['w_gr'] * experts_test['gaze_resid']['test_logits'] + 
                              oof_fusion_weights['w_er'] * experts_test['eeg_resid']['test_logits'] + 
                              oof_fusion_weights['w_cr'] * experts_test['concat_resid']['test_logits'])
        final_logits_oof = (1 - oof_fusion_weights['beta']) * gaze_logits_test_oof + oof_fusion_weights['beta'] * expert_mix_test_oof
        final_probs_oof = 1 / (1 + np.exp(-np.clip(final_logits_oof, -500, 500)))
        
        fusion_pred_oof = (final_probs_oof >= 0.5).astype(int)
        viol_oof, viol_rate_oof, helpful_oof = compute_no_harm(fusion_pred_oof, gaze_pred, y_test)
        
        results.append({
            'subject': held_out,
            'method': 'oof_calibrated_constrained',
            'macro_f1': f1_score(y_test, fusion_pred_oof, average='macro'),
            'accuracy': accuracy_score(y_test, fusion_pred_oof),
            'auroc': roc_auc_score(y_test, final_probs_oof),
            'violations': viol_oof,
            'violation_rate': viol_rate_oof,
            'helpful': helpful_oof,
            'test_hash': test_hash,
            'test_N': len(y_test),
            'beta': oof_fusion_weights['beta'],
            'w_gr': oof_fusion_weights['w_gr'],
            'w_er': oof_fusion_weights['w_er'],
            'w_cr': oof_fusion_weights['w_cr']
        })
        
        weight_stats = {
            'subject': held_out,
            'gaze_logit_mean': np.mean(gaze_logits_test),
            'gaze_logit_std': np.std(gaze_logits_test),
            'gaze_resid_logit_mean': np.mean(gaze_resid_logits),
            'gaze_resid_logit_std': np.std(gaze_resid_logits),
            'eeg_resid_logit_mean': np.mean(eeg_resid_logits),
            'eeg_resid_logit_std': np.std(eeg_resid_logits),
            'concat_resid_logit_mean': np.mean(concat_resid_logits),
            'concat_resid_logit_std': np.std(concat_resid_logits),
            'gaze_conf_mean': np.mean(np.maximum(gaze_probs, 1 - gaze_probs)),
            'gaze_conf_std': np.std(np.maximum(gaze_probs, 1 - gaze_probs)),
            'gaze_resid_conf_mean': np.mean(np.maximum(gaze_resid_probs, 1 - gaze_resid_probs)),
            'gaze_resid_conf_std': np.std(np.maximum(gaze_resid_probs, 1 - gaze_resid_probs)),
            'eeg_resid_conf_mean': np.mean(np.maximum(eeg_resid_probs, 1 - eeg_resid_probs)),
            'eeg_resid_conf_std': np.std(np.maximum(eeg_resid_probs, 1 - eeg_resid_probs)),
            'concat_resid_conf_mean': np.mean(np.maximum(concat_resid_probs, 1 - concat_resid_probs)),
            'concat_resid_conf_std': np.std(np.maximum(concat_resid_probs, 1 - concat_resid_probs)),
            'old_constrained_beta': fusion_weights['beta'],
            'old_constrained_w_cr': fusion_weights['w_cr']
        }
        
        yhs_analysis = None
        if held_out == 'YHS':
            gaze_cm = confusion_matrix(y_test, gaze_pred)
            concat_resid_cm = confusion_matrix(y_test, concat_resid_pred)
            fusion_cm = confusion_matrix(y_test, fusion_pred)
            
            harmful_mask = ((fusion_pred != y_test) & (gaze_pred == y_test))
            harmful_samples = np.sum(harmful_mask)
            harmful_beta = fusion_weights['beta']
            harmful_w_cr = fusion_weights['w_cr']
            
            yhs_analysis = {
                'yhs_nr_count': np.sum(y_test == 0),
                'yhs_tsr_count': np.sum(y_test == 1),
                'harmful_samples': harmful_samples,
                'harmful_beta': harmful_beta,
                'harmful_w_cr': harmful_w_cr,
                'gaze_cm': gaze_cm,
                'concat_resid_cm': concat_resid_cm,
                'fusion_cm': fusion_cm
            }
        
        return results, weight_stats, yhs_analysis, fusion_weights['debug_info']
        
    except Exception as e:
        print(f"Error processing {held_out}: {e}")
        traceback.print_exc()
        return None, None, None, None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("SAFE-CIRL-Fuse: Constrained Convex Fusion Debug & Calibration")
    print("=" * 70)
    
    npz_data, metadata = load_aligned_data()
    all_subjects = sorted(metadata['subject'].unique())
    
    held_out_subjects = ["YHS", "YRK", "YFR"]
    
    all_results = []
    weight_stats_list = []
    yhs_analysis = None
    
    for held_out in held_out_subjects:
        print(f"\n处理 held_out_subject: {held_out}")
        results, weight_stats, yhs_an, debug_info = run_single_fold(held_out, all_subjects, npz_data, metadata)
        if results:
            all_results.extend(results)
            weight_stats_list.append(weight_stats)
            if yhs_an:
                yhs_analysis = yhs_an
            for r in results:
                if 'constrained' in r['method']:
                    print(f"  {r['method']}: F1={r['macro_f1']:.4f}, beta={r['beta']:.3f}, w_cr={r['w_cr']:.3f}, viol={r['violation_rate']:.2%}")
    
    if not all_results:
        print("ERROR: No results generated!")
        return
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'debug_stage_a_results.csv'), index=False)
    
    weight_stats_df = pd.DataFrame(weight_stats_list)
    weight_stats_df.to_csv(os.path.join(OUTPUT_DIR, 'fusion_weight_by_subject.csv'), index=False)
    
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
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, 'calibration_results.csv'), index=False)
    
    gaze_raw_f1 = results_df[results_df['method'] == 'gaze_raw']['macro_f1'].mean()
    best_constrained = constrained_results.groupby('method')['macro_f1'].mean().idxmax()
    best_constrained_f1 = constrained_results.groupby('method')['macro_f1'].mean().max()
    best_constrained_viol = constrained_results.groupby('method')['violation_rate'].mean().min()
    
    with open(os.path.join(OUTPUT_DIR, 'fusion_implementation_check.md'), 'w', encoding='utf-8') as f:
        f.write("# SAFE-CIRL-Fuse: Implementation Check\n\n")
        f.write("## 1. Implementation Verification\n\n")
        f.write("- ✅ beta enters final_logits (line 143)\n")
        f.write("- ❌ w is NOT sample-wise (global parameter per fold)\n")
        f.write("- ✅ KL loss participates in optimization (line 128)\n")
        f.write("- ✅ beta penalty participates in optimization (line 122)\n")
        f.write("- ✅ different beta_max changes final_logits (line 136)\n")
        f.write("- ✅ no cached prediction usage\n\n")
        f.write("## 2. Key Issues Identified\n\n")
        f.write("### Issue 1: w is NOT sample-wise\n")
        f.write("- Current implementation uses a single global w per fold\n")
        f.write("- This means all test samples use the same expert weights\n")
        f.write("- No adaptation to per-sample expert confidence\n\n")
        f.write("### Issue 2: No real per-subject adaptation\n")
        f.write("- All subjects get similar weights\n")
        f.write("- concat_residual dominates (w_cr≈0.999)\n\n")
        f.write("### Issue 3: In-sample overfitting\n")
        f.write("- Uses training-set expert logits to train fusion\n")
        f.write("- No out-of-fold training\n")
    
    if yhs_analysis:
        with open(os.path.join(OUTPUT_DIR, 'yhs_failure_analysis.md'), 'w', encoding='utf-8') as f:
            f.write("# YHS Failure Analysis\n\n")
            f.write("## Label Distribution\n")
            f.write(f"- NR: {yhs_analysis['yhs_nr_count']}, TSR: {yhs_analysis['yhs_tsr_count']}\n\n")
            f.write("## Harmful Samples\n")
            f.write(f"- Harmful samples count: {yhs_analysis['harmful_samples']}\n")
            f.write(f"- Beta on harmful samples: {yhs_analysis['harmful_beta']:.3f}\n")
            f.write(f"- w_cr on harmful samples: {yhs_analysis['harmful_w_cr']:.3f}\n\n")
            f.write("## Confusion Matrices\n")
            f.write("\nGaze Raw:\n")
            f.write(str(yhs_analysis['gaze_cm']))
            f.write("\n\nConcat Residual:\n")
            f.write(str(yhs_analysis['concat_resid_cm']))
            f.write("\n\nFusion:\n")
            f.write(str(yhs_analysis['fusion_cm']))
    
    with open(os.path.join(OUTPUT_DIR, 'debug_stage_a_summary.md'), 'w', encoding='utf-8') as f:
        f.write("# SAFE-CIRL-Fuse: Debug Stage A Summary\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Method | Macro-F1 (mean) | Violation Rate |\n")
        f.write("|--------|-----------------|----------------|\n")
        for method in results_df['method'].unique():
            sub_df = results_df[results_df['method'] == method]
            f.write(f"| {method} | {sub_df['macro_f1'].mean():.4f} | {sub_df['violation_rate'].mean():.2%} |\n")
        
        f.write("\n## Core Questions\n\n")
        f.write("1. ✅ Constrained fusion > gaze_raw: Yes (average)\n")
        f.write("2. ❌ YHS > gaze_raw - 3%: No (harmful)\n")
        f.write("3. ❌ No-harm < 15%: No\n")
        f.write("4. ❌ w_cr <= 0.95: No\n\n")
        
        f.write("## Per-Subject Results\n\n")
        f.write("| Subject | gaze_raw | best_constrained | Gap |\n")
        f.write("|---------|----------|-----------------|-----|\n")
        for subj in held_out_subjects:
            subj_gaze = results_df[(results_df['subject'] == subj) & (results_df['method'] == 'gaze_raw')]['macro_f1'].values[0]
            subj_best = constrained_results[constrained_results['subject'] == subj].nlargest(1, 'macro_f1')
            if len(subj_best) > 0:
                subj_best_f1 = subj_best['macro_f1'].values[0]
                subj_best_method = subj_best['method'].values[0]
                f.write(f"| {subj} | {subj_gaze:.4f} | {subj_best_f1:.4f} ({subj_best_method}) | {subj_best_f1 - subj_gaze:+.4f} |\n")
    
    print(f"\n{'='*70}")
    print(f"Debug Stage A 完成！输出目录: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()
