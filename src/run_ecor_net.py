import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import warnings
warnings.filterwarnings('ignore')

SUBJECTS_16 = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS',
                'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_aligned_data(subject):
    eeg_path = f'features/{subject}_electrode_features_all.npy'
    gaze_path = f'features/{subject}_sent_gaze_sacc.npy'
    
    if not os.path.exists(eeg_path) or not os.path.exists(gaze_path):
        return None, None, None, None
    
    eeg_data = np.load(eeg_path, allow_pickle=True).item()
    gaze_data = np.load(gaze_path, allow_pickle=True).item()
    
    gaze_by_label_sent = {}
    for key in gaze_data.keys():
        parts = key.split('_')
        if len(parts) >= 3:
            label = parts[1]
            sent_idx = int(parts[2])
            gaze_by_label_sent[(label, sent_idx)] = key
    
    X_eeg, X_gaze, texts, y = [], [], [], []
    
    for eeg_key in eeg_data.keys():
        parts = eeg_key.split('_')
        if len(parts) < 3:
            continue
        
        label = parts[1]
        sent_idx = int(parts[2])
        
        gaze_key = gaze_by_label_sent.get((label, sent_idx))
        if gaze_key is None:
            continue
        
        eeg_feat = np.array(eeg_data[eeg_key])
        gaze_feat = np.array(gaze_data[gaze_key])
        
        if eeg_feat[-1] in ['NR', 'TSR']:
            eeg_feat = eeg_feat[:-1]
        if gaze_feat[-1] in ['NR', 'TSR']:
            gaze_feat = gaze_feat[:-1]
        
        X_eeg.append(eeg_feat.astype(float))
        X_gaze.append(gaze_feat.astype(float))
        texts.append(f'{label}_{sent_idx}')
        y.append(0 if label == 'NR' else 1)
    
    return np.array(X_eeg), np.array(X_gaze), np.array(texts), np.array(y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def entropy(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def train_base_models(X_text, X_eeg, X_gaze, y):
    models = {}
    
    # Text_only
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500)
    clf.fit(X_text, y)
    models['text'] = clf
    
    # EEG_only
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500)
    clf.fit(X_eeg, y)
    models['eeg'] = clf
    
    # Gaze_only
    clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500)
    clf.fit(X_gaze, y)
    models['gaze'] = clf
    
    # Text+EEG
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500)
    clf.fit(np.hstack([X_text, X_eeg]), y)
    models['te'] = clf
    
    # Text+Gaze
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    clf.fit(np.hstack([X_text, X_gaze]), y)
    models['tg'] = clf
    
    # EEG+Gaze
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    clf.fit(np.hstack([X_eeg, X_gaze]), y)
    models['eg'] = clf
    
    # Text+EEG+Gaze
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500)
    clf.fit(np.hstack([X_text, X_eeg, X_gaze]), y)
    models['all'] = clf
    
    return models


def get_predictions(models, X_text, X_eeg, X_gaze):
    preds = {}
    
    preds['p_text'] = models['text'].predict_proba(X_text)[:, 1]
    preds['p_eeg'] = models['eeg'].predict_proba(X_eeg)[:, 1]
    preds['p_gaze'] = models['gaze'].predict_proba(X_gaze)[:, 1]
    preds['p_te'] = models['te'].predict_proba(np.hstack([X_text, X_eeg]))[:, 1]
    preds['p_tg'] = models['tg'].predict_proba(np.hstack([X_text, X_gaze]))[:, 1]
    preds['p_eg'] = models['eg'].predict_proba(np.hstack([X_eeg, X_gaze]))[:, 1]
    preds['p_all'] = models['all'].predict_proba(np.hstack([X_text, X_eeg, X_gaze]))[:, 1]
    
    return preds


def construct_features(preds, mode='full'):
    p_text = preds['p_text']
    p_eeg = preds['p_eeg']
    p_gaze = preds['p_gaze']
    p_te = preds['p_te']
    p_tg = preds['p_tg']
    p_eg = preds['p_eg']
    p_all = preds['p_all']
    
    features_list = []
    
    if mode in ['full', 'z_only']:
        features_list.extend([p_text, p_eeg, p_gaze, p_te, p_tg, p_eg, p_all])
    
    if mode in ['full', 'conflict_only']:
        features_list.extend([
            np.abs(p_eeg - p_gaze),
            np.abs(p_text - p_eeg),
            np.abs(p_text - p_gaze),
            np.abs(p_all - p_eg)
        ])
    
    if mode == 'full':
        features_list.extend([
            p_eeg * p_gaze,
            p_text * p_eeg,
            p_text * p_gaze,
            entropy(p_all),
            np.abs(p_all - 0.5),
            4 * p_all * (1 - p_all)
        ])
    
    return np.column_stack(features_list)


def train_ecor_net(X_text_train, X_eeg_train, X_gaze_train, y_train, mode='full', lambda_corr=0.3):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = []
    oof_y = []
    
    for train_idx, val_idx in kf.split(X_text_train, y_train):
        X_t_tr, X_t_val = X_text_train[train_idx], X_text_train[val_idx]
        X_e_tr, X_e_val = X_eeg_train[train_idx], X_eeg_train[val_idx]
        X_g_tr, X_g_val = X_gaze_train[train_idx], X_gaze_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        models = train_base_models(X_t_tr, X_e_tr, X_g_tr, y_tr)
        preds = get_predictions(models, X_t_val, X_e_val, X_g_val)
        
        oof_preds.append(preds)
        oof_y.append(y_val)
    
    oof_p_all = np.concatenate([p['p_all'] for p in oof_preds])
    oof_features = construct_features({
        'p_text': np.concatenate([p['p_text'] for p in oof_preds]),
        'p_eeg': np.concatenate([p['p_eeg'] for p in oof_preds]),
        'p_gaze': np.concatenate([p['p_gaze'] for p in oof_preds]),
        'p_te': np.concatenate([p['p_te'] for p in oof_preds]),
        'p_tg': np.concatenate([p['p_tg'] for p in oof_preds]),
        'p_eg': np.concatenate([p['p_eg'] for p in oof_preds]),
        'p_all': oof_p_all
    }, mode=mode)
    oof_y_all = np.concatenate(oof_y)
    
    corrector = RidgeClassifier()
    corrector.fit(oof_features, oof_y_all)
    
    final_models = train_base_models(X_text_train, X_eeg_train, X_gaze_train, y_train)
    
    return {
        'base_models': final_models,
        'corrector': corrector,
        'mode': mode,
        'lambda_corr': lambda_corr
    }


def predict_ecor_net(model_dict, X_text, X_eeg, X_gaze, use_uncertainty=True):
    preds = get_predictions(model_dict['base_models'], X_text, X_eeg, X_gaze)
    features = construct_features(preds, mode=model_dict['mode'])
    
    p_base = preds['p_all']
    u = 4 * p_base * (1 - p_base)
    
    delta_logit = model_dict['corrector'].decision_function(features)
    
    if use_uncertainty:
        logit_final = logit(p_base) + model_dict['lambda_corr'] * u * delta_logit
    else:
        logit_final = logit(p_base) + model_dict['lambda_corr'] * delta_logit
    
    p_final = sigmoid(logit_final)
    y_pred = (p_final >= 0.5).astype(int)
    
    return y_pred, p_final


def run_experiment():
    all_data = {}
    for subject in SUBJECTS_16:
        X_eeg, X_gaze, texts, y = load_aligned_data(subject)
        if X_eeg is not None:
            all_data[subject] = {'X_eeg': X_eeg, 'X_gaze': X_gaze, 'texts': texts, 'y': y}
    
    X_eeg_all = np.vstack([d['X_eeg'] for d in all_data.values()])
    X_gaze_all = np.vstack([d['X_gaze'] for d in all_data.values()])
    texts_all = np.concatenate([d['texts'] for d in all_data.values()])
    y_all = np.concatenate([d['y'] for d in all_data.values()])
    
    results = []
    
    for seed in range(3):
        print(f"\n=== Seed {seed} ===")
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_idx, test_idx = next(sss.split(X_eeg_all, y_all))
        
        X_eeg_train, X_eeg_test = X_eeg_all[train_idx], X_eeg_all[test_idx]
        X_gaze_train, X_gaze_test = X_gaze_all[train_idx], X_gaze_all[test_idx]
        texts_train, texts_test = texts_all[train_idx], texts_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        tfidf = TfidfVectorizer(max_features=200)
        X_text_train = tfidf.fit_transform(texts_train).toarray()
        X_text_test = tfidf.transform(texts_test).toarray()
        
        scaler_eeg = StandardScaler()
        X_eeg_train_s = scaler_eeg.fit_transform(X_eeg_train)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        
        scaler_gaze = StandardScaler()
        X_gaze_train_s = scaler_gaze.fit_transform(X_gaze_train)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)
        
        # Baseline: Text+EEG+Gaze_concat
        X_all_train = np.hstack([X_text_train, X_eeg_train_s, X_gaze_train_s])
        X_all_test = np.hstack([X_text_test, X_eeg_test_s, X_gaze_test_s])
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500)
        clf.fit(X_all_train, y_train)
        y_pred = clf.predict(X_all_test)
        y_proba = clf.predict_proba(X_all_test)[:, 1]
        results.append({'method': 'Text+EEG+Gaze_concat', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        print(f"Text+EEG+Gaze_concat: {accuracy_score(y_test, y_pred):.4f}")
        
        # ECOR-Net_z_only
        model = train_ecor_net(X_text_train, X_eeg_train_s, X_gaze_train_s, y_train, 
                              mode='z_only', lambda_corr=0.3)
        y_pred, y_proba = predict_ecor_net(model, X_text_test, X_eeg_test_s, X_gaze_test_s)
        results.append({'method': 'ECOR-Net_z_only', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        print(f"ECOR-Net_z_only: {accuracy_score(y_test, y_pred):.4f}")
        
        # ECOR-Net_conflict_only
        model = train_ecor_net(X_text_train, X_eeg_train_s, X_gaze_train_s, y_train,
                              mode='conflict_only', lambda_corr=0.3)
        y_pred, y_proba = predict_ecor_net(model, X_text_test, X_eeg_test_s, X_gaze_test_s)
        results.append({'method': 'ECOR-Net_conflict_only', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        print(f"ECOR-Net_conflict_only: {accuracy_score(y_test, y_pred):.4f}")
        
        # ECOR-Net_full_no_uncertainty
        model = train_ecor_net(X_text_train, X_eeg_train_s, X_gaze_train_s, y_train,
                              mode='full', lambda_corr=0.3)
        y_pred, y_proba = predict_ecor_net(model, X_text_test, X_eeg_test_s, X_gaze_test_s, use_uncertainty=False)
        results.append({'method': 'ECOR-Net_full_no_uncertainty', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        print(f"ECOR-Net_full_no_uncertainty: {accuracy_score(y_test, y_pred):.4f}")
        
        # ECOR-Net_full_with_uncertainty
        y_pred, y_proba = predict_ecor_net(model, X_text_test, X_eeg_test_s, X_gaze_test_s, use_uncertainty=True)
        results.append({'method': 'ECOR-Net_full_with_uncertainty', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        print(f"ECOR-Net_full_with_uncertainty: {accuracy_score(y_test, y_pred):.4f}")
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("=" * 80)
    print("ECOR-Net: Error-Corrected Residual Fusion")
    print("=" * 80)
    
    df = run_experiment()
    
    print("\n" + "=" * 80)
    print("Results Summary (mean over 3 seeds):")
    print("=" * 80)
    
    summary = df.groupby('method').agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        f1_mean=('macro_f1', 'mean'),
        f1_std=('macro_f1', 'std'),
        balanced_acc_mean=('balanced_acc', 'mean'),
        auroc_mean=('auroc', 'mean')
    ).round(4)
    
    print(summary)
    
    os.makedirs('results/final', exist_ok=True)
    df.to_csv('results/final/ecor_net_results.csv', index=False)
    
    print("\n\nFiles saved:")
    print("  - results/final/ecor_net_results.csv")
    
    print("\n" + "=" * 80)
    print("ECOR-Net Experiment Complete")
    print("=" * 80)