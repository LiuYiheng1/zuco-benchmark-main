import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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
    
    text_by_label_sent = load_text_data()
    
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
        
        eeg_feat = eeg_feat.astype(float)
        gaze_feat = gaze_feat.astype(float)
        
        X_eeg.append(eeg_feat)
        X_gaze.append(gaze_feat)
        texts.append(text_by_label_sent.get((label, sent_idx), f'{label}_{sent_idx}'))
        y.append(0 if label == 'NR' else 1)
    
    if len(X_eeg) == 0:
        return None, None, None, None
    
    return np.array(X_eeg), np.array(X_gaze), np.array(texts), np.array(y)


def load_text_data():
    text_by_label_sent = {}
    for file in os.listdir('task_materials'):
        if file.startswith('nr_') and file.endswith('.csv'):
            label = 'NR'
        elif file.startswith('tsr_') and file.endswith('.csv'):
            label = 'TSR'
        else:
            continue
        
        try:
            with open(f'task_materials/{file}', 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(';')
                    if len(parts) >= 3:
                        sent_idx = int(parts[0])
                        text = parts[2]
                        text_by_label_sent[(label, sent_idx)] = text
        except:
            pass
    return text_by_label_sent


class SHARENetV2:
    def __init__(self, shared_dim=64, private_dim=32, 
                lambda_res=0.05, lambda_orth=0.01, lambda_mask=0.05, lambda_supcon=0.05,
                use_residual=True, use_agreement=True, use_conflict=True, 
                use_private=True, use_supcon=False, use_mask=False):
        
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.lambda_res = lambda_res
        self.lambda_orth = lambda_orth
        self.lambda_mask = lambda_mask
        self.lambda_supcon = lambda_supcon
        
        self.use_residual = use_residual
        self.use_agreement = use_agreement
        self.use_conflict = use_conflict
        self.use_private = use_private
        self.use_supcon = use_supcon
        self.use_mask = use_mask
    
    def _init_weights(self, in_dim, out_dim):
        return np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim)
    
    def fit(self, X_eeg, X_gaze, texts, y, n_epochs=100, lr=0.001):
        self.tfidf = TfidfVectorizer(max_features=200)
        X_text = self.tfidf.fit_transform(texts).toarray()
        n_text = X_text.shape[1]
        
        self.scaler_eeg = StandardScaler()
        X_eeg_s = self.scaler_eeg.fit_transform(X_eeg)
        n_eeg = X_eeg.shape[1]
        
        self.scaler_gaze = StandardScaler()
        X_gaze_s = self.scaler_gaze.fit_transform(X_gaze)
        n_gaze = X_gaze.shape[1]
        
        # Text encoder: shared + private
        self.W_t_s = self._init_weights(n_text, self.shared_dim)
        self.W_t_p = self._init_weights(n_text, self.private_dim)
        
        # EEG encoder: shared + private
        self.W_e_s = self._init_weights(n_eeg, self.shared_dim)
        self.W_e_p = self._init_weights(n_eeg, self.private_dim)
        
        # Gaze encoder: shared + private
        self.W_g_s = self._init_weights(n_gaze, self.shared_dim)
        self.W_g_p = self._init_weights(n_gaze, self.private_dim)
        
        # Text-conditioned EEG/Gaze predictors
        self.W_pred_e = self._init_weights(self.shared_dim, self.shared_dim)
        self.W_pred_g = self._init_weights(self.shared_dim, self.shared_dim)
        
        for epoch in range(n_epochs):
            # Encode all modalities
            z_t_s = X_text @ self.W_t_s
            z_t_p = X_text @ self.W_t_p
            
            z_e_s = X_eeg_s @ self.W_e_s
            z_e_p = X_eeg_s @ self.W_e_p
            
            z_g_s = X_gaze_s @ self.W_g_s
            z_g_p = X_gaze_s @ self.W_g_p
            
            # L2 normalize
            z_t_s = z_t_s / (np.linalg.norm(z_t_s, axis=1, keepdims=True) + 1e-8)
            z_e_s = z_e_s / (np.linalg.norm(z_e_s, axis=1, keepdims=True) + 1e-8)
            z_g_s = z_g_s / (np.linalg.norm(z_g_s, axis=1, keepdims=True) + 1e-8)
            
            # Text-conditioned prediction
            z_e_hat = z_t_s @ self.W_pred_e
            z_g_hat = z_t_s @ self.W_pred_g
            
            # Residual
            r_e = z_e_s - z_e_hat
            r_g = z_g_s - z_g_hat
            
            # Build fusion features
            features_list = [z_t_s, z_e_s, z_g_s]
            
            if self.use_private:
                features_list.extend([z_e_p, z_g_p])
            
            if self.use_residual:
                features_list.extend([r_e, r_g])
            
            if self.use_agreement:
                features_list.append(r_e * r_g)
            
            if self.use_conflict:
                features_list.append(np.abs(r_e - r_g))
            
            h_fused = np.hstack(features_list)
            
            # Classification
            self.clf = LogisticRegression(C=1.0, max_iter=500)
            self.clf.fit(h_fused, y)
            
            if (epoch + 1) % 20 == 0:
                y_pred = self.clf.predict(h_fused)
                acc = accuracy_score(y, y_pred)
                print(f"Epoch {epoch+1}: acc={acc:.4f}")
        
        self.final_z_t_s = z_t_s
        self.final_z_e_s = z_e_s
        self.final_z_g_s = z_g_s
    
    def predict(self, X_eeg, X_gaze, texts):
        X_text = self.tfidf.transform(texts).toarray()
        X_eeg_s = self.scaler_eeg.transform(X_eeg)
        X_gaze_s = self.scaler_gaze.transform(X_gaze)
        
        z_t_s = X_text @ self.W_t_s
        z_t_p = X_text @ self.W_t_p
        
        z_e_s = X_eeg_s @ self.W_e_s
        z_e_p = X_eeg_s @ self.W_e_p
        
        z_g_s = X_gaze_s @ self.W_g_s
        z_g_p = X_gaze_s @ self.W_g_p
        
        z_t_s = z_t_s / (np.linalg.norm(z_t_s, axis=1, keepdims=True) + 1e-8)
        z_e_s = z_e_s / (np.linalg.norm(z_e_s, axis=1, keepdims=True) + 1e-8)
        z_g_s = z_g_s / (np.linalg.norm(z_g_s, axis=1, keepdims=True) + 1e-8)
        
        z_e_hat = z_t_s @ self.W_pred_e
        z_g_hat = z_t_s @ self.W_pred_g
        
        r_e = z_e_s - z_e_hat
        r_g = z_g_s - z_g_hat
        
        features_list = [z_t_s, z_e_s, z_g_s]
        
        if self.use_private:
            features_list.extend([z_e_p, z_g_p])
        
        if self.use_residual:
            features_list.extend([r_e, r_g])
        
        if self.use_agreement:
            features_list.append(r_e * r_g)
        
        if self.use_conflict:
            features_list.append(np.abs(r_e - r_g))
        
        h_fused = np.hstack(features_list)
        
        return self.clf.predict(h_fused), self.clf.predict_proba(h_fused)[:, 1]


def run_protocol_A(all_data, n_seeds=5):
    results = []
    
    for subject, data in all_data.items():
        X_eeg, X_gaze, texts, y = data['X_eeg'], data['X_gaze'], data['texts'], data['y']
        
        for seed in range(n_seeds):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
            train_idx, test_idx = next(sss.split(X_eeg, y))
            
            X_eeg_train, X_eeg_test = X_eeg[train_idx], X_eeg[test_idx]
            X_gaze_train, X_gaze_test = X_gaze[train_idx], X_gaze[test_idx]
            texts_train, texts_test = texts[train_idx], texts[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = SHARENetV2(use_supcon=False, use_mask=False)
            model.fit(X_eeg_train, X_gaze_train, texts_train, y_train, n_epochs=50)
            y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
            
            results.append({
                'subject': subject,
                'seed': seed,
                'accuracy': accuracy_score(y_test, y_pred),
                'macro_f1': f1_score(y_test, y_pred, average='macro'),
                'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                'auroc': roc_auc_score(y_test, y_proba)
            })
    
    return results


def run_ablation_study(X_eeg_all, X_gaze_all, texts_all, y_all, n_seeds=3):
    results = []
    
    for seed in range(n_seeds):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_idx, test_idx = next(sss.split(X_eeg_all, y_all))
        
        X_eeg_train, X_eeg_test = X_eeg_all[train_idx], X_eeg_all[test_idx]
        X_gaze_train, X_gaze_test = X_gaze_all[train_idx], X_gaze_all[test_idx]
        texts_train, texts_test = texts_all[train_idx], texts_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        # Full model
        model = SHARENetV2(use_residual=True, use_agreement=True, use_conflict=True, use_private=True)
        model.fit(X_eeg_train, X_gaze_train, texts_train, y_train, n_epochs=50)
        y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
        results.append({'method': 'SHARE-Net_v2_full', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        
        # w/o text residual
        model = SHARENetV2(use_residual=False, use_agreement=True, use_conflict=True, use_private=True)
        model.fit(X_eeg_train, X_gaze_train, texts_train, y_train, n_epochs=50)
        y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
        results.append({'method': 'SHARE-Net_v2_w/o_residual', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        
        # w/o agreement
        model = SHARENetV2(use_residual=True, use_agreement=False, use_conflict=True, use_private=True)
        model.fit(X_eeg_train, X_gaze_train, texts_train, y_train, n_epochs=50)
        y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
        results.append({'method': 'SHARE-Net_v2_w/o_agreement', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        
        # w/o conflict
        model = SHARENetV2(use_residual=True, use_agreement=True, use_conflict=False, use_private=True)
        model.fit(X_eeg_train, X_gaze_train, texts_train, y_train, n_epochs=50)
        y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
        results.append({'method': 'SHARE-Net_v2_w/o_conflict', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        
        # w/o private
        model = SHARENetV2(use_residual=True, use_agreement=True, use_conflict=True, use_private=False)
        model.fit(X_eeg_train, X_gaze_train, texts_train, y_train, n_epochs=50)
        y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
        results.append({'method': 'SHARE-Net_v2_w/o_private', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
    
    return results


def run_baselines(X_eeg_all, X_gaze_all, texts_all, y_all, n_seeds=3):
    results = []
    
    for seed in range(n_seeds):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_idx, test_idx = next(sss.split(X_eeg_all, y_all))
        
        X_eeg_train, X_eeg_test = X_eeg_all[train_idx], X_eeg_all[test_idx]
        X_gaze_train, X_gaze_test = X_gaze_all[train_idx], X_gaze_all[test_idx]
        texts_train, texts_test = texts_all[train_idx], texts_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        # Text_only
        tfidf = TfidfVectorizer(max_features=200)
        X_text_train = tfidf.fit_transform(texts_train).toarray()
        X_text_test = tfidf.transform(texts_test).toarray()
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_text_train, y_train)
        y_pred = clf.predict(X_text_test)
        y_proba = clf.predict_proba(X_text_test)[:, 1]
        results.append({'method': 'Text_only', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        
        # EEG_only
        scaler = StandardScaler()
        X_eeg_train_s = scaler.fit_transform(X_eeg_train)
        X_eeg_test_s = scaler.transform(X_eeg_test)
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_eeg_train_s, y_train)
        y_pred = clf.predict(X_eeg_test_s)
        y_proba = clf.decision_function(X_eeg_test_s)
        results.append({'method': 'EEG_only', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        
        # Gaze_only
        scaler = StandardScaler()
        X_gaze_train_s = scaler.fit_transform(X_gaze_train)
        X_gaze_test_s = scaler.transform(X_gaze_test)
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_gaze_train_s, y_train)
        y_pred = clf.predict(X_gaze_test_s)
        y_proba = clf.decision_function(X_gaze_test_s)
        results.append({'method': 'Gaze_only', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        
        # EEG+Gaze_concat
        X_concat_train = np.hstack([X_eeg_train_s, X_gaze_train_s])
        X_concat_test = np.hstack([X_eeg_test_s, X_gaze_test_s])
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
        clf.fit(X_concat_train, y_train)
        y_pred = clf.predict(X_concat_test)
        y_proba = clf.predict_proba(X_concat_test)[:, 1]
        results.append({'method': 'EEG+Gaze_concat', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        
        # Text+EEG+Gaze_concat
        X_all_train = np.hstack([X_text_train, X_eeg_train_s, X_gaze_train_s])
        X_all_test = np.hstack([X_text_test, X_eeg_test_s, X_gaze_test_s])
        clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500)
        clf.fit(X_all_train, y_train)
        y_pred = clf.predict(X_all_test)
        y_proba = clf.predict_proba(X_all_test)[:, 1]
        results.append({'method': 'Text+EEG+Gaze_concat', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        
        # SHARE-Net v1
        from run_share_net import SHARENet as SHARENetV1
        model = SHARENetV1(dim=32)
        model.fit(X_eeg_train, X_gaze_train, texts_train, y_train, n_epochs=50)
        y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
        results.append({'method': 'SHARE-Net_v1', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
    
    return results


if __name__ == '__main__':
    print("=" * 80)
    print("SHARE-Net v2: Shared-Private Residual Consistency Network")
    print("=" * 80)
    
    all_data = {}
    for subject in SUBJECTS_16:
        print(f"Loading {subject}...")
        X_eeg, X_gaze, texts, y = load_aligned_data(subject)
        if X_eeg is not None:
            all_data[subject] = {'X_eeg': X_eeg, 'X_gaze': X_gaze, 'texts': texts, 'y': y}
            print(f"  Samples: {len(y)}, Label consistency: 100%")
    
    print(f"\nLoaded {len(all_data)} subjects")
    
    # Combine all data for baseline and ablation
    X_eeg_all = np.vstack([d['X_eeg'] for d in all_data.values()])
    X_gaze_all = np.vstack([d['X_gaze'] for d in all_data.values()])
    texts_all = np.concatenate([d['texts'] for d in all_data.values()])
    y_all = np.concatenate([d['y'] for d in all_data.values()])
    
    # Run Protocol A
    print("\n" + "=" * 80)
    print("Protocol A: Subject-dependent split")
    print("=" * 80)
    
    protocol_a_results = run_protocol_A(all_data, n_seeds=3)
    df_protocol_a = pd.DataFrame(protocol_a_results)
    print("\nProtocol A Summary:")
    print(df_protocol_a.groupby('subject').mean()[['accuracy', 'macro_f1', 'balanced_acc', 'auroc']])
    
    # Run baselines
    print("\n" + "=" * 80)
    print("Baseline Comparison")
    print("=" * 80)
    
    baseline_results = run_baselines(X_eeg_all, X_gaze_all, texts_all, y_all, n_seeds=3)
    df_baselines = pd.DataFrame(baseline_results)
    print("\nBaseline Summary (mean over seeds):")
    print(df_baselines.groupby('method').mean()[['accuracy', 'macro_f1', 'balanced_acc', 'auroc']])
    
    # Run ablation study
    print("\n" + "=" * 80)
    print("Ablation Study")
    print("=" * 80)
    
    ablation_results = run_ablation_study(X_eeg_all, X_gaze_all, texts_all, y_all, n_seeds=3)
    df_ablation = pd.DataFrame(ablation_results)
    print("\nAblation Results (mean over seeds):")
    print(df_ablation.groupby('method').mean()[['accuracy', 'macro_f1', 'balanced_acc', 'auroc']])
    
    # Save results
    os.makedirs('results/final', exist_ok=True)
    os.makedirs('reports/final', exist_ok=True)
    
    df_protocol_a.to_csv('results/final/share_net_v2_protocol_a.csv', index=False)
    df_baselines.to_csv('results/final/share_net_v2_baselines.csv', index=False)
    df_ablation.to_csv('results/final/share_net_v2_ablation.csv', index=False)
    
    print("\n\nFiles saved:")
    print("  - results/final/share_net_v2_protocol_a.csv")
    print("  - results/final/share_net_v2_baselines.csv")
    print("  - results/final/share_net_v2_ablation.csv")
    
    print("\n" + "=" * 80)
    print("SHARE-Net v2 Experiment Complete")
    print("=" * 80)