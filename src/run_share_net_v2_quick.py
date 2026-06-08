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


class SHARENetV2:
    def __init__(self, use_residual=True, use_agreement=True, use_conflict=True, use_private=True):
        self.use_residual = use_residual
        self.use_agreement = use_agreement
        self.use_conflict = use_conflict
        self.use_private = use_private
    
    def fit(self, X_eeg, X_gaze, texts, y):
        self.tfidf = TfidfVectorizer(max_features=200)
        X_text = self.tfidf.fit_transform(texts).toarray()
        
        self.scaler_eeg = StandardScaler()
        X_eeg_s = self.scaler_eeg.fit_transform(X_eeg)
        
        self.scaler_gaze = StandardScaler()
        X_gaze_s = self.scaler_gaze.fit_transform(X_gaze)
        
        n_text, n_eeg, n_gaze = X_text.shape[1], X_eeg.shape[1], X_gaze.shape[1]
        
        self.W_t_s = np.random.randn(n_text, 64) * 0.01
        self.W_e_s = np.random.randn(n_eeg, 64) * 0.01
        self.W_g_s = np.random.randn(n_gaze, 64) * 0.01
        
        z_t_s = X_text @ self.W_t_s
        z_e_s = X_eeg_s @ self.W_e_s
        z_g_s = X_gaze_s @ self.W_g_s
        
        z_t_s = z_t_s / (np.linalg.norm(z_t_s, axis=1, keepdims=True) + 1e-8)
        z_e_s = z_e_s / (np.linalg.norm(z_e_s, axis=1, keepdims=True) + 1e-8)
        z_g_s = z_g_s / (np.linalg.norm(z_g_s, axis=1, keepdims=True) + 1e-8)
        
        self.W_pred_e = np.random.randn(64, 64) * 0.01
        self.W_pred_g = np.random.randn(64, 64) * 0.01
        
        z_e_hat = z_t_s @ self.W_pred_e
        z_g_hat = z_t_s @ self.W_pred_g
        
        r_e = z_e_s - z_e_hat
        r_g = z_g_s - z_g_hat
        
        features_list = [z_t_s, z_e_s, z_g_s]
        
        if self.use_private:
            self.W_e_p = np.random.randn(n_eeg, 32) * 0.01
            self.W_g_p = np.random.randn(n_gaze, 32) * 0.01
            z_e_p = X_eeg_s @ self.W_e_p
            z_g_p = X_gaze_s @ self.W_g_p
            features_list.extend([z_e_p, z_g_p])
        
        if self.use_residual:
            features_list.extend([r_e, r_g])
        
        if self.use_agreement:
            features_list.append(r_e * r_g)
        
        if self.use_conflict:
            features_list.append(np.abs(r_e - r_g))
        
        h_fused = np.hstack(features_list)
        self.clf = LogisticRegression(C=1.0, max_iter=500)
        self.clf.fit(h_fused, y)
    
    def predict(self, X_eeg, X_gaze, texts):
        X_text = self.tfidf.transform(texts).toarray()
        X_eeg_s = self.scaler_eeg.transform(X_eeg)
        X_gaze_s = self.scaler_gaze.transform(X_gaze)
        
        z_t_s = X_text @ self.W_t_s
        z_e_s = X_eeg_s @ self.W_e_s
        z_g_s = X_gaze_s @ self.W_g_s
        
        z_t_s = z_t_s / (np.linalg.norm(z_t_s, axis=1, keepdims=True) + 1e-8)
        z_e_s = z_e_s / (np.linalg.norm(z_e_s, axis=1, keepdims=True) + 1e-8)
        z_g_s = z_g_s / (np.linalg.norm(z_g_s, axis=1, keepdims=True) + 1e-8)
        
        z_e_hat = z_t_s @ self.W_pred_e
        z_g_hat = z_t_s @ self.W_pred_g
        
        r_e = z_e_s - z_e_hat
        r_g = z_g_s - z_g_hat
        
        features_list = [z_t_s, z_e_s, z_g_s]
        
        if self.use_private:
            z_e_p = X_eeg_s @ self.W_e_p
            z_g_p = X_gaze_s @ self.W_g_p
            features_list.extend([z_e_p, z_g_p])
        
        if self.use_residual:
            features_list.extend([r_e, r_g])
        
        if self.use_agreement:
            features_list.append(r_e * r_g)
        
        if self.use_conflict:
            features_list.append(np.abs(r_e - r_g))
        
        h_fused = np.hstack(features_list)
        return self.clf.predict(h_fused), self.clf.predict_proba(h_fused)[:, 1]


def run_comparison():
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
        
        # Text+EEG+Gaze_concat
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
        
        # SHARE-Net v2 full
        model = SHARENetV2(use_residual=True, use_agreement=True, use_conflict=True, use_private=True)
        model.fit(X_eeg_train, X_gaze_train, texts_train, y_train)
        y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
        results.append({'method': 'SHARE-Net_v2_full', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        
        # SHARE-Net v2 w/o agreement
        model = SHARENetV2(use_residual=True, use_agreement=False, use_conflict=True, use_private=True)
        model.fit(X_eeg_train, X_gaze_train, texts_train, y_train)
        y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
        results.append({'method': 'SHARE-Net_v2_w/o_agreement', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        
        # SHARE-Net v2 w/o conflict
        model = SHARENetV2(use_residual=True, use_agreement=True, use_conflict=False, use_private=True)
        model.fit(X_eeg_train, X_gaze_train, texts_train, y_train)
        y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
        results.append({'method': 'SHARE-Net_v2_w/o_conflict', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("=" * 80)
    print("SHARE-Net v2: Quick Comparison")
    print("=" * 80)
    
    df = run_comparison()
    
    print("\nResults (mean over 3 seeds):")
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
    df.to_csv('results/final/share_net_v2_results.csv', index=False)
    
    print("\n\nFiles saved:")
    print("  - results/final/share_net_v2_results.csv")
    
    print("\n" + "=" * 80)
    print("SHARE-Net v2 Quick Comparison Complete")
    print("=" * 80)