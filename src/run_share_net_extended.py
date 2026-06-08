import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
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
    
    X_eeg, X_gaze, texts, y, keys = [], [], [], [], []
    
    for eeg_key in eeg_data.keys():
        parts = eeg_key.split('_')
        if len(parts) < 3:
            continue
        
        label = parts[1]
        sent_idx = int(parts[2])
        
        gaze_key = gaze_by_label_sent.get((label, sent_idx))
        text_key = (label, sent_idx)
        
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
        texts.append(text_by_label_sent.get(text_key, f'{label}_{sent_idx}'))
        y.append(0 if label == 'NR' else 1)
        keys.append((subject, label, sent_idx))
    
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


class SHARENet:
    def __init__(self, dim=32, lambda_align=0.1, use_align_loss=True, use_eg_consistency=True):
        self.dim = dim
        self.lambda_align = lambda_align
        self.use_align_loss = use_align_loss
        self.use_eg_consistency = use_eg_consistency
    
    def fit(self, X_eeg, X_gaze, texts, y):
        self.tfidf = TfidfVectorizer(max_features=100)
        X_text = self.tfidf.fit_transform(texts).toarray()
        
        self.scaler_eeg = StandardScaler()
        X_eeg_s = self.scaler_eeg.fit_transform(X_eeg)
        self.scaler_gaze = StandardScaler()
        X_gaze_s = self.scaler_gaze.fit_transform(X_gaze)
        
        self.W_text = np.random.randn(X_text.shape[1], self.dim) * 0.01
        self.W_eeg = np.random.randn(X_eeg.shape[1], self.dim) * 0.01
        self.W_gaze = np.random.randn(X_gaze.shape[1], self.dim) * 0.01
        
        z_t = X_text @ self.W_text
        z_e = X_eeg_s @ self.W_eeg
        z_g = X_gaze_s @ self.W_gaze
        
        z_t = z_t / (np.linalg.norm(z_t, axis=1, keepdims=True) + 1e-8)
        z_e = z_e / (np.linalg.norm(z_e, axis=1, keepdims=True) + 1e-8)
        z_g = z_g / (np.linalg.norm(z_g, axis=1, keepdims=True) + 1e-8)
        
        r_te = np.hstack([z_t, z_e, np.abs(z_t - z_e), z_t * z_e])
        r_tg = np.hstack([z_t, z_g, np.abs(z_t - z_g), z_t * z_g])
        r_eg = np.hstack([z_e, z_g, np.abs(z_e - z_g), z_e * z_g])
        
        if self.use_eg_consistency:
            h_relation = np.hstack([r_te, r_tg, r_eg])
        else:
            h_relation = np.hstack([r_te, r_tg])
        
        n_hidden = h_relation.shape[1]
        self.W_gate = np.random.randn(n_hidden, self.dim) * 0.01
        
        for epoch in range(50):
            z_t = X_text @ self.W_text
            z_e = X_eeg_s @ self.W_eeg
            z_g = X_gaze_s @ self.W_gaze
            
            z_t = z_t / (np.linalg.norm(z_t, axis=1, keepdims=True) + 1e-8)
            z_e = z_e / (np.linalg.norm(z_e, axis=1, keepdims=True) + 1e-8)
            z_g = z_g / (np.linalg.norm(z_g, axis=1, keepdims=True) + 1e-8)
            
            r_te = np.hstack([z_t, z_e, np.abs(z_t - z_e), z_t * z_e])
            r_tg = np.hstack([z_t, z_g, np.abs(z_t - z_g), z_t * z_g])
            r_eg = np.hstack([z_e, z_g, np.abs(z_e - z_g), z_e * z_g])
            
            if self.use_eg_consistency:
                h_relation = np.hstack([r_te, r_tg, r_eg])
            else:
                h_relation = np.hstack([r_te, r_tg])
            
            gamma = 1 / (1 + np.exp(-(h_relation @ self.W_gate)))
            h_eg = gamma * z_e + (1 - gamma) * z_g
            
            h_fused = np.hstack([z_t, h_eg, np.abs(z_t - h_eg), z_t * h_eg, h_relation])
            self.clf = LogisticRegression(C=1.0, max_iter=500)
            self.clf.fit(h_fused, y)
        
        self.final_z_t = z_t
        self.final_z_e = z_e
        self.final_z_g = z_g
    
    def predict(self, X_eeg, X_gaze, texts):
        X_text = self.tfidf.transform(texts).toarray()
        X_eeg_s = self.scaler_eeg.transform(X_eeg)
        X_gaze_s = self.scaler_gaze.transform(X_gaze)
        
        z_t = X_text @ self.W_text
        z_e = X_eeg_s @ self.W_eeg
        z_g = X_gaze_s @ self.W_gaze
        
        z_t = z_t / (np.linalg.norm(z_t, axis=1, keepdims=True) + 1e-8)
        z_e = z_e / (np.linalg.norm(z_e, axis=1, keepdims=True) + 1e-8)
        z_g = z_g / (np.linalg.norm(z_g, axis=1, keepdims=True) + 1e-8)
        
        r_te = np.hstack([z_t, z_e, np.abs(z_t - z_e), z_t * z_e])
        r_tg = np.hstack([z_t, z_g, np.abs(z_t - z_g), z_t * z_g])
        r_eg = np.hstack([z_e, z_g, np.abs(z_e - z_g), z_e * z_g])
        
        if self.use_eg_consistency:
            h_relation = np.hstack([r_te, r_tg, r_eg])
        else:
            h_relation = np.hstack([r_te, r_tg])
        
        gamma = 1 / (1 + np.exp(-(h_relation @ self.W_gate)))
        h_eg = gamma * z_e + (1 - gamma) * z_g
        
        h_fused = np.hstack([z_t, h_eg, np.abs(z_t - h_eg), z_t * h_eg, h_relation])
        
        return self.clf.predict(h_fused), self.clf.predict_proba(h_fused)[:, 1]


def run_protocol_C(X_eeg_all, X_gaze_all, texts_all, y_all, keys_all, n_splits=5):
    label_sent_set = set([(k[1], k[2]) for k in keys_all])
    label_sent_list = list(label_sent_set)
    
    results = []
    for seed in range(n_splits):
        np.random.seed(seed)
        np.random.shuffle(label_sent_list)
        split_idx = int(len(label_sent_list) * 0.7)
        train_sent = set(label_sent_list[:split_idx])
        test_sent = set(label_sent_list[split_idx:])
        
        train_mask = [(k[1], k[2]) in train_sent for k in keys_all]
        test_mask = [(k[1], k[2]) in test_sent for k in keys_all]
        
        X_eeg_train = X_eeg_all[train_mask]
        X_gaze_train = X_gaze_all[train_mask]
        texts_train = texts_all[train_mask]
        y_train = y_all[train_mask]
        
        X_eeg_test = X_eeg_all[test_mask]
        X_gaze_test = X_gaze_all[test_mask]
        texts_test = texts_all[test_mask]
        y_test = y_all[test_mask]
        
        model = SHARENet()
        model.fit(X_eeg_train, X_gaze_train, texts_train, y_train)
        y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
        
        results.append({
            'seed': seed,
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'balanced_acc': balanced_accuracy_score(y_test, y_pred),
            'auroc': roc_auc_score(y_test, y_proba),
            'train_sentences': len(train_sent),
            'test_sentences': len(test_sent)
        })
    
    return results


def run_ablation_study(X_eeg, X_gaze, texts, y):
    results = []
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_idx, test_idx = next(sss.split(X_eeg, y))
    
    X_eeg_train, X_eeg_test = X_eeg[train_idx], X_eeg[test_idx]
    X_gaze_train, X_gaze_test = X_gaze[train_idx], X_gaze[test_idx]
    texts_train, texts_test = texts[train_idx], texts[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Full SHARE-Net
    model = SHARENet(dim=32)
    model.fit(X_eeg_train, X_gaze_train, texts_train, y_train)
    y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
    results.append({'method': 'SHARE-Net_full', 'accuracy': accuracy_score(y_test, y_pred),
                    'macro_f1': f1_score(y_test, y_pred, average='macro'),
                    'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                    'auroc': roc_auc_score(y_test, y_proba)})
    
    # Without EEG-Gaze consistency
    model = SHARENet(dim=32, use_eg_consistency=False)
    model.fit(X_eeg_train, X_gaze_train, texts_train, y_train)
    y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
    results.append({'method': 'SHARE-Net_w/o_eg_consistency', 'accuracy': accuracy_score(y_test, y_pred),
                    'macro_f1': f1_score(y_test, y_pred, average='macro'),
                    'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                    'auroc': roc_auc_score(y_test, y_proba)})
    
    # Concat-only fusion
    tfidf = TfidfVectorizer(max_features=100)
    X_text_train = tfidf.fit_transform(texts_train).toarray()
    X_text_test = tfidf.transform(texts_test).toarray()
    
    scaler_eeg = StandardScaler()
    X_eeg_train_s = scaler_eeg.fit_transform(X_eeg_train)
    X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
    
    scaler_gaze = StandardScaler()
    X_gaze_train_s = scaler_gaze.fit_transform(X_gaze_train)
    X_gaze_test_s = scaler_gaze.transform(X_gaze_test)
    
    X_concat_train = np.hstack([X_text_train, X_eeg_train_s, X_gaze_train_s])
    X_concat_test = np.hstack([X_text_test, X_eeg_test_s, X_gaze_test_s])
    
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    clf.fit(X_concat_train, y_train)
    y_pred = clf.predict(X_concat_test)
    y_proba = clf.predict_proba(X_concat_test)[:, 1]
    results.append({'method': 'SHARE-Net_concat_only', 'accuracy': accuracy_score(y_test, y_pred),
                    'macro_f1': f1_score(y_test, y_pred, average='macro'),
                    'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                    'auroc': roc_auc_score(y_test, y_proba)})
    
    return results


if __name__ == '__main__':
    print("=" * 80)
    print("SHARE-Net Protocol C and Ablation Study")
    print("=" * 80)
    
    all_data = {}
    for subject in SUBJECTS_16:
        X_eeg, X_gaze, texts, y = load_aligned_data(subject)
        if X_eeg is not None:
            all_data[subject] = {'X_eeg': X_eeg, 'X_gaze': X_gaze, 'texts': texts, 'y': y}
    
    X_eeg_all = np.vstack([d['X_eeg'] for d in all_data.values()])
    X_gaze_all = np.vstack([d['X_gaze'] for d in all_data.values()])
    texts_all = np.concatenate([d['texts'] for d in all_data.values()])
    y_all = np.concatenate([d['y'] for d in all_data.values()])
    keys_all = []
    for subject, d in all_data.items():
        keys_all.extend([(subject, 'NR' if y == 0 else 'TSR', i) 
                        for i, y in enumerate(d['y'])])
    
    # Protocol C: Held-out sentence split
    print("\n" + "=" * 80)
    print("Protocol C: Held-out sentence split")
    print("=" * 80)
    
    protocol_c_results = run_protocol_C(X_eeg_all, X_gaze_all, texts_all, y_all, keys_all)
    df_protocol_c = pd.DataFrame(protocol_c_results)
    print("\nProtocol C Summary:")
    print(df_protocol_c.mean())
    
    # Ablation study
    print("\n" + "=" * 80)
    print("Ablation Study")
    print("=" * 80)
    
    ablation_results = run_ablation_study(X_eeg_all, X_gaze_all, texts_all, y_all)
    df_ablation = pd.DataFrame(ablation_results)
    print("\nAblation Results:")
    print(df_ablation)
    
    # Save results
    os.makedirs('results/final', exist_ok=True)
    df_protocol_c.to_csv('results/final/share_net_protocol_c.csv', index=False)
    df_ablation.to_csv('results/final/share_net_ablation_results.csv', index=False)
    
    print("\n\nFiles saved:")
    print("  - results/final/share_net_protocol_c.csv")
    print("  - results/final/share_net_ablation_results.csv")
    
    print("\n" + "=" * 80)
    print("SHARE-Net Extended Experiments Complete")
    print("=" * 80)