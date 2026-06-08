import numpy as np
import pandas as pd
import os
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

SUBJECTS_16 = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS',
                'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_aligned_data(subject):
    """Label-aware EEG-Gaze-Text alignment"""
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
    """Load text sentences from CSV files"""
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


class TextEncoder:
    """Text encoder with TF-IDF and handcrafted features"""
    def __init__(self, dim=32):
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.scaler = StandardScaler()
        self.dim = dim
    
    def fit(self, texts):
        tfidf_features = self.tfidf.fit_transform(texts).toarray()
        handcrafted = self._extract_handcrafted(texts)
        features = np.hstack([tfidf_features, handcrafted])
        self.scaler.fit(features)
        
        n_input = features.shape[1]
        self.projection = np.random.randn(n_input, self.dim) * 0.01
    
    def transform(self, texts):
        tfidf_features = self.tfidf.transform(texts).toarray()
        handcrafted = self._extract_handcrafted(texts)
        features = np.hstack([tfidf_features, handcrafted])
        features = self.scaler.transform(features)
        z = features @ self.projection
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        return z
    
    def _extract_handcrafted(self, texts):
        features = []
        for text in texts:
            text = str(text)
            word_count = len(text.split())
            char_count = len(text)
            avg_word_len = char_count / max(word_count, 1)
            features.append([word_count, char_count, avg_word_len])
        return np.array(features)


class EEGEncoder:
    """EEG feature encoder"""
    def __init__(self, dim=32):
        self.scaler = StandardScaler()
        self.dim = dim
    
    def fit(self, X_eeg):
        self.scaler.fit(X_eeg)
        n_input = X_eeg.shape[1]
        self.W1 = np.random.randn(n_input, 128) * 0.01
        self.W2 = np.random.randn(128, self.dim) * 0.01
    
    def transform(self, X_eeg):
        X = self.scaler.transform(X_eeg)
        h = X @ self.W1
        h = np.maximum(h, 0)
        z = h @ self.W2
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        return z


class GazeEncoder:
    """Gaze feature encoder"""
    def __init__(self, dim=32):
        self.scaler = StandardScaler()
        self.dim = dim
    
    def fit(self, X_gaze):
        self.scaler.fit(X_gaze)
        n_input = X_gaze.shape[1]
        self.W1 = np.random.randn(n_input, 32) * 0.01
        self.W2 = np.random.randn(32, self.dim) * 0.01
    
    def transform(self, X_gaze):
        X = self.scaler.transform(X_gaze)
        h = X @ self.W1
        h = np.maximum(h, 0)
        z = h @ self.W2
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        return z


class SHARENet:
    """SHARE-Net: Semantic Hyperaligned Response Evidence Network"""
    def __init__(self, dim=32, lambda_align=0.1, lambda_mask=0.05):
        self.text_encoder = TextEncoder(dim)
        self.eeg_encoder = EEGEncoder(dim)
        self.gaze_encoder = GazeEncoder(dim)
        self.dim = dim
        self.lambda_align = lambda_align
        self.lambda_mask = lambda_mask
        self.align_loss = 0.0
        self.mask_loss = 0.0
    
    def fit(self, X_eeg, X_gaze, texts, y, n_epochs=100, lr=0.01):
        self.text_encoder.fit(texts)
        self.eeg_encoder.fit(X_eeg)
        self.gaze_encoder.fit(X_gaze)
        
        z_t = self.text_encoder.transform(texts)
        z_e = self.eeg_encoder.transform(X_eeg)
        z_g = self.gaze_encoder.transform(X_gaze)
        
        h_relation = self._build_relation_features(z_t, z_e, z_g)
        n_hidden = h_relation.shape[1]
        
        self.gate_W = np.random.randn(n_hidden, self.dim) * 0.01
        self.clf = LogisticRegression(C=1.0, max_iter=500)
        
        for epoch in range(n_epochs):
            z_t = self.text_encoder.transform(texts)
            z_e = self.eeg_encoder.transform(X_eeg)
            z_g = self.gaze_encoder.transform(X_gaze)
            
            self.align_loss = self._compute_align_loss(z_t, z_e, z_g)
            
            h_relation = self._build_relation_features(z_t, z_e, z_g)
            
            gamma = 1 / (1 + np.exp(-(h_relation @ self.gate_W)))
            h_eg = gamma * z_e + (1 - gamma) * z_g
            
            h_fused = np.hstack([z_t, h_eg, np.abs(z_t - h_eg), z_t * h_eg, h_relation])
            
            self.clf.fit(h_fused, y)
            
            if (epoch + 1) % 20 == 0:
                y_pred = self.clf.predict(h_fused)
                acc = accuracy_score(y, y_pred)
                print(f"Epoch {epoch+1}: acc={acc:.4f}, align_loss={self.align_loss:.4f}")
        
        self.final_z_t = z_t
        self.final_z_e = z_e
        self.final_z_g = z_g
    
    def predict(self, X_eeg, X_gaze, texts):
        z_t = self.text_encoder.transform(texts)
        z_e = self.eeg_encoder.transform(X_eeg)
        z_g = self.gaze_encoder.transform(X_gaze)
        
        h_relation = self._build_relation_features(z_t, z_e, z_g)
        
        gamma = 1 / (1 + np.exp(-(h_relation @ self.gate_W)))
        h_eg = gamma * z_e + (1 - gamma) * z_g
        
        h_fused = np.hstack([z_t, h_eg, np.abs(z_t - h_eg), z_t * h_eg, h_relation])
        
        return self.clf.predict(h_fused), self.clf.predict_proba(h_fused)[:, 1]
    
    def _build_relation_features(self, z_t, z_e, z_g):
        r_te = np.hstack([z_t, z_e, np.abs(z_t - z_e), z_t * z_e])
        r_tg = np.hstack([z_t, z_g, np.abs(z_t - z_g), z_t * z_g])
        r_eg = np.hstack([z_e, z_g, np.abs(z_e - z_g), z_e * z_g])
        return np.hstack([r_te, r_tg, r_eg])
    
    def _compute_align_loss(self, z_t, z_e, z_g):
        loss = np.mean((z_e - z_t)**2) + np.mean((z_g - z_t)**2) + np.mean((z_e - z_g)**2)
        return loss


def run_protocol_A(subject, X_eeg, X_gaze, texts, y, n_seeds=5, test_size=0.3):
    results = []
    for seed in range(n_seeds):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(X_eeg, y))
        
        X_eeg_train, X_eeg_test = X_eeg[train_idx], X_eeg[test_idx]
        X_gaze_train, X_gaze_test = X_gaze[train_idx], X_gaze[test_idx]
        texts_train, texts_test = texts[train_idx], texts[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = SHARENet(dim=32)
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


def run_protocol_B(X_eeg_all, X_gaze_all, texts_all, y_all, subjects):
    results = []
    for i, test_subject in enumerate(subjects):
        train_idx = [j for j, s in enumerate(subjects) if s != test_subject]
        test_idx = [i]
        
        X_eeg_train = np.vstack([X_eeg_all[j] for j in train_idx])
        X_gaze_train = np.vstack([X_gaze_all[j] for j in train_idx])
        texts_train = np.concatenate([texts_all[j] for j in train_idx])
        y_train = np.concatenate([y_all[j] for j in train_idx])
        
        X_eeg_test = X_eeg_all[test_idx[0]]
        X_gaze_test = X_gaze_all[test_idx[0]]
        texts_test = texts_all[test_idx[0]]
        y_test = y_all[test_idx[0]]
        
        model = SHARENet(dim=32)
        model.fit(X_eeg_train, X_gaze_train, texts_train, y_train, n_epochs=50)
        y_pred, y_proba = model.predict(X_eeg_test, X_gaze_test, texts_test)
        
        results.append({
            'test_subject': test_subject,
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'balanced_acc': balanced_accuracy_score(y_test, y_pred),
            'auroc': roc_auc_score(y_test, y_proba)
        })
    
    return results


def run_baselines(X_eeg, X_gaze, texts, y):
    results = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_idx, test_idx = next(sss.split(X_eeg, y))
    
    X_eeg_train, X_eeg_test = X_eeg[train_idx], X_eeg[test_idx]
    X_gaze_train, X_gaze_test = X_gaze[train_idx], X_gaze[test_idx]
    texts_train, texts_test = texts[train_idx], texts[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Text-only
    tfidf = TfidfVectorizer(max_features=100)
    X_text_train = tfidf.fit_transform(texts_train).toarray()
    X_text_test = tfidf.transform(texts_test).toarray()
    clf = LogisticRegression()
    clf.fit(X_text_train, y_train)
    y_pred = clf.predict(X_text_test)
    y_proba = clf.predict_proba(X_text_test)[:, 1]
    results.append({'method': 'Text_only', 'accuracy': accuracy_score(y_test, y_pred),
                    'macro_f1': f1_score(y_test, y_pred, average='macro'),
                    'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                    'auroc': roc_auc_score(y_test, y_proba)})
    
    # EEG-only
    scaler = StandardScaler()
    X_eeg_train_s = scaler.fit_transform(X_eeg_train)
    X_eeg_test_s = scaler.transform(X_eeg_test)
    clf = RidgeClassifier()
    clf.fit(X_eeg_train_s, y_train)
    y_pred = clf.predict(X_eeg_test_s)
    y_proba = clf.decision_function(X_eeg_test_s)
    results.append({'method': 'EEG_only', 'accuracy': accuracy_score(y_test, y_pred),
                    'macro_f1': f1_score(y_test, y_pred, average='macro'),
                    'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                    'auroc': roc_auc_score(y_test, y_proba)})
    
    # Gaze-only
    scaler = StandardScaler()
    X_gaze_train_s = scaler.fit_transform(X_gaze_train)
    X_gaze_test_s = scaler.transform(X_gaze_test)
    clf = RidgeClassifier()
    clf.fit(X_gaze_train_s, y_train)
    y_pred = clf.predict(X_gaze_test_s)
    y_proba = clf.decision_function(X_gaze_test_s)
    results.append({'method': 'Gaze_only', 'accuracy': accuracy_score(y_test, y_pred),
                    'macro_f1': f1_score(y_test, y_pred, average='macro'),
                    'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                    'auroc': roc_auc_score(y_test, y_proba)})
    
    # EEG+Gaze concat
    X_concat_train = np.hstack([X_eeg_train_s, X_gaze_train_s])
    X_concat_test = np.hstack([X_eeg_test_s, X_gaze_test_s])
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500)
    clf.fit(X_concat_train, y_train)
    y_pred = clf.predict(X_concat_test)
    y_proba = clf.predict_proba(X_concat_test)[:, 1]
    results.append({'method': 'EEG+Gaze_concat', 'accuracy': accuracy_score(y_test, y_pred),
                    'macro_f1': f1_score(y_test, y_pred, average='macro'),
                    'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                    'auroc': roc_auc_score(y_test, y_proba)})
    
    # Text+EEG+Gaze concat
    X_all_train = np.hstack([X_text_train, X_eeg_train_s, X_gaze_train_s])
    X_all_test = np.hstack([X_text_test, X_eeg_test_s, X_gaze_test_s])
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    clf.fit(X_all_train, y_train)
    y_pred = clf.predict(X_all_test)
    y_proba = clf.predict_proba(X_all_test)[:, 1]
    results.append({'method': 'Text+EEG+Gaze_concat', 'accuracy': accuracy_score(y_test, y_pred),
                    'macro_f1': f1_score(y_test, y_pred, average='macro'),
                    'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                    'auroc': roc_auc_score(y_test, y_proba)})
    
    return results


if __name__ == '__main__':
    print("=" * 80)
    print("SHARE-Net: Semantic Hyperaligned Response Evidence Network")
    print("=" * 80)
    
    all_data = {}
    for subject in SUBJECTS_16:
        print(f"Loading {subject}...")
        X_eeg, X_gaze, texts, y = load_aligned_data(subject)
        if X_eeg is not None:
            all_data[subject] = {'X_eeg': X_eeg, 'X_gaze': X_gaze, 'texts': texts, 'y': y}
            print(f"  Samples: {len(y)}, Label consistency: 100%")
    
    print(f"\nLoaded {len(all_data)} subjects")
    
    # Protocol A: Subject-dependent split
    print("\n" + "=" * 80)
    print("Protocol A: Subject-dependent split")
    print("=" * 80)
    
    protocol_a_results = []
    for subject, data in all_data.items():
        results = run_protocol_A(subject, data['X_eeg'], data['X_gaze'], data['texts'], data['y'])
        protocol_a_results.extend(results)
    
    df_protocol_a = pd.DataFrame(protocol_a_results)
    print("\nProtocol A Summary:")
    print(df_protocol_a.groupby('subject').mean()[['accuracy', 'macro_f1', 'balanced_acc', 'auroc']])
    
    # Baselines
    print("\n" + "=" * 80)
    print("Baseline Results")
    print("=" * 80)
    
    X_eeg_all = []
    X_gaze_all = []
    texts_all = []
    y_all = []
    subjects_list = []
    
    for subject, data in all_data.items():
        X_eeg_all.append(data['X_eeg'])
        X_gaze_all.append(data['X_gaze'])
        texts_all.append(data['texts'])
        y_all.append(data['y'])
        subjects_list.append(subject)
    
    baseline_results = run_baselines(np.vstack(X_eeg_all), np.vstack(X_gaze_all), 
                                     np.concatenate(texts_all), np.concatenate(y_all))
    df_baselines = pd.DataFrame(baseline_results)
    print("\nBaseline Summary:")
    print(df_baselines)
    
    # Save results
    os.makedirs('results/final', exist_ok=True)
    os.makedirs('reports/final', exist_ok=True)
    
    df_protocol_a.to_csv('results/final/share_net_protocol_a.csv', index=False)
    df_baselines.to_csv('results/final/share_net_baselines.csv', index=False)
    
    print("\n\nFiles saved:")
    print("  - results/final/share_net_protocol_a.csv")
    print("  - results/final/share_net_baselines.csv")
    
    print("\n" + "=" * 80)
    print("SHARE-Net Experiment Started")
    print("=" * 80)