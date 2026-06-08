import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
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


def run_fast_experiment():
    """优化版实验：减少模型数量和训练时间"""
    
    print("=" * 80)
    print("优化版实验：减少计算量")
    print("=" * 80)
    
    all_data = {}
    for subject in SUBJECTS_16:
        print(f"Loading {subject}...", end=' ')
        X_eeg, X_gaze, texts, y = load_aligned_data(subject)
        if X_eeg is not None:
            all_data[subject] = {'X_eeg': X_eeg, 'X_gaze': X_gaze, 'texts': texts, 'y': y}
            print(f"{len(y)} samples")
    
    X_eeg_all = np.vstack([d['X_eeg'] for d in all_data.values()])
    X_gaze_all = np.vstack([d['X_gaze'] for d in all_data.values()])
    texts_all = np.concatenate([d['texts'] for d in all_data.values()])
    y_all = np.concatenate([d['y'] for d in all_data.values()])
    
    print(f"\n总样本数: {len(y_all)}")
    
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
        
        # Baseline: Text+EEG+Gaze_concat (使用更快的 LogisticRegression)
        X_all_train = np.hstack([X_text_train, X_eeg_train_s, X_gaze_train_s])
        X_all_test = np.hstack([X_text_test, X_eeg_test_s, X_gaze_test_s])
        
        # 优化1: 使用 LogisticRegression 代替 MLP（快10倍）
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_all_train, y_train)
        y_pred = clf.predict(X_all_test)
        y_proba = clf.predict_proba(X_all_test)[:, 1]
        
        results.append({'method': 'Text+EEG+Gaze_LR', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        print(f"Text+EEG+Gaze_LR: {accuracy_score(y_test, y_pred):.4f}")
        
        # 优化2: 使用更小 MLP
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100)  # 减小网络
        clf.fit(X_all_train, y_train)
        y_pred = clf.predict(X_all_test)
        y_proba = clf.predict_proba(X_all_test)[:, 1]
        
        results.append({'method': 'Text+EEG+Gaze_MLP_small', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        print(f"Text+EEG+Gaze_MLP_small: {accuracy_score(y_test, y_pred):.4f}")
        
        # Single modality baselines (使用 LogisticRegression)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_text_train, y_train)
        y_pred = clf.predict(X_text_test)
        y_proba = clf.predict_proba(X_text_test)[:, 1]
        results.append({'method': 'Text_only_LR', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        print(f"Text_only_LR: {accuracy_score(y_test, y_pred):.4f}")
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_eeg_train_s, y_train)
        y_pred = clf.predict(X_eeg_test_s)
        y_proba = clf.predict_proba(X_eeg_test_s)[:, 1]
        results.append({'method': 'EEG_only_LR', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        print(f"EEG_only_LR: {accuracy_score(y_test, y_pred):.4f}")
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_gaze_train_s, y_train)
        y_pred = clf.predict(X_gaze_test_s)
        y_proba = clf.predict_proba(X_gaze_test_s)[:, 1]
        results.append({'method': 'Gaze_only_LR', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        print(f"Gaze_only_LR: {accuracy_score(y_test, y_pred):.4f}")
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    import time
    start = time.time()
    
    df = run_fast_experiment()
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 80)
    print(f"实验完成！总时间: {elapsed:.1f}秒")
    print("=" * 80)
    
    print("\nResults Summary (mean over 3 seeds):")
    summary = df.groupby('method').mean()[['accuracy', 'macro_f1', 'balanced_acc', 'auroc']].round(4)
    print(summary)
    
    os.makedirs('results/final', exist_ok=True)
    df.to_csv('results/final/fast_baseline_results.csv', index=False)
    
    print("\n\nFiles saved:")
    print("  - results/final/fast_baseline_results.csv")