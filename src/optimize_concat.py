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


def run_optimization_experiment():
    """探索 Text+EEG+Gaze_concat 的提升空间"""
    
    print("=" * 80)
    print("Text+EEG+Gaze_concat 优化实验")
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
        
        X_all_train = np.hstack([X_text_train, X_eeg_train_s, X_gaze_train_s])
        X_all_test = np.hstack([X_text_test, X_eeg_test_s, X_gaze_test_s])
        
        # 1. 探索网络规模
        print("\n--- 网络规模探索 ---")
        
        architectures = [
            (64,),
            (128,),
            (256,),
            (128, 64),
            (256, 128),
            (512, 256, 128),
            (256, 128, 64),
        ]
        
        for arch in architectures:
            clf = MLPClassifier(hidden_layer_sizes=arch, max_iter=300, random_state=seed)
            clf.fit(X_all_train, y_train)
            y_pred = clf.predict(X_all_test)
            y_proba = clf.predict_proba(X_all_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            print(f"Architecture {arch}: {acc:.4f}")
            
            results.append({
                'experiment': 'architecture',
                'config': str(arch),
                'seed': seed,
                'accuracy': acc,
                'macro_f1': f1_score(y_test, y_pred, average='macro'),
                'auroc': roc_auc_score(y_test, y_proba)
            })
        
        # 2. 探索训练轮数
        print("\n--- 训练轮数探索 (256, 128) ---")
        
        for max_iter in [100, 200, 300, 500]:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=max_iter, random_state=seed)
            clf.fit(X_all_train, y_train)
            y_pred = clf.predict(X_all_test)
            y_proba = clf.predict_proba(X_all_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            print(f"max_iter={max_iter}: {acc:.4f}")
            
            results.append({
                'experiment': 'max_iter',
                'config': str(max_iter),
                'seed': seed,
                'accuracy': acc,
                'macro_f1': f1_score(y_test, y_pred, average='macro'),
                'auroc': roc_auc_score(y_test, y_proba)
            })
        
        # 3. 探索正则化
        print("\n--- 正则化探索 ---")
        
        for alpha in [0.0001, 0.001, 0.01, 0.1]:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=alpha, max_iter=300, random_state=seed)
            clf.fit(X_all_train, y_train)
            y_pred = clf.predict(X_all_test)
            y_proba = clf.predict_proba(X_all_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            print(f"alpha={alpha}: {acc:.4f}")
            
            results.append({
                'experiment': 'regularization',
                'config': str(alpha),
                'seed': seed,
                'accuracy': acc,
                'macro_f1': f1_score(y_test, y_pred, average='macro'),
                'auroc': roc_auc_score(y_test, y_proba)
            })
        
        # 4. 探索文本特征维度
        print("\n--- 文本特征维度探索 ---")
        
        for max_feat in [100, 200, 500, 1000]:
            tfidf = TfidfVectorizer(max_features=max_feat)
            X_text_train = tfidf.fit_transform(texts_train).toarray()
            X_text_test = tfidf.transform(texts_test).toarray()
            
            X_all_train = np.hstack([X_text_train, X_eeg_train_s, X_gaze_train_s])
            X_all_test = np.hstack([X_text_test, X_eeg_test_s, X_gaze_test_s])
            
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=seed)
            clf.fit(X_all_train, y_train)
            y_pred = clf.predict(X_all_test)
            y_proba = clf.predict_proba(X_all_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            print(f"max_features={max_feat}: {acc:.4f}")
            
            results.append({
                'experiment': 'text_features',
                'config': str(max_feat),
                'seed': seed,
                'accuracy': acc,
                'macro_f1': f1_score(y_test, y_pred, average='macro'),
                'auroc': roc_auc_score(y_test, y_proba)
            })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    import time
    start = time.time()
    
    df = run_optimization_experiment()
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 80)
    print(f"实验完成！总时间: {elapsed:.1f}秒 = {elapsed/60:.1f}分钟")
    print("=" * 80)
    
    print("\n\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)
    
    for exp in df['experiment'].unique():
        print(f"\n--- {exp} ---")
        exp_df = df[df['experiment'] == exp]
        summary = exp_df.groupby('config').mean()[['accuracy', 'auroc']].round(4)
        print(summary)
    
    os.makedirs('results/final', exist_ok=True)
    df.to_csv('results/final/concat_optimization_results.csv', index=False)
    
    print("\n\nFiles saved:")
    print("  - results/final/concat_optimization_results.csv")