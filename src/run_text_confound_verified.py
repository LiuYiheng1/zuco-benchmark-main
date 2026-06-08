"""
Text Confound Controls - 真实实验
使用可用的文本代理特征，避免虚假数据

可用的文本特征：
1. Sentence Length - 从sent_gaze_sacc提取
2. Word Count - 从sent_gaze_sacc提取
3. FRE-like - 基于句子属性的可读性指标

注意：由于项目中没有真正的BERT embedding，
Text+EEG实验将使用sentence length + EEG，而不是真正的BERT + EEG
"""

import os
import sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None
    data = np.load(path, allow_pickle=True).item()
    X, y = [], []
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 2 and parts[1] == "NR":
            label = 1
        elif len(parts) >= 2 and parts[1] == "TSR":
            label = 0
        else:
            continue
        X.append(values[:-1])
        y.append(label)
    return np.array(X, dtype=np.float64), np.array(y)

def load_gaze_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze_sacc.npy")
    if not os.path.exists(path):
        return None, None
    data = np.load(path, allow_pickle=True).item()
    X, y = [], []
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 2 and parts[1] == "NR":
            label = 1
        elif len(parts) >= 2 and parts[1] == "TSR":
            label = 0
        else:
            continue
        X.append([float(v) for v in values[:-1]])
        y.append(label)
    return np.array(X, dtype=np.float64), np.array(y)

def compute_text_features_from_gaze(gaze_data):
    """
    从gaze数据中提取文本代理特征
    sent_gaze_sacc格式：[fix_duration, fixation_ratio, saccade_length, fixation_count, total_reading_time, mean_time_per_word, total_saccade_distance, mean_saccade_distance, sentence_length, label]
    """
    text_features = []
    for sample in gaze_data:
        sentence_length = sample[-2]  # sentence_length是倒数第二个特征
        
        word_count = sentence_length / 6.0 if sentence_length > 0 else 1
        
        fre_proxy = 206.835 - 1.015 * word_count - 84.6 * (sample[0] / sample[4]) if sample[4] > 0 else 50
        
        text_features.append([word_count, sentence_length, fre_proxy])
    
    return np.array(text_features)

def balanced_random_sampling(y_pool, n_per_class):
    class0 = np.where(y_pool == 0)[0]
    class1 = np.where(y_pool == 1)[0]
    if len(class0) < n_per_class or len(class1) < n_per_class:
        return None
    np.random.shuffle(class0)
    np.random.shuffle(class1)
    selected = np.concatenate([class0[:n_per_class], class1[:n_per_class]])
    np.random.shuffle(selected)
    return selected

def run_text_confound_experiments():
    """
    运行文本混淆控制实验
    
    实验方法：
    1. Random - 随机基线
    2. Sentence Length - 句子长度
    3. FRE Proxy - 可读性指标代理
    4. Word Count - 词数
    5. Text MLP (Sentence Length + FRE) - 文本MLP
    6. EEG only - EEG基线（对比）
    7. Text + EEG - 文本+EEG组合（使用sentence length代理）
    8. Text + Random EEG - 文本+随机EEG（控制组）
    """
    results = []
    shot_settings = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]
    
    print("Running text confound experiments...")
    print("=" * 60)
    
    for seed in seeds:
        np.random.seed(seed)
        print(f"\nSeed {seed}:")
        
        for subject in Y_SUBJECTS:
            Xe, ye = load_eeg_data(subject)
            Xg, yg = load_gaze_data(subject)
            
            if Xe is None or Xg is None:
                print(f"  {subject}: Data not found, skipping...")
                continue
            
            if len(Xe) != len(Xg):
                min_len = min(len(Xe), len(Xg))
                Xe = Xe[:min_len]
                ye = ye[:min_len]
                Xg = Xg[:min_len]
                yg = yg[:min_len]
            
            text_features = compute_text_features_from_gaze(Xg)
            n_samples = len(ye)
            indices = np.random.permutation(n_samples)
            test_size = n_samples // 2
            test_idx = indices[:test_size]
            cal_pool_idx = indices[test_size:]
            
            X_cal_text = text_features[cal_pool_idx]
            X_cal_eeg = Xe[cal_pool_idx]
            X_cal_gaze = Xg[cal_pool_idx]
            y_cal_pool = ye[cal_pool_idx]
            
            X_test_text = text_features[test_idx]
            X_test_eeg = Xe[test_idx]
            X_test_gaze = Xg[test_idx]
            y_test = ye[test_idx]
            
            for n_cal in shot_settings:
                selected = balanced_random_sampling(y_cal_pool, n_cal)
                if selected is None:
                    continue
                
                X_text_cal = X_cal_text[selected]
                X_eeg_cal = X_cal_eeg[selected]
                y_cal = y_cal_pool[selected]
                
                scaler_text = StandardScaler()
                X_text_cal_s = scaler_text.fit_transform(X_text_cal)
                X_text_test_s = scaler_text.transform(X_test_text)
                
                scaler_eeg = StandardScaler()
                X_eeg_cal_s = scaler_eeg.fit_transform(X_eeg_cal)
                X_eeg_test_s = scaler_eeg.transform(X_test_eeg)
                
                row = {
                    'seed': seed, 
                    'subject': subject, 
                    'n_cal': n_cal,
                    'n_samples': n_samples
                }
                
                y_ratio = np.mean(y_cal)
                y_pred_random = np.random.choice([0, 1], size=len(y_test), p=[1-y_ratio, y_ratio])
                row['Random_acc'] = accuracy_score(y_test, y_pred_random)
                
                ridge_text = RidgeClassifier(alpha=1.0)
                ridge_text.fit(X_text_cal_s[:, 1:2], y_cal)
                row['SentenceLength_acc'] = accuracy_score(y_test, ridge_text.predict(X_text_test_s[:, 1:2]))
                
                ridge_fre = RidgeClassifier(alpha=1.0)
                ridge_fre.fit(X_text_cal_s[:, 2:3], y_cal)
                row['FRE_acc'] = accuracy_score(y_test, ridge_fre.predict(X_text_test_s[:, 2:3]))
                
                ridge_wc = RidgeClassifier(alpha=1.0)
                ridge_wc.fit(X_text_cal_s[:, 0:1], y_cal)
                row['WordCount_acc'] = accuracy_score(y_test, ridge_wc.predict(X_text_test_s[:, 0:1]))
                
                mlp_text = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
                mlp_text.fit(X_text_cal_s, y_cal)
                row['TextMLP_acc'] = accuracy_score(y_test, mlp_text.predict(X_text_test_s))
                
                mlp_eeg = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                mlp_eeg.fit(X_eeg_cal_s, y_cal)
                row['EEG_only_acc'] = accuracy_score(y_test, mlp_eeg.predict(X_eeg_test_s))
                
                X_concat = np.hstack([X_text_cal_s, X_eeg_cal_s])
                X_concat_test = np.hstack([X_text_test_s, X_eeg_test_s])
                mlp_concat = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                mlp_concat.fit(X_concat, y_cal)
                row['Text_EEG_acc'] = accuracy_score(y_test, mlp_concat.predict(X_concat_test))
                
                np.random.seed(seed + 100)
                X_random_eeg_cal = np.random.randn(*X_eeg_cal_s.shape)
                X_random_eeg_test = np.random.randn(*X_eeg_test_s.shape)
                X_random_concat = np.hstack([X_text_cal_s, X_random_eeg_cal])
                X_random_concat_test = np.hstack([X_text_test_s, X_random_eeg_test])
                mlp_random = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                mlp_random.fit(X_random_concat, y_cal)
                row['Text_RandomEEG_acc'] = accuracy_score(y_test, mlp_random.predict(X_random_concat_test))
                
                results.append(row)
        
        print(f"  Completed subject loop for seed {seed}")
    
    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, 'text_confound_verified_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return df

def generate_summary_report(df):
    """生成汇总报告"""
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    metrics = ['Random_acc', 'SentenceLength_acc', 'FRE_acc', 'WordCount_acc', 
               'TextMLP_acc', 'EEG_only_acc', 'Text_EEG_acc', 'Text_RandomEEG_acc']
    
    method_names = {
        'Random_acc': 'Random',
        'SentenceLength_acc': 'Sentence Length',
        'FRE_acc': 'FRE Proxy',
        'WordCount_acc': 'Word Count',
        'TextMLP_acc': 'Text MLP',
        'EEG_only_acc': 'EEG only',
        'Text_EEG_acc': 'Text + EEG',
        'Text_RandomEEG_acc': 'Text + Random EEG'
    }
    
    print("\nMean Accuracy (%) by Shot Setting:")
    print("-" * 80)
    print(f"{'Method':<25} {'3-shot':>8} {'5-shot':>8} {'10-shot':>8} {'20-shot':>8} {'50-shot':>8}")
    print("-" * 80)
    
    for metric in metrics:
        row_data = []
        for n_cal in [3, 5, 10, 20, 50]:
            sub = df[df['n_cal'] == n_cal]
            mean_val = sub[metric].mean() * 100
            row_data.append(mean_val)
        print(f"{method_names[metric]:<25} " + " ".join([f"{v:>8.1f}" for v in row_data]))
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    
    for n_cal in [3, 5, 10, 20, 50]:
        sub = df[df['n_cal'] == n_cal]
        text_eeg = sub['Text_EEG_acc'].mean() * 100
        text_only = sub['TextMLP_acc'].mean() * 100
        eeg_only = sub['EEG_only_acc'].mean() * 100
        text_random = sub['Text_RandomEEG_acc'].mean() * 100
        
        print(f"\n{n_cal}-shot:")
        print(f"  Text+EEG ({text_eeg:.1f}%) vs Text-only ({text_only:.1f}%): {'+' if text_eeg > text_only else ''}{text_eeg - text_only:.1f}%")
        print(f"  Text+EEG ({text_eeg:.1f}%) vs EEG-only ({eeg_only:.1f}%): {'+' if text_eeg > eeg_only else ''}{text_eeg - eeg_only:.1f}%")
        print(f"  Text+EEG ({text_eeg:.1f}%) vs Text+RandomEEG ({text_random:.1f}%): {'+' if text_eeg > text_random else ''}{text_eeg - text_random:.1f}%")
    
    summary_data = []
    for n_cal in [3, 5, 10, 20, 50]:
        sub = df[df['n_cal'] == n_cal]
        summary_data.append({
            'Shot': n_cal,
            'Random': sub['Random_acc'].mean() * 100,
            'Sentence_Length': sub['SentenceLength_acc'].mean() * 100,
            'FRE_Proxy': sub['FRE_acc'].mean() * 100,
            'Word_Count': sub['WordCount_acc'].mean() * 100,
            'Text_MLP': sub['TextMLP_acc'].mean() * 100,
            'EEG_only': sub['EEG_only_acc'].mean() * 100,
            'Text_EEG': sub['Text_EEG_acc'].mean() * 100,
            'Text_RandomEEG': sub['Text_RandomEEG_acc'].mean() * 100
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(RESULTS_DIR, 'text_confound_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    return summary_df

if __name__ == '__main__':
    print("Text Confound Controls - Verified Experiments")
    print("=" * 60)
    print("\nIMPORTANT NOTES:")
    print("1. This script uses TEXT PROXY features extracted from gaze data")
    print("2. 'Sentence Length' and 'Word Count' are extracted from sent_gaze_sacc")
    print("3. 'FRE Proxy' is calculated from gaze features as readability proxy")
    print("4. NO REAL BERT EMBEDDINGS are used (not available in this project)")
    print("5. 'Text + EEG' means 'Sentence Length + EEG', not 'BERT + EEG'")
    print("=" * 60)
    
    df = run_text_confound_experiments()
    summary_df = generate_summary_report(df)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {os.path.join(RESULTS_DIR, 'text_confound_verified_results.csv')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'text_confound_summary.csv')}")
    print("\nNote: All 'Text' features are PROXIES, not real text embeddings.")
