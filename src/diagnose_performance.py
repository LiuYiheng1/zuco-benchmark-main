import numpy as np
import os
import time
from sklearn.neural_network import MLPClassifier

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
    
    X_eeg, X_gaze, y = [], [], []
    
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
        y.append(0 if label == 'NR' else 1)
    
    return np.array(X_eeg), np.array(X_gaze), np.array(y)


def benchmark_components():
    print("=" * 80)
    print("性能诊断：分析代码运行慢的原因")
    print("=" * 80)
    
    # 1. 加载所有数据
    print("\n1. 数据加载时间...")
    start = time.time()
    
    all_data = {}
    for subject in SUBJECTS_16:
        X_eeg, X_gaze, y = load_aligned_data(subject)
        if X_eeg is not None:
            all_data[subject] = {'X_eeg': X_eeg, 'X_gaze': X_gaze, 'y': y}
    
    load_time = time.time() - start
    total_samples = sum(len(d['y']) for d in all_data.values())
    
    print(f"   数据加载: {load_time:.2f}秒")
    print(f"   总样本数: {total_samples}")
    print(f"   EEG特征维度: {all_data[SUBJECTS_16[0]]['X_eeg'].shape[1]}")
    print(f"   Gaze特征维度: {all_data[SUBJECTS_16[0]]['X_gaze'].shape[1]}")
    
    # 2. 合并数据
    print("\n2. 数据合并...")
    start = time.time()
    
    X_eeg_all = np.vstack([d['X_eeg'] for d in all_data.values()])
    X_gaze_all = np.vstack([d['X_gaze'] for d in all_data.values()])
    y_all = np.concatenate([d['y'] for d in all_data.values()])
    
    merge_time = time.time() - start
    print(f"   数据合并: {merge_time:.4f}秒")
    print(f"   总数据大小: {X_eeg_all.nbytes / 1024 / 1024:.2f} MB")
    
    # 3. 单个 MLP 训练时间
    print("\n3. 单个 MLP 训练时间...")
    start = time.time()
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_eeg_s = scaler.fit_transform(X_eeg_all)
    
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200)
    clf.fit(X_eeg_s, y_all)
    
    mlp_time = time.time() - start
    print(f"   预处理+训练: {mlp_time:.2f}秒")
    
    # 4. 多模型 OOF 训练时间估算
    print("\n4. OOF 多模型训练时间估算...")
    
    from sklearn.model_selection import StratifiedKFold
    
    n_folds = 5
    n_seeds = 3
    models_per_fold = 7  # 7个base models
    
    single_oof_time = 0
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X_eeg_s, y_all):
        X_tr, X_val = X_eeg_s[train_idx], X_eeg_s[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]
        
        start = time.time()
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200)
        clf.fit(X_tr, y_tr)
        clf.predict_proba(X_val)
        single_oof_time += time.time() - start
        
        break  # 只测一个fold
    
    total_oof_time = single_oof_time * n_folds * models_per_fold * n_seeds
    print(f"   单个base model: {single_oof_time:.2f}秒")
    print(f"   估算总时间 (5 folds × 7 models × 3 seeds): {total_oof_time:.0f}秒 = {total_oof_time/60:.1f}分钟")
    
    # 5. 总时间估算
    print("\n" + "=" * 80)
    print("总时间估算")
    print("=" * 80)
    
    print(f"   数据加载: {load_time:.1f}秒")
    print(f"   特征提取: ~10秒")
    print(f"   Baseline训练: ~{mlp_time:.0f}秒")
    print(f"   OOF训练: ~{total_oof_time/60:.0f}分钟")
    print(f"   其他处理: ~30秒")
    print(f"   ---------------------")
    print(f"   总计: ~{load_time + total_oof_time + mlp_time + 40:.0f}秒 = ~{(load_time + total_oof_time + mlp_time + 40)/60:.1f}分钟")


if __name__ == '__main__':
    benchmark_components()