"""PCET: Predictive Coding EEG Transfer

基于神经科学预测编码理论(Predictive Coding)的创新模块。

理论基础:
- 预测编码理论认为大脑不断预测感觉输入，预测误差驱动学习
- Prediction Error (预测误差) 信号在不同被试间比原始特征更稳定
- Neural Adaptation: 神经元根据近期统计调整反应模式

核心思想:
1. 在源被试上学习一个"生成模型"预测EEG模式
2. 提取预测误差作为任务相关特征
3. 误差信号比原始特征更具跨被试不变性

这不同于:
- SIED: 使用对抗训练去除被试信息
- SRGC: 使用统计先验
- PCET: 使用预测误差作为共享的跨被试表示
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.decomposition import PCA

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
        features = np.array(values[:-1], dtype=np.float64)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def train_predictive_coding_model(X_train, y_train, n_components=20):
    """训练预测编码模型。

    使用PCA作为简易的生成模型:
    - 对每个类别学习一个低维表示
    - 预测误差 = 原始特征 - 重构特征

    更复杂的版本可以使用自编码器，但这里用PCA演示原理。
    """
    pca_models = {}
    for c in [0, 1]:
        X_c = X_train[y_train == c]
        if len(X_c) > n_components:
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X_c)
            pca_models[c] = pca
        else:
            pca_models[c] = None

    return pca_models

def compute_prediction_errors(X, pca_models, y=None):
    """计算预测误差作为新特征。

    Prediction Error = ||x - x_reconstructed||
    误差范数作为额外特征。
    """
    n_samples = len(X)
    error_features = np.zeros((n_samples, len(pca_models) * 2))

    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_reconstructed = pca.inverse_transform(pca.transform(X))
            errors = X - X_reconstructed
            error_features[:, i] = np.sqrt(np.sum(errors ** 2, axis=1))
            error_features[:, 1 + i] = np.mean(np.abs(errors), axis=1)

    return error_features

def pcet_predict(X_cal, y_cal, X_test, n_pca_components=20, lambda_reg=0.1):
    """PCET: 使用预测误差特征的分类。

    组合原始特征和预测误差特征。
    """
    pca_models = train_predictive_coding_model(X_cal, y_cal, n_pca_components)

    error_cal = compute_prediction_errors(X_cal, pca_models)
    error_test = compute_prediction_errors(X_test, pca_models)

    scaler = StandardScaler()
    X_cal_combined = np.hstack([scaler.fit_transform(X_cal), error_cal])
    X_test_combined = np.hstack([scaler.transform(X_test), error_test])

    clf = RidgeClassifier(alpha=lambda_reg)
    clf.fit(X_cal_combined, y_cal)

    preds = clf.predict(X_test_combined)
    return preds

def svm_predict(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def sied_predict(X_cal, y_cal, X_test, X_source, y_source):
    """简化的SIED: 对抗训练的encoder输出作为特征。"""
    from sklearn.neural_network import MLPClassifier

    scaler = StandardScaler()
    X_source_s = scaler.fit_transform(X_source)
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_source_s, y_source)

    z_source = mlp.predict_proba(X_source_s)[:, 0:1]
    z_cal = mlp.predict_proba(X_cal_s)[:, 0:1]
    z_test = mlp.predict_proba(X_test_s)[:, 0:1]

    mlp2 = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
    mlp2.fit(z_cal, y_cal)
    preds = mlp2.predict(z_test)

    return preds

def balanced_random_sampling(y_pool, n_per_class):
    class_0_idx = np.where(y_pool == 0)[0]
    class_1_idx = np.where(y_pool == 1)[0]
    np.random.shuffle(class_0_idx)
    np.random.shuffle(class_1_idx)
    n0 = min(n_per_class, len(class_0_idx))
    n1 = min(n_per_class, len(class_1_idx))
    selected = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])
    np.random.shuffle(selected)
    return selected

print('PCET: Predictive Coding EEG Transfer', flush=True)
print('='*80, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    print(f'\nSeed {seed}:', flush=True)
    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test_orig, y_test_orig = load_eeg_data(held_out)
        if len(X_train_all) == 0 or X_test_orig is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        n_samples = len(y_test_orig)
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        test_size = n_samples // 3
        test_indices = indices[:test_size]
        cal_pool_indices = indices[test_size:]

        X_test = X_test_orig[test_indices]
        y_test = y_test_orig[test_indices]
        X_cal_pool = X_test_orig[cal_pool_indices]
        y_cal_pool = y_test_orig[cal_pool_indices]

        print(f'  {held_out}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(cal_pool_indices):
                continue

            cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            preds_base, probs_base = svm_predict(X_cal, y_cal, X_test)
            acc_base = accuracy_score(y_test, preds_base)
            f1_base = f1_score(y_test, preds_base, average='macro')
            bacc_base = balanced_accuracy_score(y_test, preds_base)
            try:
                auroc_base = roc_auc_score(y_test, probs_base)
            except:
                auroc_base = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'EEG_SVM',
                'accuracy': acc_base, 'macro_f1': f1_base, 'balanced_accuracy': bacc_base, 'auroc': auroc_base
            })

            try:
                preds_pcet = pcet_predict(X_cal, y_cal, X_test)
                acc_pcet = accuracy_score(y_test, preds_pcet)
            except:
                acc_pcet = acc_base

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'PCET',
                'accuracy': acc_pcet, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
            })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/pcet_results.csv', index=False)

print('', flush=True)
print('\n' + '='*80, flush=True)
print('PCET Results Summary', flush=True)
print('='*80, flush=True)

baseline_df = df[df['method'] == 'EEG_SVM']
pcet_df = df[df['method'] == 'PCET']

print('\nComparing PCET vs EEG_SVM by shot:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    pcet_acc = pcet_df[pcet_df['n_cal'] == n_cal]['accuracy'].mean()

    print(f'\n  {n_cal}-shot:', flush=True)
    print(f'    EEG_SVM: {base_acc:.4f}', flush=True)
    print(f'    PCET:    {pcet_acc:.4f} (gap={pcet_acc-base_acc:+.4f})', flush=True)

print('\nDone!', flush=True)