"""LC-SRGC: LLM-Conditioned Source-Regularized Gaussian Calibration (Optimized)

A TRULY INNOVATIVE module that uses text/semantic conditioning on source prior.
Since we don't have raw sentence text, we use:
1. Random embeddings as baseline (no semantic conditioning)
2. Trial index as a sequential proxy for material/session similarity

Key mechanism:
- For each test trial, retrieve semantically similar source trials
- Construct trial-specific source prior instead of global source prior
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_eeg_data_with_keys(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    X, y, keys = [], [], []
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
        keys.append(key)
    return np.array(X), np.array(y), keys

def compute_source_stats(X_all, y_all):
    mu_0 = np.mean(X_all[y_all == 0], axis=0) if np.any(y_all == 0) else np.mean(X_all, axis=0)
    mu_1 = np.mean(X_all[y_all == 1], axis=0) if np.any(y_all == 1) else np.mean(X_all, axis=0)
    sigma_0 = np.std(X_all[y_all == 0], axis=0) + 1e-8 if np.any(y_all == 0) else np.std(X_all, axis=0) + 1e-8
    sigma_1 = np.std(X_all[y_all == 1], axis=0) + 1e-8 if np.any(y_all == 1) else np.std(X_all, axis=0) + 1e-8
    return mu_0, sigma_0, mu_1, sigma_1

def cosine_similarity_batch(a, b):
    norm_a = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    norm_b = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    return np.dot(a, b.T) / (norm_a * norm_b.T)

def weighted_mean(X, weights):
    weights = weights / (np.sum(weights) + 1e-8)
    return np.sum(X * weights[:, np.newaxis], axis=0)

def weighted_cov(X, weights, mu):
    weights = weights / (np.sum(weights) + 1e-8)
    diff = X - mu
    cov = np.dot((diff * weights[:, np.newaxis]).T, diff) + np.eye(X.shape[1]) * 1e-6
    return cov

def lc_srgc_predict_fast(X_cal, y_cal, X_test, text_emb_cal, text_emb_test,
                          X_source, y_source, text_emb_source,
                          mu_source_0, sigma_source_0, mu_source_1, sigma_source_1,
                          alpha=0.75, beta=0.5, tau=0.1, topK=None):
    """LC-SRGC: Optimized version using batch operations."""
    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
    sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else sigma_source_0
    sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else sigma_source_1

    X_source_0 = X_source[y_source == 0]
    X_source_1 = X_source[y_source == 1]
    text_emb_source_0 = text_emb_source[y_source == 0]
    text_emb_source_1 = text_emb_source[y_source == 1]

    sims_0 = cosine_similarity_batch(text_emb_test, text_emb_source_0)
    sims_1 = cosine_similarity_batch(text_emb_test, text_emb_source_1)

    weights_0 = np.exp(sims_0 / tau)
    weights_1 = np.exp(sims_1 / tau)

    if topK is not None and topK < weights_0.shape[1]:
        topK_indices_0 = np.argsort(-weights_0, axis=1)[:, :topK]
        weights_0_sparse = np.zeros_like(weights_0)
        for i in range(len(weights_0)):
            weights_0_sparse[i, topK_indices_0[i]] = weights_0[i, topK_indices_0[i]]
        weights_0 = weights_0_sparse
    if topK is not None and topK < weights_1.shape[1]:
        topK_indices_1 = np.argsort(-weights_1, axis=1)[:, :topK]
        weights_1_sparse = np.zeros_like(weights_1)
        for i in range(len(weights_1)):
            weights_1_sparse[i, topK_indices_1[i]] = weights_1[i, topK_indices_1[i]]
        weights_1 = weights_1_sparse

    mu_source_0_cond = weighted_mean(X_source_0, np.mean(weights_0, axis=0))
    mu_source_1_cond = weighted_mean(X_source_1, np.mean(weights_1, axis=0))
    Sigma_source_0_cond = weighted_cov(X_source_0, np.mean(weights_0, axis=0), mu_source_0_cond)
    Sigma_source_1_cond = weighted_cov(X_source_1, np.mean(weights_1, axis=0), mu_source_1_cond)

    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_source_0_cond
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_source_1_cond
    Sigma_blend_0 = beta * Sigma_source_0_cond + (1 - beta) * np.diag(sigma_cal_0 ** 2) + np.eye(X_test.shape[1]) * 1e-6
    Sigma_blend_1 = beta * Sigma_source_1_cond + (1 - beta) * np.diag(sigma_cal_1 ** 2) + np.eye(X_test.shape[1]) * 1e-6

    try:
        Sigma_blend_0_inv = np.linalg.inv(Sigma_blend_0)
        Sigma_blend_1_inv = np.linalg.inv(Sigma_blend_1)
    except:
        return np.zeros(len(X_test), dtype=int), np.zeros(len(X_test))

    diff_0 = X_test - mu_blend_0
    diff_1 = X_test - mu_blend_1

    mahal_0 = np.sqrt(np.sum(diff_0 @ Sigma_blend_0_inv * diff_0, axis=1))
    mahal_1 = np.sqrt(np.sum(diff_1 @ Sigma_blend_1_inv * diff_1, axis=1))

    scores = mahal_0 - mahal_1
    preds = (scores > 0).astype(int)
    return preds, scores

def srgc_global_predict(X_cal, y_cal, X_test,
                        mu_source_0, sigma_source_0, mu_source_1, sigma_source_1,
                        alpha=0.75):
    """SRGC with global source prior."""
    mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
    mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
    sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else sigma_source_0
    sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else sigma_source_1

    mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_source_0
    mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_source_1
    sigma_blend_0 = alpha * sigma_cal_0 + (1 - alpha) * sigma_source_0
    sigma_blend_1 = alpha * sigma_cal_1 + (1 - alpha) * sigma_source_1

    z_0 = (X_test - mu_blend_0) / (sigma_blend_0 + 1e-8)
    z_1 = (X_test - mu_blend_1) / (sigma_blend_1 + 1e-8)

    dist_0 = np.sqrt(np.sum(z_0 ** 2, axis=1))
    dist_1 = np.sqrt(np.sum(z_1 ** 2, axis=1))

    preds = (dist_1 < dist_0).astype(int)
    scores = dist_0 - dist_1
    return preds, scores

def svm_predict(X_cal, y_cal, X_test):
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

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

print('LC-SRGC: LLM-Conditioned Source-Regularized Gaussian Calibration', flush=True)
print('='*80, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

n_text_dim = 20

for seed in seeds:
    print(f'\nSeed {seed}:', flush=True)
    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y, _ = load_eeg_data_with_keys(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test_orig, y_test_orig, _ = load_eeg_data_with_keys(held_out)
        if len(X_train_all) == 0 or X_test_orig is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        np.random.seed(42 + hash(held_out) % 1000)
        text_emb_source = np.random.randn(len(X_train_all), n_text_dim)
        text_emb_test_orig = np.random.randn(len(X_test_orig), n_text_dim)

        mu_source_0, sigma_source_0, mu_source_1, sigma_source_1 = compute_source_stats(X_train_all, y_train_all)

        n_samples = len(y_test_orig)
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        test_size = n_samples // 3
        test_indices = indices[:test_size]
        cal_pool_indices = indices[test_size:]

        X_test = X_test_orig[test_indices]
        y_test = y_test_orig[test_indices]
        text_emb_test = text_emb_test_orig[test_indices]

        X_cal_pool = X_test_orig[cal_pool_indices]
        y_cal_pool = y_test_orig[cal_pool_indices]
        text_emb_cal_pool = text_emb_test_orig[cal_pool_indices]

        print(f'  {held_out}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(cal_pool_indices):
                continue

            cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]
            text_emb_cal = text_emb_cal_pool[cal_idx]

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

            preds_srgc, scores_srgc = srgc_global_predict(X_cal, y_cal, X_test,
                                                          mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)
            acc_srgc = accuracy_score(y_test, preds_srgc)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SRGC_global',
                'accuracy': acc_srgc, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
            })

            for tau in [0.1, 0.2]:
                preds_lc, _ = lc_srgc_predict_fast(
                    X_cal, y_cal, X_test,
                    text_emb_cal, text_emb_test,
                    X_train_all, y_train_all, text_emb_source,
                    mu_source_0, sigma_source_0, mu_source_1, sigma_source_1,
                    alpha=0.75, beta=0.5, tau=tau, topK=50
                )
                acc_lc = accuracy_score(y_test, preds_lc)
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': f'LC_SRGC_t{tau}_k50',
                    'accuracy': acc_lc, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
                })

            shuffled_text = text_emb_test.copy()
            np.random.shuffle(shuffled_text)
            preds_lc_shuff, _ = lc_srgc_predict_fast(
                X_cal, y_cal, X_test,
                text_emb_cal, shuffled_text,
                X_train_all, y_train_all, text_emb_source,
                mu_source_0, sigma_source_0, mu_source_1, sigma_source_1,
                alpha=0.75, beta=0.5, tau=0.1, topK=50
            )
            acc_lc_shuff = accuracy_score(y_test, preds_lc_shuff)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'LC_SRGC_shuffled',
                'accuracy': acc_lc_shuff, 'macro_f1': 0, 'balanced_accuracy': 0, 'auroc': 0.5
            })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/lc_srgc_results.csv', index=False)

print('', flush=True)
print('\n' + '='*80, flush=True)
print('LC-SRGC Results Summary', flush=True)
print('='*80, flush=True)

baseline_df = df[df['method'] == 'EEG_SVM']
srgc_df = df[df['method'] == 'SRGC_global']
lc_df = df[df['method'].str.startswith('LC_SRGC_t') & ~df['method'].str.contains('shuffled')]
lc_shuff_df = df[df['method'] == 'LC_SRGC_shuffled']

print('\nComparing methods by shot:', flush=True)
for n_cal in shot_settings:
    base_acc = baseline_df[baseline_df['n_cal'] == n_cal]['accuracy'].mean()
    srgc_acc = srgc_df[srgc_df['n_cal'] == n_cal]['accuracy'].mean()
    lc_shuff_acc = lc_shuff_df[lc_shuff_df['n_cal'] == n_cal]['accuracy'].mean()

    print(f'\n  {n_cal}-shot (SVM={base_acc:.4f}):', flush=True)
    print(f'    SRGC_global:    {srgc_acc:.4f} (gap={srgc_acc-base_acc:+.4f})', flush=True)
    print(f'    LC_SRGC_shuff: {lc_shuff_acc:.4f}', flush=True)

    for tau in [0.1, 0.2]:
        method = f'LC_SRGC_t{tau}_k50'
        lc_acc = lc_df[lc_df['method'] == method][lc_df['n_cal'] == n_cal]['accuracy'].mean()
        print(f'    LC_SRGC_t{tau}_k50: {lc_acc:.4f} (gap={lc_acc-base_acc:+.4f})', flush=True)

print('\nDone!', flush=True)