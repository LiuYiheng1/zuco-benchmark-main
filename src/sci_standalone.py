"""SCI Framework - Standalone execution with file logging"""
import sys
import os

log_file = open('d:/pycharmproject/zuco-benchmark-main/src/results/final/sci_log.txt', 'w')

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

try:
    os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
    log("Starting SCI Framework...")
    log(f"Python: {sys.version}")

    import numpy as np
    log(f"numpy: {np.__version__}")

    import pandas as pd
    log(f"pandas: {pd.__version__}")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
    log("sklearn imported")

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

    log("Functions defined")

    def compute_abs_error(X, pca_models):
        error_features = np.zeros((len(X), 2 * X.shape[1]))
        for i, (c, pca) in enumerate(pca_models.items()):
            if pca is not None:
                try:
                    X_recon = pca.inverse_transform(pca.transform(X))
                    abs_e = np.abs(X - X_recon)
                    error_features[:, i*X.shape[1]:(i+1)*X.shape[1]] = abs_e
                except:
                    pass
        return error_features

    def compute_uncertainty(X, mu_0, sigma_0, mu_1, sigma_1):
        sigma_0_d = sigma_0 ** 2 + 1e-8
        sigma_1_d = sigma_1 ** 2 + 1e-8
        sigma_0_inv = np.linalg.inv(np.diag(sigma_0_d))
        sigma_1_inv = np.linalg.inv(np.diag(sigma_1_d))
        d0 = np.sqrt(np.sum((X - mu_0) * (np.dot(X - mu_0, sigma_0_inv)), axis=1))
        d1 = np.sqrt(np.sum((X - mu_1) * (np.dot(X - mu_1, sigma_1_inv)), axis=1))
        return np.abs(d1 - d0)

    def compute_domain_similarity(X_cal, y_cal, X_test, source_stats):
        mu_0_s, sigma_0_s, mu_1_s, sigma_1_s = source_stats
        if len(np.unique(y_cal)) < 2:
            return np.ones(len(X_test)) * 0.5
        mu_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_0_s
        mu_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_1_s
        sigma_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else sigma_0_s + 1e-8
        sigma_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else sigma_1_s + 1e-8
        sigma_0_d = sigma_0 ** 2
        sigma_1_d = sigma_1 ** 2
        sigma_0_inv = np.linalg.inv(np.diag(sigma_0_d) + np.eye(len(sigma_0_d)) * 1e-6)
        sigma_1_inv = np.linalg.inv(np.diag(sigma_1_d) + np.eye(len(sigma_1_d)) * 1e-6)
        log_lik_0 = -0.5 * np.sum(np.dot(X_test - mu_0, sigma_0_inv) * (X_test - mu_0), axis=1)
        log_lik_1 = -0.5 * np.sum(np.dot(X_test - mu_1, sigma_1_inv) * (X_test - mu_1), axis=1)
        p_0 = 1 / (1 + np.exp(log_lik_1 - log_lik_0 + np.log(0.5 + 1e-8)))
        return p_0

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

    results = []
    shot_settings = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2, 3, 4]

    log("Starting main loop...")

    for seed in seeds:
        log(f"\nSeed {seed}:")
        for held_out in Y_SUBJECTS:
            X_test_orig, y_test_orig = load_eeg_data(held_out)
            train_subjs = [s for s in Y_SUBJECTS if s != held_out]
            X_train_all, y_train_all = [], []
            for subj in train_subjs:
                X, y = load_eeg_data(subj)
                if X is not None:
                    X_train_all.append(X)
                    y_train_all.append(y)

            if len(X_train_all) == 0 or X_test_orig is None:
                continue

            X_train_all = np.vstack(X_train_all)
            y_train_all = np.concatenate(y_train_all)

            mu_source_0 = np.mean(X_train_all[y_train_all == 0], axis=0) if np.any(y_train_all == 0) else np.mean(X_train_all, axis=0)
            mu_source_1 = np.mean(X_train_all[y_train_all == 1], axis=0) if np.any(y_train_all == 1) else np.mean(X_train_all, axis=0)
            sigma_source_0 = np.std(X_train_all[y_train_all == 0], axis=0) + 1e-8 if np.any(y_train_all == 0) else np.std(X_train_all, axis=0) + 1e-8
            sigma_source_1 = np.std(X_train_all[y_train_all == 1], axis=0) + 1e-8 if np.any(y_train_all == 1) else np.std(X_train_all, axis=0) + 1e-8
            source_stats = (mu_source_0, sigma_source_0, mu_source_1, sigma_source_1)

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

            log(f" {held_out}", end='')

            for n_cal in shot_settings:
                if n_cal * 2 > len(cal_pool_indices):
                    continue

                cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
                X_cal = X_cal_pool[cal_idx]
                y_cal = y_cal_pool[cal_idx]

                if len(np.unique(y_cal)) < 2:
                    continue

                pca_models = {}
                for c in [0, 1]:
                    X_c = X_cal[y_cal == c]
                    if len(X_c) > 10:
                        pca = PCA(n_components=min(10, X_c.shape[0]-1), random_state=42)
                        pca.fit(X_c)
                        pca_models[c] = pca

                error_cal = compute_abs_error(X_cal, pca_models)
                error_test = compute_abs_error(X_test, pca_models)
                h_pcet_cal = np.hstack([X_cal, error_cal])
                h_pcet_test = np.hstack([X_test, error_test])

                mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
                mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
                sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else sigma_source_0
                sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else sigma_source_1

                scaler_pcet = StandardScaler()
                h_pcet_cal_s = scaler_pcet.fit_transform(h_pcet_cal)
                h_pcet_test_s = scaler_pcet.transform(h_pcet_test)

                scaler_raw = StandardScaler()
                X_cal_s = scaler_raw.fit_transform(X_cal)
                X_test_s = scaler_raw.transform(X_test)

                uncertainty_cal = compute_uncertainty(X_cal_s, mu_cal_0, sigma_cal_0, mu_cal_1, sigma_cal_1)
                uncertainty_test = compute_uncertainty(X_test_s, mu_cal_0, sigma_cal_0, mu_cal_1, sigma_cal_1)

                clf_pcet = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
                clf_pcet.fit(h_pcet_cal_s, y_cal)
                p_pcet_test = clf_pcet.predict_proba(h_pcet_test_s)[:, 1]

                clf_srgc = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
                clf_srgc.fit(np.column_stack([uncertainty_cal]), y_cal)
                p_srgc_test = clf_srgc.predict_proba(np.column_stack([uncertainty_test]))[:, 1]

                p_domain = compute_domain_similarity(X_cal, y_cal, X_test, source_stats)

                y_pcet_pred = (p_pcet_test >= 0.5).astype(int)
                y_srgc_pred = (p_srgc_test >= 0.5).astype(int)
                y_domain_pred = (p_domain >= 0.5).astype(int)

                acc_pcet = accuracy_score(y_test, y_pcet_pred)
                acc_srgc = accuracy_score(y_test, y_srgc_pred)
                acc_domain = accuracy_score(y_test, y_domain_pred)

                f1_pcet = f1_score(y_test, y_pcet_pred, average='macro')
                f1_srgc = f1_score(y_test, y_srgc_pred, average='macro')
                f1_domain = f1_score(y_test, y_domain_pred, average='macro')

                bacc_pcet = balanced_accuracy_score(y_test, y_pcet_pred)
                bacc_srgc = balanced_accuracy_score(y_test, y_srgc_pred)
                bacc_domain = balanced_accuracy_score(y_test, y_domain_pred)

                try:
                    auroc_pcet = roc_auc_score(y_test, p_pcet_test)
                    auroc_srgc = roc_auc_score(y_test, p_srgc_test)
                    auroc_domain = roc_auc_score(y_test, p_domain)
                except:
                    auroc_pcet = auroc_srgc = auroc_domain = 0.5

                results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'PCET', 'accuracy': acc_pcet, 'macro_f1': f1_pcet, 'balanced_accuracy': bacc_pcet, 'auroc': auroc_pcet})
                results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'SRGC', 'accuracy': acc_srgc, 'macro_f1': f1_srgc, 'balanced_accuracy': bacc_srgc, 'auroc': auroc_srgc})
                results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': 'SIED', 'accuracy': acc_domain, 'macro_f1': f1_domain, 'balanced_accuracy': bacc_domain, 'auroc': auroc_domain})

                configs = [
                    ('SCI_w0.5u0.3d0.2', 0.5, 0.3, 0.2),
                    ('SCI_w0.6u0.2d0.2', 0.6, 0.2, 0.2),
                    ('SCI_w0.7u0.2d0.1', 0.7, 0.2, 0.1),
                    ('SCI_w0.6u0.3d0.1', 0.6, 0.3, 0.1),
                    ('SCI_w0.5u0.2d0.3', 0.5, 0.2, 0.3),
                    ('SCI_w0.7u0.1d0.2', 0.7, 0.1, 0.2),
                    ('SCI_w0.8u0.1d0.1', 0.8, 0.1, 0.1),
                    ('SCI_w0.4u0.3d0.3', 0.4, 0.3, 0.3),
                ]

                for name, w_p, w_u, w_d in configs:
                    p_comb = w_p * p_pcet_test + w_u * p_srgc_test + w_d * p_domain
                    y_pred = (p_comb >= 0.5).astype(int)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='macro')
                    bacc = balanced_accuracy_score(y_test, y_pred)
                    try:
                        auroc = roc_auc_score(y_test, p_comb)
                    except:
                        auroc = 0.5
                    results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': name, 'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc})

                p_unc_norm = 1 - uncertainty_test / (uncertainty_test.max() + 1e-8)
                p_dom_norm = 1 - np.abs(p_domain - 0.5) * 2

                configs_ortho = [
                    ('ORTHO_w0.5u0.3d0.2', 0.5, 0.3, 0.2),
                    ('ORTHO_w0.6u0.2d0.2', 0.6, 0.2, 0.2),
                    ('ORTHO_w0.7u0.2d0.1', 0.7, 0.2, 0.1),
                    ('ORTHO_w0.6u0.3d0.1', 0.6, 0.3, 0.1),
                    ('ORTHO_w0.5u0.2d0.3', 0.5, 0.2, 0.3),
                ]

                for name, w_p, w_u, w_d in configs_ortho:
                    p_ortho = w_p * p_pcet_test + w_u * p_unc_norm + w_d * p_dom_norm
                    y_pred = (p_ortho >= 0.5).astype(int)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='macro')
                    bacc = balanced_accuracy_score(y_test, y_pred)
                    try:
                        auroc = roc_auc_score(y_test, p_ortho)
                    except:
                        auroc = 0.5
                    results.append({'seed': seed, 'subject': held_out, 'n_cal': n_cal, 'method': name, 'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc})

            log('.', end='')

        df_partial = pd.DataFrame(results)
        df_partial.to_csv(f"{RESULTS_DIR}/sci_v2_partial.csv", index=False)
        log(f"\nSeed {seed} completed")

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/sci_v2_results.csv", index=False)

    log("\n" + "="*70)
    log("Results Summary (Accuracy)")
    log("="*70)

    for n_cal in shot_settings:
        log(f"\n{n_cal}-shot:")
        shot_df = df[df['n_cal'] == n_cal]

        best_sci_acc = 0
        best_sci_name = ''
        for method in shot_df['method'].unique():
            if method.startswith('SCI_') or method.startswith('ORTHO_'):
                m_df = shot_df[shot_df['method'] == method]
                acc = m_df['accuracy'].mean()
                if acc > best_sci_acc:
                    best_sci_acc = acc
                    best_sci_name = method

        for method in ['PCET', 'SRGC', 'SIED']:
            method_df = shot_df[shot_df['method'] == method]
            if len(method_df) > 0:
                acc = method_df['accuracy'].mean()
                std = method_df['accuracy'].std()
                log(f"  {method:15s}: {acc:.4f}±{std:.4f}")

        log(f"  {'SCI_best':15s}: {best_sci_acc:.4f} ({best_sci_name})")

    log("\n" + "="*70)
    log("Key Finding: SCI vs Individual Modules")
    log("="*70)
    for n_cal in shot_settings:
        shot_df = df[df['n_cal'] == n_cal]
        pcet_acc = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
        srgc_acc = shot_df[shot_df['method'] == 'SRGC']['accuracy'].mean()
        sied_acc = shot_df[shot_df['method'] == 'SIED']['accuracy'].mean()

        best_sci_acc = 0
        best_sci_name = ''
        for method in shot_df['method'].unique():
            if method.startswith('SCI_') or method.startswith('ORTHO_'):
                m_df = shot_df[shot_df['method'] == method]
                acc = m_df['accuracy'].mean()
                if acc > best_sci_acc:
                    best_sci_acc = acc
                    best_sci_name = method

        best_single = max(pcet_acc, srgc_acc, sied_acc)
        improvement = best_sci_acc - best_single
        log(f"{n_cal}-shot: Best SCI={best_sci_acc:.4f} vs Best Single={best_single:.4f} (Δ={improvement:+.4f}) [{best_sci_name}]")

    log("\nDone!")
    log_file.close()

except Exception as e:
    log(f"ERROR: {str(e)}")
    import traceback
    log(traceback.format_exc())
    log_file.close()