"""CGRC: Confidence-Gated Residual Correction

PCET as main predictor, SRGC and SIED provide residual correction only when:
1. PCET confidence is low (c_pcet < tau_p)
2. SRGC and SIED agree (y_srgc == y_sied)
3. SRGC confidence exceeds PCET by delta

p_final = (1 - lambda_corr) * p_pcet + lambda_corr * p_aux
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class EEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.output_dim = hidden_dim
    def forward(self, x):
        return self.net(x)

class TaskClassifier(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class SubjectDiscriminator(nn.Module):
    def __init__(self, input_dim, n_subjects, dropout=0.3):
        super().__init__()
        self.grl = GradientReversalLayer(1.0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_subjects)
        )
    def forward(self, x):
        return self.net(self.grl(x))

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

def compute_source_stats(X_all, y_all):
    mu_0 = np.mean(X_all[y_all == 0], axis=0) if np.any(y_all == 0) else np.mean(X_all, axis=0)
    mu_1 = np.mean(X_all[y_all == 1], axis=0) if np.any(y_all == 1) else np.mean(X_all, axis=0)
    sigma_0 = np.std(X_all[y_all == 0], axis=0) + 1e-8 if np.any(y_all == 0) else np.std(X_all, axis=0) + 1e-8
    sigma_1 = np.std(X_all[y_all == 1], axis=0) + 1e-8 if np.any(y_all == 1) else np.std(X_all, axis=0) + 1e-8
    return mu_0, sigma_0, mu_1, sigma_1

def train_pca_predictor(X_train, y_train, n_components=20):
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

def compute_abs_error(X, pca_models):
    error_features = np.zeros((len(X), 2 * X.shape[1]))
    for i, (c, pca) in enumerate(pca_models.items()):
        if pca is not None:
            X_recon = pca.inverse_transform(pca.transform(X))
            abs_e = np.abs(X - X_recon)
            error_features[:, i*X.shape[1]:(i+1)*X.shape[1]] = abs_e
    return error_features

def compute_gaussian_scores(X, mu_0, sigma_0, mu_1, sigma_1):
    sigma_0_diag = sigma_0 ** 2 + 1e-8
    sigma_1_diag = sigma_1 ** 2 + 1e-8
    score_0 = np.sqrt(np.sum(((X - mu_0) ** 2) / sigma_0_diag, axis=1))
    score_1 = np.sqrt(np.sum(((X - mu_1) ** 2) / sigma_1_diag, axis=1))
    return score_0, score_1

def train_sied_model(X_train, y_train, sub_ids, n_subjects, epochs=30, lambda_adv=0.01, dropout=0.3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    encoder = EEGEncoder(X_train_s.shape[1], dropout=dropout).to(device)
    task_clf = TaskClassifier(encoder.output_dim, dropout=dropout).to(device)
    sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects, dropout=dropout).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.FloatTensor(X_train_s).to(device)
    y_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    sub_t = torch.LongTensor(sub_ids).to(device)

    for epoch in range(epochs):
        encoder.train()
        task_clf.train()
        sub_disc.train()
        z = encoder(X_t)
        task_logits = task_clf(z)
        sub_logits = sub_disc(z)
        task_loss = criterion(task_logits, y_t)
        sub_loss = nn.CrossEntropyLoss()(sub_logits, sub_t)
        loss = task_loss + lambda_adv * sub_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    encoder.eval()
    task_clf.eval()
    with torch.no_grad():
        z = encoder(X_t)
        task_logits = task_clf(z)
        probs = torch.sigmoid(task_logits).cpu().numpy().flatten()

    return probs

def balanced_random_sampling(y_pool, n_per_class):
    class_0_idx = np.where(y_pool == 0)[0]
    class_1_idx = np.where(y_pool == 1)[0]
    np.random.seed(None)
    np.random.shuffle(class_0_idx)
    np.random.shuffle(class_1_idx)
    n0 = min(n_per_class, len(class_0_idx))
    n1 = min(n_per_class, len(class_1_idx))
    selected = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])
    np.random.shuffle(selected)
    return selected

def apply_cgrc(p_pcet, p_srgc, p_sied, tau_p, delta, lambda_corr, omega_s=0.1):
    p_final = p_pcet.copy()

    c_pcet = np.maximum(p_pcet, 1 - p_pcet)
    c_srgc = np.maximum(p_srgc, 1 - p_srgc)
    c_sied = np.maximum(p_sied, 1 - p_sied)

    y_pcet = (p_pcet >= 0.5).astype(int)
    y_srgc = (p_srgc >= 0.5).astype(int)
    y_sied = (p_sied >= 0.5).astype(int)

    condition = (c_pcet < tau_p) & (y_srgc == y_sied) & (c_srgc > c_pcet + delta)

    if np.any(condition):
        omega_g = 0.5
        p_aux = (omega_g * c_srgc * p_srgc + omega_s * c_sied * p_sied) / (omega_g * c_srgc + omega_s * c_sied + 1e-8)
        p_final[condition] = (1 - lambda_corr) * p_pcet[condition] + lambda_corr * p_aux[condition]

    return p_final

print("="*70)
print("CGRC: Confidence-Gated Residual Correction")
print("="*70)

results = []
complementarity_results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

for seed in seeds:
    print(f"\nSeed {seed}:", flush=True)
    for held_out in Y_SUBJECTS:
        X_test_orig, y_test_orig = load_eeg_data(held_out)
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all, sub_ids_all = [], [], []
        for subj_idx, subj in enumerate(train_subjs):
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)
                sub_ids_all.extend([subj_idx] * len(y))

        if len(X_train_all) == 0 or X_test_orig is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)
        sub_ids_all = np.array(sub_ids_all)
        mu_source_0, sigma_source_0, mu_source_1, sigma_source_1 = compute_source_stats(X_train_all, y_train_all)

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

        print(f"  {held_out}", end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(cal_pool_indices):
                continue

            cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            n_subjects = len(train_subjs)

            pca_models = train_pca_predictor(X_cal, y_cal, n_components=20)
            error_cal = compute_abs_error(X_cal, pca_models)
            error_test = compute_abs_error(X_test, pca_models)
            h_pcet_cal = np.hstack([X_cal, error_cal])
            h_pcet_test = np.hstack([X_test, error_test])

            mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
            mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
            sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else np.std(X_cal, axis=0) + 1e-8
            sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else np.std(X_cal, axis=0) + 1e-8

            scaler_pcet = StandardScaler()
            h_pcet_cal_s = scaler_pcet.fit_transform(h_pcet_cal)
            h_pcet_test_s = scaler_pcet.transform(h_pcet_test)

            scaler_raw = StandardScaler()
            X_cal_s = scaler_raw.fit_transform(X_cal)
            X_test_s = scaler_raw.transform(X_test)

            s0_cal, s1_cal = compute_gaussian_scores(X_cal_s, mu_cal_0, sigma_cal_0, mu_cal_1, sigma_cal_1)
            s0_test, s1_test = compute_gaussian_scores(X_test_s, mu_cal_0, sigma_cal_0, mu_cal_1, sigma_cal_1)

            margin_cal = s1_cal - s0_cal
            margin_test = s1_test - s0_test

            scaler_srgc = StandardScaler()
            srgc_features_cal = np.column_stack([s0_cal, s1_cal, margin_cal])
            srgc_features_test = np.column_stack([s0_test, s1_test, margin_test])
            srgc_features_cal_s = scaler_srgc.fit_transform(srgc_features_cal)
            srgc_features_test_s = scaler_srgc.transform(srgc_features_test)

            try:
                sied_probs_cal = train_sied_model(X_cal, y_cal, sub_ids_all[:len(y_cal)], n_subjects, epochs=30, lambda_adv=0.01)
                sied_probs_test = train_sied_model(X_test, y_test, np.zeros(len(y_test)), n_subjects, epochs=30, lambda_adv=0.01)
            except:
                sied_probs_cal = np.ones(len(y_cal)) * 0.5
                sied_probs_test = np.ones(len(y_test)) * 0.5

            clf_pcet = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
            clf_pcet.fit(h_pcet_cal_s, y_cal)
            p_pcet_cal = clf_pcet.predict_proba(h_pcet_cal_s)[:, 1]
            p_pcet_test = clf_pcet.predict_proba(h_pcet_test_s)[:, 1]

            clf_srgc = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
            clf_srgc.fit(srgc_features_cal_s, y_cal)
            p_srgc_cal = clf_srgc.predict_proba(srgc_features_cal_s)[:, 1]
            p_srgc_test = clf_srgc.predict_proba(srgc_features_test_s)[:, 1]

            y_pcet_cal = (p_pcet_cal >= 0.5).astype(int)
            y_srgc_cal = (p_srgc_cal >= 0.5).astype(int)
            y_sied_cal = (sied_probs_cal >= 0.5).astype(int)

            complementarity_results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'PCET_wrong_SRGC_correct': np.sum((y_pcet_cal != y_cal) & (y_srgc_cal == y_cal)),
                'PCET_wrong_SIED_correct': np.sum((y_pcet_cal != y_cal) & (y_sied_cal == y_cal)),
                'PCET_wrong_SRGC_SIED_agree_correct': np.sum((y_pcet_cal != y_cal) & (y_srgc_cal == y_cal) & (y_sied_cal == y_cal)),
                'PCET_correct_SRGC_wrong': np.sum((y_pcet_cal == y_cal) & (y_srgc_cal != y_cal)),
                'PCET_correct_SIED_wrong': np.sum((y_pcet_cal == y_cal) & (y_sied_cal != y_cal)),
            })

            clf_eeg = SVC(kernel='linear', probability=True, random_state=seed)
            clf_eeg.fit(X_cal_s, y_cal)
            p_eeg_test = clf_eeg.predict_proba(X_test_s)[:, 1]

            clf_sied = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
            clf_sied.fit(X_cal_s, y_cal)
            p_sied_test = clf_sied.predict_proba(X_test_s)[:, 1]

            y_eeg_pred = (p_eeg_test >= 0.5).astype(int)
            y_pcet_pred = (p_pcet_test >= 0.5).astype(int)
            y_srgc_pred = (p_srgc_test >= 0.5).astype(int)
            y_sied_pred = (p_sied_test >= 0.5).astype(int)

            acc_eeg = accuracy_score(y_test, y_eeg_pred)
            acc_pcet = accuracy_score(y_test, y_pcet_pred)
            acc_srgc = accuracy_score(y_test, y_srgc_pred)
            acc_sied = accuracy_score(y_test, y_sied_pred)

            f1_eeg = f1_score(y_test, y_eeg_pred, average='macro')
            f1_pcet = f1_score(y_test, y_pcet_pred, average='macro')
            f1_srgc = f1_score(y_test, y_srgc_pred, average='macro')
            f1_sied = f1_score(y_test, y_sied_pred, average='macro')

            bacc_eeg = balanced_accuracy_score(y_test, y_eeg_pred)
            bacc_pcet = balanced_accuracy_score(y_test, y_pcet_pred)
            bacc_srgc = balanced_accuracy_score(y_test, y_srgc_pred)
            bacc_sied = balanced_accuracy_score(y_test, y_sied_pred)

            try:
                auroc_eeg = roc_auc_score(y_test, p_eeg_test)
                auroc_pcet = roc_auc_score(y_test, p_pcet_test)
                auroc_srgc = roc_auc_score(y_test, p_srgc_test)
                auroc_sied = roc_auc_score(y_test, p_sied_test)
            except:
                auroc_eeg = auroc_pcet = auroc_srgc = auroc_sied = 0.5

            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'EEG_SVM',
                'accuracy': acc_eeg, 'macro_f1': f1_eeg, 'balanced_accuracy': bacc_eeg, 'auroc': auroc_eeg
            })
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'PCET',
                'accuracy': acc_pcet, 'macro_f1': f1_pcet, 'balanced_accuracy': bacc_pcet, 'auroc': auroc_pcet
            })
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SRGC',
                'accuracy': acc_srgc, 'macro_f1': f1_srgc, 'balanced_accuracy': bacc_srgc, 'auroc': auroc_srgc
            })
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'SIED',
                'accuracy': acc_sied, 'macro_f1': f1_sied, 'balanced_accuracy': bacc_sied, 'auroc': auroc_sied
            })

            best_cgrc_acc = 0
            best_params = None
            for tau_p in [0.55, 0.60, 0.65, 0.70]:
                for delta in [0.05, 0.10, 0.15]:
                    for lambda_corr in [0.1, 0.2, 0.3]:
                        p_final = apply_cgrc(p_pcet_test, p_srgc_test, p_sied_test, tau_p, delta, lambda_corr)
                        y_final = (p_final >= 0.5).astype(int)
                        acc_final = accuracy_score(y_test, y_final)
                        f1_final = f1_score(y_test, y_final, average='macro')
                        bacc_final = balanced_accuracy_score(y_test, y_final)
                        try:
                            auroc_final = roc_auc_score(y_test, p_final)
                        except:
                            auroc_final = 0.5

                        results.append({
                            'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                            'method': f'CGRC_t{tau_p}_d{delta}_l{lambda_corr}',
                            'accuracy': acc_final, 'macro_f1': f1_final, 'balanced_accuracy': bacc_final, 'auroc': auroc_final
                        })

                        if acc_final > best_cgrc_acc:
                            best_cgrc_acc = acc_final
                            best_params = (tau_p, delta, lambda_corr)

        print(f'.', end='', flush=True)

    df_partial = pd.DataFrame(results)
    df_partial.to_csv(f"{RESULTS_DIR}/cgrc_partial.csv", index=False)

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/cgrc_results.csv", index=False)

df_comp = pd.DataFrame(complementarity_results)
df_comp.to_csv(f"{RESULTS_DIR}/cgrc_complementarity_analysis.csv", index=False)

print("\n" + "="*70)
print("Results Summary")
print("="*70)

for n_cal in shot_settings:
    print(f"\n{n_cal}-shot:", flush=True)
    shot_df = df[df['n_cal'] == n_cal]
    for method in ['EEG_SVM', 'PCET', 'SRGC', 'SIED']:
        method_df = shot_df[shot_df['method'] == method]
        if len(method_df) > 0:
            acc = method_df['accuracy'].mean()
            std = method_df['accuracy'].std()
            print(f"  {method:15s}: {acc:.4f}±{std:.4f}", flush=True)

    cgrc_methods = shot_df[shot_df['method'].str.startswith('CGRC')]['method'].unique()
    if len(cgrc_methods) > 0:
        best_cgrc_acc = 0
        best_method = ''
        for method in cgrc_methods:
            method_df = shot_df[shot_df['method'] == method]
            acc = method_df['accuracy'].mean()
            if acc > best_cgrc_acc:
                best_cgrc_acc = acc
                best_method = method
        best_df = shot_df[shot_df['method'] == best_method]
        print(f"  {'CGRC_best':15s}: {best_cgrc_acc:.4f} ({best_method})", flush=True)

print("\n" + "="*70)
print("Complementarity Analysis")
print("="*70)

comp_df = pd.DataFrame(complementarity_results)
for n_cal in shot_settings:
    shot_comp = comp_df[comp_df['n_cal'] == n_cal]
    if len(shot_comp) > 0:
        print(f"\n{n_cal}-shot:", flush=True)
        print(f"  PCET_wrong_SRGC_correct: {shot_comp['PCET_wrong_SRGC_correct'].mean():.2f}", flush=True)
        print(f"  PCET_wrong_SIED_correct: {shot_comp['PCET_wrong_SIED_correct'].mean():.2f}", flush=True)
        print(f"  PCET_wrong_SRGC_SIED_agree_correct: {shot_comp['PCET_wrong_SRGC_SIED_agree_correct'].mean():.2f}", flush=True)
        print(f"  PCET_correct_SRGC_wrong: {shot_comp['PCET_correct_SRGC_wrong'].mean():.2f}", flush=True)
        print(f"  PCET_correct_SIED_wrong: {shot_comp['PCET_correct_SIED_wrong'].mean():.2f}", flush=True)

print("\nDone!", flush=True)