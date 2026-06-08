"""PCET-centered Auxiliary Joint Training - Fixed Version

PCET as main prediction path, SRGC and SIED as auxiliary constraints.

Loss function:
    L_total = L_pcet + eta * L_srgc_aux + lambda_adv * L_sied_aux

where:
    - PCET is the final prediction path
    - SRGC only as low-shot source-prior auxiliary constraint (on raw features)
    - SIED only as source-side regularization
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.special import softmax
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

class PCETClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
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

def compute_gaussian_scores_np(X, mu_0, sigma_0, mu_1, sigma_1):
    sigma_0_diag = sigma_0 ** 2 + 1e-8
    sigma_1_diag = sigma_1 ** 2 + 1e-8
    score_0 = np.sqrt(np.sum(((X - mu_0) ** 2) / sigma_0_diag, axis=1))
    score_1 = np.sqrt(np.sum(((X - mu_1) ** 2) / sigma_1_diag, axis=1))
    return score_0, score_1

def compute_lambda_warmup(p, lambda_max, gamma):
    return lambda_max * (2 / (1 + np.exp(-gamma * p)) - 1)

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

print("="*70)
print("PCET-centered Auxiliary Joint Training")
print("="*70)

results = []
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
        raw_dim = X_train_all.shape[1]
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

            pca_models = train_pca_predictor(X_cal, y_cal, n_components=20)
            error_cal = compute_abs_error(X_cal, pca_models)
            error_test = compute_abs_error(X_test, pca_models)
            h_pcet_cal = np.hstack([X_cal, error_cal])
            h_pcet_test = np.hstack([X_test, error_test])

            mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
            mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
            sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else np.std(X_cal, axis=0) + 1e-8
            sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else np.std(X_cal, axis=0) + 1e-8

            sub_ids_cal = sub_ids_all[cal_idx] if len(sub_ids_all) > len(cal_idx) else np.zeros(len(y_cal), dtype=int)
            n_subjects = len(train_subjs)

            scaler_raw = StandardScaler()
            X_cal_s = scaler_raw.fit_transform(X_cal)
            X_test_s = scaler_raw.transform(X_test)

            s0_cal, s1_cal = compute_gaussian_scores_np(X_cal_s, mu_cal_0, sigma_cal_0, mu_cal_1, sigma_cal_1)
            s0_test, s1_test = compute_gaussian_scores_np(X_test_s, mu_cal_0, sigma_cal_0, mu_cal_1, sigma_cal_1)

            margin_cal = s1_cal - s0_cal
            margin_test = s1_test - s0_test
            conf_cal = np.abs(margin_cal)
            conf_test = np.abs(margin_test)
            srgc_features_cal = np.column_stack([s0_cal, s1_cal, margin_cal, conf_cal])
            srgc_features_test = np.column_stack([s0_test, s1_test, margin_test, conf_test])

            scaler_srgc = StandardScaler()
            srgc_features_cal_s = scaler_srgc.fit_transform(srgc_features_cal)
            srgc_features_test_s = scaler_srgc.transform(srgc_features_test)

            scaler_pcet = StandardScaler()
            h_pcet_cal_s = scaler_pcet.fit_transform(h_pcet_cal)
            h_pcet_test_s = scaler_pcet.transform(h_pcet_test)

            h_pcet_cal_t = torch.FloatTensor(h_pcet_cal_s).to(device)
            h_pcet_test_t = torch.FloatTensor(h_pcet_test_s).to(device)
            srgc_cal_t = torch.FloatTensor(srgc_features_cal_s).to(device)
            srgc_test_t = torch.FloatTensor(srgc_features_test_s).to(device)
            y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).to(device)
            y_cal_long = torch.LongTensor(y_cal).to(device)

            kappa = 5
            eta = kappa / (kappa + n_cal)

            def train_with_aux(use_srgc, use_sied, lambda_max=0.01, gamma=5, epochs=50):
                input_dim = h_pcet_cal_s.shape[1]
                clf = PCETClassifier(input_dim).to(device)
                sub_disc = SubjectDiscriminator(input_dim, n_subjects).to(device) if use_sied else None

                params = list(clf.parameters())
                if sub_disc is not None:
                    params += list(sub_disc.parameters())
                optimizer = optim.Adam(params, lr=0.001, weight_decay=1e-4)
                criterion = nn.BCEWithLogitsLoss()

                best_loss = float('inf')
                patience_counter = 0

                for epoch in range(epochs):
                    p = epoch / epochs
                    current_lambda = compute_lambda_warmup(p, lambda_max, gamma) if use_sied else 0

                    clf.train()
                    if sub_disc is not None:
                        sub_disc.train()

                    logits_pcet = clf(h_pcet_cal_t)
                    L_pcet = criterion(logits_pcet, y_cal_t)

                    L_srgc_aux = torch.tensor(0.0).to(device)
                    if use_srgc:
                        srgc_logits = srgc_cal_t[:, 0] - srgc_cal_t[:, 1]
                        srgc_probs = torch.sigmoid(srgc_logits)
                        L_srgc_aux = F.binary_cross_entropy(srgc_probs, y_cal_t.squeeze())

                    L_sied_aux = torch.tensor(0.0).to(device)
                    if use_sied and sub_disc is not None:
                        sub_logits = sub_disc(h_pcet_cal_t)
                        L_sied_aux = F.cross_entropy(sub_logits, torch.LongTensor(sub_ids_cal).to(device))

                    L_total = L_pcet + eta * L_srgc_aux + current_lambda * L_sied_aux

                    optimizer.zero_grad()
                    L_total.backward()
                    optimizer.step()

                    if L_total.item() < best_loss:
                        best_loss = L_total.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= 10:
                            break

                clf.eval()
                with torch.no_grad():
                    test_logits = clf(h_pcet_test_t)
                    test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
                    test_preds = (test_probs >= 0.5).astype(int)

                try:
                    auroc = roc_auc_score(y_test, test_probs)
                except:
                    auroc = 0.5

                acc = accuracy_score(y_test, test_preds)
                f1 = f1_score(y_test, test_preds, average='macro')
                bacc = balanced_accuracy_score(y_test, test_preds)

                return acc, f1, bacc, auroc

            methods = {
                'PCET': (False, False),
                'PCET_SRGC_aux': (True, False),
                'PCET_SIED_aux': (False, True),
                'PCET_SRGC_SIED_aux': (True, True),
            }

            for method_name, (use_srgc, use_sied) in methods.items():
                acc, f1, bacc, auroc = train_with_aux(use_srgc, use_sied)
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': method_name,
                    'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
                })

        print(f'.', end='', flush=True)

    df_partial = pd.DataFrame(results)
    df_partial.to_csv(f"{RESULTS_DIR}/auxiliary_joint_partial.csv", index=False)

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/auxiliary_joint_training.csv", index=False)

print("\n" + "="*70)
print("Results Summary")
print("="*70)

for n_cal in shot_settings:
    print(f"\n{n_cal}-shot:", flush=True)
    shot_df = df[df['n_cal'] == n_cal]
    for method in sorted(shot_df['method'].unique()):
        method_df = shot_df[shot_df['method'] == method]
        acc = method_df['accuracy'].mean()
        std = method_df['accuracy'].std()
        print(f"  {method:25s}: {acc:.4f}±{std:.4f}", flush=True)

print("\n" + "="*70)
print("Success Criteria Check")
print("="*70)

pcet_baseline = df[df['method'] == 'PCET']
for n_cal in shot_settings:
    pcet_acc = pcet_baseline[pcet_baseline['n_cal'] == n_cal]['accuracy'].mean()
    full_acc = df[(df['method'] == 'PCET_SRGC_SIED_aux') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    gap = full_acc - pcet_acc
    print(f"{n_cal}-shot: PCET={pcet_acc:.4f}, Full={full_acc:.4f}, gap={gap:+.4f}", flush=True)

print("\nDone!", flush=True)