"""SAFE: Score-Augmented Feature Enhancement

Non-destructive three-module combination:
- PCET as main features (raw_eeg + abs_error)
- SRGC provides low-dimensional Gaussian scores
- SIED provides low-dimensional logits/probs
- Separate standardization before concatenation
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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

def compute_gaussian_scores(X, mu_0, sigma_0, mu_1, sigma_1, cov_0_inv=None, cov_1_inv=None, use_full_cov=False):
    """Compute Gaussian class scores (Mahalanobis-based)"""
    if use_full_cov and cov_0_inv is not None and cov_1_inv is not None:
        score_0 = np.array([np.sqrt(np.dot(np.dot(x - mu_0, cov_0_inv), x - mu_0)) for x in X])
        score_1 = np.array([np.sqrt(np.dot(np.dot(x - mu_1, cov_1_inv), x - mu_1)) for x in X])
    else:
        sigma_0_diag = sigma_0 ** 2 + 1e-8
        sigma_1_diag = sigma_1 ** 2 + 1e-8
        score_0 = np.sqrt(np.sum(((X - mu_0) ** 2) / sigma_0_diag, axis=1))
        score_1 = np.sqrt(np.sum(((X - mu_1) ** 2) / sigma_1_diag, axis=1))

    margin = score_1 - score_0
    confidence = np.abs(margin)
    return score_0, score_1, margin, confidence

def train_sied(X_train, y_train, sub_ids, n_subjects, epochs=30, lambda_adv=0.01, dropout=0.3):
    """Train SIED model and return predictions"""
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
        logits = task_logits.cpu().numpy().flatten()

    return logits, probs

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

print("="*70)
print("SAFE: Score-Augmented Feature Enhancement")
print("="*70)

results = []
shot_settings = [3, 5, 10, 20, 50]
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    print(f"\nSeed {seed}:", flush=True)
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

            sub_ids_cal = []
            for i, subj in enumerate(train_subjs):
                X_subj, _ = load_eeg_data(subj)
                if X_subj is not None:
                    sub_ids_cal.extend([i] * len(X_subj))
            sub_ids_cal = np.array(sub_ids_cal[:len(y_train_all)])
            sub_ids_train = sub_ids_cal

            pca_models = train_pca_predictor(X_cal, y_cal, n_components=20)
            error_cal = compute_abs_error(X_cal, pca_models)
            error_test = compute_abs_error(X_test, pca_models)

            h_pcet_cal = np.hstack([X_cal, error_cal])
            h_pcet_test = np.hstack([X_test, error_test])

            mu_cal_0 = np.mean(X_cal[y_cal == 0], axis=0) if np.any(y_cal == 0) else mu_source_0
            mu_cal_1 = np.mean(X_cal[y_cal == 1], axis=0) if np.any(y_cal == 1) else mu_source_1
            sigma_cal_0 = np.std(X_cal[y_cal == 0], axis=0) + 1e-8 if np.any(y_cal == 0) else np.std(X_cal, axis=0) + 1e-8
            sigma_cal_1 = np.std(X_cal[y_cal == 1], axis=0) + 1e-8 if np.any(y_cal == 1) else np.std(X_cal, axis=0) + 1e-8

            alpha, beta = 0.75, 0.75
            mu_blend_0 = alpha * mu_cal_0 + (1 - alpha) * mu_source_0
            mu_blend_1 = alpha * mu_cal_1 + (1 - alpha) * mu_source_1
            cov_source_0 = np.diag(sigma_source_0 ** 2) + 1e-6
            cov_source_1 = np.diag(sigma_source_1 ** 2) + 1e-6
            cov_cal_0 = np.cov(X_cal[y_cal == 0].T) + np.eye(X_cal.shape[1]) * 1e-4 if np.any(y_cal == 0) else np.eye(X_cal.shape[1])
            cov_cal_1 = np.cov(X_cal[y_cal == 1].T) + np.eye(X_cal.shape[1]) * 1e-4 if np.any(y_cal == 1) else np.eye(X_cal.shape[1])
            cov_blend_0 = beta * cov_source_0 + (1 - beta) * cov_cal_0
            cov_blend_1 = beta * cov_source_1 + (1 - beta) * cov_cal_1

            try:
                cov_blend_0_inv = np.linalg.inv(cov_blend_0)
                cov_blend_1_inv = np.linalg.inv(cov_blend_1)
                use_full_cov = True
            except:
                cov_blend_0_inv = None
                cov_blend_1_inv = None
                use_full_cov = False

            s0_cal, s1_cal, margin_cal, conf_cal = compute_gaussian_scores(X_cal, mu_blend_0, sigma_cal_0, mu_blend_1, sigma_cal_1, cov_blend_0_inv, cov_blend_1_inv, use_full_cov)
            s0_test, s1_test, margin_test, conf_test = compute_gaussian_scores(X_test, mu_blend_0, sigma_cal_0, mu_blend_1, sigma_cal_1, cov_blend_0_inv, cov_blend_1_inv, use_full_cov)
            h_srgc_cal = np.column_stack([s0_cal, s1_cal, margin_cal, conf_cal])
            h_srgc_test = np.column_stack([s0_test, s1_test, margin_test, conf_test])

            try:
                sied_logits_cal, sied_probs_cal = train_sied(X_cal, y_cal, sub_ids_train[:len(y_cal)], n_subjects, epochs=30, lambda_adv=0.01)
                sied_logits_test, sied_probs_test = train_sied(X_test, y_test, np.zeros(len(y_test)), n_subjects, epochs=30, lambda_adv=0.01)
                h_sied_cal = np.column_stack([sied_logits_cal, sied_probs_cal, np.zeros(len(sied_probs_cal))])
                h_sied_test = np.column_stack([sied_logits_test, sied_probs_test, np.zeros(len(sied_probs_test))])
            except:
                h_sied_cal = np.zeros((len(X_cal), 3))
                h_sied_test = np.zeros((len(X_test), 3))

            scaler_pcet = StandardScaler()
            h_pcet_cal_s = scaler_pcet.fit_transform(h_pcet_cal)
            h_pcet_test_s = scaler_pcet.transform(h_pcet_test)

            scaler_srgc = StandardScaler()
            h_srgc_cal_s = scaler_srgc.fit_transform(h_srgc_cal)
            h_srgc_test_s = scaler_srgc.transform(h_srgc_test)

            scaler_sied = StandardScaler()
            h_sied_cal_s = scaler_sied.fit_transform(h_sied_cal)
            h_sied_test_s = scaler_sied.transform(h_sied_test)

            h_safe_cal = np.hstack([h_pcet_cal_s, h_srgc_cal_s, h_sied_cal_s])
            h_safe_test = np.hstack([h_pcet_test_s, h_srgc_test_s, h_sied_test_s])

            methods = {
                'EEG_SVM': (X_cal, X_test, 'svm'),
                'PCET': (h_pcet_cal_s, h_pcet_test_s, 'ridge'),
                'SRGC': (h_srgc_cal_s, h_srgc_test_s, 'ridge'),
                'SIED': (h_sied_cal_s, h_sied_test_s, 'ridge'),
                'PCET_SRGC_score': (np.hstack([h_pcet_cal_s, h_srgc_cal_s]), np.hstack([h_pcet_test_s, h_srgc_test_s]), 'ridge'),
                'PCET_SIED_score': (np.hstack([h_pcet_cal_s, h_sied_cal_s]), np.hstack([h_pcet_test_s, h_sied_test_s]), 'ridge'),
                'SAFE': (h_safe_cal, h_safe_test, 'ridge'),
            }

            for method_name, (X_tr, X_te, clf_type) in methods.items():
                try:
                    if clf_type == 'svm':
                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_cal)
                        X_te = scaler.transform(X_test)
                        clf = SVC(kernel='linear', random_state=seed)
                        clf.fit(X_tr, y_cal)
                    else:
                        clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
                        clf.fit(X_tr, y_cal)

                    y_pred = clf.predict(X_te)
                    try:
                        y_prob = clf.predict_proba(X_te)[:, 1]
                        auroc = roc_auc_score(y_test, y_prob)
                    except:
                        auroc = 0.5

                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='macro')
                    bacc = balanced_accuracy_score(y_test, y_pred)

                    results.append({
                        'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                        'method': method_name,
                        'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
                    })
                except Exception as e:
                    pass

        print(f'.', end='', flush=True)

    df_partial = pd.DataFrame(results)
    df_partial.to_csv(f"{RESULTS_DIR}/safe_partial_results.csv", index=False)

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/safe_module_combination.csv", index=False)

print("\n" + "="*70)
print("Results Summary")
print("="*70)

summary_data = []
for n_cal in shot_settings:
    row = {'Shot': n_cal}
    shot_df = df[df['n_cal'] == n_cal]
    for method in ['EEG_SVM', 'PCET', 'SRGC', 'SIED', 'PCET_SRGC_score', 'PCET_SIED_score', 'SAFE']:
        method_df = shot_df[shot_df['method'] == method]
        if len(method_df) > 0:
            acc = method_df['accuracy'].mean()
            std = method_df['accuracy'].std()
            row[method] = f"{acc:.4f}±{std:.4f}"
            row[f'{method}_raw'] = acc
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("Success Criteria Check")
print("="*70)

for n_cal in shot_settings:
    shot_df = df[df['n_cal'] == n_cal]
    pcet_acc = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
    safe_acc = shot_df[shot_df['method'] == 'SAFE']['accuracy'].mean()
    gap = safe_acc - pcet_acc
    print(f"{n_cal}-shot: PCET={pcet_acc:.4f}, SAFE={safe_acc:.4f}, gap={gap:+.4f}")

print("\nDone!", flush=True)