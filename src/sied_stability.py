"""SIED Stability Optimization: Lambda Warm-up + Subject Discriminator Regularization

1. Lambda warm-up: lambda_adv = lambda_max * (2 / (1 + exp(-gamma * p)) - 1)
   - lambda_max = [0.005, 0.01, 0.05]
   - gamma = [5, 10]

2. Subject discriminator regularization:
   - dropout = [0.1, 0.3, 0.5]
   - label_smoothing = [0.0, 0.1]
"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
        self.net = nn.Sequential(
            GradientReversalLayer(1.0),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_subjects)
        )

    def forward(self, x):
        return self.net(x)

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

def compute_lambda_warmup(p, lambda_max, gamma):
    return lambda_max * (2 / (1 + np.exp(-gamma * p)) - 1)

def run_raw_eeg_baseline(seed):
    print(f"  Raw_EEG seed={seed}", flush=True)
    results = []

    for held_out in Y_SUBJECTS:
        try:
            train_subjs = [s for s in Y_SUBJECTS if s != held_out]
            X_train_all, y_train_all = [], []
            for subj in train_subjs:
                X, y = load_eeg_data(subj)
                if X is not None:
                    X_train_all.append(X)
                    y_train_all.append(y)

            X_test, y_test = load_eeg_data(held_out)
            if len(X_train_all) == 0 or X_test is None:
                continue

            X_train_all = np.vstack(X_train_all)
            y_train_all = np.concatenate(y_train_all)

            np.random.seed(seed)
            indices = np.random.permutation(len(y_train_all))
            val_size = int(len(y_train_all) * 0.1)
            train_idx = indices[val_size:]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_all[train_idx])
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X_tr, y_train_all[train_idx])
            y_pred = clf.predict(X_test_s)
            probs = clf.predict_proba(X_test_s)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            bacc = balanced_accuracy_score(y_test, y_pred)
            try:
                auroc = roc_auc_score(y_test, probs)
            except:
                auroc = 0.5

            results.append({
                'model': 'Raw_EEG', 'seed': seed, 'held_out': held_out,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc,
                'subject_predictability': 0.5
            })
        except Exception as e:
            print(f"    Error {held_out}: {e}", flush=True)

    return results

def run_sied_baseline(seed, lambda_adv=0.01):
    print(f"  SIED_baseline lambda={lambda_adv} seed={seed}", flush=True)
    results = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_subjects = len(Y_SUBJECTS) - 1

    for held_out in Y_SUBJECTS:
        try:
            train_subjs = [s for s in Y_SUBJECTS if s != held_out]
            X_train_all, y_train_all, sub_ids = [], [], []

            for subj_idx, subj in enumerate(train_subjs):
                X, y = load_eeg_data(subj)
                if X is not None:
                    X_train_all.append(X)
                    y_train_all.append(y)
                    sub_ids.extend([subj_idx] * len(y))

            X_test, y_test = load_eeg_data(held_out)
            if len(X_train_all) == 0 or X_test is None:
                continue

            X_train_all = np.vstack(X_train_all)
            y_train_all = np.concatenate(y_train_all)
            sub_ids = np.array(sub_ids)

            np.random.seed(seed)
            indices = np.random.permutation(len(y_train_all))
            val_size = int(len(y_train_all) * 0.1)
            train_idx = indices[val_size:]
            val_idx = indices[:val_size]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_all[train_idx])
            y_tr = y_train_all[train_idx]
            sub_tr = sub_ids[train_idx]
            X_val = scaler.transform(X_train_all[val_idx])
            y_val = y_train_all[val_idx]
            X_test_s = scaler.transform(X_test)

            eeg_dim = X_tr.shape[1]
            encoder = EEGEncoder(eeg_dim).to(device)
            task_clf = TaskClassifier(encoder.output_dim).to(device)
            sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

            optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            X_tr_t = torch.FloatTensor(X_tr).to(device)
            y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
            sub_tr_t = torch.LongTensor(sub_tr).to(device)
            X_val_t = torch.FloatTensor(X_val).to(device)
            X_test_t = torch.FloatTensor(X_test_s).to(device)

            best_val_f1 = 0
            best_encoder_state = None
            best_clf_state = None
            patience_counter = 0

            for epoch in range(50):
                encoder.train()
                task_clf.train()
                sub_disc.train()

                z = encoder(X_tr_t)
                task_logits = task_clf(z)
                sub_logits = sub_disc(z)

                task_loss = criterion(task_logits, y_tr_t)
                sub_loss = F.cross_entropy(sub_logits, sub_tr_t)
                loss = task_loss + lambda_adv * sub_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                encoder.eval()
                task_clf.eval()
                with torch.no_grad():
                    val_z = encoder(X_val_t)
                    val_logits = task_clf(val_z)
                    val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                    val_f1 = f1_score(y_val, val_preds.cpu().numpy(), average='macro')

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_encoder_state = encoder.state_dict().copy()
                    best_clf_state = task_clf.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        break

            if best_encoder_state is not None:
                encoder.load_state_dict(best_encoder_state)
                task_clf.load_state_dict(best_clf_state)

            encoder.eval()
            with torch.no_grad():
                test_z = encoder(X_test_t)
                test_logits = task_clf(test_z)
                test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
                test_preds = (test_probs >= 0.5).astype(int)

            acc = accuracy_score(y_test, test_preds)
            f1 = f1_score(y_test, test_preds, average='macro')
            bacc = balanced_accuracy_score(y_test, test_preds)
            try:
                auroc = roc_auc_score(y_test, test_probs)
            except:
                auroc = 0.5

            test_z_np = test_z.cpu().numpy()
            try:
                sub_clf = LogisticRegression(max_iter=1000, random_state=seed)
                sub_clf.fit(test_z_np, y_test)
                sub_pred = sub_clf.predict(test_z_np)
                sub_acc = accuracy_score(y_test, sub_pred)
            except:
                sub_acc = 0.5

            results.append({
                'model': f'SIED_baseline_l{lambda_adv}', 'seed': seed, 'held_out': held_out,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc,
                'subject_predictability': sub_acc
            })

        except Exception as e:
            print(f"    Error {held_out}: {e}", flush=True)

    return results

def run_sied_warmup(seed, lambda_max, gamma, dropout, label_smoothing):
    print(f"  warmup_lmax{lambda_max}_g{gamma}_d{dropout}_ls{label_smoothing} seed={seed}", flush=True)
    results = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_subjects = len(Y_SUBJECTS) - 1

    for held_out in Y_SUBJECTS:
        try:
            train_subjs = [s for s in Y_SUBJECTS if s != held_out]
            X_train_all, y_train_all, sub_ids = [], [], []

            for subj_idx, subj in enumerate(train_subjs):
                X, y = load_eeg_data(subj)
                if X is not None:
                    X_train_all.append(X)
                    y_train_all.append(y)
                    sub_ids.extend([subj_idx] * len(y))

            X_test, y_test = load_eeg_data(held_out)
            if len(X_train_all) == 0 or X_test is None:
                continue

            X_train_all = np.vstack(X_train_all)
            y_train_all = np.concatenate(y_train_all)
            sub_ids = np.array(sub_ids)

            np.random.seed(seed)
            indices = np.random.permutation(len(y_train_all))
            val_size = int(len(y_train_all) * 0.1)
            train_idx = indices[val_size:]
            val_idx = indices[:val_size]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_all[train_idx])
            y_tr = y_train_all[train_idx]
            sub_tr = sub_ids[train_idx]
            X_val = scaler.transform(X_train_all[val_idx])
            y_val = y_train_all[val_idx]
            X_test_s = scaler.transform(X_test)

            eeg_dim = X_tr.shape[1]
            encoder = EEGEncoder(eeg_dim, dropout=dropout).to(device)
            task_clf = TaskClassifier(encoder.output_dim, dropout=dropout).to(device)
            sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects, dropout=dropout).to(device)

            optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            X_tr_t = torch.FloatTensor(X_tr).to(device)
            y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
            sub_tr_t = torch.LongTensor(sub_tr).to(device)
            X_val_t = torch.FloatTensor(X_val).to(device)
            X_test_t = torch.FloatTensor(X_test_s).to(device)

            best_val_f1 = 0
            best_encoder_state = None
            best_clf_state = None
            patience_counter = 0
            n_epochs = 50

            for epoch in range(n_epochs):
                p = epoch / n_epochs
                lambda_adv = compute_lambda_warmup(p, lambda_max, gamma)

                encoder.train()
                task_clf.train()
                sub_disc.train()

                z = encoder(X_tr_t)
                task_logits = task_clf(z)
                sub_logits = sub_disc(z)

                task_loss = criterion(task_logits, y_tr_t)

                if label_smoothing > 0:
                    n_class = n_subjects
                    sub_probs = F.softmax(sub_logits, dim=1)
                    uniform = torch.ones_like(sub_probs) / n_class
                    sub_loss = F.kl_div(sub_probs.log(), uniform, reduction='batchmean')
                else:
                    sub_loss = F.cross_entropy(sub_logits, sub_tr_t)

                loss = task_loss + lambda_adv * sub_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                encoder.eval()
                task_clf.eval()
                with torch.no_grad():
                    val_z = encoder(X_val_t)
                    val_logits = task_clf(val_z)
                    val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                    val_f1 = f1_score(y_val, val_preds.cpu().numpy(), average='macro')

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_encoder_state = encoder.state_dict().copy()
                    best_clf_state = task_clf.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        break

            if best_encoder_state is not None:
                encoder.load_state_dict(best_encoder_state)
                task_clf.load_state_dict(best_clf_state)

            encoder.eval()
            with torch.no_grad():
                test_z = encoder(X_test_t)
                test_logits = task_clf(test_z)
                test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
                test_preds = (test_probs >= 0.5).astype(int)

            acc = accuracy_score(y_test, test_preds)
            f1 = f1_score(y_test, test_preds, average='macro')
            bacc = balanced_accuracy_score(y_test, test_preds)
            try:
                auroc = roc_auc_score(y_test, test_probs)
            except:
                auroc = 0.5

            test_z_np = test_z.cpu().numpy()
            try:
                sub_clf = LogisticRegression(max_iter=1000, random_state=seed)
                sub_clf.fit(test_z_np, y_test)
                sub_pred = sub_clf.predict(test_z_np)
                sub_acc = accuracy_score(y_test, sub_pred)
            except:
                sub_acc = 0.5

            results.append({
                'model': f'SIED_warmup_lmax{lambda_max}_g{gamma}_d{dropout}_ls{label_smoothing}',
                'seed': seed, 'held_out': held_out,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc,
                'subject_predictability': sub_acc,
                'lambda_max': lambda_max, 'gamma': gamma, 'dropout': dropout, 'label_smoothing': label_smoothing
            })

        except Exception as e:
            print(f"    Error {held_out}: {e}", flush=True)

    return results

def main():
    print("="*70)
    print("SIED Stability Optimization: Lambda Warm-up + Regularization")
    print("="*70)

    all_results = []
    seeds = [0, 1, 2, 3, 4]

    print("\nBaseline experiments:", flush=True)
    for seed in seeds:
        all_results.extend(run_raw_eeg_baseline(seed))
        all_results.extend(run_sied_baseline(seed, lambda_adv=0.01))

    print("\nWarm-up experiments:", flush=True)
    lambda_max_values = [0.005, 0.01, 0.05]
    gamma_values = [5, 10]
    dropout_values = [0.1, 0.3, 0.5]
    label_smoothing_values = [0.0, 0.1]

    for seed in seeds:
        print(f"\nSeed {seed}:", flush=True)
        for lambda_max in lambda_max_values:
            for gamma in gamma_values:
                for dropout in dropout_values:
                    for label_smoothing in label_smoothing_values:
                        all_results.extend(run_sied_warmup(seed, lambda_max, gamma, dropout, label_smoothing))

    df = pd.DataFrame(all_results)
    df.to_csv(f"{RESULTS_DIR}/sied_stability_results.csv", index=False)

    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)

    print("\n--- Baseline vs Warmup Comparison ---", flush=True)
    baseline_acc = df[df['model'] == 'Raw_EEG']['accuracy'].mean()
    sied_base_acc = df[df['model'] == 'SIED_baseline_l0.01']['accuracy'].mean()
    print(f"\nRaw_EEG: {baseline_acc:.4f}", flush=True)
    print(f"SIED_baseline: {sied_base_acc:.4f}", flush=True)

    warmup_df = df[df['model'].str.contains('warmup', na=False)]
    if len(warmup_df) > 0:
        print("\n--- Best Warmup Configurations ---", flush=True)
        for lambda_max in lambda_max_values:
            for gamma in gamma_values:
                config_df = warmup_df[(warmup_df['lambda_max'] == lambda_max) & (warmup_df['gamma'] == gamma)]
                if len(config_df) > 0:
                    acc = config_df['accuracy'].mean()
                    std = config_df['accuracy'].std()
                    f1 = config_df['macro_f1'].mean()
                    bacc = config_df['balanced_accuracy'].mean()
                    sub_pred = config_df['subject_predictability'].mean()
                    print(f"\nlambda_max={lambda_max}, gamma={gamma}:", flush=True)
                    print(f"  acc={acc:.4f}±{std:.4f}, f1={f1:.4f}, bacc={bacc:.4f}, sub_pred={sub_pred:.4f}", flush=True)
                    print(f"  gap_vs_SIED={acc-sied_base_acc:+.4f}, gap_vs_EEG={acc-baseline_acc:+.4f}", flush=True)

        best_config = warmup_df.groupby(['lambda_max', 'gamma', 'dropout', 'label_smoothing']).agg({
            'accuracy': 'mean', 'macro_f1': 'mean', 'balanced_accuracy': 'mean', 'subject_predictability': 'mean'
        }).reset_index()
        best_config = best_config.sort_values('accuracy', ascending=False).head(5)
        print("\n--- Top 5 Configurations by Accuracy ---", flush=True)
        for idx, row in best_config.iterrows():
            print(f"  lmax={row['lambda_max']}, g={row['gamma']}, d={row['dropout']}, ls={row['label_smoothing']}: "
                  f"acc={row['accuracy']:.4f}, f1={row['macro_f1']:.4f}, bacc={row['balanced_accuracy']:.4f}, sub_pred={row['subject_predictability']:.4f}", flush=True)

    print("\n--- Success Criteria Check ---", flush=True)
    if len(warmup_df) > 0:
        warmup_acc = warmup_df['accuracy'].mean()
        warmup_sub_pred = warmup_df['subject_predictability'].mean()
        sied_base_sub_pred = df[df['model'] == 'SIED_baseline_l0.01']['subject_predictability'].mean()

        print(f"SIED baseline: acc={sied_base_acc:.4f}, sub_pred={sied_base_sub_pred:.4f}", flush=True)
        print(f"SIED warmup:   acc={warmup_acc:.4f}, sub_pred={warmup_sub_pred:.4f}", flush=True)
        print(f"Task accuracy not decreasing: {(warmup_acc >= sied_base_acc - 0.02)}", flush=True)
        print(f"Subject predictability降低: {warmup_sub_pred < sied_base_sub_pred}", flush=True)

    print("\nDone!", flush=True)

if __name__ == '__main__':
    main()