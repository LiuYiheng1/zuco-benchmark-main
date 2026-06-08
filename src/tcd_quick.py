"""
TCD Quick Test - 1 seed for quick validation
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
RESULTS_DIR = "results/domain_generalization"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

class GRLLayer(nn.Module):
    def forward(self, x):
        return x

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
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class SubjectAdvClassifier(nn.Module):
    def __init__(self, input_dim, n_subjects):
        super().__init__()
        self.net = nn.Sequential(
            GRLLayer(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_subjects)
        )

    def forward(self, x):
        return self.net(x)

class SubjectClassifier(nn.Module):
    def __init__(self, input_dim, n_subjects):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_subjects)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, concat_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
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

def correlation_loss(z1, z2):
    z1_centered = z1 - z1.mean(dim=1, keepdim=True)
    z2_centered = z2 - z2.mean(dim=1, keepdim=True)
    z1_norm = F.normalize(z1_centered, dim=1)
    z2_norm = F.normalize(z2_centered, dim=1)
    corr = (z1_norm * z2_norm).sum(dim=1).mean()
    return corr ** 2

def run_model(seed, model_name, use_conf_branch=True, use_corr=True, use_recon=True, use_adv=True,
              lambda_adv=0.01, lambda_conf=0.05, lambda_corr=0.05, lambda_recon=0.01):
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_subjects = len(Y_SUBJECTS) - 1

    for held_out in Y_SUBJECTS:
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

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_all[train_idx])
        X_test_s = scaler.transform(X_test)

        y_tr = y_train_all[train_idx]
        sub_tr = sub_ids[train_idx]

        eeg_dim = X_tr.shape[1]

        task_enc = EEGEncoder(eeg_dim).to(device)
        task_clf = TaskClassifier(task_enc.output_dim).to(device)
        sub_adv = SubjectAdvClassifier(task_enc.output_dim, n_subjects).to(device)

        conf_enc = None
        conf_clf = None
        decoder = None

        if use_conf_branch:
            conf_enc = EEGEncoder(eeg_dim).to(device)
            conf_clf = SubjectClassifier(conf_enc.output_dim, n_subjects).to(device)
            decoder = Decoder(task_enc.output_dim + conf_enc.output_dim, eeg_dim).to(device)

        params = list(task_enc.parameters()) + list(task_clf.parameters()) + list(sub_adv.parameters())
        if use_conf_branch:
            params += list(conf_enc.parameters()) + list(conf_clf.parameters()) + list(decoder.parameters())

        optimizer = optim.Adam(params, lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        X_tr_t = torch.FloatTensor(X_tr).to(device)
        y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
        sub_tr_t = torch.LongTensor(sub_tr).to(device)
        X_test_t = torch.FloatTensor(X_test_s).to(device)

        best_val_f1 = 0
        best_task_enc_state = None
        best_task_clf_state = None
        patience_counter = 0

        for epoch in range(30):
            task_enc.train()
            task_clf.train()
            sub_adv.train()
            if use_conf_branch:
                conf_enc.train()
                conf_clf.train()
                decoder.train()

            z_task = task_enc(X_tr_t)
            task_logits = task_clf(z_task)
            task_loss = criterion(task_logits, y_tr_t)

            loss = task_loss

            if use_adv:
                sub_logits = sub_adv(z_task)
                sub_loss = F.cross_entropy(sub_logits, sub_tr_t)
                loss = loss + lambda_adv * sub_loss

            if use_conf_branch:
                z_conf = conf_enc(X_tr_t)
                conf_logits = conf_clf(z_conf)
                conf_loss = F.cross_entropy(conf_logits, sub_tr_t)
                loss = loss + lambda_conf * conf_loss

                if use_corr:
                    corr = correlation_loss(z_task, z_conf)
                    loss = loss + lambda_corr * corr

                if use_recon:
                    recon = decoder(torch.cat([z_task, z_conf], dim=1))
                    recon_loss = F.mse_loss(recon, X_tr_t)
                    loss = loss + lambda_recon * recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            task_enc.eval()
            task_clf.eval()
            with torch.no_grad():
                val_z = task_enc(X_tr_t[:100])
                val_logits = task_clf(val_z)
                val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                val_f1 = f1_score(y_tr[:100], val_preds.cpu().numpy(), average='macro')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_task_enc_state = task_enc.state_dict().copy()
                best_task_clf_state = task_clf.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break

        if best_task_enc_state is not None:
            task_enc.load_state_dict(best_task_enc_state)
            task_clf.load_state_dict(best_task_clf_state)

        task_enc.eval()
        task_clf.eval()
        with torch.no_grad():
            test_z = task_enc(X_test_t)
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

        results.append({
            'model': model_name, 'seed': seed, 'held_out': held_out,
            'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
        })

    return results

def run_raw_eeg(seed):
    results = []
    for held_out in Y_SUBJECTS:
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

        clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
        clf.fit(X_tr, y_train_all[train_idx])
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bacc = balanced_accuracy_score(y_test, y_pred)
        try:
            auroc = roc_auc_score(y_test, clf.decision_function(X_test_s))
        except:
            auroc = 0.5

        results.append({
            'model': 'Raw_EEG', 'seed': seed, 'held_out': held_out,
            'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
        })

    return results

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    all_results = []
    seed = 0

    print("Running experiments (seed=0 only for quick test)...", flush=True)

    print("Raw_EEG...", flush=True)
    all_results.extend(run_raw_eeg(seed))

    print("SIED...", flush=True)
    all_results.extend(run_model(seed, 'SIED', use_conf_branch=False, use_adv=True, lambda_adv=0.01))

    print("TCD_no_conf_branch...", flush=True)
    all_results.extend(run_model(seed, 'TCD_no_conf_branch', use_conf_branch=False, use_adv=True, lambda_adv=0.01))

    print("TCD_no_corr...", flush=True)
    all_results.extend(run_model(seed, 'TCD_no_corr', use_conf_branch=True, use_corr=False, use_recon=True, use_adv=True,
                                lambda_adv=0.01, lambda_conf=0.05, lambda_corr=0.0, lambda_recon=0.01))

    print("TCD_no_recon...", flush=True)
    all_results.extend(run_model(seed, 'TCD_no_recon', use_conf_branch=True, use_corr=True, use_recon=False, use_adv=True,
                                lambda_adv=0.01, lambda_conf=0.05, lambda_corr=0.05, lambda_recon=0.0))

    print("TCD_no_adv...", flush=True)
    all_results.extend(run_model(seed, 'TCD_no_adv', use_conf_branch=True, use_corr=True, use_recon=True, use_adv=False,
                                lambda_adv=0.0, lambda_conf=0.05, lambda_corr=0.05, lambda_recon=0.01))

    print("TCD_full...", flush=True)
    all_results.extend(run_model(seed, 'TCD_full', use_conf_branch=True, use_corr=True, use_recon=True, use_adv=True,
                                lambda_adv=0.01, lambda_conf=0.05, lambda_corr=0.05, lambda_recon=0.01))

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS_DIR, "tcd_full_results.csv"), index=False)

    print("\n" + "="*60, flush=True)
    print("Results (seed=0 only)", flush=True)
    print("="*60, flush=True)

    for model in ['Raw_EEG', 'SIED', 'TCD_no_conf_branch', 'TCD_no_corr', 'TCD_no_recon', 'TCD_no_adv', 'TCD_full']:
        data = df[df['model'] == model]
        if len(data) > 0:
            acc = data['accuracy'].mean()
            std = data['accuracy'].std()
            f1 = data['macro_f1'].mean()
            bacc = data['balanced_accuracy'].mean()
            print(f"  {model:20s}: acc={acc:.4f}+-{std:.4f}, f1={f1:.4f}, bacc={bacc:.4f}", flush=True)

    sied_acc = df[df['model'] == 'SIED']['accuracy'].mean()
    tcd_acc = df[df['model'] == 'TCD_full']['accuracy'].mean()

    print(f"\nSIED: {sied_acc:.4f}", flush=True)
    print(f"TCD_full: {tcd_acc:.4f} (gap={tcd_acc-sied_acc:+.4f})", flush=True)
    print(f"Target: SIED + 1.5% = {sied_acc + 0.015:.4f}", flush=True)

    if tcd_acc >= sied_acc + 0.015:
        print("RESULT: TCD_full exceeds SIED by >= 1.5%!", flush=True)
    elif tcd_acc >= sied_acc + 0.005:
        print("RESULT: TCD_full exceeds SIED by < 1.5% (marginal)", flush=True)
    else:
        print("RESULT: TCD_full does NOT exceed SIED by 1.5%", flush=True)

if __name__ == '__main__':
    main()