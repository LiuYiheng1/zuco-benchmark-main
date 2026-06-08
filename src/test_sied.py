"""
Minimal SIED + SupCon Test
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

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return x  # No reversal for now

class EEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
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

class SubjectDiscriminator(nn.Module):
    def __init__(self, input_dim, n_subjects):
        super().__init__()
        self.net = nn.Sequential(
            GradientReversalLayer(1.0),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
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

def supervised_contrastive_loss(z, y, temperature=0.1):
    device = z.device
    batch_size = z.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=device)

    y = y.view(-1)
    z = F.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T) / temperature

    mask = torch.eq(y.unsqueeze(0), y.unsqueeze(1)).float().to(device)
    mask_no_self = mask - torch.eye(batch_size).to(device)
    pos_mask = mask_no_self
    neg_mask = 1.0 - mask

    pos_sim = (sim_matrix * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    neg_sim = (sim_matrix * neg_mask).sum(dim=1) / (neg_mask.sum(dim=1) + 1e-8)

    loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8))
    return loss.mean()

def run_model(seed, model_name, lambda_adv=1.0, beta=0.0, temperature=0.1):
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
        val_idx = indices[:val_size]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_all[train_idx])
        y_tr = y_train_all[train_idx]
        sub_tr = sub_ids[train_idx]
        X_val = scaler.transform(X_train_all[val_idx])
        y_val = y_train_all[val_idx]
        X_test_s = scaler.transform(X_test)

        if model_name == 'Raw_EEG':
            clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
            clf.fit(X_tr, y_tr)
            test_probs = clf.decision_function(X_test_s)
            test_preds = clf.predict(X_test_s)
            acc = accuracy_score(y_test, test_preds)
            f1 = f1_score(y_test, test_preds, average='macro')
            bacc = balanced_accuracy_score(y_test, test_preds)
            try:
                auroc = roc_auc_score(y_test, test_probs)
            except:
                auroc = 0.5
        else:
            eeg_dim = X_tr.shape[1]
            encoder = EEGEncoder(eeg_dim).to(device)
            task_clf = TaskClassifier(encoder.output_dim).to(device)
            sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

            optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            X_tr_t = torch.FloatTensor(X_tr).to(device)
            y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1).to(device)
            y_tr_labels = torch.LongTensor(y_tr).to(device)
            sub_tr_t = torch.LongTensor(sub_tr).to(device)
            X_val_t = torch.FloatTensor(X_val).to(device)
            X_test_t = torch.FloatTensor(X_test_s).to(device)

            best_val_f1 = 0
            best_encoder_state = None
            best_clf_state = None
            patience_counter = 0

            for epoch in range(30):
                encoder.train()
                task_clf.train()
                sub_disc.train()

                z = encoder(X_tr_t)
                task_logits = task_clf(z)
                sub_logits = sub_disc(z)

                task_loss = criterion(task_logits, y_tr_t)
                sub_loss = F.cross_entropy(sub_logits, sub_tr_t)

                supcon = supervised_contrastive_loss(z, y_tr_labels, temperature) if beta > 0 else torch.tensor(0.0)

                loss = task_loss + lambda_adv * sub_loss + beta * supcon

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
                    if patience_counter >= 5:
                        break

            if best_encoder_state is not None:
                encoder.load_state_dict(best_encoder_state)
                task_clf.load_state_dict(best_clf_state)

            encoder.eval()
            task_clf.eval()
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

        results.append({
            'model': model_name, 'seed': seed, 'held_out': held_out,
            'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
        })

    return results

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    all_results = []

    print("Running Raw_EEG...", flush=True)
    for seed in [0]:
        all_results.extend(run_model(seed, 'Raw_EEG'))

    print("Running SIED...", flush=True)
    for seed in [0]:
        all_results.extend(run_model(seed, 'SIED', lambda_adv=1.0))

    print("Running SIED_TaskSupCon...", flush=True)
    for seed in [0]:
        all_results.extend(run_model(seed, 'SIED_TaskSupCon_b0.1_t0.1', lambda_adv=1.0, beta=0.1, temperature=0.1))

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "sied_supcon_results.csv")
    df.to_csv(output_path, index=False)

    print("\n" + "="*60, flush=True)
    print("Results Summary", flush=True)
    print("="*60, flush=True)

    for model in df['model'].unique():
        data = df[df['model'] == model]
        acc = data['accuracy'].mean()
        std = data['accuracy'].std()
        f1 = data['macro_f1'].mean()
        print(f"  {model}: acc={acc:.4f}±{std:.4f}, f1={f1:.4f}", flush=True)

    print(f"\nResults saved to {output_path}", flush=True)

if __name__ == '__main__':
    main()