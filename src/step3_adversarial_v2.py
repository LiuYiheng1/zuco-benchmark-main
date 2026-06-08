"""
EEG Adaptation - Adversarial with incremental save
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURES_DIR = "features"
RESULTS_DIR = "results/eeg_adaptation"
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

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class EEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
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
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_subjects)
        )
    def forward(self, x):
        return self.net(x)

def run_adversarial_for_subject(data, seed, lambda_adv, held_out, device, n_subjects):
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
        return None

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
        reversed_z = GradientReversalFunction.apply(z, lambda_adv)
        sub_logits = sub_disc(reversed_z)

        task_loss = criterion(task_logits, y_tr_t)
        sub_loss = nn.CrossEntropyLoss()(sub_logits, sub_tr_t)
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
    task_clf.eval()
    with torch.no_grad():
        test_z = encoder(X_test_t)
        test_logits = task_clf(test_z)
        test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
        test_preds = (test_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds, average='macro')
    bacc = balanced_accuracy_score(y_test, test_preds)
    prec, rec, _, _ = precision_recall_fscore_support(y_test, test_preds, average='macro', warn_for=[])
    cm = confusion_matrix(y_test, test_preds)

    try:
        auroc = roc_auc_score(y_test, test_probs)
    except:
        auroc = 0.5

    return {
        'model': f'EEG_Adversarial_lamb{lambda_adv}',
        'seed': seed,
        'held_out': held_out,
        'accuracy': acc,
        'macro_f1': f1,
        'balanced_accuracy': bacc,
        'precision_macro': prec,
        'recall_macro': rec,
        'auroc': auroc,
        'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_subjects = len(Y_SUBJECTS) - 1

    all_results = []
    for seed in [0, 1, 2, 3, 4]:
        for lambda_adv in [0.01, 0.05, 0.1]:
            model_name = f'EEG_Adversarial_lamb{lambda_adv}'
            print(f"Adversarial lambda={lambda_adv} (seed={seed})...", flush=True)

            for held_out in Y_SUBJECTS:
                result = run_adversarial_for_subject(None, seed, lambda_adv, held_out, device, n_subjects)
                if result:
                    all_results.append(result)
                    print(f"  {held_out}: {result['accuracy']:.4f}", flush=True)

                if len(all_results) % 10 == 0:
                    df = pd.DataFrame(all_results)
                    df.to_csv(os.path.join(RESULTS_DIR, "adversarial_partial.csv"), index=False)

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "adversarial_5seeds.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(df.groupby('model')['accuracy'].agg(['mean', 'std']))

if __name__ == '__main__':
    main()