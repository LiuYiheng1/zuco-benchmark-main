"""Zero-shot Cross-User SIED Experiments"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

FEATURES_DIR = "features"
GAZE_DIR = "gaze_features"
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

def load_gaze_data(subject):
    path = os.path.join(GAZE_DIR, f"{subject}_gaze_features.npy")
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
        if isinstance(values, np.ndarray) and len(values) > 0:
            features = np.array(values, dtype=np.float64)
            X.append(features)
            y.append(label)
    if len(X) == 0:
        return None, None
    return np.array(X), np.array(y)

def run_raw_eeg(seed):
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

            clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
            clf.fit(X_tr, y_train_all[train_idx])
            y_pred = clf.predict(X_test_s)
            probs = clf.decision_function(X_test_s)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            bacc = balanced_accuracy_score(y_test, y_pred)
            try:
                auroc = roc_auc_score(y_test, probs)
            except:
                auroc = 0.5
            cm = confusion_matrix(y_test, y_pred)

            results.append({
                'model': 'Raw_EEG', 'seed': seed, 'held_out': held_out,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc,
                'cm_0_0': cm[0, 0] if cm.shape == (2, 2) else 0,
                'cm_0_1': cm[0, 1] if cm.shape == (2, 2) else 0,
                'cm_1_0': cm[1, 0] if cm.shape == (2, 2) else 0,
                'cm_1_1': cm[1, 1] if cm.shape == (2, 2) else 0
            })
        except Exception as e:
            print(f"    Error {held_out}: {e}", flush=True)

    return results

def run_sied(seed, lambda_adv=1.0):
    print(f"  SIED lambda={lambda_adv} seed={seed}", flush=True)
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
            cm = confusion_matrix(y_test, test_preds)

            results.append({
                'model': f'SIED', 'seed': seed, 'held_out': held_out,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc,
                'cm_0_0': cm[0, 0] if cm.shape == (2, 2) else 0,
                'cm_0_1': cm[0, 1] if cm.shape == (2, 2) else 0,
                'cm_1_0': cm[1, 0] if cm.shape == (2, 2) else 0,
                'cm_1_1': cm[1, 1] if cm.shape == (2, 2) else 0
            })

        except Exception as e:
            print(f"    Error {held_out}: {e}", flush=True)

    return results

def run_gaze_only(seed):
    print(f"  Gaze_only seed={seed}", flush=True)
    results = []

    for held_out in Y_SUBJECTS:
        try:
            train_subjs = [s for s in Y_SUBJECTS if s != held_out]
            X_train_all, y_train_all = [], []
            for subj in train_subjs:
                X, y = load_gaze_data(subj)
                if X is not None:
                    X_train_all.append(X)
                    y_train_all.append(y)

            X_test, y_test = load_gaze_data(held_out)
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
            probs = clf.decision_function(X_test_s)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            bacc = balanced_accuracy_score(y_test, y_pred)
            try:
                auroc = roc_auc_score(y_test, probs)
            except:
                auroc = 0.5
            cm = confusion_matrix(y_test, y_pred)

            results.append({
                'model': 'Gaze_only', 'seed': seed, 'held_out': held_out,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc,
                'cm_0_0': cm[0, 0] if cm.shape == (2, 2) else 0,
                'cm_0_1': cm[0, 1] if cm.shape == (2, 2) else 0,
                'cm_1_0': cm[1, 0] if cm.shape == (2, 2) else 0,
                'cm_1_1': cm[1, 1] if cm.shape == (2, 2) else 0
            })
        except Exception as e:
            print(f"    Error {held_out}: {e}", flush=True)

    return results

def run_eeg_gaze_baseline(seed):
    print(f"  EEG_Gaze_baseline seed={seed}", flush=True)
    results = []

    for held_out in Y_SUBJECTS:
        try:
            train_subjs = [s for s in Y_SUBJECTS if s != held_out]
            X_train_eeg, y_train_eeg = [], []
            X_train_gaze, y_train_gaze = [], []

            for subj in train_subjs:
                Xe, ye = load_eeg_data(subj)
                Xg, yg = load_gaze_data(subj)
                if Xe is not None and Xg is not None and len(Xe) == len(Xg):
                    X_train_eeg.append(Xe)
                    y_train_eeg.append(ye)
                    X_train_gaze.append(Xg)
                    y_train_gaze.append(yg)

            X_test_eeg, y_test = load_eeg_data(held_out)
            X_test_gaze, _ = load_gaze_data(held_out)
            if len(X_train_eeg) == 0 or X_test_eeg is None or X_test_gaze is None:
                continue

            X_train_eeg = np.vstack(X_train_eeg)
            X_train_gaze = np.vstack(X_train_gaze)
            y_train_eeg = np.concatenate(y_train_eeg)

            np.random.seed(seed)
            indices = np.random.permutation(len(y_train_eeg))
            val_size = int(len(y_train_eeg) * 0.1)
            train_idx = indices[val_size:]

            scaler_eeg = StandardScaler()
            X_tr_eeg = scaler_eeg.fit_transform(X_train_eeg[train_idx])
            X_test_eeg_s = scaler_eeg.transform(X_test_eeg)

            scaler_gaze = StandardScaler()
            X_tr_gaze = scaler_gaze.fit_transform(X_train_gaze[train_idx])
            X_test_gaze_s = scaler_gaze.transform(X_test_gaze)

            clf_eeg = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
            clf_eeg.fit(X_tr_eeg, y_train_eeg[train_idx])
            probs_eeg = clf_eeg.decision_function(X_test_eeg_s)

            clf_gaze = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
            clf_gaze.fit(X_tr_gaze, y_train_eeg[train_idx])
            probs_gaze = clf_gaze.decision_function(X_test_gaze_s)

            probs_avg = (probs_eeg + probs_gaze) / 2
            y_pred = (probs_avg >= 0).astype(int)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            bacc = balanced_accuracy_score(y_test, y_pred)
            try:
                auroc = roc_auc_score(y_test, probs_avg)
            except:
                auroc = 0.5
            cm = confusion_matrix(y_test, y_pred)

            results.append({
                'model': 'EEG_Gaze_baseline', 'seed': seed, 'held_out': held_out,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc,
                'cm_0_0': cm[0, 0] if cm.shape == (2, 2) else 0,
                'cm_0_1': cm[0, 1] if cm.shape == (2, 2) else 0,
                'cm_1_0': cm[1, 0] if cm.shape == (2, 2) else 0,
                'cm_1_1': cm[1, 1] if cm.shape == (2, 2) else 0
            })
        except Exception as e:
            print(f"    Error {held_out}: {e}", flush=True)

    return results

def main():
    print("="*60)
    print("Zero-shot Cross-User SIED Experiments")
    print("="*60)

    all_results = []
    seeds = [0, 1, 2, 3, 4]

    for seed in seeds:
        print(f"\nSeed {seed}:", flush=True)
        all_results.extend(run_raw_eeg(seed))
        all_results.extend(run_sied(seed, lambda_adv=1.0))
        all_results.extend(run_gaze_only(seed))
        all_results.extend(run_eeg_gaze_baseline(seed))

    df = pd.DataFrame(all_results)
    df.to_csv("results/final/zero_shot_loso_results.csv", index=False)

    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)

    for model in ['Raw_EEG', 'SIED', 'Gaze_only', 'EEG_Gaze_baseline']:
        data = df[df['model'] == model]
        if len(data) > 0:
            acc = data['accuracy'].mean()
            std = data['accuracy'].std()
            f1 = data['macro_f1'].mean()
            bacc = data['balanced_accuracy'].mean()
            print(f"  {model:20s}: acc={acc:.4f}±{std:.4f}, f1={f1:.4f}, bacc={bacc:.4f}")

    print("\nDone!")

if __name__ == '__main__':
    main()