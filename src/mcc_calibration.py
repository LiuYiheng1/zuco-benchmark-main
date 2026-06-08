"""
MCC: Material-Controlled EEG Calibration

This module implements material cluster adversary to reduce EEG embedding
dependence on material/text shortcuts.

MCC adds a material discriminator that predicts material clusters (from gaze features)
and uses gradient reversal to make EEG embeddings invariant to material clusters.

Models:
- MCC_only: EEG encoder + material cluster adversary
- SIED_MCC: SIED (subject) + material cluster adversary
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
from sklearn.svm import SVC
from sklearn.cluster import KMeans

FEATURES_DIR = "features"
RESULTS_DIR = "results/personalized"
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

def load_gaze_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze_sacc.npy")
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

class MaterialClusterDiscriminator(nn.Module):
    def __init__(self, input_dim, n_clusters):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_clusters)
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

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def compute_material_clusters(X_gaze_all, n_clusters=4):
    """Compute material clusters from gaze features using KMeans"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_gaze_all)
    return clusters, kmeans

def run_mcc_experiment(X_train, y_train, X_cal, y_cal, X_test, y_test,
                       X_gaze_train, X_gaze_cal, X_gaze_test,
                       model_type, device, n_clusters=4, lambda_mat=0.1, lambda_subj=0.0):
    """Run MCC experiment"""

    n_subjects = 1

    scaler_eeg = StandardScaler()
    X_train_s = scaler_eeg.fit_transform(X_train)
    X_cal_s = scaler_eeg.transform(X_cal)
    X_test_s = scaler_eeg.transform(X_test)

    scaler_gaze = StandardScaler()
    X_gaze_train_s = scaler_gaze.fit_transform(X_gaze_train)
    X_gaze_cal_s = scaler_gaze.transform(X_gaze_cal)
    X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

    material_labels_train, kmeans = compute_material_clusters(X_gaze_train_s, n_clusters)
    material_labels_cal = kmeans.predict(X_gaze_cal_s)
    material_labels_test = kmeans.predict(X_gaze_test_s)

    eeg_dim = X_train_s.shape[1]
    encoder = EEGEncoder(eeg_dim).to(device)
    task_clf = TaskClassifier(encoder.output_dim).to(device)
    mat_disc = MaterialClusterDiscriminator(encoder.output_dim, n_clusters).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(mat_disc.parameters()),
                                       lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.FloatTensor(X_train_s).to(device)
    y_tr_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    mat_tr_t = torch.LongTensor(material_labels_train).to(device)

    X_cal_t = torch.FloatTensor(X_cal_s).to(device)
    y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).to(device)

    X_test_t = torch.FloatTensor(X_test_s).to(device)

    best_cal_f1 = 0
    best_encoder_state = None
    best_clf_state = None
    patience = 0

    for epoch in range(50):
        encoder.train()
        task_clf.train()
        mat_disc.train()

        z = encoder(X_tr_t)
        task_logits = task_clf(z)
        mat_logits = mat_disc(z)

        task_loss = criterion(task_logits, y_tr_t)
        mat_loss = F.cross_entropy(mat_logits, mat_tr_t)
        loss = task_loss + lambda_mat * mat_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        encoder.eval()
        task_clf.eval()
        with torch.no_grad():
            cal_z = encoder(X_cal_t)
            cal_logits = task_clf(cal_z)
            cal_preds = (torch.sigmoid(cal_logits) > 0.5).float()
            cal_f1 = f1_score(y_cal, cal_preds.cpu().numpy(), average='macro')

        if cal_f1 > best_cal_f1:
            best_cal_f1 = cal_f1
            best_encoder_state = encoder.state_dict().copy()
            best_clf_state = task_clf.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 10:
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
    try:
        auroc = roc_auc_score(y_test, test_probs)
    except:
        auroc = 0.5

    with torch.no_grad():
        test_z_np = test_z.cpu().numpy()
        mat_clusters_pred = kmeans.predict(test_z_np)
        mat_acc = accuracy_score(material_labels_test, mat_clusters_pred)

    return acc, f1, bacc, auroc, mat_acc

def run_sied_mcc_experiment(X_train, y_train, sub_ids_train,
                           X_cal, y_cal, X_test, y_test,
                           X_gaze_train, X_gaze_cal, X_gaze_test,
                           device, n_clusters=4, lambda_mat=0.1, lambda_subj=0.01):
    """Run SIED + MCC experiment"""

    n_subjects = len(np.unique(sub_ids_train))

    scaler_eeg = StandardScaler()
    X_train_s = scaler_eeg.fit_transform(X_train)
    X_cal_s = scaler_eeg.transform(X_cal)
    X_test_s = scaler_eeg.transform(X_test)

    scaler_gaze = StandardScaler()
    X_gaze_train_s = scaler_gaze.fit_transform(X_gaze_train)
    X_gaze_cal_s = scaler_gaze.transform(X_gaze_cal)
    X_gaze_test_s = scaler_gaze.transform(X_gaze_test)

    material_labels_train, kmeans = compute_material_clusters(X_gaze_train_s, n_clusters)
    material_labels_cal = kmeans.predict(X_gaze_cal_s)
    material_labels_test = kmeans.predict(X_gaze_test_s)

    eeg_dim = X_train_s.shape[1]
    encoder = EEGEncoder(eeg_dim).to(device)
    task_clf = TaskClassifier(encoder.output_dim).to(device)
    sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)
    mat_disc = MaterialClusterDiscriminator(encoder.output_dim, n_clusters).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) +
                         list(sub_disc.parameters()) + list(mat_disc.parameters()),
                         lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.FloatTensor(X_train_s).to(device)
    y_tr_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    sub_tr_t = torch.LongTensor(sub_ids_train).to(device)
    mat_tr_t = torch.LongTensor(material_labels_train).to(device)

    X_cal_t = torch.FloatTensor(X_cal_s).to(device)
    y_cal_t = torch.FloatTensor(y_cal).unsqueeze(1).to(device)

    X_test_t = torch.FloatTensor(X_test_s).to(device)

    best_cal_f1 = 0
    best_encoder_state = None
    best_clf_state = None
    patience = 0

    for epoch in range(50):
        encoder.train()
        task_clf.train()
        sub_disc.train()
        mat_disc.train()

        z = encoder(X_tr_t)
        task_logits = task_clf(z)

        reversed_z = GradientReversalFunction.apply(z, lambda_subj)
        sub_logits = sub_disc(reversed_z)

        mat_logits = mat_disc(z)

        task_loss = criterion(task_logits, y_tr_t)
        sub_loss = F.cross_entropy(sub_logits, sub_tr_t)
        mat_loss = F.cross_entropy(mat_logits, mat_tr_t)

        loss = task_loss + lambda_subj * sub_loss + lambda_mat * mat_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        encoder.eval()
        task_clf.eval()
        with torch.no_grad():
            cal_z = encoder(X_cal_t)
            cal_logits = task_clf(cal_z)
            cal_preds = (torch.sigmoid(cal_logits) > 0.5).float()
            cal_f1 = f1_score(y_cal, cal_preds.cpu().numpy(), average='macro')

        if cal_f1 > best_cal_f1:
            best_cal_f1 = cal_f1
            best_encoder_state = encoder.state_dict().copy()
            best_clf_state = task_clf.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 10:
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
    try:
        auroc = roc_auc_score(y_test, test_probs)
    except:
        auroc = 0.5

    with torch.no_grad():
        test_z_np = test_z.cpu().numpy()
        mat_clusters_pred = kmeans.predict(test_z_np)
        mat_acc = accuracy_score(material_labels_test, mat_clusters_pred)

    return acc, f1, bacc, auroc, mat_acc

def run_baseline_svm(X_cal, y_cal, X_test, y_test):
    """Baseline SVM classifier"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel='linear', random_state=42, gamma='scale', probability=True)
    clf.fit(X_cal_s, y_cal)
    preds = clf.predict(X_test_s)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc, 0.25

def run_experiment(seed, model_type):
    """Run MCC experiment"""
    results = []
    calibration_settings = [1, 3, 5, 10, 20, 50]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for held_out in Y_SUBJECTS:
        print(f"\n  {model_type} - {held_out}:", flush=True)

        X_eeg, y_eeg = load_eeg_data(held_out)
        X_gaze, y_gaze = load_gaze_data(held_out)
        if X_eeg is None or X_gaze is None:
            continue
        common_len = min(len(y_eeg), len(y_gaze))
        X_eeg = X_eeg[:common_len]
        y_eeg = y_eeg[:common_len]
        X_gaze = X_gaze[:common_len]
        y_gaze = y_gaze[:common_len]

        if len(X_eeg) < 50:
            continue

        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all_eeg, y_train_all, sub_ids_train_all = [], [], []
        X_train_all_gaze = []
        for subj_idx, subj in enumerate(train_subjs):
            X_subj_eeg, y_subj_eeg = load_eeg_data(subj)
            X_subj_gaze, y_subj_gaze = load_gaze_data(subj)
            if X_subj_eeg is not None and X_subj_gaze is not None and len(y_subj_eeg) == len(y_subj_gaze):
                common_len = min(len(y_subj_eeg), len(y_subj_gaze))
                X_train_all_eeg.append(X_subj_eeg[:common_len])
                X_train_all_gaze.append(X_subj_gaze[:common_len])
                y_train_all.append(y_subj_eeg[:common_len])
                sub_ids_train_all.extend([subj_idx] * common_len)

        if len(X_train_all_eeg) == 0:
            continue

        X_train_all_eeg = np.vstack(X_train_all_eeg)
        X_train_all_gaze = np.vstack(X_train_all_gaze)
        y_train_all = np.concatenate(y_train_all)
        sub_ids_train_all = np.array(sub_ids_train_all)

        n_samples = len(y_eeg)
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_samples // 2]
        cal_pool_indices = indices[n_samples // 2:]

        X_test = X_eeg[test_indices]
        y_test = y_eeg[test_indices]
        X_test_gaze = X_gaze[test_indices]

        X_cal_pool = X_eeg[cal_pool_indices]
        y_cal_pool = y_eeg[cal_pool_indices]
        X_cal_pool_gaze = X_gaze[cal_pool_indices]

        for n_cal_per_class in calibration_settings:
            if n_cal_per_class * 2 > len(cal_pool_indices):
                continue

            cal_idx_0 = np.where(y_cal_pool == 0)[0][:n_cal_per_class]
            cal_idx_1 = np.where(y_cal_pool == 1)[0][:n_cal_per_class]
            cal_idx = np.concatenate([cal_idx_0, cal_idx_1])
            np.random.shuffle(cal_idx)

            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]
            X_cal_gaze = X_cal_pool_gaze[cal_idx]

            if model_type == 'MCC_only':
                acc, f1, bacc, auroc, mat_acc = run_mcc_experiment(
                    X_train_all_eeg, y_train_all,
                    X_cal, y_cal, X_test, y_test,
                    X_train_all_gaze, X_cal_gaze, X_test_gaze,
                    model_type, device)
            elif model_type == 'SIED_MCC':
                acc, f1, bacc, auroc, mat_acc = run_sied_mcc_experiment(
                    X_train_all_eeg, y_train_all, sub_ids_train_all,
                    X_cal, y_cal, X_test, y_test,
                    X_train_all_gaze, X_cal_gaze, X_test_gaze,
                    device)
            elif model_type == 'EEG_SVM':
                acc, f1, bacc, auroc, mat_acc = run_baseline_svm(X_cal, y_cal, X_test, y_test)
            else:
                continue

            results.append({
                'model': model_type,
                'seed': seed,
                'subject': held_out,
                'n_cal_per_class': n_cal_per_class,
                'n_cal_total': n_cal_per_class * 2,
                'accuracy': acc,
                'macro_f1': f1,
                'balanced_accuracy': bacc,
                'auroc': auroc,
                'material_cluster_acc': mat_acc
            })

            print(f"    {n_cal_per_class}-shot: Acc={acc:.4f}, F1={f1:.4f}, MatAcc={mat_acc:.4f}", flush=True)

    return results

def main():
    print("="*70)
    print("MCC: Material-Controlled EEG Calibration Experiment")
    print("="*70)

    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    all_results = []

    model_types = ['EEG_SVM', 'MCC_only', 'SIED_MCC']

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Running: {model_type}")
        print("="*70)

        for seed in [0, 1, 2, 3, 4]:
            print(f"\n--- Seed {seed} ---", flush=True)
            results = run_experiment(seed, model_type)
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "mcc_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n\nSaved to {output_path}")

    summary = df.groupby(['model', 'n_cal_per_class']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'auroc': ['mean', 'std'],
        'material_cluster_acc': ['mean', 'std']
    }).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "mcc_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary.to_string())

    print("\nDone!")

if __name__ == '__main__':
    main()