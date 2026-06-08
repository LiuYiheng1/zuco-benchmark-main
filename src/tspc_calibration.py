"""
TSPC: Task-Set Personalized Prototype Calibration

This module implements personalized prototype-based calibration for EEG classification.

Modes:
- TSPC_proto_only: Direct prototype computation on scaled EEG features
- TSPC_pretrained_encoder: Pretrained encoder for embedding + prototypes
- TSPC_SIED_encoder: SIED adversarial encoder + personalized prototypes

The key innovation is computing class prototypes from calibration samples
and classifying test samples based on distance to prototypes.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

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

def compute_prototypes(embeddings, labels):
    """Compute class prototypes (mean embedding per class)"""
    class_0_mask = labels == 0
    class_1_mask = labels == 1

    prototype_0 = np.mean(embeddings[class_0_mask], axis=0) if np.any(class_0_mask) else None
    prototype_1 = np.mean(embeddings[class_1_mask], axis=0) if np.any(class_1_mask) else None

    return prototype_0, prototype_1

def classify_by_prototype(test_embedding, proto_0, proto_1, metric='euclidean'):
    """Classify test sample based on distance to prototypes"""
    if proto_0 is None or proto_1 is None:
        return 0, [0.5, 0.5]

    if metric == 'euclidean':
        dist_0 = np.linalg.norm(test_embedding - proto_0)
        dist_1 = np.linalg.norm(test_embedding - proto_1)
    elif metric == 'cosine':
        cos_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        dist_0 = 1 - cos_sim(test_embedding, proto_0)
        dist_1 = 1 - cos_sim(test_embedding, proto_1)
    else:
        dist_0 = np.linalg.norm(test_embedding - proto_0)
        dist_1 = np.linalg.norm(test_embedding - proto_1)

    logit_0 = -dist_0
    logit_1 = -dist_1

    exp_0 = np.exp(logit_0 - np.max([logit_0, logit_1]))
    exp_1 = np.exp(logit_1 - np.max([logit_0, logit_1]))
    prob_0 = exp_0 / (exp_0 + exp_1)
    prob_1 = exp_1 / (exp_0 + exp_1)

    pred = 0 if dist_0 < dist_1 else 1
    return pred, [prob_0, prob_1]

def train_sied_encoder(X_train, y_train, sub_ids_train, n_subjects, device, lambda_adv=0.01, epochs=30):
    """Train SIED adversarial encoder"""
    eeg_dim = X_train.shape[1]
    encoder = EEGEncoder(eeg_dim).to(device)
    sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.FloatTensor(X_train).to(device)
    y_tr_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    sub_tr_t = torch.LongTensor(sub_ids_train).to(device)

    for epoch in range(epochs):
        encoder.train()
        sub_disc.train()

        z = encoder(X_tr_t)
        reversed_z = GradientReversalFunction.apply(z, lambda_adv)
        sub_logits = sub_disc(reversed_z)

        sub_loss = F.cross_entropy(sub_logits, sub_tr_t)

        optimizer.zero_grad()
        sub_loss.backward()
        optimizer.step()

    encoder.eval()
    return encoder

def run_tspc_proto_only(X_cal, y_cal, X_test, y_test):
    """TSPC_proto_only: Direct prototype on scaled EEG features"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    proto_0, proto_1 = compute_prototypes(X_cal_s, y_cal)

    preds, probs = [], []
    for i in range(len(X_test_s)):
        pred, prob = classify_by_prototype(X_test_s[i], proto_0, proto_1)
        preds.append(pred)
        probs.append(prob[1])

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_tspc_pretrained_encoder(X_train_all, y_train_all, sub_ids_train_all,
                                X_cal, y_cal, X_test, y_test,
                                device, hidden_dim=64, epochs=20):
    """TSPC with pretrained encoder"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_all)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    eeg_dim = X_train_s.shape[1]
    encoder = EEGEncoder(eeg_dim, hidden_dim=hidden_dim).to(device)
    clf_head = nn.Sequential(nn.Linear(hidden_dim, 1)).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(clf_head.parameters()), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.FloatTensor(X_train_s)
    y_tr_t = torch.FloatTensor(y_train_all).unsqueeze(1)

    for epoch in range(epochs):
        encoder.train()
        clf_head.train()
        z = encoder(X_tr_t)
        logits = clf_head(z)
        loss = criterion(logits, y_tr_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    encoder.eval()
    clf_head.eval()

    with torch.no_grad():
        z_cal = encoder(torch.FloatTensor(X_cal_s).to(device)).cpu().numpy()
        z_test = encoder(torch.FloatTensor(X_test_s).to(device)).cpu().numpy()

    proto_0, proto_1 = compute_prototypes(z_cal, y_cal)

    preds, probs = [], []
    for i in range(len(z_test)):
        pred, prob = classify_by_prototype(z_test[i], proto_0, proto_1)
        preds.append(pred)
        probs.append(prob[1])

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_tspc_sied_encoder(X_train_all, y_train_all, sub_ids_train_all,
                          X_cal, y_cal, X_test, y_test,
                          device, lambda_adv=0.01):
    """TSPC with SIED adversarial encoder"""
    n_subjects = len(np.unique(sub_ids_train_all))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_all)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    encoder = train_sied_encoder(X_train_s, y_train_all, sub_ids_train_all, n_subjects, device, lambda_adv)
    encoder.eval()

    with torch.no_grad():
        z_cal = encoder(torch.FloatTensor(X_cal_s).to(device)).cpu().numpy()
        z_test = encoder(torch.FloatTensor(X_test_s).to(device)).cpu().numpy()

    proto_0, proto_1 = compute_prototypes(z_cal, y_cal)

    preds, probs = [], []
    for i in range(len(z_test)):
        pred, prob = classify_by_prototype(z_test[i], proto_0, proto_1)
        preds.append(pred)
        probs.append(prob[1])

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

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

    return acc, f1, bacc, auroc

def run_baseline_mlp(X_cal, y_cal, X_test, y_test):
    """Baseline MLP classifier"""
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    clf.fit(X_cal_s, y_cal)
    preds = clf.predict(X_test_s)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_experiment(seed, model_type):
    """Run TSPC experiment for all subjects and calibration settings"""
    results = []
    calibration_settings = [1, 3, 5, 10, 20, 50]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for held_out in Y_SUBJECTS:
        print(f"\n  {model_type} - {held_out}:", flush=True)

        X_eeg, y_eeg = load_eeg_data(held_out)
        if X_eeg is None or len(X_eeg) < 50:
            continue

        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all, sub_ids_train_all = [], [], []
        for subj_idx, subj in enumerate(train_subjs):
            X_subj, y_subj = load_eeg_data(subj)
            if X_subj is not None:
                X_train_all.append(X_subj)
                y_train_all.append(y_subj)
                sub_ids_train_all.extend([subj_idx] * len(y_subj))

        if len(X_train_all) == 0:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)
        sub_ids_train_all = np.array(sub_ids_train_all)

        n_samples = len(y_eeg)
        n_class_0 = np.sum(y_eeg == 0)
        n_class_1 = np.sum(y_eeg == 1)
        min_class_size = min(n_class_0, n_class_1)

        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_samples // 2]
        cal_pool_indices = indices[n_samples // 2:]

        X_test = X_eeg[test_indices]
        y_test = y_eeg[test_indices]
        X_cal_pool = X_eeg[cal_pool_indices]
        y_cal_pool = y_eeg[cal_pool_indices]

        for n_cal_per_class in calibration_settings:
            if n_cal_per_class * 2 > len(cal_pool_indices):
                continue

            cal_idx_0 = np.where(y_cal_pool == 0)[0][:n_cal_per_class]
            cal_idx_1 = np.where(y_cal_pool == 1)[0][:n_cal_per_class]
            cal_idx = np.concatenate([cal_idx_0, cal_idx_1])
            np.random.shuffle(cal_idx)

            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if model_type == 'TSPC_proto_only':
                acc, f1, bacc, auroc = run_tspc_proto_only(X_cal, y_cal, X_test, y_test)
            elif model_type == 'TSPC_pretrained':
                acc, f1, bacc, auroc = run_tspc_pretrained_encoder(
                    X_train_all, y_train_all, sub_ids_train_all,
                    X_cal, y_cal, X_test, y_test, device)
            elif model_type == 'TSPC_SIED':
                acc, f1, bacc, auroc = run_tspc_sied_encoder(
                    X_train_all, y_train_all, sub_ids_train_all,
                    X_cal, y_cal, X_test, y_test, device)
            elif model_type == 'EEG_SVM':
                acc, f1, bacc, auroc = run_baseline_svm(X_cal, y_cal, X_test, y_test)
            elif model_type == 'EEG_MLP':
                acc, f1, bacc, auroc = run_baseline_mlp(X_cal, y_cal, X_test, y_test)
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
                'auroc': auroc
            })

            print(f"    {n_cal_per_class}-shot: Acc={acc:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}", flush=True)

    return results

def main():
    print("="*70)
    print("TSPC: Task-Set Personalized Prototype Calibration Experiment")
    print("="*70)

    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    all_results = []

    model_types = [
        'EEG_SVM',
        'EEG_MLP',
        'TSPC_proto_only',
        'TSPC_pretrained',
        'TSPC_SIED'
    ]

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Running: {model_type}")
        print("="*70)

        for seed in [0, 1, 2, 3, 4]:
            print(f"\n--- Seed {seed} ---", flush=True)
            results = run_experiment(seed, model_type)
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "tspc_calibration_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n\nSaved to {output_path}")

    summary = df.groupby(['model', 'n_cal_per_class']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'auroc': ['mean', 'std']
    }).reset_index()

    summary_path = os.path.join(RESULTS_DIR, "tspc_calibration_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(summary.to_string())

    print("\nDone!")

if __name__ == '__main__':
    main()