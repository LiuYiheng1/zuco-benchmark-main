"""
EEG Diagnostic 1: Within-subject EEG classification
Tests if EEG features are informative within a single subject
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

FEATURES_DIR = "features"
RESULTS_DIR = "results/eeg_diagnostics"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None, None
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
    return np.array(X), np.array(y), data

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_val, y_val, input_dim, epochs=30, lr=0.001, device='cpu'):
    model = SimpleMLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tr_t = torch.FloatTensor(X_train).to(device)
    y_tr_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    best_val_f1 = 0
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            val_f1 = f1_score(y_val_t.cpu().numpy(), val_preds.cpu().numpy(), average='macro')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model

def evaluate_mlp(model, X_test, y_test, device='cpu'):
    model.eval()
    X_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_t)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    prec, rec, _, _ = precision_recall_fscore_support(y_test, preds, average='macro', warn_for=[])
    cm = confusion_matrix(y_test, preds)

    return {
        'accuracy': acc,
        'macro_f1': f1,
        'balanced_accuracy': bacc,
        'precision_macro': prec,
        'recall_macro': rec,
        'tn': int(cm[0, 0]) if cm.shape[0] > 1 else 0,
        'fp': int(cm[0, 1]) if cm.shape[0] > 1 else 0,
        'fn': int(cm[1, 0]) if cm.shape[0] > 1 else 0,
        'tp': int(cm[1, 1]) if cm.shape[0] > 1 else 0
    }

def run_within_subject_experiment():
    print("="*70)
    print("EEG Diagnostic 1: Within-Subject Classification")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    results = []
    n_folds = 5

    for subject in Y_SUBJECTS:
        print(f"\n--- Subject: {subject} ---")

        X, y, _ = load_eeg_data(subject)
        if X is None or len(X) == 0:
            print(f"  No data for {subject}")
            continue

        print(f"  Samples: {len(y)}, NR={sum(y==1)}, TSR={sum(y==0)}")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            svm_clf = SGDClassifier(loss='hinge', random_state=0, max_iter=1000, tol=1e-3)
            svm_clf.fit(X_train_s, y_train)
            y_pred_svm = svm_clf.predict(X_test_s)

            svm_acc = accuracy_score(y_test, y_pred_svm)
            svm_f1 = f1_score(y_test, y_pred_svm, average='macro')
            svm_bacc = balanced_accuracy_score(y_test, y_pred_svm)
            svm_cm = confusion_matrix(y_test, y_pred_svm)

            results.append({
                'subject': subject,
                'model': 'SVM',
                'fold': fold_idx,
                'accuracy': svm_acc,
                'macro_f1': svm_f1,
                'balanced_accuracy': svm_bacc,
                'n_samples': len(y),
                'n_train': len(y_train),
                'n_test': len(y_test),
                'nr_ratio_train': sum(y_train == 1) / len(y_train),
                'tn': int(svm_cm[0, 0]) if svm_cm.shape[0] > 1 else 0,
                'fp': int(svm_cm[0, 1]) if svm_cm.shape[0] > 1 else 0,
                'fn': int(svm_cm[1, 0]) if svm_cm.shape[0] > 1 else 0,
                'tp': int(svm_cm[1, 1]) if svm_cm.shape[0] > 1 else 0
            })

            model = train_mlp(X_train_s, y_train, X_test_s, y_test, X_train_s.shape[1], device=device)
            mlp_metrics = evaluate_mlp(model, X_test_s, y_test, device=device)

            results.append({
                'subject': subject,
                'model': 'MLP',
                'fold': fold_idx,
                'accuracy': mlp_metrics['accuracy'],
                'macro_f1': mlp_metrics['macro_f1'],
                'balanced_accuracy': mlp_metrics['balanced_accuracy'],
                'n_samples': len(y),
                'n_train': len(y_train),
                'n_test': len(y_test),
                'nr_ratio_train': sum(y_train == 1) / len(y_train),
                'tn': mlp_metrics['tn'],
                'fp': mlp_metrics['fp'],
                'fn': mlp_metrics['fn'],
                'tp': mlp_metrics['tp']
            })

            print(f"  Fold {fold_idx}: SVM Acc={svm_acc:.4f}, MLP Acc={mlp_metrics['accuracy']:.4f}")

    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, "within_subject_eeg.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    summary = df.groupby(['subject', 'model']).agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    }).reset_index()
    summary.columns = ['subject', 'model', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std', 'bacc_mean', 'bacc_std']

    summary_path = os.path.join(RESULTS_DIR, "within_subject_eeg_summary.csv")
    summary.to_csv(summary_path, index=False)

    overall = df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std']
    })
    overall.columns = ['accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'macro_f1_std', 'bacc_mean', 'bacc_std']
    overall = overall.reset_index()

    print("\n" + "="*70)
    print("WITHIN-SUBJECT EEG RESULTS (Overall)")
    print("="*70)
    for _, row in overall.iterrows():
        print(f"{row['model']}: Acc={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}, F1={row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f}")

    print("\nPer-subject SVM results:")
    svm_summary = summary[summary['model'] == 'SVM'].sort_values('accuracy_mean', ascending=False)
    for _, row in svm_summary.iterrows():
        print(f"  {row['subject']}: Acc={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}")

    return df, summary, overall

if __name__ == '__main__':
    df, summary, overall = run_within_subject_experiment()