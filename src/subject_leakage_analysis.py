"""
EEG Subject Leakage / Subject-Invariance Analysis
Tests whether adversarial training reduces subject identity predictability
while maintaining/improving task classification
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

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

def extract_embeddings(model_name, seed=0, lambda_adv=None):
    """Extract EEG embeddings using trained encoder"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []

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

        if model_name == 'Raw_EEG':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=128)
            train_emb = pca.fit_transform(X_tr)
            test_emb = pca.transform(X_test_s)
        else:
            n_subjects = len(Y_SUBJECTS) - 1
            eeg_dim = X_tr.shape[1]
            encoder = EEGEncoder(eeg_dim).to(device)
            task_clf = TaskClassifier(encoder.output_dim).to(device)
            sub_disc = SubjectDiscriminator(encoder.output_dim, n_subjects).to(device)

            optimizer = optim.Adam(list(encoder.parameters()) + list(task_clf.parameters()) + list(sub_disc.parameters()), lr=0.001, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            X_tr_t = torch.FloatTensor(X_tr).to(device)
            y_tr_t = torch.FloatTensor(y_train_all[train_idx]).unsqueeze(1).to(device)
            sub_tr_t = torch.LongTensor(sub_ids[train_idx]).to(device)

            for epoch in range(50):
                encoder.train()
                task_clf.train()
                sub_disc.train()

                z = encoder(X_tr_t)
                task_logits = task_clf(z)
                if lambda_adv:
                    reversed_z = GradientReversalFunction.apply(z, lambda_adv)
                else:
                    reversed_z = z
                sub_logits = sub_disc(reversed_z)

                task_loss = criterion(task_logits, y_tr_t)
                sub_loss = nn.CrossEntropyLoss()(sub_logits, sub_tr_t)
                if lambda_adv:
                    loss = task_loss + lambda_adv * sub_loss
                else:
                    loss = task_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            encoder.eval()
            with torch.no_grad():
                train_emb = encoder(torch.FloatTensor(X_tr).to(device)).cpu().numpy()
                test_emb = encoder(torch.FloatTensor(X_test_s).to(device)).cpu().numpy()

        results.append({
            'model': model_name,
            'seed': seed,
            'held_out': held_out,
            'train_embeddings': train_emb,
            'test_embeddings': test_emb,
            'train_labels': y_train_all[train_idx],
            'test_labels': y_test,
            'train_sub_ids': sub_ids[train_idx]
        })

    return results

def train_subject_classifier(embeddings, labels, sub_ids):
    """Train a subject classifier on embeddings"""
    clf = SGDClassifier(loss='hinge', random_state=0, max_iter=1000, tol=1e-3)
    clf.fit(embeddings, sub_ids)
    return clf

def train_task_classifier(embeddings, labels):
    """Train a task classifier on embeddings"""
    clf = SGDClassifier(loss='hinge', random_state=0, max_iter=1000, tol=1e-3)
    clf.fit(embeddings, labels)
    return clf

def evaluate_classifiers(subject_clf, task_clf, test_emb, test_labels, test_sub_ids):
    """Evaluate subject and task classifiers"""
    sub_preds = subject_clf.predict(test_emb)
    task_preds = task_clf.predict(test_emb)

    sub_acc = accuracy_score(test_sub_ids, sub_preds)
    task_acc = accuracy_score(test_labels, task_preds)
    task_f1 = f1_score(test_labels, task_preds, average='macro')

    return {
        'subject_accuracy': sub_acc,
        'task_accuracy': task_acc,
        'task_f1': task_f1
    }

def main():
    print("="*70)
    print("EEG Subject Leakage / Subject-Invariance Analysis")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    all_results = []

    print("\n--- Extracting embeddings and evaluating ---")

    for seed in [0, 1, 2]:
        print(f"\nSeed {seed}:")

        print("  Raw EEG...")
        raw_embs = extract_embeddings('Raw_EEG', seed=seed)

        print("  EEG Adversarial λ=0.01...")
        adv_01_embs = extract_embeddings('EEG_Adversarial', seed=seed, lambda_adv=0.01)

        print("  EEG Adversarial λ=0.1...")
        adv_10_embs = extract_embeddings('EEG_Adversarial', seed=seed, lambda_adv=0.1)

        for i, (raw, adv_01, adv_10) in enumerate(zip(raw_embs, adv_01_embs, adv_10_embs)):
            held_out = raw['held_out']

            raw_sub_clf = train_subject_classifier(raw['train_embeddings'], raw['train_labels'], raw['train_sub_ids'])
            raw_task_clf = train_task_classifier(raw['train_embeddings'], raw['train_labels'])
            raw_eval = evaluate_classifiers(raw_sub_clf, raw_task_clf, raw['test_embeddings'], raw['test_labels'], raw['train_sub_ids'])

            adv_01_sub_clf = train_subject_classifier(adv_01['train_embeddings'], adv_01['train_labels'], adv_01['train_sub_ids'])
            adv_01_task_clf = train_task_classifier(adv_01['train_embeddings'], adv_01['train_labels'])
            adv_01_eval = evaluate_classifiers(adv_01_sub_clf, adv_01_task_clf, adv_01['test_embeddings'], adv_01['test_labels'], adv_01['train_sub_ids'])

            adv_10_sub_clf = train_subject_classifier(adv_10['train_embeddings'], adv_10['train_labels'], adv_10['train_sub_ids'])
            adv_10_task_clf = train_task_classifier(adv_10['train_embeddings'], adv_10['train_labels'])
            adv_10_eval = evaluate_classifiers(adv_10_sub_clf, adv_10_task_clf, adv_10['test_embeddings'], adv_10['test_labels'], adv_10['train_sub_ids'])

            all_results.append({
                'seed': seed,
                'held_out': held_out,
                'Raw_EEG_subject_acc': raw_eval['subject_accuracy'],
                'Raw_EEG_task_acc': raw_eval['task_accuracy'],
                'Raw_EEG_task_f1': raw_eval['task_f1'],
                'Adv_01_subject_acc': adv_01_eval['subject_accuracy'],
                'Adv_01_task_acc': adv_01_eval['task_accuracy'],
                'Adv_01_task_f1': adv_01_eval['task_f1'],
                'Adv_10_subject_acc': adv_10_eval['subject_accuracy'],
                'Adv_10_task_acc': adv_10_eval['task_accuracy'],
                'Adv_10_task_f1': adv_10_eval['task_f1'],
            })

            print(f"    {held_out}: Raw Sub={raw_eval['subject_accuracy']:.3f} Task={raw_eval['task_accuracy']:.3f} | "
                  f"Adv01 Sub={adv_01_eval['subject_accuracy']:.3f} Task={adv_01_eval['task_accuracy']:.3f} | "
                  f"Adv10 Sub={adv_10_eval['subject_accuracy']:.3f} Task={adv_10_eval['task_accuracy']:.3f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "subject_leakage_analysis.csv"), index=False)

    print("\n" + "="*70)
    print("SUMMARY (Subject Leakage Analysis)")
    print("="*70)

    summary = {
        'Model': ['Raw_EEG', 'EEG_Adversarial λ=0.01', 'EEG_Adversarial λ=0.1'],
        'Subject_Acc_Mean': [
            results_df['Raw_EEG_subject_acc'].mean(),
            results_df['Adv_01_subject_acc'].mean(),
            results_df['Adv_10_subject_acc'].mean()
        ],
        'Subject_Acc_Std': [
            results_df['Raw_EEG_subject_acc'].std(),
            results_df['Adv_01_subject_acc'].std(),
            results_df['Adv_10_subject_acc'].std()
        ],
        'Task_Acc_Mean': [
            results_df['Raw_EEG_task_acc'].mean(),
            results_df['Adv_01_task_acc'].mean(),
            results_df['Adv_10_task_acc'].mean()
        ],
        'Task_Acc_Std': [
            results_df['Raw_EEG_task_acc'].std(),
            results_df['Adv_01_task_acc'].std(),
            results_df['Adv_10_task_acc'].std()
        ],
        'Task_F1_Mean': [
            results_df['Raw_EEG_task_f1'].mean(),
            results_df['Adv_01_task_f1'].mean(),
            results_df['Adv_10_task_f1'].mean()
        ],
        'Task_F1_Std': [
            results_df['Raw_EEG_task_f1'].std(),
            results_df['Adv_01_task_f1'].std(),
            results_df['Adv_10_task_f1'].std()
        ]
    }

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    raw_sub = results_df['Raw_EEG_subject_acc'].mean()
    adv01_sub = results_df['Adv_01_subject_acc'].mean()
    adv10_sub = results_df['Adv_10_subject_acc'].mean()

    raw_task = results_df['Raw_EEG_task_acc'].mean()
    adv01_task = results_df['Adv_01_task_acc'].mean()
    adv10_task = results_df['Adv_10_task_acc'].mean()

    print(f"\n1. Subject Identity Predictability (lower = more subject-invariant):")
    print(f"   Raw EEG:           {raw_sub:.4f}")
    print(f"   EEG_Adversarial λ=0.01: {adv01_sub:.4f} (Δ={adv01_sub-raw_sub:.4f})")
    print(f"   EEG_Adversarial λ=0.1:  {adv10_sub:.4f} (Δ={adv10_sub-raw_sub:.4f})")

    print(f"\n2. Task Classification Accuracy (higher = better):")
    print(f"   Raw EEG:           {raw_task:.4f}")
    print(f"   EEG_Adversarial λ=0.01: {adv01_task:.4f} (Δ={adv01_task-raw_task:.4f})")
    print(f"   EEG_Adversarial λ=0.1:  {adv10_task:.4f} (Δ={adv10_task-raw_task:.4f})")

    if adv01_sub < raw_sub and adv01_task >= raw_task:
        print("\n✓ Adversarial training (λ=0.01) reduces subject identity predictability")
        print("  while maintaining/improving task classification!")
    elif adv10_sub < raw_sub and adv10_task >= raw_task:
        print("\n✓ Adversarial training (λ=0.1) reduces subject identity predictability")
        print("  while maintaining/improving task classification!")
    else:
        print("\n⚠ Results suggest trade-off between subject-invariance and task accuracy")

    print(f"\nSaved to {RESULTS_DIR}/subject_leakage_analysis.csv")
    print("Done!")

if __name__ == '__main__':
    main()