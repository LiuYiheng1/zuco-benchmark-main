"""
Create EvalAI Submission for ZuCo 2.0 Hidden Test Set
Trains on labeled Y-subjects and generates predictions for unlabeled X-subjects.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if os.path.basename(PROJECT_ROOT) == 'src':
    SRC_DIR = PROJECT_ROOT
else:
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

FEATURES_DIR = os.path.join(SRC_DIR, "features")
SUBMISSION_DIR = os.path.join(SRC_DIR, "submissions")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
X_SUBJECTS = ["XBB", "XDT", "XLS", "XPB", "XSE", "XTR", "XWS", "XAH", "XBD", "XSS"]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_features(subject, feature_name):
    path = os.path.join(FEATURES_DIR, f"{subject}_{feature_name}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None

def parse_key(key):
    parts = key.split("_")
    subj = parts[0]
    if len(parts) >= 2 and parts[1] == "NR":
        label, sent_idx, full_idx, is_labeled = "NR", int(parts[2]), int(parts[3]), True
    elif len(parts) >= 2 and parts[1] == "TSR":
        label, sent_idx, full_idx, is_labeled = "TSR", int(parts[2]), int(parts[3]), True
    else:
        label, sent_idx, full_idx, is_labeled = "", int(parts[-2]) if len(parts) >= 2 else 0, int(parts[-1]) if len(parts) >= 1 else 0, False
    return subj, label, sent_idx, full_idx, is_labeled

def load_labeled_data(subjects, feature_name):
    all_X, all_y, all_meta = [], [], []
    for subj in subjects:
        feats = load_features(subj, feature_name)
        if feats is None:
            continue
        for key, values in feats.items():
            subj_id, label, sent_idx, full_idx, is_labeled = parse_key(key)
            if not is_labeled:
                continue
            features = np.array(values[:-1], dtype=np.float64)
            label_binary = 1 if label == "NR" else 0
            all_X.append(features)
            all_y.append(label_binary)
            all_meta.append({'subject_id': subj_id, 'sentence_id': sent_idx, 'full_idx': full_idx, 'label': label_binary, 'original_key': key})
    return np.array(all_X), np.array(all_y), all_meta if all_X else ([], [], [])

def load_unlabeled_data(subjects, feature_name):
    all_X, all_meta = [], []
    for subj in subjects:
        feats = load_features(subj, feature_name)
        if feats is None:
            continue
        for key, values in feats.items():
            subj_id, label, sent_idx, full_idx, is_labeled = parse_key(key)
            if is_labeled:
                continue
            features = np.array(values[:-1], dtype=np.float64)
            all_X.append(features)
            all_meta.append({'subject_id': subj_id, 'sentence_id': sent_idx, 'full_idx': full_idx, 'original_key': key})
    return np.array(all_X), all_meta if all_X else ([], [])

def load_eeg_gaze_paired_labeled(subjects):
    eeg_X, eeg_y, eeg_meta = load_labeled_data(subjects, 'electrode_features_all')
    gaze_X, gaze_y, gaze_meta = load_labeled_data(subjects, 'sent_gaze_sacc')
    if len(eeg_X) == 0 or len(gaze_X) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), []
    eeg_keys = set((m['subject_id'], m['full_idx']) for m in eeg_meta)
    gaze_keys = set((m['subject_id'], m['full_idx']) for m in gaze_meta)
    common_keys = eeg_keys & gaze_keys
    eeg_lookup = {(m['subject_id'], m['full_idx']): i for i, m in enumerate(eeg_meta)}
    gaze_lookup = {(m['subject_id'], m['full_idx']): i for i, m in enumerate(gaze_meta)}
    common_eeg_idx = [eeg_lookup[k] for k in common_keys if k in eeg_lookup]
    common_gaze_idx = [gaze_lookup[k] for k in common_keys if k in gaze_lookup]
    return eeg_X[common_eeg_idx], gaze_X[common_gaze_idx], eeg_y[common_eeg_idx], gaze_y[common_gaze_idx], [eeg_meta[i] for i in common_eeg_idx]

def load_eeg_gaze_paired_unlabeled(subjects):
    eeg_X, eeg_meta = load_unlabeled_data(subjects, 'electrode_features_all')
    gaze_X, gaze_meta = load_unlabeled_data(subjects, 'sent_gaze_sacc')
    if len(eeg_X) == 0 or len(gaze_X) == 0:
        return np.array([]), np.array([]), []
    eeg_keys = set((m['subject_id'], m['full_idx']) for m in eeg_meta)
    gaze_keys = set((m['subject_id'], m['full_idx']) for m in gaze_meta)
    common_keys = eeg_keys & gaze_keys
    eeg_lookup = {(m['subject_id'], m['full_idx']): i for i, m in enumerate(eeg_meta)}
    gaze_lookup = {(m['subject_id'], m['full_idx']): i for i, m in enumerate(gaze_meta)}
    common_eeg_idx = [eeg_lookup[k] for k in common_keys if k in eeg_lookup]
    common_gaze_idx = [gaze_lookup[k] for k in common_keys if k in gaze_lookup]
    return eeg_X[common_eeg_idx], gaze_X[common_gaze_idx], [eeg_meta[i] for i in common_eeg_idx]

class EEGMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):
        return self.net(x).squeeze()

class FusionMLP(nn.Module):
    def __init__(self, eeg_dim, gaze_dim):
        super().__init__()
        self.eeg_enc = nn.Sequential(nn.Linear(eeg_dim, 256), nn.ReLU(), nn.Dropout(0.3))
        self.gaze_enc = nn.Sequential(nn.Linear(gaze_dim, 64), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(nn.Linear(256+64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, eeg, gaze):
        z_eeg = self.eeg_enc(eeg)
        z_gaze = self.gaze_enc(gaze)
        return self.classifier(torch.cat([z_eeg, z_gaze], dim=1)).squeeze()

def train_svm_model(X_train, y_train, seed=0):
    set_seed(seed)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = SVC(random_state=seed, kernel='linear', gamma='scale', probability=True)
    clf.fit(X_train_s, y_train)
    return clf, scaler

def train_eeg_mlp(X_train, y_train, input_dim, seed=0, epochs=100, device='cpu'):
    set_seed(seed)
    model = EEGMLP(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def train_fusion_mlp(eeg_train, gaze_train, y_train, eeg_dim, gaze_dim, seed=0, epochs=100, device='cpu'):
    set_seed(seed)
    model = FusionMLP(eeg_dim, gaze_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(eeg_train), torch.FloatTensor(gaze_train), torch.FloatTensor(y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for eeg, gaze, labels in loader:
            eeg, gaze, labels = eeg.to(device), gaze.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(eeg, gaze)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def create_submission(model_type, submission_name, seed=0, epochs=100, device='cpu'):
    print(f"\n{'='*60}")
    print(f"Creating {submission_name} submission")
    print(f"{'='*60}")

    print("Loading labeled training data (Y-subjects)...")
    if model_type == 'svm_eeg':
        X_train, y_train, _ = load_labeled_data(Y_SUBJECTS, 'electrode_features_all')
        if len(X_train) == 0:
            print("ERROR: No training data found!")
            return None
        print(f"Train: {len(X_train)} samples, {X_train.shape[1]} features")

        print("Loading unlabeled test data (X-subjects)...")
        X_test, test_meta = load_unlabeled_data(X_SUBJECTS, 'electrode_features_all')
        if len(X_test) == 0:
            print("ERROR: No test data found!")
            return None
        print(f"Test: {len(X_test)} samples")

        print("Training SVM...")
        clf, scaler = train_svm_model(X_train, y_train, seed)
        X_test_s = scaler.transform(X_test)
        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)[:, 1]

    elif model_type == 'svm_gaze':
        X_train, y_train, _ = load_labeled_data(Y_SUBJECTS, 'sent_gaze_sacc')
        if len(X_train) == 0:
            print("ERROR: No training data found!")
            return None
        print(f"Train: {len(X_train)} samples, {X_train.shape[1]} features")

        print("Loading unlabeled test data (X-subjects)...")
        X_test, test_meta = load_unlabeled_data(X_SUBJECTS, 'sent_gaze_sacc')
        if len(X_test) == 0:
            print("ERROR: No test data found!")
            return None
        print(f"Test: {len(X_test)} samples")

        print("Training SVM...")
        clf, scaler = train_svm_model(X_train, y_train, seed)
        X_test_s = scaler.transform(X_test)
        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)[:, 1]

    elif model_type == 'eeg_mlp':
        X_train, y_train, _ = load_labeled_data(Y_SUBJECTS, 'electrode_features_all')
        if len(X_train) == 0:
            print("ERROR: No training data found!")
            return None
        print(f"Train: {len(X_train)} samples, {X_train.shape[1]} features")

        print("Loading unlabeled test data (X-subjects)...")
        X_test, test_meta = load_unlabeled_data(X_SUBJECTS, 'electrode_features_all')
        if len(X_test) == 0:
            print("ERROR: No test data found!")
            return None
        print(f"Test: {len(X_test)} samples")

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        print("Training EEG MLP...")
        model = train_eeg_mlp(X_train_s, y_train, X_train.shape[1], seed, epochs, device)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test_s).to(device)
            outputs = model(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_pred = (probs > 0.5).astype(int)
            y_prob = probs

    elif model_type in ['early_concat', 'late_fusion', 'attention_fusion']:
        eeg_train, gaze_train, y_train, _, _ = load_eeg_gaze_paired_labeled(Y_SUBJECTS)
        if len(eeg_train) == 0:
            print("ERROR: No training data found!")
            return None
        print(f"Train: {len(eeg_train)} samples, EEG:{eeg_train.shape[1]} features, Gaze:{gaze_train.shape[1]} features")

        print("Loading unlabeled test data (X-subjects)...")
        eeg_test, gaze_test, test_meta = load_eeg_gaze_paired_unlabeled(X_SUBJECTS)
        if len(eeg_test) == 0:
            print("ERROR: No test data found!")
            return None
        print(f"Test: {len(eeg_test)} samples")

        scaler_eeg = StandardScaler()
        scaler_gaze = StandardScaler()
        eeg_train_s = scaler_eeg.fit_transform(eeg_train)
        eeg_test_s = scaler_eeg.transform(eeg_test)
        gaze_train_s = scaler_gaze.fit_transform(gaze_train)
        gaze_test_s = scaler_gaze.transform(gaze_test)

        print(f"Training {model_type} MLP...")
        model = train_fusion_mlp(eeg_train_s, gaze_train_s, y_train, eeg_train.shape[1], gaze_train.shape[1], seed, epochs, device)
        model.eval()
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_test_s).to(device)
            gaze_tensor = torch.FloatTensor(gaze_test_s).to(device)
            outputs = model(eeg_tensor, gaze_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_pred = (probs > 0.5).astype(int)
            y_prob = probs

    else:
        print(f"ERROR: Unknown model type: {model_type}")
        return None

    print(f"Predictions: NR={sum(y_pred==1)}, TSR={sum(y_pred==0)}")

    submission = {}
    for i, meta in enumerate(test_meta):
        subject_id = meta['subject_id']
        sentence_id = meta['sentence_id']
        if subject_id not in submission:
            submission[subject_id] = {}
        submission[subject_id][sentence_id] = int(y_pred[i])

    submission_path = os.path.join(SUBMISSION_DIR, f"{submission_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"Saved submission to {submission_path}")

    return submission_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', help='Model to create submission for: svm_eeg, svm_gaze, eeg_mlp, early_concat, late_fusion, attention_fusion, all')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print("="*70)
    print("ZuCo 2.0 EvalAI Submission Creator")
    print("="*70)
    print(f"Training on Y-subjects ({len(Y_SUBJECTS)} labeled subjects)")
    print(f"Predicting on X-subjects ({len(X_SUBJECTS)} hidden test subjects)")
    print(f"NO ground-truth labels available for X-subjects - for EvalAI evaluation only")

    models_to_run = []
    if args.model == 'all':
        models_to_run = [
            ('svm_eeg', 'baseline_svm_eeg'),
            ('svm_gaze', 'baseline_svm_gaze'),
            ('eeg_mlp', 'baseline_eeg_mlp'),
            ('early_concat', 'baseline_early_concat'),
            ('late_fusion', 'baseline_late_fusion'),
            ('attention_fusion', 'baseline_attention_fusion'),
        ]
    else:
        models_to_run = [(args.model, f'{args.model}_submission')]

    for model_type, submission_name in models_to_run:
        create_submission(model_type, submission_name, args.seed, args.epochs, args.device)

    print("\n" + "="*70)
    print("SUBMISSION CREATION COMPLETE!")
    print("="*70)
    print(f"Submissions saved to: {SUBMISSION_DIR}")
    print("Upload to EvalAI for official evaluation on hidden test set.")

if __name__ == '__main__':
    main()