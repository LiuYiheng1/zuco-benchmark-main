import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

SUBJECTS_16 = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS',
                'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_aligned_data(subject):
    eeg_path = f'features/{subject}_electrode_features_all.npy'
    gaze_path = f'features/{subject}_sent_gaze_sacc.npy'
    
    if not os.path.exists(eeg_path) or not os.path.exists(gaze_path):
        return None, None, None, None
    
    eeg_data = np.load(eeg_path, allow_pickle=True).item()
    gaze_data = np.load(gaze_path, allow_pickle=True).item()
    
    gaze_by_label_sent = {}
    for key in gaze_data.keys():
        parts = key.split('_')
        if len(parts) >= 3:
            label = parts[1]
            sent_idx = int(parts[2])
            gaze_by_label_sent[(label, sent_idx)] = key
    
    X_eeg, X_gaze, texts, y = [], [], [], []
    
    for eeg_key in eeg_data.keys():
        parts = eeg_key.split('_')
        if len(parts) < 3:
            continue
        
        label = parts[1]
        sent_idx = int(parts[2])
        
        gaze_key = gaze_by_label_sent.get((label, sent_idx))
        if gaze_key is None:
            continue
        
        eeg_feat = np.array(eeg_data[eeg_key])
        gaze_feat = np.array(gaze_data[gaze_key])
        
        if eeg_feat[-1] in ['NR', 'TSR']:
            eeg_feat = eeg_feat[:-1]
        if gaze_feat[-1] in ['NR', 'TSR']:
            gaze_feat = gaze_feat[:-1]
        
        X_eeg.append(eeg_feat.astype(float))
        X_gaze.append(gaze_feat.astype(float))
        texts.append(f'{label}_{sent_idx}')
        y.append(0 if label == 'NR' else 1)
    
    return np.array(X_eeg), np.array(X_gaze), np.array(texts), np.array(y)


class SRFNet(nn.Module):
    def __init__(self, text_dim, eeg_dim, gaze_dim, 
                proj_dim=64, hidden_dim=128,
                use_agreement=True, use_conflict=True,
                lambda_inter=0.0, lambda_learnable=False):
        super().__init__()
        
        self.use_agreement = use_agreement
        self.use_conflict = use_conflict
        self.lambda_learnable = lambda_learnable
        
        concat_dim = text_dim + eeg_dim + gaze_dim
        
        self.backbone = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.backbone_head = nn.Linear(128, 2)
        
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.eeg_proj = nn.Linear(eeg_dim, proj_dim)
        self.gaze_proj = nn.Linear(gaze_dim, proj_dim)
        
        inter_input_dim = proj_dim * 3  # z_t, z_e, z_g
        inter_input_dim += proj_dim * 2  # text_eeg_diff, text_gaze_diff
        if self.use_agreement:
            inter_input_dim += proj_dim
        if self.use_conflict:
            inter_input_dim += proj_dim
        
        self.inter_mlp = nn.Sequential(
            nn.Linear(inter_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.inter_head = nn.Linear(hidden_dim // 2, 2)
        
        if self.lambda_learnable:
            self.lambda_param = nn.Parameter(torch.tensor(lambda_inter))
        else:
            self.lambda_value = lambda_inter
    
    def forward(self, x_text, x_eeg, x_gaze):
        x_concat = torch.cat([x_text, x_eeg, x_gaze], dim=1)
        h_base = self.backbone(x_concat)
        logit_base = self.backbone_head(h_base)
        
        z_t = torch.relu(self.text_proj(x_text))
        z_e = torch.relu(self.eeg_proj(x_eeg))
        z_g = torch.relu(self.gaze_proj(x_gaze))
        
        text_eeg_diff = torch.abs(z_t - z_e)
        text_gaze_diff = torch.abs(z_t - z_g)
        
        inter_features = [z_t, z_e, z_g, text_eeg_diff, text_gaze_diff]
        if self.use_agreement:
            inter_features.append(z_e * z_g)
        if self.use_conflict:
            inter_features.append(torch.abs(z_e - z_g))
        
        h_inter = torch.cat(inter_features, dim=1)
        h_inter = self.inter_mlp(h_inter)
        delta_logit = self.inter_head(h_inter)
        
        if self.lambda_learnable:
            lambda_val = torch.sigmoid(self.lambda_param) * 0.5
        else:
            lambda_val = self.lambda_value
        
        logit_final = logit_base + lambda_val * torch.tanh(delta_logit)
        
        return logit_final, logit_base, delta_logit


def train_stage1(model, X_text_train, X_eeg_train, X_gaze_train, y_train,
                X_text_val, X_eeg_val, X_gaze_val, y_val, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_text_train = torch.tensor(X_text_train, dtype=torch.float32).to(device)
    X_eeg_train = torch.tensor(X_eeg_train, dtype=torch.float32).to(device)
    X_gaze_train = torch.tensor(X_gaze_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    
    X_text_val = torch.tensor(X_text_val, dtype=torch.float32).to(device)
    X_eeg_val = torch.tensor(X_eeg_val, dtype=torch.float32).to(device)
    X_gaze_val = torch.tensor(X_gaze_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        
        logit_final, _, _ = model(X_text_train, X_eeg_train, X_gaze_train)
        loss = criterion(logit_final, y_train)
        
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        logit_final, _, _ = model(X_text_val, X_eeg_val, X_gaze_val)
        y_pred = torch.argmax(logit_final, dim=1).cpu().numpy()
        y_proba = torch.softmax(logit_final, dim=1)[:, 1].cpu().numpy()
    
    return {
        'accuracy': accuracy_score(y_val.cpu().numpy(), y_pred),
        'macro_f1': f1_score(y_val.cpu().numpy(), y_pred, average='macro'),
        'balanced_acc': balanced_accuracy_score(y_val.cpu().numpy(), y_pred),
        'auroc': roc_auc_score(y_val.cpu().numpy(), y_proba)
    }


def train_stage2(model, X_text_train, X_eeg_train, X_gaze_train, y_train,
                X_text_val, X_eeg_val, X_gaze_val, y_val, device):
    model = model.to(device)
    
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone_head.parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    X_text_train = torch.tensor(X_text_train, dtype=torch.float32).to(device)
    X_eeg_train = torch.tensor(X_eeg_train, dtype=torch.float32).to(device)
    X_gaze_train = torch.tensor(X_gaze_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    
    X_text_val = torch.tensor(X_text_val, dtype=torch.float32).to(device)
    X_eeg_val = torch.tensor(X_eeg_val, dtype=torch.float32).to(device)
    X_gaze_val = torch.tensor(X_gaze_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        
        logit_final, _, _ = model(X_text_train, X_eeg_train, X_gaze_train)
        loss = criterion(logit_final, y_train)
        
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        logit_final, _, _ = model(X_text_val, X_eeg_val, X_gaze_val)
        y_pred = torch.argmax(logit_final, dim=1).cpu().numpy()
        y_proba = torch.softmax(logit_final, dim=1)[:, 1].cpu().numpy()
    
    return {
        'accuracy': accuracy_score(y_val.cpu().numpy(), y_pred),
        'macro_f1': f1_score(y_val.cpu().numpy(), y_pred, average='macro'),
        'balanced_acc': balanced_accuracy_score(y_val.cpu().numpy(), y_pred),
        'auroc': roc_auc_score(y_val.cpu().numpy(), y_proba)
    }


def run_experiment():
    all_data = {}
    for subject in SUBJECTS_16:
        X_eeg, X_gaze, texts, y = load_aligned_data(subject)
        if X_eeg is not None:
            all_data[subject] = {'X_eeg': X_eeg, 'X_gaze': X_gaze, 'texts': texts, 'y': y}
    
    X_eeg_all = np.vstack([d['X_eeg'] for d in all_data.values()])
    X_gaze_all = np.vstack([d['X_gaze'] for d in all_data.values()])
    texts_all = np.concatenate([d['texts'] for d in all_data.values()])
    y_all = np.concatenate([d['y'] for d in all_data.values()])
    
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    for seed in range(3):
        print(f"\n=== Seed {seed} ===")
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_idx, test_idx = next(sss.split(X_eeg_all, y_all))
        
        X_eeg_train, X_eeg_test = X_eeg_all[train_idx], X_eeg_all[test_idx]
        X_gaze_train, X_gaze_test = X_gaze_all[train_idx], X_gaze_all[test_idx]
        texts_train, texts_test = texts_all[train_idx], texts_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        tfidf = TfidfVectorizer(max_features=200)
        X_text_train = tfidf.fit_transform(texts_train).toarray()
        X_text_test = tfidf.transform(texts_test).toarray()
        
        scaler_eeg = StandardScaler()
        X_eeg_train_s = scaler_eeg.fit_transform(X_eeg_train)
        X_eeg_test_s = scaler_eeg.transform(X_eeg_test)
        
        scaler_gaze = StandardScaler()
        X_gaze_train_s = scaler_gaze.fit_transform(X_gaze_train)
        X_gaze_test_s = scaler_gaze.transform(X_gaze_test)
        
        text_dim = X_text_train.shape[1]
        eeg_dim = X_eeg_train.shape[1]
        gaze_dim = X_gaze_train.shape[1]
        
        # Baseline: Text+EEG+Gaze_concat (sklearn MLP)
        X_all_train = np.hstack([X_text_train, X_eeg_train_s, X_gaze_train_s])
        X_all_test = np.hstack([X_text_test, X_eeg_test_s, X_gaze_test_s])
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500)
        clf.fit(X_all_train, y_train)
        y_pred = clf.predict(X_all_test)
        y_proba = clf.predict_proba(X_all_test)[:, 1]
        results.append({'method': 'Text+EEG+Gaze_concat', 'seed': seed,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'macro_f1': f1_score(y_test, y_pred, average='macro'),
                        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
                        'auroc': roc_auc_score(y_test, y_proba)})
        print(f"Text+EEG+Gaze_concat: {accuracy_score(y_test, y_pred):.4f}")
        
        # SRF_base_only (same architecture as baseline)
        model = SRFNet(text_dim, eeg_dim, gaze_dim, lambda_inter=0.0)
        result = train_stage1(model, X_text_train, X_eeg_train_s, X_gaze_train_s, y_train,
                            X_text_test, X_eeg_test_s, X_gaze_test_s, y_test, device)
        result['method'] = 'SRF_base_only'
        result['seed'] = seed
        results.append(result)
        print(f"SRF_base_only: {result['accuracy']:.4f}")
        
        # SRF_full (stage1 + stage2)
        model = SRFNet(text_dim, eeg_dim, gaze_dim, lambda_inter=0.3)
        train_stage1(model, X_text_train, X_eeg_train_s, X_gaze_train_s, y_train,
                    X_text_test, X_eeg_test_s, X_gaze_test_s, y_test, device)
        result = train_stage2(model, X_text_train, X_eeg_train_s, X_gaze_train_s, y_train,
                            X_text_test, X_eeg_test_s, X_gaze_test_s, y_test, device)
        result['method'] = 'SRF_full'
        result['seed'] = seed
        results.append(result)
        print(f"SRF_full: {result['accuracy']:.4f}")
        
        # SRF_w/o_agreement
        model = SRFNet(text_dim, eeg_dim, gaze_dim, use_agreement=False, lambda_inter=0.3)
        train_stage1(model, X_text_train, X_eeg_train_s, X_gaze_train_s, y_train,
                    X_text_test, X_eeg_test_s, X_gaze_test_s, y_test, device)
        result = train_stage2(model, X_text_train, X_eeg_train_s, X_gaze_train_s, y_train,
                            X_text_test, X_eeg_test_s, X_gaze_test_s, y_test, device)
        result['method'] = 'SRF_w/o_agreement'
        result['seed'] = seed
        results.append(result)
        print(f"SRF_w/o_agreement: {result['accuracy']:.4f}")
        
        # SRF_w/o_conflict
        model = SRFNet(text_dim, eeg_dim, gaze_dim, use_conflict=False, lambda_inter=0.3)
        train_stage1(model, X_text_train, X_eeg_train_s, X_gaze_train_s, y_train,
                    X_text_test, X_eeg_test_s, X_gaze_test_s, y_test, device)
        result = train_stage2(model, X_text_train, X_eeg_train_s, X_gaze_train_s, y_train,
                            X_text_test, X_eeg_test_s, X_gaze_test_s, y_test, device)
        result['method'] = 'SRF_w/o_conflict'
        result['seed'] = seed
        results.append(result)
        print(f"SRF_w/o_conflict: {result['accuracy']:.4f}")
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("=" * 80)
    print("SRF-Net: Safe Residual Fusion Network")
    print("=" * 80)
    
    df = run_experiment()
    
    print("\n" + "=" * 80)
    print("Results Summary (mean over 3 seeds):")
    print("=" * 80)
    
    summary = df.groupby('method').agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        f1_mean=('macro_f1', 'mean'),
        f1_std=('macro_f1', 'std'),
        balanced_acc_mean=('balanced_acc', 'mean'),
        auroc_mean=('auroc', 'mean')
    ).round(4)
    
    print(summary)
    
    os.makedirs('results/final', exist_ok=True)
    df.to_csv('results/final/srf_net_results.csv', index=False)
    
    print("\n\nFiles saved:")
    print("  - results/final/srf_net_results.csv")
    
    print("\n" + "=" * 80)
    print("SRF-Net Experiment Complete")
    print("=" * 80)