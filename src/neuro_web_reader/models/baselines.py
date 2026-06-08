"""
PyTorch Baseline Models for ZuCo 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EEGOnlyMLP(nn.Module):
    def __init__(self, eeg_dim: int, hidden_dims: list = [256, 128], dropout: float = 0.3):
        super().__init__()
        self.eeg_dim = eeg_dim
        self.encoder = SimpleMLP(eeg_dim, hidden_dims, dropout)

    def forward(self, eeg):
        return self.encoder(eeg)


class GazeOnlyMLP(nn.Module):
    def __init__(self, gaze_dim: int, hidden_dims: list = [64, 32], dropout: float = 0.3):
        super().__init__()
        self.gaze_dim = gaze_dim
        self.encoder = SimpleMLP(gaze_dim, hidden_dims, dropout)

    def forward(self, gaze):
        return self.encoder(gaze)


class EarlyConcatMLP(nn.Module):
    def __init__(self, eeg_dim: int, gaze_dim: int, hidden_dims: list = [256, 128], dropout: float = 0.3):
        super().__init__()
        self.eeg_dim = eeg_dim
        self.gaze_dim = gaze_dim
        concat_dim = eeg_dim + gaze_dim
        self.encoder = SimpleMLP(concat_dim, hidden_dims, dropout)

    def forward(self, eeg, gaze):
        x = torch.cat([eeg, gaze], dim=1)
        return self.encoder(x)


class LateFusionModel(nn.Module):
    def __init__(self, eeg_dim: int, gaze_dim: int, eeg_hidden: list = [256, 128],
                 gaze_hidden: list = [64, 32], fusion_hidden: list = [64], dropout: float = 0.3):
        super().__init__()
        self.eeg_encoder = SimpleMLP(eeg_dim, eeg_hidden, dropout)
        self.gaze_encoder = SimpleMLP(gaze_dim, gaze_hidden, dropout)

        fusion_input_dim = eeg_hidden[-1] + gaze_hidden[-1]
        fusion_layers = []
        prev_dim = fusion_input_dim
        for h_dim in fusion_hidden:
            fusion_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        fusion_layers.append(nn.Linear(prev_dim, 1))
        self.fusion = nn.Sequential(*fusion_layers) if fusion_layers else nn.Linear(prev_dim, 1)

    def forward(self, eeg, gaze):
        z_eeg = self.eeg_encoder(eeg)
        z_gaze = self.gaze_encoder(gaze)
        z_fusion = torch.cat([z_eeg, z_gaze], dim=1)
        return self.fusion(z_fusion)


class AttentionFusion(nn.Module):
    def __init__(self, eeg_dim: int, gaze_dim: int, eeg_hidden: list = [256, 128],
                 gaze_hidden: list = [64, 32], dropout: float = 0.3):
        super().__init__()
        self.eeg_encoder = SimpleMLP(eeg_dim, eeg_hidden, dropout)
        self.gaze_encoder = SimpleMLP(gaze_dim, gaze_hidden, dropout)

        eeg_out_dim = eeg_hidden[-1]
        gaze_out_dim = gaze_hidden[-1]

        self.gate = nn.Sequential(
            nn.Linear(eeg_out_dim + gaze_out_dim, eeg_out_dim + gaze_out_dim),
            nn.Sigmoid()
        )

        fusion_input_dim = eeg_out_dim + gaze_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, eeg, gaze):
        z_eeg = self.eeg_encoder(eeg)
        z_gaze = self.gaze_encoder(gaze)

        z_concat = torch.cat([z_eeg, z_gaze], dim=1)
        gate = self.gate(z_concat)
        z_fused = z_concat * gate

        return self.classifier(z_fused)


class ZuCoDataset(Dataset):
    def __init__(self, eeg_X: np.ndarray, gaze_X: np.ndarray, y: np.ndarray):
        self.eeg_X = torch.FloatTensor(eeg_X)
        self.gaze_X = torch.FloatTensor(gaze_X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'eeg': self.eeg_X[idx],
            'gaze': self.gaze_X[idx],
            'label': self.y[idx]
        }


class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'features': self.X[idx], 'label': self.y[idx]}


class GazeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'features': self.X[idx], 'label': self.y[idx]}


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        eeg = batch['eeg'].to(device)
        gaze = batch['gaze'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(eeg, gaze).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_eeg_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    probabilities = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            gaze = batch['gaze'].to(device)
            batch_labels = batch['label']

            outputs = model(eeg, gaze).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            labels.extend(batch_labels.numpy())

    return np.array(predictions), np.array(probabilities), np.array(labels)


def evaluate_eeg(model, dataloader, device):
    model.eval()
    predictions = []
    probabilities = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            batch_labels = batch['label']

            outputs = model(features).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            labels.extend(batch_labels.numpy())

    return np.array(predictions), np.array(probabilities), np.array(labels)