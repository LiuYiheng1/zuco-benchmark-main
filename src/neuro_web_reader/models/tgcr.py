"""
Task-Schema Guided Cognitive Router v1 (TGCR v1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Tuple, Optional

class EEGEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128], output_dim: int = 64, dropout: float = 0.3):
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
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GazeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32], output_dim: int = 32, dropout: float = 0.3):
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
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    def __init__(self, eeg_dim: int = 64, gaze_dim: int = 32, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Sequential(
            nn.Linear(eeg_dim + gaze_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, z_eeg, z_gaze):
        combined = torch.cat([z_eeg, z_gaze], dim=1)
        weights = self.router(combined)
        return weights


class EEGExpert(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_eeg):
        return self.net(z_eeg)


class GazeExpert(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_gaze):
        return self.net(z_gaze)


class FusionExpert(nn.Module):
    def __init__(self, eeg_dim: int = 64, gaze_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(eeg_dim + gaze_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z_eeg, z_gaze):
        combined = torch.cat([z_eeg, z_gaze], dim=1)
        return self.net(combined)


class TGCRv1(nn.Module):
    def __init__(self, eeg_dim: int, gaze_dim: int, eeg_hidden: list = [256, 128],
                 gaze_hidden: list = [64, 32], dropout: float = 0.3):
        super().__init__()
        self.eeg_encoder = EEGEncoder(eeg_dim, eeg_hidden, dropout=dropout)
        self.gaze_encoder = GazeEncoder(gaze_dim, gaze_hidden, dropout=dropout)

        self.router = Router(eeg_dim=64, gaze_dim=32, num_experts=4)

        self.eeg_expert = EEGExpert(input_dim=64, hidden_dim=32)
        self.gaze_expert = GazeExpert(input_dim=32, hidden_dim=16)
        self.fusion_expert = FusionExpert(eeg_dim=64, gaze_dim=32, hidden_dim=32)

        self.eeg_out_dim = 64
        self.gaze_out_dim = 32

    def forward(self, eeg, gaze, return_router_weights=False):
        z_eeg = self.eeg_encoder(eeg)
        z_gaze = self.gaze_encoder(gaze)

        router_weights = self.router(z_eeg, z_gaze)

        expert_eeg = self.eeg_expert(z_eeg)
        expert_gaze = self.gaze_expert(z_gaze)
        expert_fusion = self.fusion_expert(z_eeg, z_gaze)

        alpha_eeg = router_weights[:, 0].unsqueeze(1)
        alpha_gaze = router_weights[:, 1].unsqueeze(1)
        alpha_fusion = router_weights[:, 2].unsqueeze(1)

        logits = alpha_eeg * expert_eeg + alpha_gaze * expert_gaze + alpha_fusion * expert_fusion

        if return_router_weights:
            return logits.squeeze(), router_weights
        return logits.squeeze()

    def get_router_weights(self, eeg, gaze):
        z_eeg = self.eeg_encoder(eeg)
        z_gaze = self.gaze_encoder(gaze)
        return self.router(z_eeg, z_gaze)


class TGCRv1WithoutRouter(nn.Module):
    def __init__(self, eeg_dim: int, gaze_dim: int, eeg_hidden: list = [256, 128],
                 gaze_hidden: list = [64, 32], dropout: float = 0.3):
        super().__init__()
        self.eeg_encoder = EEGEncoder(eeg_dim, eeg_hidden, dropout=dropout)
        self.gaze_encoder = GazeEncoder(gaze_dim, gaze_hidden, dropout=dropout)

        self.fusion_expert = FusionExpert(eeg_dim=64, gaze_dim=32, hidden_dim=32)

    def forward(self, eeg, gaze):
        z_eeg = self.eeg_encoder(eeg)
        z_gaze = self.gaze_encoder(gaze)
        logits = self.fusion_expert(z_eeg, z_gaze)
        return logits.squeeze()


class TGCRv1EEGonlY(nn.Module):
    def __init__(self, eeg_dim: int, eeg_hidden: list = [256, 128], dropout: float = 0.3):
        super().__init__()
        self.eeg_encoder = EEGEncoder(eeg_dim, eeg_hidden, dropout=dropout)
        self.eeg_expert = EEGExpert(input_dim=64, hidden_dim=32)

    def forward(self, eeg, gaze=None):
        z_eeg = self.eeg_encoder(eeg)
        return self.eeg_expert(z_eeg).squeeze()


class TGCRv1GazeOnly(nn.Module):
    def __init__(self, gaze_dim: int, gaze_hidden: list = [64, 32], dropout: float = 0.3):
        super().__init__()
        self.gaze_encoder = GazeEncoder(gaze_dim, gaze_hidden, dropout=dropout)
        self.gaze_expert = GazeExpert(input_dim=32, hidden_dim=16)

    def forward(self, eeg=None, gaze=None):
        z_gaze = self.gaze_encoder(gaze)
        return self.gaze_expert(z_gaze).squeeze()


class TGCRv1RandomRouter(nn.Module):
    def __init__(self, eeg_dim: int, gaze_dim: int, eeg_hidden: list = [256, 128],
                 gaze_hidden: list = [64, 32], dropout: float = 0.3):
        super().__init__()
        self.eeg_encoder = EEGEncoder(eeg_dim, eeg_hidden, dropout=dropout)
        self.gaze_encoder = GazeEncoder(gaze_dim, gaze_hidden, dropout=dropout)

        self.register_buffer('fixed_weights', torch.tensor([0.25, 0.25, 0.25, 0.25]))

        self.eeg_expert = EEGExpert(input_dim=64, hidden_dim=32)
        self.gaze_expert = GazeExpert(input_dim=32, hidden_dim=16)
        self.fusion_expert = FusionExpert(eeg_dim=64, gaze_dim=32, hidden_dim=32)

    def forward(self, eeg, gaze, return_router_weights=False):
        z_eeg = self.eeg_encoder(eeg)
        z_gaze = self.gaze_encoder(gaze)

        router_weights = self.fixed_weights.unsqueeze(0).expand(z_eeg.size(0), -1)

        expert_eeg = self.eeg_expert(z_eeg)
        expert_gaze = self.gaze_expert(z_gaze)
        expert_fusion = self.fusion_expert(z_eeg, z_gaze)

        alpha_eeg = router_weights[:, 0].unsqueeze(1)
        alpha_gaze = router_weights[:, 1].unsqueeze(1)
        alpha_fusion = router_weights[:, 2].unsqueeze(1)

        logits = alpha_eeg * expert_eeg + alpha_gaze * expert_gaze + alpha_fusion * expert_fusion

        if return_router_weights:
            return logits.squeeze(), router_weights
        return logits.squeeze()


class TGCRDataset(Dataset):
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