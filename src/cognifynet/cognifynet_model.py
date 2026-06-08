"""
CognifyNet: Cognitive-Neuro Inspired Multimodal Framework
========================================================
Pure PyTorch implementation (no torch_geometric dependency)

Modules:
  1. Frequency-aware EEG NeuroGraph Encoder
     - Train-only correlation → learnable adjacency
     - Multi-head GAT-style attention across frequency bands
  2. Gaze Behavior Encoder (MLP)
  3. Causal Subject Disentanglement (GRL + orthogonality)
  4. Energy-Based Multimodal Fusion
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lamda, None


def grl_layer(x, lamda=1.0):
    return GradientReversal.apply(x, lamda)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.3):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.a = nn.Parameter(torch.randn(n_heads, 2 * out_dim) * 0.05)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim * n_heads, in_dim)

    def forward(self, x, adj):
        B, N, D = x.shape
        Wh = self.W(x).view(B, N, self.n_heads, self.out_dim)

        a_src = self.a[:, :self.out_dim]
        a_dst = self.a[:, self.out_dim:]

        a1 = torch.einsum('bnhd,hd->bnh', Wh, a_src)
        a2 = torch.einsum('bnhd,hd->bnh', Wh, a_dst)

        e = self.leaky_relu(a1.unsqueeze(2) + a2.unsqueeze(1))
        e = e.masked_fill(adj.unsqueeze(0).unsqueeze(-1) == 0, -1e9)

        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)

        h = torch.einsum('bnmh,bmhd->bnhd', alpha, Wh)
        h = h.reshape(B, N, self.n_heads * self.out_dim)
        return self.out_proj(h), alpha


class EEGNeuroGraphEncoder(nn.Module):
    def __init__(self, n_bands=4, n_channels=105, hidden=128, gcn_hidden=256, dropout=0.3):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.hidden = hidden

        self.node_proj = nn.Linear(n_bands, hidden)
        self.node_norm = nn.LayerNorm(hidden)

        self.gat1 = GraphAttentionLayer(hidden, hidden // 4, n_heads=4, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden)

        self.gat2 = GraphAttentionLayer(hidden, hidden // 4, n_heads=4, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden)

        self.ffn = nn.Sequential(
            nn.Linear(hidden, gcn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden, hidden),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        B = x.shape[0]
        x = x.view(B, self.n_channels, self.n_bands)
        x = self.node_proj(x)
        x = self.node_norm(x)

        h, _ = self.gat1(x, adj)
        h = self.norm1(x + self.dropout(h))

        h, _ = self.gat2(h, adj)
        h = self.norm2(h + self.dropout(h))

        h = self.norm3(h + self.ffn(h))
        return h.mean(dim=1)


class GazeEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden=64, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class TaskDisentangler(nn.Module):
    def __init__(self, eeg_dim=128, gaze_dim=128, hidden=256, task_dim=128, subj_dim=128):
        super().__init__()
        in_dim = eeg_dim + gaze_dim
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),
        )
        self.task_proj = nn.Sequential(
            nn.Linear(hidden, task_dim),
            nn.GELU(),
            nn.LayerNorm(task_dim),
        )
        self.subj_proj = nn.Sequential(
            nn.Linear(hidden, subj_dim),
            nn.GELU(),
            nn.LayerNorm(subj_dim),
        )

    def forward(self, z_eeg, z_gaze, lamda=0.1):
        z_concat = torch.cat([z_eeg, z_gaze], dim=-1)
        h = self.shared(z_concat)
        z_task = self.task_proj(h)
        z_subj_raw = self.subj_proj(h)
        z_task_grl = grl_layer(z_task, lamda)
        return z_task, z_subj_raw, z_task_grl


class EnergyFusion(nn.Module):
    def __init__(self, eeg_dim=128, gaze_dim=128, task_dim=128, n_classes=2):
        super().__init__()
        self.eeg_energy = nn.Sequential(
            nn.Linear(eeg_dim, 64), nn.GELU(), nn.Linear(64, n_classes))
        self.gaze_energy = nn.Sequential(
            nn.Linear(gaze_dim, 64), nn.GELU(), nn.Linear(64, n_classes))
        self.task_energy = nn.Sequential(
            nn.Linear(task_dim, 64), nn.GELU(), nn.Linear(64, n_classes))
        self.reliability = nn.Sequential(
            nn.Linear(eeg_dim + gaze_dim + task_dim, 64),
            nn.GELU(), nn.Linear(64, 3), nn.Softmax(dim=-1))

    def forward(self, z_eeg, z_gaze, z_task):
        E_eeg = self.eeg_energy(z_eeg)
        E_gaze = self.gaze_energy(z_gaze)
        E_task = self.task_energy(z_task)
        w = self.reliability(torch.cat([z_eeg, z_gaze, z_task], dim=-1))
        E_fused = w[:, 0:1] * E_eeg + w[:, 1:2] * E_gaze + w[:, 2:3] * E_task
        return E_fused, w, (E_eeg, E_gaze, E_task)


class CognifyNet(nn.Module):
    """CognifyNet: Cognitive-Neuro Inspired Multimodal Framework"""

    def __init__(self,
                 n_bands=4, n_channels=105,
                 eeg_hidden=128, gcn_hidden=256,
                 gaze_hidden=64, gaze_dim=128,
                 task_dim=128, subj_dim=128,
                 n_classes=2, dropout=0.3):
        super().__init__()

        self.eeg_encoder = EEGNeuroGraphEncoder(
            n_bands=n_bands, n_channels=n_channels,
            hidden=eeg_hidden, gcn_hidden=gcn_hidden, dropout=dropout)
        self.gaze_encoder = GazeEncoder(
            input_dim=9, hidden=gaze_hidden, output_dim=gaze_dim)
        self.disentangler = TaskDisentangler(
            eeg_dim=eeg_hidden, gaze_dim=gaze_dim,
            hidden=256, task_dim=task_dim, subj_dim=subj_dim)
        self.energy_fusion = EnergyFusion(
            eeg_dim=eeg_hidden, gaze_dim=gaze_dim,
            task_dim=task_dim, n_classes=n_classes)
        self.task_head = nn.Linear(task_dim, n_classes)
        self.subj_head = None
        self.eeg_hidden = eeg_hidden

    def set_subj_head(self, n_subjects):
        self.subj_head = nn.Linear(128, n_subjects)

    def forward(self, x_eeg, x_gaze, adj, lamda=0.1):
        z_eeg = self.eeg_encoder(x_eeg, adj)
        z_gaze = self.gaze_encoder(x_gaze)
        z_task, z_subj, z_task_grl = self.disentangler(z_eeg, z_gaze, lamda)
        E_fused, w, energies = self.energy_fusion(z_eeg, z_gaze, z_task)
        task_logits = self.task_head(z_task) - E_fused
        return task_logits, z_task, z_subj, z_task_grl, w, energies