"""NOCS model: Neuro-Oculomotor Controlled State Model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return GradientReverseFn.apply(x, scale)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom


def gaze_stat_pool(gaze: torch.Tensor, valid_mask: torch.Tensor, gaze_mask: torch.Tensor) -> torch.Tensor:
    mask_f = gaze_mask.float().unsqueeze(-1)
    valid_count = mask_f.sum(dim=1)
    sequence_length = valid_mask.float().sum(dim=1, keepdim=True)
    denom = valid_count.clamp_min(1.0)
    mean = (gaze * mask_f).sum(dim=1) / denom
    centered = (gaze - mean.unsqueeze(1)) * mask_f
    var = centered.pow(2).sum(dim=1) / denom
    std = torch.sqrt(var.clamp_min(0.0))
    std = torch.where(valid_count > 1.0, std, torch.zeros_like(std))
    valid_ratio = valid_count / sequence_length.clamp_min(1.0)
    return torch.cat([mean, std, valid_ratio, sequence_length, valid_count], dim=-1)


class MaskedMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        h = h * mask.float().unsqueeze(-1)
        z = masked_mean(h, mask)
        return z, h


class GazeBranch(nn.Module):
    def __init__(self, gaze_dim: int, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = MaskedMLPEncoder(gaze_dim, d_model, hidden_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, gaze: torch.Tensor, gaze_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, tokens = self.encoder(gaze, gaze_mask)
        return z, tokens, self.classifier(z)


class GazeStatBranch(nn.Module):
    def __init__(self, gaze_dim: int, hidden_dim: int, dropout: float, stat_head: str = "linear") -> None:
        super().__init__()
        self.feature_dim = 2 * gaze_dim + 3
        self.norm = nn.LayerNorm(self.feature_dim)
        self.repr = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if stat_head == "linear":
            self.classifier = nn.Linear(self.feature_dim, 2)
        elif stat_head == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2),
            )
        else:
            raise ValueError("Unsupported stat_head: %s" % stat_head)

    def forward(
        self,
        gaze: torch.Tensor,
        valid_mask: torch.Tensor,
        gaze_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = gaze_stat_pool(gaze, valid_mask, gaze_mask)
        normed = self.norm(features)
        z = self.repr(normed)
        logits = self.classifier(normed)
        return z, features, logits


class EEGBranch(nn.Module):
    def __init__(self, eeg_dim: int, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = MaskedMLPEncoder(eeg_dim, d_model, hidden_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, eeg: torch.Tensor, eeg_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, tokens = self.encoder(eeg, eeg_mask)
        return z, tokens, self.classifier(z)


class GazeControlledState(nn.Module):
    def __init__(
        self,
        eeg_dim: int,
        gaze_dim: int,
        hidden_dim: int,
        dropout: float,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.eeg_proj = nn.Linear(eeg_dim, hidden_dim)
        self.gaze_proj = nn.Linear(gaze_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(gaze_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), 2)

    def _scan(
        self,
        eeg: torch.Tensor,
        gaze: torch.Tensor,
        valid_mask: torch.Tensor,
        eeg_mask: torch.Tensor,
        gaze_mask: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            eeg = torch.flip(eeg, dims=[1])
            gaze = torch.flip(gaze, dims=[1])
            valid_mask = torch.flip(valid_mask, dims=[1])
            eeg_mask = torch.flip(eeg_mask, dims=[1])
            gaze_mask = torch.flip(gaze_mask, dims=[1])

        batch, steps, _ = eeg.shape
        h = eeg.new_zeros(batch, self.eeg_proj.out_features)
        states = []
        gates = []
        for idx in range(steps):
            valid_t = valid_mask[:, idx].float().unsqueeze(-1)
            eeg_t = eeg[:, idx] * eeg_mask[:, idx].float().unsqueeze(-1)
            gaze_t = gaze[:, idx] * gaze_mask[:, idx].float().unsqueeze(-1)
            gate_in = torch.cat([gaze_t, gaze_mask[:, idx].float().unsqueeze(-1)], dim=-1)
            alpha = torch.sigmoid(self.gate(gate_in))
            candidate = torch.tanh(self.eeg_proj(eeg_t) + self.gaze_proj(gaze_t))
            h_new = alpha * h + (1.0 - alpha) * candidate
            h = valid_t * h_new + (1.0 - valid_t) * h
            states.append(h)
            gates.append(alpha.squeeze(-1))

        states_t = torch.stack(states, dim=1)
        gates_t = torch.stack(gates, dim=1)
        if reverse:
            states_t = torch.flip(states_t, dims=[1])
            gates_t = torch.flip(gates_t, dims=[1])
        return states_t, gates_t

    def forward(
        self,
        eeg: torch.Tensor,
        gaze: torch.Tensor,
        valid_mask: torch.Tensor,
        eeg_mask: torch.Tensor,
        gaze_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states_f, gates = self._scan(eeg, gaze, valid_mask, eeg_mask, gaze_mask, reverse=False)
        if self.bidirectional:
            states_b, _ = self._scan(eeg, gaze, valid_mask, eeg_mask, gaze_mask, reverse=True)
            states = torch.cat([states_f, states_b], dim=-1)
        else:
            states = states_f
        z = masked_mean(states, valid_mask)
        return z, gates, self.classifier(z)


class NOCSModel(nn.Module):
    def __init__(
        self,
        eeg_dim: int,
        gaze_dim: int,
        n_subjects: int,
        d_model: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        bidirectional: bool = True,
        ablation: str = "full",
        residual_beta: float = 0.3,
        stat_head: str = "linear",
    ) -> None:
        super().__init__()
        self.ablation = ablation
        self.residual_beta = residual_beta
        self.n_subjects = n_subjects
        self.gaze_branch = GazeBranch(gaze_dim, d_model, hidden_dim, dropout)
        self.gaze_stat_branch = GazeStatBranch(gaze_dim, hidden_dim, dropout, stat_head)
        self.eeg_branch = EEGBranch(eeg_dim, d_model, hidden_dim, dropout)
        self.controlled_state = GazeControlledState(eeg_dim, gaze_dim, hidden_dim, dropout, bidirectional)
        c_dim = hidden_dim * (2 if bidirectional else 1)
        self.c_to_hidden = nn.Linear(c_dim, hidden_dim)
        self.logvar_g = nn.Linear(hidden_dim, 1)
        self.logvar_e = nn.Linear(hidden_dim, 1)
        self.logvar_c = nn.Linear(hidden_dim, 1)
        self.full_classifier = nn.Linear(hidden_dim, 2)
        residual_dim = hidden_dim * 3
        self.residual_head = nn.Sequential(
            nn.Linear(residual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(residual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        stat_residual_dim = hidden_dim * 3
        self.stat_residual_head = nn.Sequential(
            nn.Linear(stat_residual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.stat_residual_gate = nn.Sequential(
            nn.Linear(stat_residual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        stat_full_dim = hidden_dim * 4
        self.stat_full_residual_head = nn.Sequential(
            nn.Linear(stat_full_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.stat_full_residual_gate = nn.Sequential(
            nn.Linear(stat_full_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.subject_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_subjects),
        )

    def _fusion(
        self,
        z_g: torch.Tensor,
        z_e: torch.Tensor,
        z_c: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logvar_g = self.logvar_g(z_g).clamp(-5.0, 5.0)
        logvar_e = self.logvar_e(z_e).clamp(-5.0, 5.0)
        logvar_c = self.logvar_c(z_c).clamp(-5.0, 5.0)
        precision_g = torch.exp(-logvar_g)
        precision_e = torch.exp(-logvar_e)
        precision_c = torch.exp(-logvar_c)

        if self.ablation == "no_uncertainty":
            precision_g = torch.ones_like(precision_g)
            precision_e = torch.ones_like(precision_e)
            precision_c = torch.ones_like(precision_c)
        if self.ablation == "no_eeg":
            precision_e = torch.zeros_like(precision_e)
        if self.ablation == "gaze_only":
            precision_e = torch.zeros_like(precision_e)
            precision_c = torch.zeros_like(precision_c)
        if self.ablation == "eeg_only":
            precision_g = torch.zeros_like(precision_g)
            precision_c = torch.zeros_like(precision_c)

        denom = (precision_g + precision_e + precision_c).clamp_min(1e-6)
        z_full = (precision_g * z_g + precision_e * z_e + precision_c * z_c) / denom
        return z_full, {
            "logvar_g": logvar_g,
            "logvar_e": logvar_e,
            "logvar_c": logvar_c,
            "precision_g": precision_g,
            "precision_e": precision_e,
            "precision_c": precision_c,
        }

    def forward(self, batch: dict[str, torch.Tensor], grl_scale: float = 1.0) -> dict[str, torch.Tensor]:
        eeg = batch["eeg"]
        gaze = batch["gaze"]
        valid_mask = batch["valid_mask"].bool()
        eeg_mask = valid_mask & (~batch["eeg_missing"].bool())
        gaze_mask = valid_mask & (~batch["gaze_missing"].bool())

        if self.ablation == "stat_gaze":
            z_stat, gaze_stat_feat, logits_stat = self.gaze_stat_branch(gaze, valid_mask, gaze_mask)
            zeros_one = logits_stat.new_zeros(logits_stat.shape[0], 1)
            zeros_gate = valid_mask.float().new_zeros(valid_mask.shape)
            return {
                "logits_full": logits_stat,
                "logits_stat": logits_stat,
                "z_full": z_stat,
                "z_stat": z_stat,
                "gaze_stat_feat": gaze_stat_feat,
                "gate": zeros_gate,
                "delta_logits_eeg": torch.zeros_like(logits_stat),
                "residual_gate": zeros_one,
                "residual_correction": torch.zeros_like(logits_stat),
                "subject_logits": logits_stat.new_zeros(logits_stat.shape[0], self.n_subjects),
                "precision_g": zeros_one,
                "precision_e": zeros_one,
                "precision_c": zeros_one,
            }

        if self.ablation == "stat_residual":
            z_g = gaze.new_zeros(gaze.shape[0], self.full_classifier.in_features)
            logits_g = gaze.new_zeros(gaze.shape[0], 2)
        else:
            z_g, _, logits_g = self.gaze_branch(gaze, gaze_mask)
        z_e, _, logits_e = self.eeg_branch(eeg, eeg_mask)
        if self.ablation == "no_gaze_control":
            z_c = 0.5 * (z_g + z_e)
            gates = torch.full_like(valid_mask.float(), 0.5)
            logits_c = self.full_classifier(z_c)
        else:
            z_c_raw, gates, logits_c = self.controlled_state(eeg, gaze, valid_mask, eeg_mask, gaze_mask)
            z_c = F.gelu(self.c_to_hidden(z_c_raw))

        if self.ablation in {"stat_residual", "stat_full"}:
            z_stat, gaze_stat_feat, logits_stat = self.gaze_stat_branch(gaze, valid_mask, gaze_mask)
            if self.ablation == "stat_residual":
                residual_in = torch.cat([z_e, z_c, z_stat], dim=-1)
                delta_logits_eeg = self.stat_residual_head(residual_in)
                residual_gate = torch.sigmoid(self.stat_residual_gate(residual_in))
            else:
                residual_in = torch.cat([z_stat, z_g, z_e, z_c], dim=-1)
                delta_logits_eeg = self.stat_full_residual_head(residual_in)
                residual_gate = torch.sigmoid(self.stat_full_residual_gate(residual_in))
            residual_correction = self.residual_beta * residual_gate * delta_logits_eeg
            z_full = z_stat
            logits_full = logits_stat + residual_correction
            subject_logits = self.subject_classifier(gradient_reverse(z_full, grl_scale))
            out = {
                "logits_full": logits_full,
                "logits_stat": logits_stat,
                "logits_e": logits_e,
                "logits_c": logits_c,
                "z_full": z_full,
                "z_stat": z_stat,
                "z_e": z_e,
                "z_c": z_c,
                "gaze_stat_feat": gaze_stat_feat,
                "gate": gates,
                "delta_logits_eeg": delta_logits_eeg,
                "residual_gate": residual_gate,
                "residual_correction": residual_correction,
                "subject_logits": subject_logits,
            }
            if self.ablation == "stat_full":
                out["logits_g"] = logits_g
                out["z_g"] = z_g
            out.update(
                {
                    "logvar_g": torch.zeros_like(residual_gate),
                    "logvar_e": torch.zeros_like(residual_gate),
                    "logvar_c": torch.zeros_like(residual_gate),
                    "precision_g": torch.ones_like(residual_gate),
                    "precision_e": torch.ones_like(residual_gate),
                    "precision_c": torch.ones_like(residual_gate),
                }
            )
            return out

        z_full, uncertainty = self._fusion(z_g, z_e, z_c)
        residual_in = torch.cat([z_e, z_c, z_g], dim=-1)
        delta_logits_eeg = self.residual_head(residual_in)
        residual_gate = torch.sigmoid(self.residual_gate(residual_in))
        residual_correction = self.residual_beta * residual_gate * delta_logits_eeg

        if self.ablation == "residual":
            logits_full = logits_g + residual_correction
            z_full = z_g
        elif self.ablation == "gaze_only":
            logits_full = logits_g
            z_full = z_g
        elif self.ablation == "eeg_only":
            logits_full = logits_e
            z_full = z_e
        else:
            logits_full = self.full_classifier(z_full)

        subject_logits = self.subject_classifier(gradient_reverse(z_full, grl_scale))
        out = {
            "logits_full": logits_full,
            "logits_g": logits_g,
            "logits_e": logits_e,
            "logits_c": logits_c,
            "z_full": z_full,
            "z_g": z_g,
            "z_e": z_e,
            "z_c": z_c,
            "gate": gates,
            "delta_logits_eeg": delta_logits_eeg,
            "residual_gate": residual_gate,
            "residual_correction": residual_correction,
            "subject_logits": subject_logits,
        }
        out.update(uncertainty)
        return out
