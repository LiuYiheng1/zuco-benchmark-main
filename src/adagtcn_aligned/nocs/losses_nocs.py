"""Losses for NOCS training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def branch_weights(ablation: str) -> dict[str, float]:
    if ablation == "gaze_only":
        return {"g": 0.5, "e": 0.0, "c": 0.0}
    if ablation == "eeg_only":
        return {"g": 0.0, "e": 0.3, "c": 0.0}
    if ablation == "no_eeg":
        return {"g": 0.5, "e": 0.0, "c": 0.5}
    if ablation == "residual":
        return {"g": 0.5, "e": 0.3, "c": 0.0}
    return {"g": 0.5, "e": 0.3, "c": 0.5}


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    subjects: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    if features.shape[0] < 2:
        return features.new_tensor(0.0)
    z = F.normalize(features, dim=-1)
    logits = torch.matmul(z, z.T) / temperature
    eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    positive = (labels[:, None] == labels[None, :]) & (subjects[:, None] != subjects[None, :]) & (~eye)
    if not positive.any():
        return features.new_tensor(0.0)
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = positive.float().sum(dim=1).clamp_min(1.0)
    loss = -(log_prob * positive.float()).sum(dim=1) / pos_count
    valid_rows = positive.any(dim=1)
    return loss[valid_rows].mean()


def uncertainty_penalty(outputs: dict[str, torch.Tensor], margin: float) -> torch.Tensor:
    precision_e = outputs["precision_e"]
    precision_g = outputs["precision_g"]
    return torch.relu(precision_e - precision_g - margin).mean()


def residual_penalties(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    residual_norm = outputs["residual_correction"].pow(2).sum(dim=-1).mean()
    gate_penalty = outputs["residual_gate"].mean()
    return residual_norm, gate_penalty


def nocs_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    subjects: torch.Tensor,
    class_weights: torch.Tensor | None,
    ablation: str,
    lambda_mono: float,
    lambda_adv: float,
    lambda_uncert: float,
    lambda_supcon: float,
    lambda_residual_norm: float,
    lambda_gate: float,
    mono_margin: float,
    uncert_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if ablation == "stat_gaze":
        ce_stat = F.cross_entropy(outputs["logits_stat"], labels, weight=class_weights)
        logs = {
            "loss": float(ce_stat.detach().cpu()),
            "cls_loss": float(ce_stat.detach().cpu()),
            "stat_loss": float(ce_stat.detach().cpu()),
            "mono_loss": 0.0,
            "subject_adv_loss": 0.0,
            "uncertainty_penalty": 0.0,
            "supcon_loss": 0.0,
            "residual_norm_penalty": 0.0,
            "residual_gate_penalty": 0.0,
            "precision_g": 0.0,
            "precision_e": 0.0,
            "precision_c": 0.0,
            "gate_mean": 0.0,
            "residual_gate_mean": 0.0,
            "residual_correction_norm": 0.0,
        }
        return ce_stat, logs

    ce_full = F.cross_entropy(outputs["logits_full"], labels, weight=class_weights, reduction="none")
    if ablation in {"stat_residual", "stat_full"}:
        ce_stat = F.cross_entropy(outputs["logits_stat"], labels, weight=class_weights, reduction="none")
        ce_e = F.cross_entropy(outputs["logits_e"], labels, weight=class_weights, reduction="none")
        cls_loss = ce_full.mean() + 0.5 * ce_stat.mean() + 0.3 * ce_e.mean()
        if ablation == "stat_full":
            ce_g = F.cross_entropy(outputs["logits_g"], labels, weight=class_weights, reduction="none")
            cls_loss = cls_loss + 0.3 * ce_g.mean()
        residual_norm, gate_penalty = residual_penalties(outputs)
        total = cls_loss + lambda_residual_norm * residual_norm + lambda_gate * gate_penalty
        logs = {
            "loss": float(total.detach().cpu()),
            "cls_loss": float(cls_loss.detach().cpu()),
            "stat_loss": float(ce_stat.mean().detach().cpu()),
            "mono_loss": 0.0,
            "subject_adv_loss": 0.0,
            "uncertainty_penalty": 0.0,
            "supcon_loss": 0.0,
            "residual_norm_penalty": float(residual_norm.detach().cpu()),
            "residual_gate_penalty": float(gate_penalty.detach().cpu()),
            "precision_g": float(outputs["precision_g"].detach().mean().cpu()),
            "precision_e": float(outputs["precision_e"].detach().mean().cpu()),
            "precision_c": float(outputs["precision_c"].detach().mean().cpu()),
            "gate_mean": float(outputs["gate"].detach().mean().cpu()),
            "residual_gate_mean": float(outputs["residual_gate"].detach().mean().cpu()),
            "residual_correction_norm": float(outputs["residual_correction"].detach().norm(dim=-1).mean().cpu()),
        }
        return total, logs

    ce_g = F.cross_entropy(outputs["logits_g"], labels, weight=class_weights, reduction="none")
    ce_e = F.cross_entropy(outputs["logits_e"], labels, weight=class_weights, reduction="none")
    ce_c = F.cross_entropy(outputs["logits_c"], labels, weight=class_weights, reduction="none")
    weights = branch_weights(ablation)
    cls_loss = ce_full.mean()
    cls_loss = cls_loss + weights["g"] * ce_g.mean()
    cls_loss = cls_loss + weights["e"] * ce_e.mean()
    cls_loss = cls_loss + weights["c"] * ce_c.mean()

    mono_loss = torch.relu(ce_full - ce_g + mono_margin).mean()
    if ablation in {"gaze_only", "eeg_only"}:
        mono_loss = mono_loss * 0.0

    subject_adv = F.cross_entropy(outputs["subject_logits"], subjects)
    if ablation == "no_adv":
        subject_adv = subject_adv * 0.0

    uncert = uncertainty_penalty(outputs, uncert_margin)
    if ablation == "no_uncertainty":
        uncert = uncert * 0.0

    supcon = supervised_contrastive_loss(outputs["z_full"], labels, subjects)
    residual_norm, gate_penalty = residual_penalties(outputs)
    if ablation != "residual":
        residual_norm = residual_norm * 0.0
        gate_penalty = gate_penalty * 0.0

    total = (
        cls_loss
        + lambda_mono * mono_loss
        + lambda_adv * subject_adv
        + lambda_uncert * uncert
        + lambda_supcon * supcon
        + lambda_residual_norm * residual_norm
        + lambda_gate * gate_penalty
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "mono_loss": float(mono_loss.detach().cpu()),
        "subject_adv_loss": float(subject_adv.detach().cpu()),
        "uncertainty_penalty": float(uncert.detach().cpu()),
        "supcon_loss": float(supcon.detach().cpu()),
        "residual_norm_penalty": float(residual_norm.detach().cpu()),
        "residual_gate_penalty": float(gate_penalty.detach().cpu()),
        "precision_g": float(outputs["precision_g"].detach().mean().cpu()),
        "precision_e": float(outputs["precision_e"].detach().mean().cpu()),
        "precision_c": float(outputs["precision_c"].detach().mean().cpu()),
        "gate_mean": float(outputs["gate"].detach().mean().cpu()),
        "residual_gate_mean": float(outputs["residual_gate"].detach().mean().cpu()),
        "residual_correction_norm": float(outputs["residual_correction"].detach().norm(dim=-1).mean().cpu()),
    }
    return total, logs
