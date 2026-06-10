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
    mono_margin: float,
    uncert_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce_full = F.cross_entropy(outputs["logits_full"], labels, weight=class_weights, reduction="none")
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
    total = (
        cls_loss
        + lambda_mono * mono_loss
        + lambda_adv * subject_adv
        + lambda_uncert * uncert
        + lambda_supcon * supcon
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "mono_loss": float(mono_loss.detach().cpu()),
        "subject_adv_loss": float(subject_adv.detach().cpu()),
        "uncertainty_penalty": float(uncert.detach().cpu()),
        "supcon_loss": float(supcon.detach().cpu()),
        "precision_g": float(outputs["precision_g"].detach().mean().cpu()),
        "precision_e": float(outputs["precision_e"].detach().mean().cpu()),
        "precision_c": float(outputs["precision_c"].detach().mean().cpu()),
        "gate_mean": float(outputs["gate"].detach().mean().cpu()),
    }
    return total, logs
