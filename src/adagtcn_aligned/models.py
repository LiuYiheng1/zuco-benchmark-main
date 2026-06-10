"""Modular AdaGTCN-aligned and CNO-GSM models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def grad_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return GradientReverse.apply(x, scale)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


class AdaptiveAdjacency(nn.Module):
    def __init__(self, n_nodes: int, rank: int = 16, temperature: float = 0.25) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.temperature = temperature
        self.left = nn.Parameter(torch.randn(n_nodes, rank) * 0.02)
        self.right = nn.Parameter(torch.randn(rank, n_nodes) * 0.02)
        self.log_self_loop = nn.Parameter(torch.tensor(0.0))

    def forward(self) -> torch.Tensor:
        scores = torch.relu(self.left @ self.right)
        scores = scores / max(self.temperature, 1e-4)
        eye = torch.eye(self.n_nodes, device=scores.device)
        scores = scores + eye * self.log_self_loop.exp()
        adj = torch.softmax(scores, dim=-1)
        return adj

    def smoothness_loss(self, x: torch.Tensor) -> torch.Tensor:
        adj = self.forward()
        deg = torch.diag(adj.sum(dim=-1))
        lap = deg - adj
        lx = torch.einsum("nm,btnh->btmh", lap, x)
        return (x * lx).mean()

    def entropy_loss(self) -> torch.Tensor:
        adj = self.forward().clamp_min(1e-8)
        return -(adj * adj.log()).sum(dim=-1).mean()


class GraphConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("nm,btmh->btnh", adj, x)
        out = self.lin(agg)
        out = F.gelu(out)
        out = self.dropout(out)
        return self.norm(x + out)


class EEGGraphEncoder(nn.Module):
    def __init__(self, eeg_dim: int, hidden_dim: int, n_eeg_nodes: int, dropout: float) -> None:
        super().__init__()
        if eeg_dim >= n_eeg_nodes and eeg_dim % n_eeg_nodes == 0:
            self.n_nodes = n_eeg_nodes
            self.node_in_dim = eeg_dim // n_eeg_nodes
        else:
            self.n_nodes = 1
            self.node_in_dim = max(eeg_dim, 1)
        self.input = nn.Linear(self.node_in_dim, hidden_dim)
        self.adj = AdaptiveAdjacency(self.n_nodes)
        self.gcn1 = GraphConvBlock(hidden_dim, dropout)
        self.gcn2 = GraphConvBlock(hidden_dim, dropout)
        self.pool = nn.Linear(hidden_dim, 1)

    def reshape_nodes(self, eeg: torch.Tensor) -> torch.Tensor:
        bsz, steps, dim = eeg.shape
        if self.n_nodes == 1:
            if dim == 0:
                return eeg.new_zeros((bsz, steps, 1, self.node_in_dim))
            return eeg.view(bsz, steps, 1, dim)
        return eeg.view(bsz, steps, self.n_nodes, self.node_in_dim)

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        nodes = self.input(self.reshape_nodes(eeg))
        adj = self.adj()
        nodes = self.gcn1(nodes, adj)
        nodes = self.gcn2(nodes, adj)
        weights = torch.softmax(self.pool(nodes).squeeze(-1), dim=-1)
        pooled = (nodes * weights.unsqueeze(-1)).sum(dim=2)
        aux = {
            "graph_smooth": self.adj.smoothness_loss(nodes),
            "graph_entropy": self.adj.entropy_loss(),
        }
        return pooled, nodes, aux


class BipartiteNeuroOculomotorGraph(nn.Module):
    def __init__(self, hidden_dim: int, n_eeg_nodes: int, gaze_dim: int, dropout: float) -> None:
        super().__init__()
        self.n_nodes = n_eeg_nodes + 1
        self.gaze_node = nn.Linear(gaze_dim, hidden_dim)
        self.gaze_to_eeg_gate = nn.Linear(gaze_dim, n_eeg_nodes)
        self.adj = AdaptiveAdjacency(self.n_nodes)
        self.gcn = GraphConvBlock(hidden_dim, dropout)
        self.pool = nn.Linear(hidden_dim, 1)

    def forward(self, eeg_nodes: torch.Tensor, gaze: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gate = torch.sigmoid(self.gaze_to_eeg_gate(gaze)).unsqueeze(-1)
        gated_eeg = eeg_nodes * gate
        gaze_node = self.gaze_node(gaze).unsqueeze(2)
        nodes = torch.cat([gated_eeg, gaze_node], dim=2)
        adj = self.adj()
        nodes = self.gcn(nodes, adj)
        weights = torch.softmax(self.pool(nodes).squeeze(-1), dim=-1)
        pooled = (nodes * weights.unsqueeze(-1)).sum(dim=2)
        aux = {
            "bipartite_smooth": self.adj.smoothness_loss(nodes),
            "bipartite_entropy": self.adj.entropy_loss(),
        }
        return pooled, aux


class TemporalConvEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, step_mask: torch.Tensor, gaze: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = F.gelu(self.conv1(y))
        y = self.dropout(F.gelu(self.conv2(y))).transpose(1, 2)
        y = self.norm(y + residual)
        return y * step_mask.unsqueeze(-1)


class GazeControlledStateSpace(nn.Module):
    def __init__(self, hidden_dim: int, gaze_dim: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.h_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gaze_control = nn.Linear(gaze_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim + hidden_dim + gaze_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, step_mask: torch.Tensor, gaze: torch.Tensor) -> torch.Tensor:
        bsz, steps, hidden = x.shape
        h = x.new_zeros((bsz, hidden))
        outputs = []
        for idx in range(steps):
            xt = x[:, idx]
            gt = gaze[:, idx]
            mt = step_mask[:, idx].unsqueeze(-1)
            control = torch.sigmoid(self.gaze_control(gt))
            candidate = torch.tanh(self.in_proj(xt) + self.h_proj(h * control))
            update = torch.sigmoid(self.update(torch.cat([xt, h, gt], dim=-1)))
            new_h = (1.0 - update) * h + update * candidate
            h = mt * new_h + (1.0 - mt) * h
            outputs.append(h.unsqueeze(1))
        y = torch.cat(outputs, dim=1)
        return self.norm(self.dropout(y) + x) * step_mask.unsqueeze(-1)


class CommonUniqueDisentangler(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.eeg_common = nn.Linear(hidden_dim, hidden_dim)
        self.gaze_common = nn.Linear(hidden_dim, hidden_dim)
        self.eeg_private = nn.Linear(hidden_dim, hidden_dim)
        self.gaze_private = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, eeg_h: torch.Tensor, gaze_h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ec = F.normalize(self.eeg_common(eeg_h), dim=-1)
        gc = F.normalize(self.gaze_common(gaze_h), dim=-1)
        ep = F.normalize(self.eeg_private(eeg_h), dim=-1)
        gp = F.normalize(self.gaze_private(gaze_h), dim=-1)
        valid = mask.unsqueeze(-1)

        common = 0.5 * (ec + gc)
        fused = self.out(torch.cat([common, ep, gp], dim=-1))

        align = (1.0 - (ec * gc).sum(dim=-1)) * mask
        align_loss = align.sum() / mask.sum().clamp_min(1.0)
        decor = ((ec * ep).sum(dim=-1).abs() + (gc * gp).sum(dim=-1).abs()) * mask
        decor_loss = decor.sum() / mask.sum().clamp_min(1.0)
        return fused * valid, {"common_align": align_loss, "unique_decor": decor_loss}


class SubjectInvariantBridge(nn.Module):
    def __init__(self, hidden_dim: int, n_subjects: int, eeg_dim: int, gaze_dim: int) -> None:
        super().__init__()
        self.to_shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.subject_head = nn.Linear(hidden_dim, n_subjects)
        self.reconstruct_eeg = nn.Linear(hidden_dim, max(eeg_dim, 1))
        self.reconstruct_gaze = nn.Linear(hidden_dim, gaze_dim)

    def forward(
        self,
        seq_h: torch.Tensor,
        pooled: torch.Tensor,
        eeg_summary: torch.Tensor,
        gaze_summary: torch.Tensor,
        grl_scale: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        shared_seq = self.to_shared(seq_h)
        shared_pool = self.to_shared(pooled)
        subject_logits = self.subject_head(grad_reverse(shared_pool, grl_scale))
        eeg_rec = self.reconstruct_eeg(shared_pool)
        gaze_rec = self.reconstruct_gaze(shared_pool)
        eeg_target = eeg_summary
        if eeg_target.shape[-1] == 0:
            eeg_loss = None
        else:
            eeg_loss = F.mse_loss(eeg_rec[:, : eeg_target.shape[-1]], eeg_target)
        gaze_loss = F.mse_loss(gaze_rec, gaze_summary)
        return shared_seq, {"subject_logits": subject_logits, "bridge_eeg_recon": eeg_loss, "bridge_gaze_recon": gaze_loss}


class OculomotorControlledSSM(nn.Module):
    def __init__(self, eeg_dim: int, gaze_dim: int, hidden_dim: int, dropout: float, use_gaze_control: bool = True) -> None:
        super().__init__()
        self.use_gaze_control = use_gaze_control
        self.gaze_encoder = nn.Sequential(nn.Linear(gaze_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.eeg_encoder = nn.Sequential(nn.Linear(max(eeg_dim, 1), hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.transition = nn.Linear(hidden_dim, hidden_dim * 3)
        self.static_transition = nn.Parameter(torch.zeros(hidden_dim * 3))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gaze: torch.Tensor, eeg: torch.Tensor, step_mask: torch.Tensor, gaze_mask: torch.Tensor, eeg_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, steps, _ = gaze.shape
        hidden = self.static_transition.shape[0] // 3
        z = gaze.new_zeros((bsz, hidden))
        outputs = []
        gaze_h = self.gaze_encoder(gaze) * gaze_mask.unsqueeze(-1)
        eeg_h = self.eeg_encoder(eeg) * eeg_mask.unsqueeze(-1)

        for idx in range(steps):
            mt = step_mask[:, idx].unsqueeze(-1)
            if self.use_gaze_control:
                params = self.transition(gaze_h[:, idx])
            else:
                params = self.static_transition.unsqueeze(0).expand(bsz, -1)
            a_t, b_t, gate_t = params.chunk(3, dim=-1)
            a_t = torch.tanh(a_t)
            gate_t = torch.sigmoid(gate_t)
            candidate = torch.tanh(a_t * z + b_t * eeg_h[:, idx])
            new_z = gate_t * z + (1.0 - gate_t) * candidate
            z = mt * new_z + (1.0 - mt) * z
            outputs.append(z.unsqueeze(1))

        prior = torch.cat(outputs, dim=1)
        return self.norm(self.dropout(prior)), eeg_h, gaze_h


class SubjectInvariantManifoldRectifier(nn.Module):
    def __init__(self, hidden_dim: int, n_subjects: int, dropout: float) -> None:
        super().__init__()
        self.cognitive = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.subject_specific = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.subject_head = nn.Linear(hidden_dim, n_subjects)

    def forward(self, eeg_h: torch.Tensor, mask: torch.Tensor, grl_scale: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_c = self.cognitive(eeg_h) * mask.unsqueeze(-1)
        h_s = self.subject_specific(eeg_h) * mask.unsqueeze(-1)
        pooled_c = masked_mean(h_c, mask)
        subject_logits = self.subject_head(grad_reverse(pooled_c, grl_scale))

        denom = mask.sum().clamp_min(1.0)
        cross = torch.einsum("bth,btd->hd", h_c, h_s) / denom
        orth_loss = torch.square(cross).mean()
        aux = {
            "subject_logits": subject_logits,
            "aion_orth": orth_loss,
        }
        return h_c, aux


class PrecisionWeightedFreeEnergyFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, use_precision: bool = True) -> None:
        super().__init__()
        self.use_precision = use_precision
        self.gaze_precision = nn.Linear(hidden_dim, 1)
        self.eeg_precision = nn.Linear(hidden_dim, 1)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.constant_(self.eeg_precision.bias, -2.0)
        nn.init.constant_(self.gaze_precision.bias, 1.0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z_prior: torch.Tensor, h_c: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.use_precision:
            tau_g = F.softplus(self.gaze_precision(z_prior))
            tau_e = F.softplus(self.eeg_precision(h_c))
            alpha = tau_e / (tau_e + tau_g + 1e-6)
        else:
            alpha = torch.full_like(z_prior[..., :1], 0.5)
        residual = self.residual(h_c - z_prior)
        z = self.norm(z_prior + alpha * residual) * mask.unsqueeze(-1)
        valid_alpha = alpha.squeeze(-1) * mask
        denom = mask.sum().clamp_min(1.0)
        alpha_mean = valid_alpha.sum() / denom
        alpha_std = torch.sqrt(torch.square((alpha.squeeze(-1) - alpha_mean) * mask).sum() / denom)
        return z, {"aion_alpha_mean": alpha_mean.detach(), "aion_alpha_std": alpha_std.detach()}


class AION(nn.Module):
    def __init__(
        self,
        eeg_dim: int,
        gaze_dim: int,
        n_subjects: int,
        hidden_dim: int,
        dropout: float,
        use_manifold: bool = True,
        use_precision: bool = True,
        use_gaze_control: bool = True,
    ) -> None:
        super().__init__()
        self.use_manifold = use_manifold
        self.state = OculomotorControlledSSM(eeg_dim, gaze_dim, hidden_dim, dropout, use_gaze_control=use_gaze_control)
        self.manifold = SubjectInvariantManifoldRectifier(hidden_dim, n_subjects, dropout) if use_manifold else None
        self.eeg_rectifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.fusion = PrecisionWeightedFreeEnergyFusion(hidden_dim, dropout, use_precision=use_precision)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch: dict[str, torch.Tensor], grl_scale: float = 1.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eeg = batch["eeg"]
        gaze = batch["gaze"]
        step_mask = batch["step_mask"]
        eeg_mask = batch["eeg_mask"]
        gaze_mask = batch["gaze_mask"]
        valid_eeg = (step_mask * eeg_mask).clamp(max=1.0)
        valid_gaze = (step_mask * gaze_mask).clamp(max=1.0)

        z_prior, eeg_h, _ = self.state(gaze, eeg, step_mask, valid_gaze, valid_eeg)
        aux: dict[str, torch.Tensor] = {}
        if self.manifold is not None:
            h_c, manifold_aux = self.manifold(eeg_h, valid_eeg, grl_scale)
            aux.update(manifold_aux)
        else:
            h_c = self.eeg_rectifier(eeg_h) * valid_eeg.unsqueeze(-1)

        z, fusion_aux = self.fusion(z_prior, h_c, step_mask)
        aux.update(fusion_aux)
        pooled = masked_mean(z, step_mask)
        logits = self.classifier(pooled)
        return logits, aux


class AIONV2(nn.Module):
    def __init__(
        self,
        eeg_dim: int,
        gaze_dim: int,
        hidden_dim: int,
        n_eeg_nodes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gaze_input = nn.Sequential(
            nn.Linear(gaze_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.gaze_prior = GazeControlledStateSpace(hidden_dim, gaze_dim, dropout)
        self.eeg_encoder = EEGGraphEncoder(eeg_dim, hidden_dim, n_eeg_nodes, dropout)
        self.eeg_residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.eeg_residual[-1].weight)
        nn.init.zeros_(self.eeg_residual[-1].bias)
        self.precision_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        nn.init.constant_(self.precision_gate.bias, -3.0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch: dict[str, torch.Tensor], grl_scale: float = 1.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eeg = batch["eeg"]
        gaze = batch["gaze"]
        step_mask = batch["step_mask"]
        eeg_mask = batch["eeg_mask"]
        gaze_mask = batch["gaze_mask"]

        gaze_input = self.gaze_input(gaze) * (step_mask * gaze_mask).clamp(max=1.0).unsqueeze(-1)
        z_g = self.gaze_prior(gaze_input, step_mask, gaze)

        h_e, _, eeg_aux = self.eeg_encoder(eeg)
        h_e = h_e * (step_mask * eeg_mask).clamp(max=1.0).unsqueeze(-1)
        delta_e = self.eeg_residual(h_e)
        gate_input = torch.cat([z_g, h_e, torch.abs(z_g - h_e)], dim=-1)
        alpha = torch.sigmoid(self.precision_gate(gate_input))
        z = self.norm(z_g + alpha * delta_e) * step_mask.unsqueeze(-1)
        pooled = masked_mean(z, step_mask)
        logits = self.classifier(pooled)

        denom = step_mask.sum().clamp_min(1.0)
        alpha_valid = alpha * step_mask.unsqueeze(-1)
        alpha_mean = alpha_valid.sum() / (denom * alpha.shape[-1])
        alpha_std = torch.sqrt(torch.square((alpha - alpha_mean) * step_mask.unsqueeze(-1)).sum() / (denom * alpha.shape[-1]))
        aux = {
            "aion_v2_alpha_mean": alpha_mean.detach(),
            "aion_v2_alpha_std": alpha_std.detach(),
            "aion_v2_delta_norm": (delta_e.norm(dim=-1) * step_mask).sum().detach() / denom,
            "aion_v2_z_g_norm": (z_g.norm(dim=-1) * step_mask).sum().detach() / denom,
            "aion_v2_z_final_norm": (z.norm(dim=-1) * step_mask).sum().detach() / denom,
        }
        aux.update(eeg_aux)
        return logits, aux


@dataclass
class ModelConfig:
    eeg_dim: int
    gaze_dim: int
    n_subjects: int
    hidden_dim: int = 64
    n_eeg_nodes: int = 105
    dropout: float = 0.2
    temporal: str = "tcn"
    use_eeg: bool = True
    use_gaze: bool = True
    use_bipartite: bool = False
    use_common_unique: bool = False
    use_subject_bridge: bool = False
    architecture: str = "cnogsm"
    use_manifold: bool = True
    use_precision: bool = True
    use_gaze_control: bool = True


class CNOGSM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.eeg_encoder = EEGGraphEncoder(config.eeg_dim, config.hidden_dim, config.n_eeg_nodes, config.dropout)
        n_nodes = self.eeg_encoder.n_nodes
        self.gaze_encoder = nn.Sequential(
            nn.Linear(config.gaze_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        if config.use_bipartite:
            self.bipartite = BipartiteNeuroOculomotorGraph(config.hidden_dim, n_nodes, config.gaze_dim, config.dropout)
        else:
            self.bipartite = None

        fusion_in = config.hidden_dim * 2
        self.fusion = nn.Sequential(nn.Linear(fusion_in, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim))
        if config.temporal == "gaze_ssm":
            self.temporal = GazeControlledStateSpace(config.hidden_dim, config.gaze_dim, config.dropout)
        else:
            self.temporal = TemporalConvEncoder(config.hidden_dim, config.dropout)

        self.disentangler = CommonUniqueDisentangler(config.hidden_dim) if config.use_common_unique else None
        self.bridge = (
            SubjectInvariantBridge(config.hidden_dim, config.n_subjects, config.eeg_dim, config.gaze_dim)
            if config.use_subject_bridge
            else None
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2),
        )

    def forward(self, batch: dict[str, torch.Tensor], grl_scale: float = 1.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        eeg = batch["eeg"]
        gaze = batch["gaze"]
        step_mask = batch["step_mask"]
        eeg_mask = batch["eeg_mask"]
        gaze_mask = batch["gaze_mask"]

        eeg_h, eeg_nodes, aux = self.eeg_encoder(eeg)
        gaze_h = self.gaze_encoder(gaze)

        if not self.config.use_eeg:
            eeg_h = torch.zeros_like(eeg_h)
            eeg_nodes = torch.zeros_like(eeg_nodes)

        if self.bipartite is not None:
            graph_h, graph_aux = self.bipartite(eeg_nodes, gaze)
            aux.update(graph_aux)
            eeg_h = graph_h

        if not self.config.use_gaze:
            gaze_h = torch.zeros_like(gaze_h)

        if self.disentangler is not None:
            fusion_h, disent_aux = self.disentangler(eeg_h, gaze_h, (eeg_mask * gaze_mask * step_mask).clamp(max=1.0))
            aux.update(disent_aux)
        else:
            fusion_h = self.fusion(torch.cat([eeg_h, gaze_h], dim=-1))

        temporal_gaze = gaze if self.config.use_gaze else torch.zeros_like(gaze)
        seq_h = self.temporal(fusion_h, step_mask, temporal_gaze)
        pooled = masked_mean(seq_h, step_mask)

        if self.bridge is not None:
            eeg_summary = masked_mean(eeg, eeg_mask)
            gaze_summary = masked_mean(gaze, gaze_mask)
            seq_h, bridge_aux = self.bridge(seq_h, pooled, eeg_summary, gaze_summary, grl_scale)
            pooled = masked_mean(seq_h, step_mask)
            for key, value in bridge_aux.items():
                if value is not None:
                    aux[key] = value

        logits = self.classifier(pooled)
        return logits, aux


def build_model(name: str, eeg_dim: int, gaze_dim: int, n_subjects: int, hidden_dim: int, n_eeg_nodes: int, dropout: float) -> nn.Module:
    presets = {
        "adagtcn_aligned": dict(temporal="tcn", use_eeg=True, use_gaze=True, use_bipartite=False, use_common_unique=False, use_subject_bridge=False),
        "eeg_only_graph_tcn": dict(temporal="tcn", use_eeg=True, use_gaze=False, use_bipartite=False, use_common_unique=False, use_subject_bridge=False),
        "eeg_graph_ssm": dict(temporal="gaze_ssm", use_eeg=True, use_gaze=False, use_bipartite=False, use_common_unique=False, use_subject_bridge=False),
        "gaze_only_ssm": dict(temporal="gaze_ssm", use_eeg=False, use_gaze=True, use_bipartite=False, use_common_unique=False, use_subject_bridge=False),
        "gaze_control_ssm": dict(temporal="gaze_ssm", use_eeg=True, use_gaze=True, use_bipartite=False, use_common_unique=False, use_subject_bridge=False),
        "bipartite_graph_ssm": dict(temporal="gaze_ssm", use_eeg=True, use_gaze=True, use_bipartite=True, use_common_unique=False, use_subject_bridge=False),
        "bridge_bipartite_ssm": dict(temporal="gaze_ssm", use_eeg=True, use_gaze=True, use_bipartite=True, use_common_unique=False, use_subject_bridge=True),
        "full_cnogsm": dict(temporal="gaze_ssm", use_eeg=True, use_gaze=True, use_bipartite=True, use_common_unique=True, use_subject_bridge=True),
        "aion": dict(architecture="aion", use_manifold=True, use_precision=True, use_gaze_control=True),
        "aion_no_manifold": dict(architecture="aion", use_manifold=False, use_precision=True, use_gaze_control=True),
        "aion_no_precision": dict(architecture="aion", use_manifold=True, use_precision=False, use_gaze_control=True),
        "aion_no_gaze_control": dict(architecture="aion", use_manifold=True, use_precision=True, use_gaze_control=False),
        "aion_v2": dict(architecture="aion_v2"),
    }
    if name not in presets:
        raise ValueError("Unknown model %s. Available: %s" % (name, sorted(presets)))
    config = ModelConfig(
        eeg_dim=eeg_dim,
        gaze_dim=gaze_dim,
        n_subjects=n_subjects,
        hidden_dim=hidden_dim,
        n_eeg_nodes=n_eeg_nodes,
        dropout=dropout,
        **presets[name],
    )
    if config.architecture == "aion":
        return AION(
            eeg_dim=config.eeg_dim,
            gaze_dim=config.gaze_dim,
            n_subjects=config.n_subjects,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            use_manifold=config.use_manifold,
            use_precision=config.use_precision,
            use_gaze_control=config.use_gaze_control,
        )
    if config.architecture == "aion_v2":
        return AIONV2(
            eeg_dim=config.eeg_dim,
            gaze_dim=config.gaze_dim,
            hidden_dim=config.hidden_dim,
            n_eeg_nodes=config.n_eeg_nodes,
            dropout=config.dropout,
        )
    return CNOGSM(config)
