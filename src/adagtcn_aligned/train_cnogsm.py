"""Train AdaGTCN-aligned CNO-GSM ablations."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.adagtcn_aligned.dataset import load_split, make_datasets
from src.adagtcn_aligned.models import build_model


ABLATIONS = [
    "eeg_only_graph_tcn",
    "eeg_graph_ssm",
    "gaze_only_ssm",
    "adagtcn_aligned",
    "gaze_control_ssm",
    "bipartite_graph_ssm",
    "bridge_bipartite_ssm",
    "full_cnogsm",
]

AION_MODELS = [
    "aion",
    "aion_v2",
    "aion_no_manifold",
    "aion_no_precision",
    "aion_no_gaze_control",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def compute_aux_loss(aux: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    device = batch["y"].device
    total = torch.tensor(0.0, device=device)
    logs = {}
    if "subject_logits" in aux:
        loss = F.cross_entropy(aux["subject_logits"], batch["subject"])
        total = total + weights["subject_adv"] * loss
        logs["subject_adv"] = float(loss.detach().cpu())
    for key, weight_name in [
        ("bridge_eeg_recon", "bridge_recon"),
        ("bridge_gaze_recon", "bridge_recon"),
        ("common_align", "common_align"),
        ("unique_decor", "unique_decor"),
        ("graph_smooth", "graph_smooth"),
        ("bipartite_smooth", "graph_smooth"),
        ("graph_entropy", "graph_entropy"),
        ("bipartite_entropy", "graph_entropy"),
        ("aion_orth", "unique_decor"),
    ]:
        if key in aux:
            loss = aux[key]
            total = total + weights[weight_name] * loss
            logs[key] = float(loss.detach().cpu())
    return total, logs


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            logits, _ = model(batch, grl_scale=0.0)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            prob = torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
            pred = torch.argmax(logits, dim=-1)
            y_true.extend(batch["y"].cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())

    if len(set(y_true)) < 2:
        auroc = 0.5
    else:
        auroc = roc_auc_score(y_true, y_prob)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": float(auroc),
        "n": int(len(y_true)),
    }


def train_one_model(args: argparse.Namespace, model_name: str, datasets: tuple[Any, Any, Any, dict[str, Any]]) -> dict[str, Any]:
    train_ds, val_ds, test_ds, meta = datasets
    device = torch.device(args.device)
    if val_ds is None or meta["n_val"] == 0:
        raise RuntimeError("Validation set is empty. Refusing to use test set for validation.")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(
        model_name,
        eeg_dim=meta["eeg_dim"],
        gaze_dim=meta["gaze_dim"],
        n_subjects=len(meta["subject_to_idx"]),
        hidden_dim=args.hidden_dim,
        n_eeg_nodes=args.n_eeg_nodes,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = {
        "subject_adv": args.w_subject_adv,
        "bridge_recon": args.w_bridge_recon,
        "common_align": args.w_common_align,
        "unique_decor": args.w_unique_decor,
        "graph_smooth": args.w_graph_smooth,
        "graph_entropy": args.w_graph_entropy,
    }

    best_val = -1.0
    best_state = None
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        aux_logs: dict[str, list[float]] = {}
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            grl_scale = min(1.0, float(epoch) / max(args.epochs // 2, 1))
            logits, aux = model(batch, grl_scale=grl_scale)
            ce = F.cross_entropy(logits, batch["y"])
            aux_loss, batch_aux_logs = compute_aux_loss(aux, batch, weights)
            loss = ce + aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in batch_aux_logs.items():
                aux_logs.setdefault(key, []).append(value)
            for key, value in aux.items():
                if key in batch_aux_logs or key == "subject_logits" or not torch.is_tensor(value) or value.numel() != 1:
                    continue
                aux_logs.setdefault(key, []).append(float(value.detach().cpu()))

        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0}
        for key, values in aux_logs.items():
            row[key] = float(np.mean(values)) if values else 0.0
        row.update({"val_" + k: v for k, v in val_metrics.items()})
        history.append(row)

        score = val_metrics["macro_f1"]
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    result = {
        "model": model_name,
        "protocol": args.protocol,
        "seed": args.seed,
        "best_val_macro_f1": best_val,
        "train_examples": meta["n_train"],
        "val_examples": meta["n_val"],
        "test_examples": meta["n_test"],
        "eeg_dim": meta["eeg_dim"],
        "gaze_dim": meta["gaze_dim"],
        "debug_random_split": meta["debug_random_split"],
    }
    result.update({"val_" + k: v for k, v in val_metrics.items()})
    result.update({"test_" + k: v for k, v in test_metrics.items()})

    history_path = args.output_dir / ("%s_history_seed%d.csv" % (model_name, args.seed))
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(history[0].keys()) if history else ["epoch"])
        writer.writeheader()
        writer.writerows(history)
    return result


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AdaGTCN-aligned CNO-GSM ablations.")
    parser.add_argument("--sequence-jsonl", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=Path("reports/adagtcn_aligned/subject_splits.json"))
    parser.add_argument("--protocol", default="Y16_12_2_2_seed0")
    parser.add_argument("--model", default="full_cnogsm", choices=ABLATIONS + AION_MODELS + ["all"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/adagtcn_aligned"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-eeg-nodes", type=int, default=105)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--w-subject-adv", type=float, default=0.05)
    parser.add_argument("--w-bridge-recon", type=float, default=0.02)
    parser.add_argument("--w-common-align", type=float, default=0.05)
    parser.add_argument("--w-unique-decor", type=float, default=0.01)
    parser.add_argument("--w-graph-smooth", type=float, default=0.001)
    parser.add_argument("--w-graph-entropy", type=float, default=0.0001)
    parser.add_argument("--debug-random-split", action="store_true")
    parser.add_argument("--cache-records", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(args.split_json, args.protocol)
    datasets = make_datasets(
        args.sequence_jsonl,
        split,
        max_len=args.max_len,
        debug_random_split=args.debug_random_split,
        cache_records=args.cache_records,
    )
    _, _, _, meta = datasets
    if meta["n_train"] == 0 or meta["n_test"] == 0:
        raise RuntimeError(
            "No train/test examples for this protocol and JSONL. "
            "Use a sequence file containing all split subjects, or use --debug-random-split only for smoke tests."
        )

    models = ABLATIONS if args.model == "all" else [args.model]
    rows = []
    for model_name in models:
        print("Training %s" % model_name, flush=True)
        rows.append(train_one_model(args, model_name, datasets))
        print(json.dumps(rows[-1], indent=2), flush=True)

    out_path = args.output_dir / ("cnogsm_%s_seed%d.csv" % (args.model, args.seed))
    write_results(rows, out_path)
    meta_path = args.output_dir / ("cnogsm_%s_seed%d_meta.json" % (args.model, args.seed))
    meta_dump = dict(meta)
    meta_dump["subject_to_idx"] = meta["subject_to_idx"]
    meta_dump["protocol_note"] = split.note
    meta_dump["sequence_jsonl"] = str(args.sequence_jsonl)
    meta_path.write_text(json.dumps(meta_dump, indent=2), encoding="utf-8")
    print("Wrote %s" % out_path)
    print("Wrote %s" % meta_path)


if __name__ == "__main__":
    main()
