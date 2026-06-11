"""Train one NOCS run for a specified subject protocol."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.adagtcn_aligned.nocs.dataset_nocs import load_split, make_datasets, make_loader
from src.adagtcn_aligned.nocs.losses_nocs import nocs_loss
from src.adagtcn_aligned.nocs.model_nocs import NOCSModel


ABLATIONS = [
    "full",
    "residual",
    "stat_gaze",
    "stat_residual",
    "stat_full",
    "no_eeg",
    "no_gaze_control",
    "no_uncertainty",
    "no_adv",
    "gaze_only",
    "eeg_only",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def class_weights_from_labels(labels: list[int], device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def binary_metrics(y_true: list[int], y_pred: list[int], y_prob: list[float]) -> dict[str, float]:
    if len(set(y_true)) < 2:
        auroc = 0.5
        auprc = float(np.mean(y_true)) if y_true else 0.0
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": auroc,
        "auprc": auprc,
        "n": int(len(y_true)),
    }


def branch_prob(outputs: dict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    if key not in outputs:
        return None
    prob = torch.softmax(outputs[key], dim=-1)[:, 1]
    return torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)


def evaluate(model: NOCSModel, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    branch_probs: dict[str, list[float]] = {"stat": [], "gaze": [], "eeg": []}
    branch_preds: dict[str, list[int]] = {"stat": [], "gaze": [], "eeg": []}
    residual_gates: list[float] = []
    residual_norms: list[float] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            outputs = model(batch, grl_scale=0.0)
            prob = branch_prob(outputs, "logits_full")
            assert prob is not None
            pred = torch.argmax(outputs["logits_full"], dim=-1)
            labels = batch["labels"].detach().cpu().numpy().tolist()
            preds = pred.detach().cpu().numpy().tolist()
            probs = prob.detach().cpu().numpy().tolist()
            y_true.extend(labels)
            y_pred.extend(preds)
            y_prob.extend(probs)

            batch_branch_values: dict[str, list[float]] = {}
            for name, key in [("stat", "logits_stat"), ("gaze", "logits_g"), ("eeg", "logits_e")]:
                b_prob = branch_prob(outputs, key)
                if b_prob is None:
                    continue
                b_pred = torch.argmax(outputs[key], dim=-1)
                b_probs = b_prob.detach().cpu().numpy().tolist()
                b_preds = b_pred.detach().cpu().numpy().tolist()
                branch_probs[name].extend(b_probs)
                branch_preds[name].extend(b_preds)
                batch_branch_values[name] = b_probs

            gate = outputs.get("residual_gate")
            correction = outputs.get("residual_correction")
            if gate is not None:
                gate_vals = gate.detach().view(gate.shape[0], -1).mean(dim=1).cpu().numpy().tolist()
                residual_gates.extend(float(x) for x in gate_vals)
            else:
                gate_vals = [float("nan")] * len(labels)
            if correction is not None:
                norm_vals = correction.detach().norm(dim=-1).cpu().numpy().tolist()
                residual_norms.extend(float(x) for x in norm_vals)
            else:
                norm_vals = [float("nan")] * len(labels)

            for idx, label in enumerate(labels):
                row = {
                    "sequence_id": batch["sequence_ids"][idx],
                    "subject": batch["subject_names"][idx],
                    "y_true": int(label),
                    "y_pred": int(preds[idx]),
                    "prob": float(probs[idx]),
                    "y_prob_full": float(probs[idx]),
                    "residual_gate_mean": float(gate_vals[idx]),
                    "residual_logit_shift_norm": float(norm_vals[idx]),
                }
                for name in ["stat", "gaze", "eeg"]:
                    if name in batch_branch_values:
                        row["y_prob_" + name] = float(batch_branch_values[name][idx])
                rows.append(row)

    metrics = binary_metrics(y_true, y_pred, y_prob)
    for name in ["stat", "gaze", "eeg"]:
        if len(branch_probs[name]) == len(y_true):
            branch_metric = binary_metrics(y_true, branch_preds[name], branch_probs[name])
            for key, value in branch_metric.items():
                metrics[name + "_" + key] = value
    if "stat_auroc" in metrics:
        metrics["full_minus_stat_auroc"] = metrics["auroc"] - metrics["stat_auroc"]
        metrics["full_minus_stat_macro_f1"] = metrics["macro_f1"] - metrics["stat_macro_f1"]
    if residual_gates:
        metrics["mean_residual_gate"] = float(np.nanmean(residual_gates))
    if residual_norms:
        metrics["mean_residual_norm"] = float(np.nanmean(residual_norms))
    return metrics, rows


def write_single_row_csv(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["sequence_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device)
    split = load_split(args.split_json, args.protocol)
    train_ds, val_ds, test_ds, meta = make_datasets(args.sequence_jsonl, split, cache_records=args.cache_records)
    train_loader = make_loader(train_ds, args.batch_size, True, args.max_len, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.max_len, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.max_len, args.num_workers)

    model = NOCSModel(
        eeg_dim=meta["eeg_dim"],
        gaze_dim=meta["gaze_dim"],
        n_subjects=len(meta["subject_to_idx"]),
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        bidirectional=not args.causal,
        ablation=args.ablation,
        residual_beta=args.residual_beta,
        stat_head=args.stat_head,
    ).to(device)

    weights = None
    if args.class_weight == "balanced":
        weights = class_weights_from_labels([ex["label"] for ex in train_ds.examples], device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_auroc = -1.0
    best_epoch = 0
    best_path = args.output_dir / ("%s_%s_seed%d_best.pt" % (args.ablation, args.protocol, args.seed))
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_logs: list[dict[str, float]] = []
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            grl_scale = min(1.0, float(epoch) / max(args.epochs // 2, 1))
            outputs = model(batch, grl_scale=grl_scale)
            loss, logs = nocs_loss(
                outputs,
                batch["labels"],
                batch["subjects"],
                weights,
                args.ablation,
                args.lambda_mono,
                args.lambda_adv,
                args.lambda_uncert,
                args.lambda_supcon,
                args.lambda_residual_norm,
                args.lambda_gate,
                args.mono_margin,
                args.uncert_margin,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_logs.append(logs)

        val_metrics, _ = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "val_auroc": val_metrics["auroc"], "val_macro_f1": val_metrics["macro_f1"]}
        for key in train_logs[0] if train_logs else []:
            row[key] = float(np.mean([log[key] for log in train_logs]))
        history.append(row)

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args), "meta": meta}, best_path)
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    val_metrics, _ = evaluate(model, val_loader, device)
    test_metrics, test_predictions = evaluate(model, test_loader, device)

    metrics = {
        "protocol": args.protocol,
        "heldout_subject": args.protocol.replace("Y16_LOSO_", "") if args.protocol.startswith("Y16_LOSO_") else "",
        "seed": args.seed,
        "ablation": args.ablation,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "train_examples": meta["n_train"],
        "val_examples": meta["n_val"],
        "test_examples": meta["n_test"],
        "eeg_dim": meta["eeg_dim"],
        "gaze_dim": meta["gaze_dim"],
    }
    if args.ablation.startswith("stat_"):
        metrics["gaze_stat_feat_dim"] = 2 * meta["gaze_dim"] + 3
    metrics.update({"val_" + key: value for key, value in val_metrics.items()})
    metrics.update({"test_" + key: value for key, value in test_metrics.items()})
    for key in [
        "stat_auroc",
        "stat_macro_f1",
        "full_minus_stat_auroc",
        "full_minus_stat_macro_f1",
        "mean_residual_gate",
        "mean_residual_norm",
    ]:
        if key in test_metrics:
            metrics[key] = test_metrics[key]

    stem = "%s_%s_seed%d" % (args.ablation, args.protocol, args.seed)
    metrics_path = args.output_dir / ("%s_metrics.csv" % stem)
    preds_path = args.output_dir / ("%s_predictions.csv" % stem)
    meta_path = args.output_dir / ("%s_meta.json" % stem)
    history_path = args.output_dir / ("%s_history.csv" % stem)
    write_single_row_csv(metrics, metrics_path)
    write_rows_csv(test_predictions, preds_path)
    write_rows_csv(history, history_path)
    meta_dump = {
        "args": json_safe(vars(args)),
        "split": {
            "protocol": split.protocol,
            "train": split.train,
            "val": split.val,
            "test": split.test,
            "note": split.note,
        },
        "dataset": json_safe(meta),
        "best_checkpoint": str(best_path),
    }
    meta_path.write_text(json.dumps(meta_dump, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print("Wrote %s" % metrics_path)
    print("Wrote %s" % preds_path)
    print("Wrote %s" % meta_path)
    print("Wrote %s" % best_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NOCS on ZuCo word-level EEG+gaze JSONL.")
    parser.add_argument("--sequence-jsonl", type=Path, default=Path("data/adagtcn_aligned/zuco_word_band_gaze_sequences.jsonl"))
    parser.add_argument("--split-json", type=Path, default=Path("reports/adagtcn_aligned/subject_splits.json"))
    parser.add_argument("--protocol", default="Y16_LOSO_YAC")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=Path("outputs/nocs"))
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--stat-head", "--stat_head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--lambda-mono", type=float, default=1.0)
    parser.add_argument("--lambda-adv", type=float, default=0.1)
    parser.add_argument("--lambda-uncert", type=float, default=0.1)
    parser.add_argument("--lambda-supcon", type=float, default=0.05)
    parser.add_argument("--lambda-residual-norm", "--lambda_residual_norm", type=float, default=0.01)
    parser.add_argument("--lambda-gate", "--lambda_gate", type=float, default=0.001)
    parser.add_argument("--mono-margin", type=float, default=0.02)
    parser.add_argument("--uncert-margin", type=float, default=0.0)
    parser.add_argument("--residual-beta", "--residual_beta", type=float, default=0.3)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--cache-records", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
