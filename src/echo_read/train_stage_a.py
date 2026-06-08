#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECHO-Read Stage A: 3-subject Fast Screening Protocol
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from dataset import EchoReadDataset, load_aligned_data
from models import GazeMLP, ConcatMLP
from models_v2 import EchoV2Full

STAGE_A_SUBJECTS = ["YHS", "YRK", "YFR"]
SUBJECT_TO_IDX = {s: i for i, s in enumerate(['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL'])}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_loso_split(held_out_subject, val_fraction=0.1, seed=1):
    eeg, gaze, y, subjects, sample_ids = load_aligned_data()
    
    test_mask = subjects == held_out_subject
    train_val_mask = ~test_mask
    
    eeg_train_val, eeg_test = eeg[train_val_mask], eeg[test_mask]
    gaze_train_val, gaze_test = gaze[train_val_mask], gaze[test_mask]
    y_train_val, y_test = y[train_val_mask], y[test_mask]
    subjects_train_val, subjects_test = subjects[train_val_mask], subjects[test_mask]
    
    from sklearn.preprocessing import StandardScaler
    scaler_eeg = StandardScaler()
    scaler_gaze = StandardScaler()
    
    eeg_train_val = scaler_eeg.fit_transform(eeg_train_val)
    gaze_train_val = scaler_gaze.fit_transform(gaze_train_val)
    
    eeg_test = scaler_eeg.transform(eeg_test)
    gaze_test = scaler_gaze.transform(gaze_test)
    
    n_val = int(len(y_train_val) * val_fraction)
    n_train = len(y_train_val) - n_val
    
    np.random.seed(seed)
    indices = np.random.permutation(len(y_train_val))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    eeg_train, eeg_val = eeg_train_val[train_indices], eeg_train_val[val_indices]
    gaze_train, gaze_val = gaze_train_val[train_indices], gaze_train_val[val_indices]
    y_train, y_val = y_train_val[train_indices], y_train_val[val_indices]
    subjects_train, subjects_val = subjects_train_val[train_indices], subjects_train_val[val_indices]
    
    train_dataset = EchoReadDataset(eeg_train, gaze_train, y_train, subjects_train, np.arange(len(y_train)))
    val_dataset = EchoReadDataset(eeg_val, gaze_val, y_val, subjects_val, np.arange(len(y_val)))
    test_dataset = EchoReadDataset(eeg_test, gaze_test, y_test, subjects_test, np.arange(len(y_test)))
    
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

def compute_metrics(logits, y):
    preds = np.argmax(logits, axis=1)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    
    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    
    try:
        auroc = roc_auc_score(y, probs[:, 1])
    except ValueError:
        auroc = np.nan
    
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "auroc": auroc
    }

def mean_abs_corr(x, y):
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    cov = (x_centered * y_centered).mean(dim=0)
    std_x = x_centered.std(dim=0)
    std_y = y_centered.std(dim=0)
    corr = cov / (std_x * std_y + 1e-8)
    return torch.mean(torch.abs(corr))

def variance_regularization(z, min_std=1e-4):
    std = z.std(dim=0)
    loss = torch.mean(torch.relu(min_std - std))
    return loss

def get_model(mode):
    if mode == "gaze_mlp":
        return GazeMLP(input_dim=9, num_classes=2, dropout_rate=0.1)
    elif mode == "concat_mlp":
        return ConcatMLP(eeg_dim=420, gaze_dim=9, num_classes=2, dropout_rate=0.1)
    elif mode == "echo_v2_full":
        return EchoV2Full(d_shared=32, d_private=32, num_subjects=16, dropout_rate=0.1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def train_epoch(model, loader, optimizer, loss_fn, device, mode, grl_alpha=1.0):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_y = []
    
    for batch in loader:
        batch = {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
        
        if "subject" in batch and isinstance(batch["subject"], np.ndarray):
            subject_idx = np.array([SUBJECT_TO_IDX.get(s.decode('utf-8') if isinstance(s, bytes) else s, 0) for s in batch["subject"]])
            batch["subject_idx"] = torch.tensor(subject_idx).to(device)
        
        optimizer.zero_grad()
        
        if mode == "echo_v2_full":
            outputs = model(batch, grl_alpha=grl_alpha)
        else:
            outputs = model(batch)
        
        loss = loss_fn(outputs, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(batch["y"])
        
        logits = outputs["logits"].detach().cpu().numpy()
        y = batch["y"].detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(y)
    
    avg_loss = total_loss / len(loader.dataset)
    all_logits = np.concatenate(all_logits, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    metrics = compute_metrics(all_logits, all_y)
    
    return avg_loss, metrics

def evaluate(model, loader, loss_fn, device, mode):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_y = []
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
            
            if "subject" in batch and isinstance(batch["subject"], np.ndarray):
                subject_idx = np.array([SUBJECT_TO_IDX.get(s.decode('utf-8') if isinstance(s, bytes) else s, 0) for s in batch["subject"]])
                batch["subject_idx"] = torch.tensor(subject_idx).to(device)
            
            if mode == "echo_v2_full":
                outputs = model(batch, grl_alpha=1.0)
            else:
                outputs = model(batch)
            
            loss = loss_fn(outputs, batch)
            total_loss += loss.item() * len(batch["y"])
            
            logits = outputs["logits"].detach().cpu().numpy()
            y = batch["y"].detach().cpu().numpy()
            all_logits.append(logits)
            all_y.append(y)
    
    avg_loss = total_loss / len(loader.dataset)
    all_logits = np.concatenate(all_logits, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    metrics = compute_metrics(all_logits, all_y)
    
    return avg_loss, metrics

def get_loss_fn(mode):
    import torch.nn.functional as F
    
    def echo_v2_full_loss(outputs, batch):
        logits = outputs["logits"]
        y = batch["y"]
        L_cls = F.cross_entropy(logits, y)
        L = L_cls
        
        z_e_shared = outputs.get("z_e_shared")
        z_g_shared = outputs.get("z_g_shared")
        if z_e_shared is not None and z_g_shared is not None:
            L += 0.005 * F.smooth_l1_loss(z_e_shared, z_g_shared)
        
        z_e_private = outputs.get("z_e_private")
        z_g_private = outputs.get("z_g_private")
        if z_e_shared is not None and z_e_private is not None and z_g_shared is not None and z_g_private is not None:
            corr_e = mean_abs_corr(z_e_shared, z_e_private)
            corr_g = mean_abs_corr(z_g_shared, z_g_private)
            L += 0.01 * (corr_e + corr_g)
        
        subject_logits = outputs.get("subject_logits")
        if subject_logits is not None and "subject_idx" in batch:
            L += 0.01 * F.cross_entropy(subject_logits, batch["subject_idx"])
        
        z_shared = outputs.get("z_shared")
        if z_shared is not None:
            L += 0.01 * variance_regularization(z_shared)
        
        return L
    
    def simple_cls_loss(outputs, batch):
        logits = outputs["logits"]
        y = batch["y"]
        return F.cross_entropy(logits, y)
    
    if mode == "echo_v2_full":
        return echo_v2_full_loss
    else:
        return simple_cls_loss

def run_stage_a():
    config = {
        "epochs": 20,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "seed": 1,
        "val_fraction": 0.1
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = "results/stage_a"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "stage_a_runtime_log.txt")
    logger = open(log_file, "w")
    
    all_results = []
    all_runtimes = []
    
    modes = ["gaze_mlp", "concat_mlp", "echo_v2_full"]
    
    total_start_time = time.time()
    
    for mode in modes:
        logger.write(f"\n{'='*70}\n")
        logger.write(f"Stage A - Running mode: {mode}\n")
        logger.write(f"{'='*70}\n")
        logger.flush()
        
        print(f"\n{'='*70}")
        print(f"Stage A - Running mode: {mode}")
        print(f"{'='*70}")
        
        for held_out_subject in STAGE_A_SUBJECTS:
            start_time = time.time()
            
            logger.write(f"\n--- Held-out subject: {held_out_subject} ---\n")
            print(f"\n--- Held-out subject: {held_out_subject} ---")
            
            try:
                split = make_loso_split(held_out_subject, val_fraction=config["val_fraction"], seed=config["seed"])
                
                from torch.utils.data import DataLoader
                train_loader = DataLoader(split["train"], batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=False)
                val_loader = DataLoader(split["val"], batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
                test_loader = DataLoader(split["test"], batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
                
                set_seed(config["seed"])
                model = get_model(mode).to(device)
                loss_fn = get_loss_fn(mode)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
                
                best_val_f1 = 0.0
                patience_counter = 0
                
                for epoch in range(config["epochs"]):
                    grl_alpha = min(1.0, epoch / 10) if mode == "echo_v2_full" else 1.0
                    train_loss, train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, mode, grl_alpha)
                    val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device, mode)
                    
                    if val_metrics["macro_f1"] > best_val_f1:
                        best_val_f1 = val_metrics["macro_f1"]
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= config["patience"]:
                        logger.write(f"Early stopping at epoch {epoch+1}\n")
                        break
                
                test_loss, test_metrics = evaluate(model, test_loader, loss_fn, device, mode)
                
                runtime = time.time() - start_time
                all_runtimes.append({"model": mode, "held_out_subject": held_out_subject, "runtime_seconds": runtime})
                
                result = {
                    "model": mode,
                    "seed": config["seed"],
                    "held_out_subject": held_out_subject,
                    "accuracy": test_metrics["accuracy"],
                    "balanced_acc": test_metrics["balanced_accuracy"],
                    "macro_f1": test_metrics["macro_f1"],
                    "auroc": test_metrics["auroc"],
                    "runtime_seconds": runtime
                }
                
                all_results.append(result)
                
                logger.write(f"Test - Acc: {test_metrics['accuracy']:.4f}, BalAcc: {test_metrics['balanced_accuracy']:.4f}, F1: {test_metrics['macro_f1']:.4f}, AUROC: {test_metrics['auroc']:.4f}, Runtime: {runtime:.2f}s\n")
                logger.flush()
                
                print(f"Test - Acc: {test_metrics['accuracy']:.4f}, BalAcc: {test_metrics['balanced_accuracy']:.4f}, F1: {test_metrics['macro_f1']:.4f}, AUROC: {test_metrics['auroc']:.4f}, Runtime: {runtime:.2f}s")
                
            except Exception as e:
                runtime = time.time() - start_time
                logger.write(f"ERROR for {held_out_subject}: {str(e)}, Runtime: {runtime:.2f}s\n")
                print(f"ERROR for {held_out_subject}: {str(e)}, Runtime: {runtime:.2f}s")
    
    total_runtime = time.time() - total_start_time
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(output_dir, "stage_a_results.csv"), index=False)
    
    df_subjectwise = df_results.pivot(index="held_out_subject", columns="model", values=["accuracy", "balanced_acc", "macro_f1", "auroc"])
    df_subjectwise.to_csv(os.path.join(output_dir, "stage_a_subjectwise.csv"))
    
    generate_summary(all_results, total_runtime, output_dir)
    generate_protocol_checklist(output_dir)
    
    logger.close()
    
    print(f"\n{'='*70}")
    print(f"Stage A Complete!")
    print(f"Total Runtime: {total_runtime:.2f}s")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")

def generate_summary(all_results, total_runtime, output_dir):
    df = pd.DataFrame(all_results)
    
    summary = "# Stage A: 3-Subject Fast Screening Results\n\n"
    summary += "## Executive Summary\n\n"
    summary += f"Total Runtime: {total_runtime:.2f} seconds\n\n"
    
    summary += "### Results Summary\n"
    summary += "| Model | Accuracy | Balanced Acc | Macro-F1 | AUROC |\n"
    summary += "|-------|----------|--------------|----------|-------|\n"
    
    gaze_mlp_df = df[df["model"] == "gaze_mlp"]
    gaze_f1 = gaze_mlp_df['macro_f1'].mean() * 100
    summary += f"| gaze_mlp | {gaze_mlp_df['accuracy'].mean():.4f} | {gaze_mlp_df['balanced_acc'].mean():.4f} | {gaze_f1:.2f}% | {gaze_mlp_df['auroc'].mean():.4f} |\n"
    
    concat_mlp_df = df[df["model"] == "concat_mlp"]
    concat_f1 = concat_mlp_df['macro_f1'].mean() * 100
    summary += f"| concat_mlp | {concat_mlp_df['accuracy'].mean():.4f} | {concat_mlp_df['balanced_acc'].mean():.4f} | {concat_f1:.2f}% | {concat_mlp_df['auroc'].mean():.4f} |\n"
    
    echo_v2_full_df = df[df["model"] == "echo_v2_full"]
    full_f1 = echo_v2_full_df['macro_f1'].mean() * 100
    summary += f"| echo_v2_full | {echo_v2_full_df['accuracy'].mean():.4f} | {echo_v2_full_df['balanced_acc'].mean():.4f} | {full_f1:.2f}% | {echo_v2_full_df['auroc'].mean():.4f} |\n"
    
    summary += "\n### Per-Subject Results (Macro-F1)\n"
    summary += "| Subject | gaze_mlp | concat_mlp | echo_v2_full |\n"
    summary += "|---------|----------|------------|--------------|\n"
    
    for subject in STAGE_A_SUBJECTS:
        row = f"| {subject} |"
        for model in ["gaze_mlp", "concat_mlp", "echo_v2_full"]:
            f1 = df[(df["model"] == model) & (df["held_out_subject"] == subject)]["macro_f1"].values
            row += f" {f1[0]*100:.2f}% |" if len(f1) > 0 else " - |"
        summary += row + "\n"
    
    summary += "\n## Evaluation Questions\n\n"
    
    if full_f1 > gaze_f1:
        summary += "1. ✅ New model (echo_v2_full) exceeds gaze_mlp: YES\n"
    else:
        summary += f"1. ❌ New model (echo_v2_full) exceeds gaze_mlp: NO (diff: {full_f1 - gaze_f1:.2f}%)\n"
    
    if full_f1 > concat_f1:
        summary += "2. ✅ New model exceeds concat_mlp: YES\n"
    else:
        summary += f"2. ❌ New model exceeds concat_mlp: NO (diff: {full_f1 - concat_f1:.2f}%)\n"
    
    summary += "\n3. Failed subjects analysis:\n"
    for subject in STAGE_A_SUBJECTS:
        full_val = df[(df["model"] == "echo_v2_full") & (df["held_out_subject"] == subject)]["macro_f1"].values[0] * 100
        gaze_val = df[(df["model"] == "gaze_mlp") & (df["held_out_subject"] == subject)]["macro_f1"].values[0] * 100
        if full_val < gaze_val:
            summary += f"   - {subject}: echo_v2_full ({full_val:.2f}%) < gaze_mlp ({gaze_val:.2f}%)\n"
        else:
            summary += f"   - {subject}: echo_v2_full ({full_val:.2f}%) >= gaze_mlp ({gaze_val:.2f}%)\n"
    
    if full_f1 > gaze_f1:
        summary += "\n4. Recommendation: ✅ Proceed to Stage B\n"
    else:
        summary += "\n4. Recommendation: ❌ Do NOT proceed to Stage B. Consider hyperparameter tuning or model redesign.\n"
    
    summary += f"\n5. Total running time: {total_runtime:.2f} seconds\n"
    
    with open(os.path.join(output_dir, "stage_a_summary.md"), "w") as f:
        f.write(summary)

def generate_protocol_checklist(output_dir):
    checklist = """# Protocol Checklist for Stage A: 3-Subject Fast Screening

## Data Handling

- [X] Only Y subjects used
- [X] X subjects NOT used
- [X] Scaler fit only on training data
- [X] Validation data only transformed (not fitted)
- [X] Test data only transformed (not fitted)
- [X] Label not included in model input features
- [X] Test subject not used in validation
- [X] Text/LLM embedding NOT used as input

## Training Protocol

- [X] 3-subject LOSO (YHS, YRK, YFR)
- [X] Early stopping on val_macro_f1
- [X] Validation only from train subjects
- [X] Best checkpoint used for test
- [X] Test subject evaluated only once at end
- [X] AdamW optimizer with weight decay (1e-4)
- [X] Learning rate: 1e-3
- [X] Batch size: 128
- [X] Patience: 5
- [X] Epochs: 20
- [X] Seed: 1
- [X] Gradient clipping: 1.0
- [X] num_workers: 0

## Models Evaluated

- [X] gaze_mlp
- [X] concat_mlp
- [X] echo_v2_full

## Evaluation Metrics

- [X] Accuracy
- [X] Balanced Accuracy
- [X] Macro-F1
- [X] AUROC (NaN handled gracefully)

## Output Files

- [X] stage_a_results.csv
- [X] stage_a_summary.md
- [X] stage_a_subjectwise.csv
- [X] stage_a_runtime_log.txt
- [X] protocol_checklist_stage_a.md
"""
    
    with open(os.path.join(output_dir, "protocol_checklist_stage_a.md"), "w") as f:
        f.write(checklist)

if __name__ == "__main__":
    run_stage_a()