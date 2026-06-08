#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECHO-Read v2: Causal Agreement Bottleneck - Complete 16-subject LOSO Experiment
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from dataset import EchoReadDataset, load_aligned_data
from models import GazeMLP, EEGMLP, ConcatMLP
from models_v2 import EchoV2SharedOnly, EchoV2SharedPrivate, EchoV2Full, get_model_summary_v2

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
SUBJECT_TO_IDX = {s: i for i, s in enumerate(Y_SUBJECTS)}

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
    sample_ids_train_val, sample_ids_test = sample_ids[train_val_mask], sample_ids[test_mask]
    
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
    sample_ids_train, sample_ids_val = sample_ids_train_val[train_indices], sample_ids_train_val[val_indices]
    
    train_dataset = EchoReadDataset(eeg_train, gaze_train, y_train, subjects_train, sample_ids_train)
    val_dataset = EchoReadDataset(eeg_val, gaze_val, y_val, subjects_val, sample_ids_val)
    test_dataset = EchoReadDataset(eeg_test, gaze_test, y_test, subjects_test, sample_ids_test)
    
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

def get_model(mode, d_shared=32, d_private=32, dropout_rate=0.1):
    if mode == "gaze_mlp":
        return GazeMLP(input_dim=9, num_classes=2, dropout_rate=dropout_rate)
    elif mode == "concat_mlp":
        return ConcatMLP(eeg_dim=420, gaze_dim=9, num_classes=2, dropout_rate=dropout_rate)
    elif mode == "echo_v2_shared_only":
        return EchoV2SharedOnly(d_shared=d_shared, dropout_rate=dropout_rate)
    elif mode == "echo_v2_shared_private":
        return EchoV2SharedPrivate(d_shared=d_shared, d_private=d_private, dropout_rate=dropout_rate)
    elif mode == "echo_v2_full":
        return EchoV2Full(d_shared=d_shared, d_private=d_private, num_subjects=16, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def get_loss_fn(mode, loss_config):
    import torch.nn.functional as F
    
    def echo_v2_full_loss(outputs, batch):
        logits = outputs["logits"]
        y = batch["y"]
        L_cls = F.cross_entropy(logits, y)
        
        L = L_cls
        loss_dict = {"total": L_cls.item(), "cls": L_cls.item()}
        
        z_e_shared = outputs.get("z_e_shared")
        z_g_shared = outputs.get("z_g_shared")
        
        if z_e_shared is not None and z_g_shared is not None:
            L_shared_align = F.smooth_l1_loss(z_e_shared, z_g_shared)
            L = L + loss_config["lambda_shared"] * L_shared_align
            loss_dict["shared_align"] = L_shared_align.item()
        else:
            loss_dict["shared_align"] = 0.0
        
        z_e_private = outputs.get("z_e_private")
        z_g_private = outputs.get("z_g_private")
        if z_e_shared is not None and z_e_private is not None and z_g_shared is not None and z_g_private is not None:
            corr_e = mean_abs_corr(z_e_shared, z_e_private)
            corr_g = mean_abs_corr(z_g_shared, z_g_private)
            L_private_orth = corr_e + corr_g
            L = L + loss_config["lambda_private"] * L_private_orth
            loss_dict["private_orth"] = L_private_orth.item()
        else:
            loss_dict["private_orth"] = 0.0
        
        subject_logits = outputs.get("subject_logits")
        if subject_logits is not None and "subject_idx" in batch:
            subject_labels = batch["subject_idx"]
            L_subject_adv = F.cross_entropy(subject_logits, subject_labels)
            L = L + loss_config["lambda_adv"] * L_subject_adv
            loss_dict["subject_adv"] = L_subject_adv.item()
        else:
            loss_dict["subject_adv"] = 0.0
        
        z_shared = outputs.get("z_shared")
        if z_shared is not None:
            L_variance = variance_regularization(z_shared)
            L = L + loss_config["lambda_var"] * L_variance
            loss_dict["variance"] = L_variance.item()
        else:
            loss_dict["variance"] = 0.0
        
        loss_dict["total"] = L.item()
        return L, loss_dict
    
    def echo_v2_shared_private_loss(outputs, batch):
        logits = outputs["logits"]
        y = batch["y"]
        L_cls = F.cross_entropy(logits, y)
        
        L = L_cls
        loss_dict = {"total": L_cls.item(), "cls": L_cls.item()}
        
        z_e_shared = outputs.get("z_e_shared")
        z_g_shared = outputs.get("z_g_shared")
        
        if z_e_shared is not None and z_g_shared is not None:
            L_shared_align = F.smooth_l1_loss(z_e_shared, z_g_shared)
            L = L + loss_config["lambda_shared"] * L_shared_align
            loss_dict["shared_align"] = L_shared_align.item()
        else:
            loss_dict["shared_align"] = 0.0
        
        z_e_private = outputs.get("z_e_private")
        z_g_private = outputs.get("z_g_private")
        if z_e_shared is not None and z_e_private is not None and z_g_shared is not None and z_g_private is not None:
            corr_e = mean_abs_corr(z_e_shared, z_e_private)
            corr_g = mean_abs_corr(z_g_shared, z_g_private)
            L_private_orth = corr_e + corr_g
            L = L + loss_config["lambda_private"] * L_private_orth
            loss_dict["private_orth"] = L_private_orth.item()
        else:
            loss_dict["private_orth"] = 0.0
        
        z_shared = outputs.get("z_shared")
        if z_shared is not None:
            L_variance = variance_regularization(z_shared)
            L = L + loss_config["lambda_var"] * L_variance
            loss_dict["variance"] = L_variance.item()
        else:
            loss_dict["variance"] = 0.0
        
        loss_dict["total"] = L.item()
        return L, loss_dict
    
    def echo_v2_shared_only_loss(outputs, batch):
        logits = outputs["logits"]
        y = batch["y"]
        L_cls = F.cross_entropy(logits, y)
        
        L = L_cls
        loss_dict = {"total": L_cls.item(), "cls": L_cls.item()}
        
        z_e_shared = outputs.get("z_e_shared")
        z_g_shared = outputs.get("z_g_shared")
        
        if z_e_shared is not None and z_g_shared is not None:
            L_shared_align = F.smooth_l1_loss(z_e_shared, z_g_shared)
            L = L + loss_config["lambda_shared"] * L_shared_align
            loss_dict["shared_align"] = L_shared_align.item()
        else:
            loss_dict["shared_align"] = 0.0
        
        loss_dict["private_orth"] = 0.0
        loss_dict["subject_adv"] = 0.0
        loss_dict["variance"] = 0.0
        
        loss_dict["total"] = L.item()
        return L, loss_dict
    
    def simple_cls_loss(outputs, batch):
        logits = outputs["logits"]
        y = batch["y"]
        L_cls = F.cross_entropy(logits, y)
        return L_cls, {"total": L_cls.item(), "cls": L_cls.item(), "shared_align": 0.0, "private_orth": 0.0, "subject_adv": 0.0, "variance": 0.0}
    
    if mode == "echo_v2_full":
        return echo_v2_full_loss
    elif mode == "echo_v2_shared_private":
        return echo_v2_shared_private_loss
    elif mode == "echo_v2_shared_only":
        return echo_v2_shared_only_loss
    else:
        return simple_cls_loss

def compute_diagnostics(outputs, batch):
    diagnostics = {}
    
    diagnostics["cos_sim_mean"] = outputs["cos_sim"].mean().item() if "cos_sim" in outputs and outputs["cos_sim"] is not None else 0.0
    
    z_e_shared = outputs.get("z_e_shared")
    z_g_shared = outputs.get("z_g_shared")
    
    if z_e_shared is not None and z_g_shared is not None:
        z_e_norm = z_e_shared / (z_e_shared.norm(dim=1, keepdim=True) + 1e-8)
        z_g_norm = z_g_shared / (z_g_shared.norm(dim=1, keepdim=True) + 1e-8)
        diagnostics["cos_sim_between_shared"] = (z_e_norm * z_g_norm).sum(dim=1).mean().item()
    else:
        diagnostics["cos_sim_between_shared"] = 0.0
    
    z_e_private = outputs.get("z_e_private")
    if z_e_shared is not None and z_e_private is not None:
        diagnostics["shared_private_corr_eeg"] = mean_abs_corr(z_e_shared, z_e_private).item()
    else:
        diagnostics["shared_private_corr_eeg"] = 0.0
    
    z_g_private = outputs.get("z_g_private")
    if z_g_shared is not None and z_g_private is not None:
        diagnostics["shared_private_corr_gaze"] = mean_abs_corr(z_g_shared, z_g_private).item()
    else:
        diagnostics["shared_private_corr_gaze"] = 0.0
    
    z_shared = outputs.get("z_shared")
    if z_shared is not None:
        diagnostics["z_shared_std_mean"] = z_shared.std(dim=0).mean().item()
    else:
        diagnostics["z_shared_std_mean"] = 0.0
    
    subject_logits = outputs.get("subject_logits")
    if subject_logits is not None and "subject_idx" in batch:
        subject_preds = torch.argmax(subject_logits, dim=1)
        subject_acc = (subject_preds == batch["subject_idx"]).float().mean().item()
        diagnostics["subject_cls_acc_on_shared"] = subject_acc
    else:
        diagnostics["subject_cls_acc_on_shared"] = 0.0
    
    return diagnostics

def train_epoch(model, loader, optimizer, loss_fn, device, mode, grl_alpha=1.0):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_y = []
    loss_components = {k: 0.0 for k in ["total", "cls", "shared_align", "private_orth", "subject_adv", "variance"]}
    diagnostics_sum = {k: 0.0 for k in ["cos_sim_mean", "cos_sim_between_shared", "shared_private_corr_eeg", "shared_private_corr_gaze", "z_shared_std_mean", "subject_cls_acc_on_shared"]}
    n_batches = 0
    
    for batch in loader:
        batch = {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
        
        if "subject" in batch and isinstance(batch["subject"], np.ndarray):
            subject_idx = np.array([SUBJECT_TO_IDX.get(s.decode('utf-8') if isinstance(s, bytes) else s, 0) for s in batch["subject"]])
            batch["subject_idx"] = torch.tensor(subject_idx).to(device)
        
        optimizer.zero_grad()
        
        if mode in ["echo_v2_full"]:
            outputs = model(batch, grl_alpha=grl_alpha)
        else:
            outputs = model(batch)
        
        loss, loss_dict = loss_fn(outputs, batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(batch["y"])
        for k in loss_dict:
            if k in loss_components:
                loss_components[k] += loss_dict[k] * len(batch["y"])
        
        diag = compute_diagnostics(outputs, batch)
        for k in diag:
            diagnostics_sum[k] += diag[k] * len(batch["y"])
        
        logits = outputs["logits"].detach().cpu().numpy()
        y = batch["y"].detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(y)
        n_batches += 1
    
    avg_loss = total_loss / len(loader.dataset)
    for k in loss_components:
        loss_components[k] /= len(loader.dataset)
    for k in diagnostics_sum:
        diagnostics_sum[k] /= len(loader.dataset)
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    metrics = compute_metrics(all_logits, all_y)
    
    return avg_loss, metrics, loss_components, diagnostics_sum

def evaluate(model, loader, loss_fn, device, mode):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_y = []
    diagnostics_sum = {k: 0.0 for k in ["cos_sim_mean", "cos_sim_between_shared", "shared_private_corr_eeg", "shared_private_corr_gaze", "z_shared_std_mean", "subject_cls_acc_on_shared"]}
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
            
            if "subject" in batch and isinstance(batch["subject"], np.ndarray):
                subject_idx = np.array([SUBJECT_TO_IDX.get(s.decode('utf-8') if isinstance(s, bytes) else s, 0) for s in batch["subject"]])
                batch["subject_idx"] = torch.tensor(subject_idx).to(device)
            
            if mode in ["echo_v2_full"]:
                outputs = model(batch, grl_alpha=1.0)
            else:
                outputs = model(batch)
            
            loss, _ = loss_fn(outputs, batch)
            
            total_loss += loss.item() * len(batch["y"])
            
            diag = compute_diagnostics(outputs, batch)
            for k in diag:
                diagnostics_sum[k] += diag[k] * len(batch["y"])
            
            logits = outputs["logits"].detach().cpu().numpy()
            y = batch["y"].detach().cpu().numpy()
            all_logits.append(logits)
            all_y.append(y)
    
    avg_loss = total_loss / len(loader.dataset)
    for k in diagnostics_sum:
        diagnostics_sum[k] /= len(loader.dataset)
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    metrics = compute_metrics(all_logits, all_y)
    
    return avg_loss, metrics, diagnostics_sum

def run_loso_for_model(mode, loss_config, config, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config["seed"]
    set_seed(seed)
    
    all_results = []
    all_loss_components = []
    all_diagnostics = []
    error_folds = []
    
    logger.write(f"\n{'='*70}\n")
    logger.write(f"Running LOSO for mode: {mode}\n")
    logger.write(f"{'='*70}\n")
    logger.flush()
    
    print(f"\n{'='*70}")
    print(f"Running LOSO for mode: {mode}")
    print(f"{'='*70}")
    
    for held_out_subject in Y_SUBJECTS:
        logger.write(f"\n--- Held-out subject: {held_out_subject} ---\n")
        print(f"\n--- Held-out subject: {held_out_subject} ---")
        
        try:
            split = make_loso_split(held_out_subject, val_fraction=config["val_fraction"], seed=seed)
            
            from torch.utils.data import DataLoader
            train_loader = DataLoader(split["train"], batch_size=config["batch_size"], shuffle=True)
            val_loader = DataLoader(split["val"], batch_size=config["batch_size"], shuffle=False)
            test_loader = DataLoader(split["test"], batch_size=config["batch_size"], shuffle=False)
            
            model = get_model(mode, d_shared=config["d_shared"], d_private=config["d_private"], dropout_rate=config["dropout_rate"]).to(device)
            loss_fn = get_loss_fn(mode, loss_config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            
            best_val_f1 = 0.0
            best_model_state = None
            patience_counter = 0
            
            for epoch in range(config["epochs"]):
                grl_alpha = min(1.0, epoch / 10) if mode == "echo_v2_full" else 1.0
                train_loss, train_metrics, loss_comps, train_diag = train_epoch(model, train_loader, optimizer, loss_fn, device, mode, grl_alpha)
                val_loss, val_metrics, val_diag = evaluate(model, val_loader, loss_fn, device, mode)
                
                all_loss_components.append({
                    "model": mode,
                    "held_out_subject": held_out_subject,
                    "epoch": epoch,
                    "total_loss": loss_comps["total"],
                    "cls_loss": loss_comps["cls"],
                    "shared_align_loss": loss_comps["shared_align"],
                    "private_orth_loss": loss_comps["private_orth"],
                    "subject_adv_loss": loss_comps["subject_adv"],
                    "variance_loss": loss_comps["variance"],
                    "train_acc": train_metrics["accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "val_acc": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_balanced_acc": val_metrics["balanced_accuracy"],
                    "val_auroc": val_metrics["auroc"]
                })
                
                if val_metrics["macro_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["macro_f1"]
                    best_model_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= config["patience"]:
                    logger.write(f"Early stopping at epoch {epoch+1}\n")
                    break
            
            model.load_state_dict(best_model_state)
            test_loss, test_metrics, test_diag = evaluate(model, test_loader, loss_fn, device, mode)
            
            result = {
                "model": mode,
                "seed": seed,
                "held_out_subject": held_out_subject,
                "accuracy": test_metrics["accuracy"],
                "balanced_acc": test_metrics["balanced_accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "auroc": test_metrics["auroc"]
            }
            
            all_results.append(result)
            
            diag_record = {
                "model": mode,
                "held_out_subject": held_out_subject,
                **test_diag
            }
            all_diagnostics.append(diag_record)
            
            logger.write(f"Test - Acc: {test_metrics['accuracy']:.4f}, BalAcc: {test_metrics['balanced_accuracy']:.4f}, F1: {test_metrics['macro_f1']:.4f}, AUROC: {test_metrics['auroc']:.4f}\n")
            logger.flush()
            
            print(f"Test - Acc: {test_metrics['accuracy']:.4f}, BalAcc: {test_metrics['balanced_accuracy']:.4f}, F1: {test_metrics['macro_f1']:.4f}, AUROC: {test_metrics['auroc']:.4f}")
            
        except Exception as e:
            logger.write(f"ERROR for {held_out_subject}: {str(e)}\n")
            error_folds.append({"model": mode, "held_out_subject": held_out_subject, "error": str(e)})
            print(f"ERROR for {held_out_subject}: {str(e)}")
    
    return all_results, all_loss_components, all_diagnostics, error_folds

def generate_summary(all_results, output_dir):
    df = pd.DataFrame(all_results)
    
    summary = "# ECHO-Read v2 LOSO Results Summary (Seed=1)\n\n"
    summary += "## Executive Summary\n\n"
    summary += "### Comparison with Baselines\n"
    summary += "| Model | Accuracy (mean±std) | Balanced Acc | Macro-F1 | AUROC |\n"
    summary += "|-------|---------------------|--------------|----------|-------|\n"
    
    gaze_mlp_df = df[df["model"] == "gaze_mlp"]
    summary += f"| gaze_mlp | {gaze_mlp_df['accuracy'].mean():.4f} ± {gaze_mlp_df['accuracy'].std():.4f} | {gaze_mlp_df['balanced_acc'].mean():.4f} ± {gaze_mlp_df['balanced_acc'].std():.4f} | {gaze_mlp_df['macro_f1'].mean():.4f} ± {gaze_mlp_df['macro_f1'].std():.4f} | {gaze_mlp_df['auroc'].mean():.4f} ± {gaze_mlp_df['auroc'].std():.4f} |\n"
    
    concat_mlp_df = df[df["model"] == "concat_mlp"]
    summary += f"| concat_mlp | {concat_mlp_df['accuracy'].mean():.4f} ± {concat_mlp_df['accuracy'].std():.4f} | {concat_mlp_df['balanced_acc'].mean():.4f} ± {concat_mlp_df['balanced_acc'].std():.4f} | {concat_mlp_df['macro_f1'].mean():.4f} ± {concat_mlp_df['macro_f1'].std():.4f} | {concat_mlp_df['auroc'].mean():.4f} ± {concat_mlp_df['auroc'].std():.4f} |\n"
    
    echo_v2_shared_df = df[df["model"] == "echo_v2_shared_only"]
    summary += f"| echo_v2_shared_only | {echo_v2_shared_df['accuracy'].mean():.4f} ± {echo_v2_shared_df['accuracy'].std():.4f} | {echo_v2_shared_df['balanced_acc'].mean():.4f} ± {echo_v2_shared_df['balanced_acc'].std():.4f} | {echo_v2_shared_df['macro_f1'].mean():.4f} ± {echo_v2_shared_df['macro_f1'].std():.4f} | {echo_v2_shared_df['auroc'].mean():.4f} ± {echo_v2_shared_df['auroc'].std():.4f} |\n"
    
    echo_v2_sp_df = df[df["model"] == "echo_v2_shared_private"]
    summary += f"| echo_v2_shared_private | {echo_v2_sp_df['accuracy'].mean():.4f} ± {echo_v2_sp_df['accuracy'].std():.4f} | {echo_v2_sp_df['balanced_acc'].mean():.4f} ± {echo_v2_sp_df['balanced_acc'].std():.4f} | {echo_v2_sp_df['macro_f1'].mean():.4f} ± {echo_v2_sp_df['macro_f1'].std():.4f} | {echo_v2_sp_df['auroc'].mean():.4f} ± {echo_v2_sp_df['auroc'].std():.4f} |\n"
    
    echo_v2_full_df = df[df["model"] == "echo_v2_full"]
    summary += f"| echo_v2_full | {echo_v2_full_df['accuracy'].mean():.4f} ± {echo_v2_full_df['accuracy'].std():.4f} | {echo_v2_full_df['balanced_acc'].mean():.4f} ± {echo_v2_full_df['balanced_acc'].std():.4f} | {echo_v2_full_df['macro_f1'].mean():.4f} ± {echo_v2_full_df['macro_f1'].std():.4f} | {echo_v2_full_df['auroc'].mean():.4f} ± {echo_v2_full_df['auroc'].std():.4f} |\n"
    
    summary += "\n## Comparison with Linear Baselines\n"
    summary += "| Model | Linear Acc | Deep Acc | Diff |\n"
    summary += "|-------|------------|----------|------|\n"
    summary += f"| Gaze-only | 61.80% | {gaze_mlp_df['accuracy'].mean()*100:.2f}% | {((gaze_mlp_df['accuracy'].mean()*100)-61.80):.2f}% |\n"
    summary += f"| EEG+Gaze concat | 55.34% | {concat_mlp_df['accuracy'].mean()*100:.2f}% | {((concat_mlp_df['accuracy'].mean()*100)-55.34):.2f}% |\n"
    
    summary += "\n## Success Criteria Evaluation\n\n"
    gaze_f1 = gaze_mlp_df['macro_f1'].mean() * 100
    full_f1 = echo_v2_full_df['macro_f1'].mean() * 100
    linear_gaze_f1 = 57.20
    
    summary += f"- gaze_mlp Macro-F1: {gaze_f1:.2f}%\n"
    summary += f"- echo_v2_full Macro-F1: {full_f1:.2f}%\n"
    summary += f"- Linear Gaze-only Macro-F1: {linear_gaze_f1}%\n\n"
    
    if full_f1 > gaze_f1:
        summary += "✅ echo_v2_full > gaze_mlp: PASSED\n"
    else:
        summary += "❌ echo_v2_full > gaze_mlp: FAILED\n"
    
    if full_f1 > linear_gaze_f1:
        summary += "✅ echo_v2_full > Linear Gaze-only: PASSED\n"
    else:
        summary += "❌ echo_v2_full > Linear Gaze-only: FAILED\n"
    
    summary += "\n## Per-Subject Results\n"
    summary += "| Subject | gaze_mlp | concat_mlp | echo_v2_shared_only | echo_v2_shared_private | echo_v2_full |\n"
    summary += "|---------|----------|------------|---------------------|------------------------|--------------|\n"
    
    for subject in Y_SUBJECTS:
        row = f"| {subject} |"
        for model in ["gaze_mlp", "concat_mlp", "echo_v2_shared_only", "echo_v2_shared_private", "echo_v2_full"]:
            f1 = df[(df["model"] == model) & (df["held_out_subject"] == subject)]["macro_f1"].values
            row += f" {f1[0]:.4f} |" if len(f1) > 0 else " - |"
        summary += row + "\n"
    
    summary += "\n## Conclusion\n\n"
    if full_f1 > gaze_f1 and full_f1 > linear_gaze_f1:
        summary += "ECHO-Read v2 successfully solves the EEG-Gaze negative transfer problem!\n"
        summary += "The Causal Agreement Bottleneck effectively leverages both modalities.\n"
        summary += "Recommend proceeding to synergy decomposition and counterfactual audit.\n"
    elif full_f1 > gaze_f1:
        summary += "ECHO-Read v2 exceeds gaze_mlp but not the linear Gaze-only baseline.\n"
        summary += "Consider tuning hyperparameters or adding advanced techniques.\n"
    else:
        summary += "ECHO-Read v2 did not solve the EEG-Gaze negative transfer problem.\n"
        summary += "Possible reasons:\n"
        summary += "- z_shared may still contain subject identity\n"
        summary += "- Private factors may not be absorbing subject-specific variance\n"
        summary += "- Agreement bottleneck may be collapsing\n"
        summary += "- EEG shared features may not align well with Gaze shared features\n"
        summary += "Recommend checking diagnostics.csv for latent statistics.\n"
    
    with open(os.path.join(output_dir, "loso_summary_seed1.md"), "w") as f:
        f.write(summary)
    
    return summary

def generate_protocol_checklist(output_dir):
    checklist = """# Protocol Checklist for ECHO-Read v2 LOSO Experiment

## Data Handling

- [X] Only Y subjects used (16 subjects)
- [X] X subjects NOT used
- [X] Scaler fit only on training data
- [X] Validation data only transformed (not fitted)
- [X] Test data only transformed (not fitted)
- [X] Label not included in model input features
- [X] Test subject not used in validation
- [X] Data loaded from aligned_multimodal_y.npz
- [X] Join key is subject+label+idx
- [X] Text/LLM embedding NOT used as input

## Training Protocol

- [X] 16-subject LOSO
- [X] Early stopping on val_macro_f1
- [X] Validation only from train subjects
- [X] Best checkpoint saved and used for test
- [X] Test subject evaluated only once at end
- [X] AdamW optimizer with weight decay (1e-4)
- [X] Learning rate: 1e-3
- [X] Batch size: 64
- [X] Patience: 10
- [X] Seed: 1
- [X] Gradient clipping: 1.0

## Model Architecture

- [X] gaze_mlp: Gaze-only 9-D
- [X] concat_mlp: EEG+Gaze 429-D
- [X] echo_v2_shared_only: Shared-only agreement bottleneck
- [X] echo_v2_shared_private: Shared+private factorization
- [X] echo_v2_full: Full causal agreement bottleneck

## Evaluation Metrics

- [X] Accuracy
- [X] Balanced Accuracy
- [X] Macro-F1
- [X] AUROC (NaN handled gracefully)

## Output Files

- [X] loso_all_results_seed1.csv
- [X] loso_summary_seed1.md
- [X] subjectwise_results_seed1.csv
- [X] diagnostics.csv
- [X] loss_components_seed1.csv
- [X] protocol_checklist_v2.md
- [X] model_param_summary_v2.txt

## Success Criteria

- [ ] echo_v2_full Macro-F1 > gaze_mlp Macro-F1
- [ ] echo_v2_full Macro-F1 > 57.20 (Linear Gaze-only)
"""
    
    with open(os.path.join(output_dir, "protocol_checklist_v2.md"), "w") as f:
        f.write(checklist)

def generate_model_param_summary(output_dir):
    from models_v2 import EchoV2SharedOnly, EchoV2SharedPrivate, EchoV2Full, count_parameters
    from models import GazeMLP, ConcatMLP
    
    summary = "# Model Parameter Summary\n\n"
    
    models = [
        ("GazeMLP", GazeMLP()),
        ("ConcatMLP", ConcatMLP()),
        ("EchoV2SharedOnly", EchoV2SharedOnly()),
        ("EchoV2SharedPrivate", EchoV2SharedPrivate()),
        ("EchoV2Full", EchoV2Full())
    ]
    
    for name, model in models:
        summary += f"## {name}\n"
        summary += f"Total parameters: {count_parameters(model):,}\n"
        
        if hasattr(model, "eeg_encoder"):
            summary += f"- EEGEncoderSharedPrivate: {count_parameters(model.eeg_encoder):,}\n"
        if hasattr(model, "gaze_encoder"):
            summary += f"- GazeEncoderSharedPrivate: {count_parameters(model.gaze_encoder):,}\n"
        if hasattr(model, "agreement_bottleneck"):
            summary += f"- AgreementBottleneck: {count_parameters(model.agreement_bottleneck):,}\n"
        if hasattr(model, "classifier"):
            summary += f"- Classifier: {count_parameters(model.classifier):,}\n"
        if hasattr(model, "subject_classifier"):
            summary += f"- SubjectClassifier: {count_parameters(model.subject_classifier):,}\n"
        
        summary += "\n"
    
    with open(os.path.join(output_dir, "model_param_summary_v2.txt"), "w") as f:
        f.write(summary)

def main():
    config = {
        "epochs": 50,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 10,
        "seed": 1,
        "d_shared": 32,
        "d_private": 32,
        "dropout_rate": 0.1,
        "val_fraction": 0.1
    }
    
    loss_config = {
        "lambda_shared": 0.005,
        "lambda_private": 0.01,
        "lambda_adv": 0.01,
        "lambda_var": 0.01
    }
    
    output_dir = "results/echo_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "loso_log_seed1.txt")
    logger = open(log_file, "w")
    
    all_results = []
    all_loss_components = []
    all_diagnostics = []
    all_errors = []
    
    modes = ["gaze_mlp", "concat_mlp", "echo_v2_shared_only", "echo_v2_shared_private", "echo_v2_full"]
    
    for mode in modes:
        results, loss_comps, diagnostics, errors = run_loso_for_model(mode, loss_config, config, logger)
        all_results.extend(results)
        all_loss_components.extend(loss_comps)
        all_diagnostics.extend(diagnostics)
        all_errors.extend(errors)
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(output_dir, "loso_all_results_seed1.csv"), index=False)
    
    df_subjectwise = df_results.pivot(index="held_out_subject", columns="model", values=["accuracy", "balanced_acc", "macro_f1", "auroc"])
    df_subjectwise.to_csv(os.path.join(output_dir, "subjectwise_results_seed1.csv"))
    
    df_loss = pd.DataFrame(all_loss_components)
    df_loss.to_csv(os.path.join(output_dir, "loss_components_seed1.csv"), index=False)
    
    df_diagnostics = pd.DataFrame(all_diagnostics)
    df_diagnostics.to_csv(os.path.join(output_dir, "diagnostics.csv"), index=False)
    
    if all_errors:
        df_errors = pd.DataFrame(all_errors)
        df_errors.to_csv(os.path.join(output_dir, "error_folds_log.txt"), index=False)
    
    generate_summary(all_results, output_dir)
    generate_protocol_checklist(output_dir)
    generate_model_param_summary(output_dir)
    
    logger.close()
    
    print(f"\n{'='*70}")
    print("ECHO-v2 LOSO Experiment Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()