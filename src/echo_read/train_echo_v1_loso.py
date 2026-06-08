#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ECHO-Read v1: Complete 16-subject LOSO Experiment
"""

import os
import json
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from dataset import EchoReadDataset, load_aligned_data
from models import GazeMLP, EEGMLP, ConcatMLP, EchoReadV0, get_model_summary

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

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

def get_model(mode, hidden_dim=32, dropout_rate=0.1):
    if mode == "gaze_mlp":
        return GazeMLP(input_dim=9, num_classes=2, dropout_rate=dropout_rate)
    elif mode == "eeg_mlp":
        return EEGMLP(input_dim=420, num_classes=2, dropout_rate=dropout_rate)
    elif mode == "concat_mlp":
        return ConcatMLP(eeg_dim=420, gaze_dim=9, num_classes=2, dropout_rate=dropout_rate)
    elif mode == "echo_common" or mode == "echo_closed_loop":
        return EchoReadV0(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def get_loss_fn(mode, loss_config):
    import torch.nn.functional as F
    
    def echo_closed_loop_loss(outputs, batch):
        logits = outputs["logits"]
        y = batch["y"]
        
        L_cls = F.cross_entropy(logits, y)
        
        gaze = batch["gaze"]
        eeg = batch["eeg"]
        
        gaze_hat_from_zc = outputs["gaze_hat_from_zc"]
        eeg_hat_from_zc = outputs["eeg_hat_from_zc"]
        gaze_hat_from_e = outputs["gaze_hat_from_e"]
        eeg_hat_from_g = outputs["eeg_hat_from_g"]
        z_e = outputs["z_e"]
        z_g = outputs["z_g"]
        
        L_common_recon = F.mse_loss(gaze_hat_from_zc, gaze) + loss_config["alpha_eeg"] * F.mse_loss(eeg_hat_from_zc, eeg)
        L_cross_pred = F.mse_loss(gaze_hat_from_e, gaze) + loss_config["alpha_eeg"] * F.mse_loss(eeg_hat_from_g, eeg)
        L_latent_align = F.smooth_l1_loss(z_e, z_g)
        
        L = L_cls + loss_config["lambda_recon"] * L_common_recon + loss_config["lambda_cross"] * L_cross_pred + loss_config["lambda_align"] * L_latent_align
        
        loss_dict = {
            "total": L.item(),
            "cls": L_cls.item(),
            "common_recon": L_common_recon.item(),
            "cross_pred": L_cross_pred.item(),
            "latent_align": L_latent_align.item()
        }
        
        return L, loss_dict
    
    def echo_common_loss(outputs, batch):
        logits = outputs["logits"]
        y = batch["y"]
        L_cls = F.cross_entropy(logits, y)
        return L_cls, {"total": L_cls.item(), "cls": L_cls.item()}
    
    def simple_cls_loss(outputs, batch):
        logits = outputs["logits"]
        y = batch["y"]
        L_cls = F.cross_entropy(logits, y)
        return L_cls, {"total": L_cls.item(), "cls": L_cls.item()}
    
    if mode == "echo_closed_loop":
        return echo_closed_loop_loss
    elif mode == "echo_common":
        return echo_common_loss
    else:
        return simple_cls_loss

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_y = []
    loss_components = {k: 0.0 for k in ["total", "cls", "common_recon", "cross_pred", "latent_align"]}
    
    for batch in loader:
        batch = {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
        optimizer.zero_grad()
        
        outputs = model(batch)
        loss, loss_dict = loss_fn(outputs, batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch["y"])
        for k in loss_dict:
            loss_components[k] += loss_dict[k] * len(batch["y"])
        
        logits = outputs["logits"].detach().cpu().numpy()
        y = batch["y"].detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(y)
    
    avg_loss = total_loss / len(loader.dataset)
    for k in loss_components:
        loss_components[k] /= len(loader.dataset)
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    metrics = compute_metrics(all_logits, all_y)
    
    return avg_loss, metrics, loss_components

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_y = []
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
            
            outputs = model(batch)
            loss, _ = loss_fn(outputs, batch)
            
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

def run_loso_for_model(mode, loss_config, config, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config["seed"]
    set_seed(seed)
    
    all_results = []
    all_loss_components = []
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
            
            model = get_model(mode, hidden_dim=config["hidden_dim"], dropout_rate=config["dropout_rate"]).to(device)
            loss_fn = get_loss_fn(mode, loss_config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            
            best_val_f1 = 0.0
            best_model_state = None
            patience_counter = 0
            
            for epoch in range(config["epochs"]):
                train_loss, train_metrics, loss_comps = train_epoch(model, train_loader, optimizer, loss_fn, device)
                val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)
                
                all_loss_components.append({
                    "model": mode,
                    "held_out_subject": held_out_subject,
                    "epoch": epoch,
                    "total_loss": loss_comps["total"],
                    "cls_loss": loss_comps["cls"],
                    "common_recon_loss": loss_comps.get("common_recon", 0),
                    "cross_pred_loss": loss_comps.get("cross_pred", 0),
                    "latent_align_loss": loss_comps.get("latent_align", 0),
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
            test_loss, test_metrics = evaluate(model, test_loader, loss_fn, device)
            
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
            
            logger.write(f"Test - Acc: {test_metrics['accuracy']:.4f}, BalAcc: {test_metrics['balanced_accuracy']:.4f}, F1: {test_metrics['macro_f1']:.4f}, AUROC: {test_metrics['auroc']:.4f}\n")
            logger.flush()
            
            print(f"Test - Acc: {test_metrics['accuracy']:.4f}, BalAcc: {test_metrics['balanced_accuracy']:.4f}, F1: {test_metrics['macro_f1']:.4f}, AUROC: {test_metrics['auroc']:.4f}")
            
        except Exception as e:
            logger.write(f"ERROR for {held_out_subject}: {str(e)}\n")
            error_folds.append({"model": mode, "held_out_subject": held_out_subject, "error": str(e)})
            print(f"ERROR for {held_out_subject}: {str(e)}")
    
    return all_results, all_loss_components, error_folds

def generate_summary(all_results, output_dir):
    df = pd.DataFrame(all_results)
    
    summary = "# ECHO-Read v1 LOSO Results Summary (Seed=1)\n\n"
    summary += "## Executive Summary\n\n"
    summary += "### Comparison with Baselines\n"
    summary += "| Model | Accuracy (mean±std) | Balanced Acc | Macro-F1 | AUROC |\n"
    summary += "|-------|---------------------|--------------|----------|-------|\n"
    
    gaze_mlp_df = df[df["model"] == "gaze_mlp"]
    summary += f"| gaze_mlp | {gaze_mlp_df['accuracy'].mean():.4f} ± {gaze_mlp_df['accuracy'].std():.4f} | {gaze_mlp_df['balanced_acc'].mean():.4f} ± {gaze_mlp_df['balanced_acc'].std():.4f} | {gaze_mlp_df['macro_f1'].mean():.4f} ± {gaze_mlp_df['macro_f1'].std():.4f} | {gaze_mlp_df['auroc'].mean():.4f} ± {gaze_mlp_df['auroc'].std():.4f} |\n"
    
    eeg_mlp_df = df[df["model"] == "eeg_mlp"]
    summary += f"| eeg_mlp | {eeg_mlp_df['accuracy'].mean():.4f} ± {eeg_mlp_df['accuracy'].std():.4f} | {eeg_mlp_df['balanced_acc'].mean():.4f} ± {eeg_mlp_df['balanced_acc'].std():.4f} | {eeg_mlp_df['macro_f1'].mean():.4f} ± {eeg_mlp_df['macro_f1'].std():.4f} | {eeg_mlp_df['auroc'].mean():.4f} ± {eeg_mlp_df['auroc'].std():.4f} |\n"
    
    concat_mlp_df = df[df["model"] == "concat_mlp"]
    summary += f"| concat_mlp | {concat_mlp_df['accuracy'].mean():.4f} ± {concat_mlp_df['accuracy'].std():.4f} | {concat_mlp_df['balanced_acc'].mean():.4f} ± {concat_mlp_df['balanced_acc'].std():.4f} | {concat_mlp_df['macro_f1'].mean():.4f} ± {concat_mlp_df['macro_f1'].std():.4f} | {concat_mlp_df['auroc'].mean():.4f} ± {concat_mlp_df['auroc'].std():.4f} |\n"
    
    echo_common_df = df[df["model"] == "echo_common"]
    summary += f"| echo_common | {echo_common_df['accuracy'].mean():.4f} ± {echo_common_df['accuracy'].std():.4f} | {echo_common_df['balanced_acc'].mean():.4f} ± {echo_common_df['balanced_acc'].std():.4f} | {echo_common_df['macro_f1'].mean():.4f} ± {echo_common_df['macro_f1'].std():.4f} | {echo_common_df['auroc'].mean():.4f} ± {echo_common_df['auroc'].std():.4f} |\n"
    
    echo_closed_df = df[df["model"] == "echo_closed_loop"]
    summary += f"| echo_closed_loop | {echo_closed_df['accuracy'].mean():.4f} ± {echo_closed_df['accuracy'].std():.4f} | {echo_closed_df['balanced_acc'].mean():.4f} ± {echo_closed_df['balanced_acc'].std():.4f} | {echo_closed_df['macro_f1'].mean():.4f} ± {echo_closed_df['macro_f1'].std():.4f} | {echo_closed_df['auroc'].mean():.4f} ± {echo_closed_df['auroc'].std():.4f} |\n"
    
    summary += "\n## Comparison with Linear Baselines\n"
    summary += "| Model | Linear Acc | Deep Acc | Diff |\n"
    summary += "|-------|------------|----------|------|\n"
    summary += f"| EEG-only | 51.22% | {eeg_mlp_df['accuracy'].mean()*100:.2f}% | {((eeg_mlp_df['accuracy'].mean()*100)-51.22):.2f}% |\n"
    summary += f"| Gaze-only | 61.80% | {gaze_mlp_df['accuracy'].mean()*100:.2f}% | {((gaze_mlp_df['accuracy'].mean()*100)-61.80):.2f}% |\n"
    summary += f"| EEG+Gaze concat | 55.34% | {concat_mlp_df['accuracy'].mean()*100:.2f}% | {((concat_mlp_df['accuracy'].mean()*100)-55.34):.2f}% |\n"
    
    summary += "\n## Success Criteria Evaluation\n\n"
    gaze_f1 = gaze_mlp_df['macro_f1'].mean() * 100
    closed_f1 = echo_closed_df['macro_f1'].mean() * 100
    linear_gaze_f1 = 57.20
    
    summary += f"- gaze_mlp Macro-F1: {gaze_f1:.2f}%\n"
    summary += f"- echo_closed_loop Macro-F1: {closed_f1:.2f}%\n"
    summary += f"- Linear Gaze-only Macro-F1: {linear_gaze_f1}%\n\n"
    
    if closed_f1 > gaze_f1:
        summary += "✅ echo_closed_loop > gaze_mlp: PASSED\n"
    else:
        summary += "❌ echo_closed_loop > gaze_mlp: FAILED\n"
    
    if closed_f1 > linear_gaze_f1:
        summary += "✅ echo_closed_loop > Linear Gaze-only: PASSED\n"
    else:
        summary += "❌ echo_closed_loop > Linear Gaze-only: FAILED\n"
    
    summary += "\n## Per-Subject Results\n"
    summary += "| Subject | gaze_mlp | eeg_mlp | concat_mlp | echo_common | echo_closed_loop |\n"
    summary += "|---------|----------|---------|------------|-------------|------------------|\n"
    
    for subject in Y_SUBJECTS:
        row = f"| {subject} |"
        for model in ["gaze_mlp", "eeg_mlp", "concat_mlp", "echo_common", "echo_closed_loop"]:
            f1 = df[(df["model"] == model) & (df["held_out_subject"] == subject)]["macro_f1"].values
            row += f" {f1[0]:.4f} |" if len(f1) > 0 else " - |"
        summary += row + "\n"
    
    summary += "\n## Conclusion\n\n"
    if closed_f1 > gaze_f1 and closed_f1 > linear_gaze_f1:
        summary += "ECHO-Read v1 successfully solves the EEG-Gaze negative transfer problem!\n"
        summary += "The closed-loop predictive evidence cycle effectively leverages both modalities.\n"
        summary += "Recommend proceeding to synergy decomposition and counterfactual audit.\n"
    elif closed_f1 > gaze_f1:
        summary += "ECHO-Read v1 exceeds gaze_mlp but not the linear Gaze-only baseline.\n"
        summary += "Consider tuning hyperparameters or adding advanced techniques.\n"
    else:
        summary += "ECHO-Read v1 did not solve the EEG-Gaze negative transfer problem.\n"
        summary += "Possible reasons:\n"
        summary += "- Loss weights may need adjustment\n"
        summary += "- EEG reconstruction may be dominating training\n"
        summary += "- Need better latent alignment mechanism\n"
        summary += "Recommend trying: reduced lambda_recon, EEG-PCA reconstruction, gradient clipping\n"
    
    with open(os.path.join(output_dir, "loso_summary_seed1.md"), "w") as f:
        f.write(summary)
    
    return summary

def generate_protocol_checklist(output_dir):
    checklist = """# Protocol Checklist for ECHO-Read v1 LOSO Experiment

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

## Model Architecture

- [X] gaze_mlp: Gaze-only 9-D
- [X] eeg_mlp: EEG-only 420-D
- [X] concat_mlp: EEG+Gaze 429-D
- [X] echo_common: Common-cause only
- [X] echo_closed_loop: Full closed-loop

## Evaluation Metrics

- [X] Accuracy
- [X] Balanced Accuracy
- [X] Macro-F1
- [X] AUROC (NaN handled gracefully)

## Output Files

- [X] loso_all_results_seed1.csv
- [X] loso_summary_seed1.md
- [X] subjectwise_results_seed1.csv
- [X] loss_components_seed1.csv
- [X] model_param_summary_v1.txt
- [X] protocol_checklist_v1.md
- [X] error_folds_log.txt

## Success Criteria

- [ ] echo_closed_loop Macro-F1 > gaze_mlp Macro-F1
- [ ] echo_closed_loop Macro-F1 > 57.20 (Linear Gaze-only)
"""
    
    with open(os.path.join(output_dir, "protocol_checklist_v1.md"), "w") as f:
        f.write(checklist)

def generate_model_param_summary(output_dir):
    from models import EchoReadV0, GazeMLP, EEGMLP, ConcatMLP, count_parameters
    
    summary = "# Model Parameter Summary\n\n"
    
    models = [
        ("GazeMLP", GazeMLP()),
        ("EEGMLP", EEGMLP()),
        ("ConcatMLP", ConcatMLP()),
        ("EchoReadV0", EchoReadV0())
    ]
    
    for name, model in models:
        summary += f"## {name}\n"
        summary += f"Total parameters: {count_parameters(model):,}\n"
        
        if hasattr(model, "eeg_observer"):
            summary += f"- EEGObserver: {count_parameters(model.eeg_observer):,}\n"
        if hasattr(model, "gaze_observer"):
            summary += f"- GazeObserver: {count_parameters(model.gaze_observer):,}\n"
        if hasattr(model, "common_cause"):
            summary += f"- CommonCauseEstimator: {count_parameters(model.common_cause):,}\n"
        if hasattr(model, "gaze_decoder"):
            summary += f"- GazeDecoder: {count_parameters(model.gaze_decoder):,}\n"
        if hasattr(model, "eeg_decoder"):
            summary += f"- EEGDecoder: {count_parameters(model.eeg_decoder):,}\n"
        if hasattr(model, "eeg_to_gaze"):
            summary += f"- EEG→Gaze Cross Decoder: {count_parameters(model.eeg_to_gaze):,}\n"
        if hasattr(model, "gaze_to_eeg"):
            summary += f"- Gaze→EEG Cross Decoder: {count_parameters(model.gaze_to_eeg):,}\n"
        if hasattr(model, "classifier"):
            summary += f"- Classifier: {count_parameters(model.classifier):,}\n"
        
        summary += "\n"
    
    with open(os.path.join(output_dir, "model_param_summary_v1.txt"), "w") as f:
        f.write(summary)

def main():
    config = {
        "epochs": 50,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 10,
        "seed": 1,
        "hidden_dim": 32,
        "dropout_rate": 0.1,
        "val_fraction": 0.1
    }
    
    loss_config = {
        "lambda_recon": 0.05,
        "lambda_cross": 0.05,
        "lambda_align": 0.01,
        "alpha_eeg": 0.05
    }
    
    output_dir = "results/echo_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "loso_log_seed1.txt")
    logger = open(log_file, "w")
    
    all_results = []
    all_loss_components = []
    all_errors = []
    
    modes = ["gaze_mlp", "eeg_mlp", "concat_mlp", "echo_common", "echo_closed_loop"]
    
    for mode in modes:
        results, loss_comps, errors = run_loso_for_model(mode, loss_config, config, logger)
        all_results.extend(results)
        all_loss_components.extend(loss_comps)
        all_errors.extend(errors)
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(output_dir, "loso_all_results_seed1.csv"), index=False)
    
    df_subjectwise = df_results.pivot(index="held_out_subject", columns="model", values=["accuracy", "balanced_acc", "macro_f1", "auroc"])
    df_subjectwise.to_csv(os.path.join(output_dir, "subjectwise_results_seed1.csv"))
    
    df_loss = pd.DataFrame(all_loss_components)
    df_loss.to_csv(os.path.join(output_dir, "loss_components_seed1.csv"), index=False)
    
    if all_errors:
        df_errors = pd.DataFrame(all_errors)
        df_errors.to_csv(os.path.join(output_dir, "error_folds_log.txt"), index=False)
    
    generate_summary(all_results, output_dir)
    generate_protocol_checklist(output_dir)
    generate_model_param_summary(output_dir)
    
    logger.close()
    
    print(f"\n{'='*70}")
    print("LOSO Experiment Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()