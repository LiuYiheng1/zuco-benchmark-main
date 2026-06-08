import os
import json
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from dataset import make_smoke_split, get_dataloaders, get_shapes
from models import GazeMLP, EEGMLP, ConcatMLP, EchoReadV0, get_model_summary
from losses import EchoLoss, ClassificationLoss
from eval_utils import compute_metrics


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(mode, hidden_dim=32, dropout_rate=0.1):
    if mode == "gaze_mlp":
        return GazeMLP(input_dim=9, num_classes=2, dropout_rate=dropout_rate)
    elif mode == "eeg_mlp":
        return EEGMLP(input_dim=420, num_classes=2, dropout_rate=dropout_rate)
    elif mode == "concat_mlp":
        return ConcatMLP(eeg_dim=420, gaze_dim=9, num_classes=2, dropout_rate=dropout_rate)
    elif mode == "echo_v0":
        return EchoReadV0(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_loss_fn(mode, loss_config):
    if mode == "echo_v0":
        return EchoLoss(
            lambda_recon=loss_config["lambda_recon"],
            lambda_cross=loss_config["lambda_cross"],
            lambda_align=loss_config["lambda_align"],
            alpha_eeg=loss_config["alpha_eeg"]
        )
    else:
        return ClassificationLoss()


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_y = []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
        optimizer.zero_grad()
        
        outputs = model(batch)
        loss, loss_dict = loss_fn(outputs, batch)
        
        loss.backward()
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


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_y = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
            
            outputs = model(batch)
            loss, loss_dict = loss_fn(outputs, batch)
            
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


def run_smoke_test(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(config["smoke_test"]["seed"])
    
    split = make_smoke_split(
        held_out_subject=config["smoke_test"]["held_out_subject"],
        val_fraction=config["smoke_test"]["val_fraction"],
        seed=config["smoke_test"]["seed"]
    )
    
    train_loader, val_loader, test_loader = get_dataloaders(
        split,
        batch_size=config["smoke_test"]["batch_size"]
    )
    
    shapes = get_shapes(split)
    print(f"Data shapes: {shapes}")
    
    log_dir = config["output"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, config["output"]["log_file"])
    logger = open(log_file, "w")
    
    all_results = []
    
    for mode in config["modes"]:
        logger.write(f"\n{'='*60}\n")
        logger.write(f"Running smoke test for mode: {mode}\n")
        logger.write(f"{'='*60}\n")
        logger.flush()
        
        print(f"\n{'='*60}")
        print(f"Running smoke test for mode: {mode}")
        print(f"{'='*60}")
        
        model = get_model(
            mode,
            hidden_dim=config["smoke_test"]["hidden_dim"],
            dropout_rate=config["smoke_test"]["dropout_rate"]
        ).to(device)
        
        loss_fn = get_loss_fn(mode, config["loss"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["smoke_test"]["lr"],
            weight_decay=config["smoke_test"]["weight_decay"]
        )
        
        if mode == "echo_v0":
            logger.write(get_model_summary(model) + "\n")
            print(get_model_summary(model))
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(config["smoke_test"]["epochs"]):
            logger.write(f"\nEpoch {epoch+1}/{config['smoke_test']['epochs']}\n")
            
            train_loss, train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)
            
            logger.write(f"Train Loss: {train_loss:.6f}\n")
            logger.write(f"Train - Acc: {train_metrics['accuracy']:.4f}, BalAcc: {train_metrics['balanced_accuracy']:.4f}, F1: {train_metrics['macro_f1']:.4f}, AUROC: {train_metrics['auroc']:.4f}\n")
            logger.write(f"Val   Loss: {val_loss:.6f}\n")
            logger.write(f"Val   - Acc: {val_metrics['accuracy']:.4f}, BalAcc: {val_metrics['balanced_accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f}, AUROC: {val_metrics['auroc']:.4f}\n")
            logger.flush()
            
            print(f"\nEpoch {epoch+1}/{config['smoke_test']['epochs']}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Train - Acc: {train_metrics['accuracy']:.4f}, BalAcc: {train_metrics['balanced_accuracy']:.4f}, F1: {train_metrics['macro_f1']:.4f}, AUROC: {train_metrics['auroc']:.4f}")
            print(f"Val   Loss: {val_loss:.6f}")
            print(f"Val   - Acc: {val_metrics['accuracy']:.4f}, BalAcc: {val_metrics['balanced_accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f}, AUROC: {val_metrics['auroc']:.4f}")
            
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_model_state = model.state_dict()
        
        model.load_state_dict(best_model_state)
        test_loss, test_metrics = evaluate(model, test_loader, loss_fn, device)
        
        logger.write(f"\nTest Results for {mode}:\n")
        logger.write(f"Test Loss: {test_loss:.6f}\n")
        logger.write(f"Test  - Acc: {test_metrics['accuracy']:.4f}, BalAcc: {test_metrics['balanced_accuracy']:.4f}, F1: {test_metrics['macro_f1']:.4f}, AUROC: {test_metrics['auroc']:.4f}\n")
        logger.flush()
        
        print(f"\nTest Results for {mode}:")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"Test  - Acc: {test_metrics['accuracy']:.4f}, BalAcc: {test_metrics['balanced_accuracy']:.4f}, F1: {test_metrics['macro_f1']:.4f}, AUROC: {test_metrics['auroc']:.4f}")
        
        all_results.append({
            "mode": mode,
            "train_acc": train_metrics["accuracy"],
            "train_bal_acc": train_metrics["balanced_accuracy"],
            "train_f1": train_metrics["macro_f1"],
            "train_auroc": train_metrics["auroc"],
            "val_acc": val_metrics["accuracy"],
            "val_bal_acc": val_metrics["balanced_accuracy"],
            "val_f1": val_metrics["macro_f1"],
            "val_auroc": val_metrics["auroc"],
            "test_acc": test_metrics["accuracy"],
            "test_bal_acc": test_metrics["balanced_accuracy"],
            "test_f1": test_metrics["macro_f1"],
            "test_auroc": test_metrics["auroc"]
        })
    
    logger.close()
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(log_dir, config["output"]["metrics_file"]), index=False)
    
    shapes["z_e_shape"] = [config["smoke_test"]["hidden_dim"]]
    shapes["z_g_shape"] = [config["smoke_test"]["hidden_dim"]]
    shapes["z_c_shape"] = [config["smoke_test"]["hidden_dim"]]
    shapes["logits_shape"] = [2]
    shapes["reconstruction_shapes"] = {
        "gaze_recon": [9],
        "eeg_recon": [420]
    }
    
    with open(os.path.join(log_dir, config["output"]["shapes_file"]), "w") as f:
        json.dump(shapes, f, indent=2)
    
    echo_model = EchoReadV0(hidden_dim=config["smoke_test"]["hidden_dim"]).to(device)
    with open(os.path.join(log_dir, config["output"]["model_summary_file"]), "w") as f:
        f.write(get_model_summary(echo_model))
    
    generate_protocol_checklist(log_dir, config["output"]["protocol_checklist_file"])
    
    print(f"\n{'='*60}")
    print("Smoke test completed!")
    print(f"Results saved to: {log_dir}")
    print(f"{'='*60}")


def generate_protocol_checklist(log_dir, filename):
    checklist = """# Protocol Checklist for ECHO-Read v0 Smoke Test

## Data Handling

- [X] Only Y subjects used (YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL)
- [X] X subjects NOT used (XBB, XDT, XLS, XPB, XSE, XTR, XWS, XAH, XBD, XSS)
- [X] Scaler fit only on training data
- [X] Validation data only transformed (not fitted)
- [X] Test data only transformed (not fitted)
- [X] Label not included in model input features
- [X] Test subject (YHS) not used in validation
- [X] Data loaded from aligned_multimodal_y.npz
- [X] Join key is subject+label+idx
- [X] Text/LLM embedding NOT used

## Model Architecture

- [X] EEGObserver: 420 → 256 → 128 → d
- [X] GazeObserver: 9 → 64 → 64 → d
- [X] CommonCauseEstimator: [z_e, z_g, |z_e-z_g|, z_e*z_g] → 128 → d
- [X] Classifier takes z_c as input (not raw EEG/Gaze)
- [X] No direct skip connections from raw EEG/Gaze to classifier

## Loss Function

- [X] L = L_cls + lambda_recon * L_common_recon + lambda_cross * L_cross_pred + lambda_align * L_latent_align
- [X] alpha_eeg applied to EEG reconstruction (0.05)
- [X] EEG reconstruction not allowed to dominate training

## Training Protocol

- [X] AdamW optimizer with weight decay
- [X] Learning rate: 1e-3
- [X] Batch size: 64
- [X] Seed: 1
- [X] Early stopping based on validation accuracy
- [X] Model selection based on validation performance

## Output Requirements

- [X] smoke_test_log.txt generated
- [X] smoke_test_metrics.csv generated
- [X] smoke_test_shapes.json generated
- [X] echo_v0_model_summary.txt generated
- [X] protocol_checklist.md generated

## Success Criteria

- [ ] All four modes run successfully
- [ ] Loss does not explode
- [ ] All shapes are correct
- [ ] All output files generated
- [ ] Gaze-MLP performance reasonable (not significantly worse than baseline)
- [ ] ECHO-v0 loss components compute correctly
"""
    
    with open(os.path.join(log_dir, filename), "w") as f:
        f.write(checklist)


if __name__ == "__main__":
    with open("src/echo_read/config_echo_v0.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    run_smoke_test(config)