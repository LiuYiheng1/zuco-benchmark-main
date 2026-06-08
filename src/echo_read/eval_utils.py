import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


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


def average_metrics(metrics_list):
    if not metrics_list:
        return {}
    
    result = {}
    keys = metrics_list[0].keys()
    
    for key in keys:
        values = [m[key] for m in metrics_list if not np.isnan(m[key])]
        if values:
            result[key] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }
        else:
            result[key] = {"mean": np.nan, "std": np.nan}
    
    return result


def log_metrics(epoch, train_metrics, val_metrics, logger=None):
    log_str = f"Epoch {epoch}:"
    log_str += f"\n  Train - Acc: {train_metrics['accuracy']:.4f}, BalAcc: {train_metrics['balanced_accuracy']:.4f}, F1: {train_metrics['macro_f1']:.4f}, AUROC: {train_metrics['auroc']:.4f}"
    log_str += f"\n  Val   - Acc: {val_metrics['accuracy']:.4f}, BalAcc: {val_metrics['balanced_accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f}, AUROC: {val_metrics['auroc']:.4f}"
    
    if logger:
        logger.write(log_str + "\n")
        logger.flush()
    
    print(log_str)