# User Adapter Ablation Report

## Experiment Summary

This experiment tested whether lightweight adaptation strategies could improve EEG-based user calibration beyond the baseline MLP.

## Models Tested

| Model | Description |
|-------|-------------|
| EEG_MLP_baseline | Simple MLP trained from scratch on calibration data |
| SIED_encoder_linear_probe | Freeze SIED encoder, train only classifier head |
| SIED_encoder_finetune | Fine-tune full encoder + classifier with small LR |
| SIED_encoder_adapter | Add residual adapter module between encoder layers |
| SIED_encoder_bias_calibration | Only update classifier head weights |

## Results: 50-shot Calibration

| Model | Accuracy | Macro F1 | Balanced Accuracy | AUROC |
|-------|----------|----------|-------------------|-------|
| **EEG_MLP_baseline** | **78.62%** | **78.47%** | **78.76%** | **0.859** |
| SIED_encoder_bias_calibration | 69.25% | 69.06% | 69.40% | 0.759 |
| SIED_encoder_adapter | 69.09% | 68.75% | 69.08% | 0.761 |
| SIED_encoder_finetune | 68.90% | 68.59% | 68.95% | 0.760 |
| SIED_encoder_linear_probe | 69.54% | 69.36% | 69.65% | 0.761 |

## Results: Calibration Curve

| Shot | EEG_MLP | SIED_linear | SIED_finetune | SIED_adapter | SIED_bias |
|------|---------|-------------|---------------|--------------|-----------|
| 1-shot | 52.7% | 51.0% | 51.0% | 50.7% | 51.6% |
| 3-shot | 59.0% | 56.6% | 57.2% | 57.7% | 56.8% |
| 5-shot | 62.1% | 58.6% | 60.0% | 59.7% | 58.6% |
| 10-shot | 66.4% | 61.8% | 63.2% | 62.9% | 61.2% |
| 20-shot | 71.3% | 65.2% | 65.7% | 65.8% | 65.3% |
| 50-shot | **78.6%** | 69.5% | 68.9% | 69.1% | 69.3% |

## Key Findings

### 1. All SIED-based Methods Underperform Baseline MLP

**At 50-shot, the gap is ~9.5%:**
- EEG_MLP_baseline: 78.6%
- Best SIED method (linear_probe): 69.5%
- **Gap: -9.1%**

### 2. SIED Encoder Removes Useful Information

The SIED adversarial training removes subject-specific patterns, which are actually **useful** for within-subject prediction:
- Subject-specific EEG patterns help classify NR vs TSR within a user
- SIED's subject-invariant features remove this discriminative information

### 3. Fine-tuning Doesn't Recover Lost Information

Even with fine-tuning (lower LR, more epochs), the SIED encoder cannot recover the performance:
- The adversarial training fundamentally removes useful patterns
- Fine-tuning on user data cannot fully restore them

### 4. Adapter Modules Don't Help

Adding residual adapters or bias-only calibration doesn't close the gap:
- The problem is not capacity or optimization
- The problem is the information removed by adversarial training

## Conclusion: User Adapter is NOT an Innovation Point

**The simple EEG_MLP_baseline significantly outperforms all SIED-based adaptation methods.**

This is because:
1. Within-subject prediction benefits from subject-specific EEG patterns
2. SIED adversarial training removes these patterns (by design)
3. Fine-tuning cannot recover the removed information

**Recommendation**: Do NOT claim User Adapter as an innovation point.

The baseline MLP with few-shot calibration (78.6% at 50-shot) is already the best performing method for personalized EEG classification.