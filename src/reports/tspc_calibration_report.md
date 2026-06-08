# TSPC (Task-Set Personalized Prototype Calibration) Evaluation Report

## Experiment Overview

TSPC was proposed as a module that computes class prototypes from calibration samples and classifies test samples based on distance to prototypes. Three variants were tested:

| Model | Description |
|-------|-------------|
| TSPC_proto_only | Direct prototype computation on scaled EEG features |
| TSPC_pretrained | Pretrained encoder + prototype calibration |
| TSPC_SIED | SIED adversarial encoder + prototype calibration |

## Results Summary

### 50-shot Calibration Performance

| Model | Accuracy | Macro F1 | Balanced Accuracy | AUROC |
|-------|----------|----------|-------------------|-------|
| **EEG_MLP** | **78.62%** | **78.47%** | **78.76%** | **0.859** |
| **EEG_SVM** | **78.61%** | **78.45%** | **78.73%** | **0.858** |
| TSPC_proto_only | 66.17% | 65.81% | 66.05% | 0.720 |
| TSPC_pretrained | 61.44% | 61.02% | 61.43% | 0.662 |
| TSPC_SIED | 59.48% | 58.97% | 59.38% | 0.641 |

### Calibration Curve Comparison

| Shot | EEG_SVM | EEG_MLP | TSPC_proto | TSPC_pretrained | TSPC_SIED |
|------|---------|---------|------------|-----------------|-----------|
| 1-shot | 52.7% | 52.7% | 52.7% | 54.4% | 54.0% |
| 3-shot | 58.5% | 59.0% | 58.3% | 56.7% | 55.5% |
| 5-shot | 60.8% | 62.1% | 60.8% | 58.6% | 57.4% |
| 10-shot | 65.7% | 66.4% | 62.9% | 59.3% | 58.3% |
| 20-shot | 71.1% | 71.3% | 64.7% | 61.4% | 58.6% |
| 50-shot | **78.6%** | **78.6%** | 66.2% | 61.4% | 59.5% |

## Critical Analysis

### Finding 1: TSPC Underperforms Baselines

**TSPC does NOT exceed EEG_SVM/EEG_MLP at any shot setting.**

- At 50-shot: TSPC_proto_only (66.2%) is **12.4% worse** than EEG_SVM (78.6%)
- At 50-shot: TSPC_SIED (59.5%) is **19.1% worse** than EEG_SVM

### Finding 2: TSPC Shows No Advantage in Low-Shot Settings

Even in the 1-shot setting:
- TSPC_pretrained: 54.4% vs EEG_SVM: 52.7% (+1.7%)
- TSPC_SIED: 54.0% vs EEG_SVM: 52.7% (+1.3%)

However, this small advantage disappears completely at higher shot settings.

### Finding 3: SIED Encoder Hurts Prototype Performance

SIED adversarial training removes subject-specific information, but this also removes information useful for prototype-based classification:

- TSPC_proto_only (66.2%) > TSPC_SIED (59.5%) at 50-shot
- The SIED encoder's subject-invariant features are not suitable for prototype distance metrics

### Finding 4: TSPC_proto_only Performs Best Among TSPC Variants

The simplest approach (direct prototypes on scaled features) outperforms more complex variants:
- TSPC_proto_only > TSPC_pretrained > TSPC_SIED

This suggests that neural encoders destroy the geometric structure needed for prototype-based classification.

## Subject-Level Analysis

### Best and Worst Subjects for TSPC_proto_only (50-shot)

| Subject | Accuracy | Subject | Accuracy |
|---------|----------|---------|----------|
| YTL | 85.5% | YRP | 52.1% |
| YHS | 82.0% | YSL | 60.0% |
| YSD | 77.8% | YDR | 60.7% |
| YAK | 76.5% | YFR | 62.2% |
| YMD | 75.9% | YAC | 63.6% |

### Key Observation

TSPC works well for some subjects (YTL: 85.5%) but poorly for others (YRP: 52.1%). The high variance suggests prototype-based classification is sensitive to the geometric structure of individual subject's EEG embeddings.

## Conclusion: TSPC Cannot Be an Innovation Point

### Why TSPC Fails

1. **Prototype distance is a weak classifier for high-dimensional EEG features**
   - Direct SVM/MLP classification outperforms prototype distance by 12%+

2. **Neural encoders destroy geometric structure**
   - The encoding process doesn't preserve the class-separating geometry needed for prototypes

3. **SIED encoder removes too much information**
   - Subject-invariant features are not useful for within-subject prototype classification

### Recommendation

**TSPC should NOT be claimed as an innovation point.**

The baseline EEG_SVM/EEG_MLP with few-shot calibration already outperforms TSPC significantly. The few-shot calibration protocol itself is valuable as an **experimental methodology** for personalized EEG analysis, but the specific TSPC prototype mechanism does not provide additional benefit.

### What CAN Be Claimed as Innovation

1. **SIED (Subject-Invariant EEG Disentanglement)**: Proven +3.55% improvement over raw EEG in cross-subject transfer
2. **Few-shot personalized calibration protocol**: Valid experimental methodology for user-calibrated systems

### Final Model Architecture (Recommended)

For the paper, the recommended architecture should focus on:
1. **SIED** for cross-subject generalization
2. **EEG_MLP/SVM** with few-shot calibration for personalized within-subject prediction
3. **Text-proxy/modal control** as experimental validation, not as a model component