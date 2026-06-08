# 2025-2026 Latest Methods Proxy Baselines Report

## Protocol
- Same as standard config: LOSO, k-shot, k=3,5,10,20,50
- seeds = [0, 1, 2, 3, 4]

## Methods Implemented

### Graph-based (Main Baseline)
1. **STRG-lite**: Spectro-Topographic Relational Graphs with learnable adjacency
2. **STRE-lite**: Spatio-Temporal Relational Embeddings with 1D conv
3. **Corr-GCN**: EEG-GCN with correlation-based adjacency

### GLIM-Encoder
4. **GLIM-Encoder**: Interpretable bottleneck EEG encoder with reconstruction loss

### CognitiveDecoder (Confound/Upper-bound)
5. **Cognitive_random**: Random noise + EEG as "text" proxy
6. **Cognitive_EEGtext**: EEG-derived features + EEG (self-fusion)

## Results Summary


### STRG_lite

| k | Accuracy | Macro-F1 | BAcc | AUROC |
|---|----------|----------|------|-------|
| 3 | 56.6±7.0 | 55.4±7.6 | 57.0±6.6 | 59.4±9.9 |
| 5 | 59.4±6.7 | 58.6±6.7 | 59.7±6.2 | 63.4±8.4 |
| 10 | 65.0±6.2 | 64.6±6.2 | 65.2±6.0 | 70.4±7.0 |
| 20 | 70.3±5.8 | 70.0±5.8 | 70.5±5.7 | 77.1±6.5 |
| 50 | 78.2±6.6 | 78.0±6.6 | 78.2±6.5 | 85.8±6.3 |

### STRE_lite

| k | Accuracy | Macro-F1 | BAcc | AUROC |
|---|----------|----------|------|-------|
| 3 | 56.4±7.2 | 55.1±7.6 | 56.7±6.8 | 58.3±9.0 |
| 5 | 58.2±7.1 | 57.5±7.0 | 58.4±6.6 | 61.3±8.5 |
| 10 | 63.5±6.4 | 63.0±6.5 | 63.7±6.2 | 68.1±6.9 |
| 20 | 69.4±5.9 | 69.1±5.9 | 69.6±5.8 | 75.8±7.0 |
| 50 | 77.2±6.4 | 77.0±6.5 | 77.2±6.4 | 84.9±6.5 |

### Corr_GCN

| k | Accuracy | Macro-F1 | BAcc | AUROC |
|---|----------|----------|------|-------|
| 3 | 54.3±6.5 | 53.0±6.8 | 54.6±6.7 | 55.9±9.5 |
| 5 | 55.6±6.2 | 54.6±6.5 | 56.1±5.8 | 58.9±7.9 |
| 10 | 60.8±5.7 | 60.1±5.7 | 60.9±5.6 | 65.2±7.6 |
| 20 | 64.6±5.8 | 64.3±5.8 | 64.7±5.8 | 70.3±6.5 |
| 50 | 71.7±5.8 | 71.4±5.9 | 71.8±5.9 | 79.2±5.8 |

### GLIM_enc

| k | Accuracy | Macro-F1 | BAcc | AUROC |
|---|----------|----------|------|-------|
| 3 | 58.0±8.2 | 56.9±8.6 | 58.3±7.6 | 61.1±10.3 |
| 5 | 60.3±6.9 | 59.5±7.0 | 60.5±6.7 | 64.4±9.1 |
| 10 | 65.6±6.1 | 65.1±6.2 | 65.7±6.0 | 71.0±6.9 |
| 20 | 70.7±5.9 | 70.4±6.0 | 70.9±5.9 | 77.1±6.5 |
| 50 | 78.4±6.4 | 78.2±6.5 | 78.4±6.4 | 85.0±6.6 |

### Cognitive_random

| k | Accuracy | Macro-F1 | BAcc | AUROC |
|---|----------|----------|------|-------|
| 3 | 58.0±7.6 | 56.7±8.0 | 58.1±7.0 | 61.5±10.3 |
| 5 | 59.7±7.1 | 58.9±7.4 | 59.9±6.8 | 64.5±8.7 |
| 10 | 63.8±5.9 | 63.4±5.9 | 63.9±5.7 | 69.3±7.1 |
| 20 | 67.1±5.4 | 66.7±5.5 | 67.2±5.3 | 74.0±6.1 |
| 50 | 70.7±5.1 | 70.4±5.2 | 70.8±5.1 | 78.2±5.6 |

### Cognitive_EEGtext

| k | Accuracy | Macro-F1 | BAcc | AUROC |
|---|----------|----------|------|-------|
| 3 | 58.2±7.7 | 57.0±8.1 | 58.5±7.1 | 61.8±9.9 |
| 5 | 60.4±6.9 | 59.7±7.0 | 60.6±6.6 | 65.2±8.8 |
| 10 | 66.0±6.1 | 65.5±6.0 | 66.1±5.9 | 71.5±6.8 |
| 20 | 70.8±6.1 | 70.5±6.1 | 71.0±6.0 | 77.8±6.7 |
| 50 | 78.7±6.5 | 78.5±6.6 | 78.7±6.5 | 85.7±6.3 |

## Key Questions Answered

### 1. STRG/STRE-lite 是否超过 AdaGTCN-lite?
Compare STRG_lite and STRE_lite vs AdaGTCN-lite from previous results.

### 2. STRG/STRE-lite 是否超过 PCET+GETA+CAGF?
Compare graph-based methods vs our best model.

### 3. GLIM-Encoder-proxy 是否有效?
GLIM-Encoder provides interpretable bottleneck representations.

### 4. CognitiveDecoder中 Text+EEG 是否超过 Text-only?
(Cognitive_random = random, Cognitive_EEGtext = EEG-derived)

### 5. 哪些方法适合放主表，哪些只能放 confound/upper-bound 表?

**Main Table**:
- STRG-lite, STRE-lite, Corr-GCN
- GLIM-Encoder
- PCET+GETA+CAGF

**Confound/Upper-bound Table**:
- Cognitive_random (random baseline)
- Cognitive_EEGtext (self-fusion, not true text)
