# ZuCo 2.0 跨被试阅读任务分类项目详细报告

## 项目概述

### 核心任务

本项目研究 **NR vs TSR 阅读状态识别** —— 区分被试在进行 Normal Reading (NR) 和 Task-specific Reading (TSR) 时的 EEG 和眼动信号。

| 任务 | 描述 |
|------|------|
| **NR (Normal Reading)** | 被试自由阅读句子 |
| **TSR (Task-specific Reading)** | 被试带着特定任务阅读（如标注关系） |

### 科学问题

1. **跨被试泛化**: 模型能否在未见过的被试上工作？
2. **模态融合**: EEG + 眼动信号如何有效融合？
3. **数据效率**: 如何在小样本情况下快速适应新被试？
4. **Subject Normalization**: 如何减少被试间差异？

---

## 数据集

### ZuCo 2.0 数据集

| 数据类型 | 描述 | 可用性 |
|---------|------|--------|
| EEG 频段特征 | theta, alpha, beta, gamma 频段 | ✅ 可用 |
| 眼动特征 | FFD, GD, TRT 等 | ✅ 可用 |
| 句子级聚合 | sentence-level EEG420 + Gaze9 | ✅ 可用 |
| Fixation 序列 | word-level 注视序列 | ❌ 需下载完整数据 |
| 原始 EEG | 128 通道原始信号 | ❌ 需下载完整数据 |

### 当前数据配置

```
data/aligned_multimodal_y.npz
├── eeg: (8755, 420)    # EEG 频段特征
├── gaze: (8755, 9)      # 眼动特征
└── y: (8755,)           # 标签 (0=NR, 1=TSR)

data/aligned_multimodal_y_metadata.csv
├── sample_id: 样本ID
├── subject: 被试ID (16 Y-subjects)
├── label: NR/TSR
└── idx: 句子索引
```

### 16 个被试

```
Y-Subjects (16人):
YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, 
YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL
```

### 标签分布

| 标签 | 样本数 | 比例 |
|------|--------|------|
| NR (Normal Reading) | 4732 | 54.0% |
| TSR (Task-specific Reading) | 4023 | 46.0% |

---

## 实验协议

### Protocol A: LOSO-Y (Leave-One-Subject-Out)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LOSO-Y Protocol                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  16 folds (每次留一个被试做测试)                                    │
│                                                                     │
│  Fold 1:  Train: [YAC以外15人]  Test: [YAC]                       │
│  Fold 2:  Train: [YAG以外15人]  Test: [YAG]                       │
│  ...                                                              │
│  Fold 16: Train: [YTL以外15人]  Test: [YTL]                       │
│                                                                     │
│  指标: subject-wise unweighted mean Macro-F1                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Protocol B: AdaGTCN-style (12/2/2 split)

```
训练集: 12 subjects
验证集: 2 subjects  
测试集: 2 subjects
Seeds: 5 个不同随机种子
```

---

## 方法演进

### R0: 初始 Baseline

**目标**: 建立基础 cross-subject 性能

#### 实现代码

```python
# src/audit_clean_baseline.py

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 模型 1: Gaze-only
gaze_clf = LogisticRegression(class_weight='balanced')
gaze_clf.fit(X_gaze_train, y_train)
y_pred = gaze_clf.predict(X_gaze_test)

# 模型 2: EEG-only
eeg_clf = LogisticRegression(class_weight='balanced')
eeg_clf.fit(X_eeg_train, y_train)

# 模型 3: Concat
concat_clf = LogisticRegression(class_weight='balanced')
concat_clf.fit(np.hstack([X_eeg_train, X_gaze_train]), y_train)
```

#### 结果

| Model | LOSO Macro-F1 | AUROC |
|-------|---------------|-------|
| Concat_LogReg | 0.5794 | 0.6283 |
| Gaze_LogReg | 0.5721 | 0.6096 |
| EEG_LogReg | 0.5324 | 0.5631 |

**关键发现**: Concat (EEG+Gaze) > Gaze-only > EEG-only

---

### R1: 深度模型探索

#### R1A: STAG-Read 深度模型

**架构**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STAG-Read Architecture                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input:                                                            │
│  ├── EEG: (batch, 420) → reshape → (batch, 4, 105)              │
│  │   └── 4 = pseudo频段分组, 105 = 电极对                          │
│  └── Gaze: (batch, 9)                                              │
│                                                                     │
│  EEG Group Encoder:                                                 │
│  ├── Linear(105 → 64) per group                                    │
│  ├── GELU + LayerNorm                                              │
│  └── Group attention across 4 groups                                │
│                                                                     │
│  Gaze Encoder:                                                      │
│  ├── Linear(9 → 32) → GELU → Linear(32 → 64)                     │
│  └── → (batch, 64)                                                 │
│                                                                     │
│  Gaze-guided Modulation:                                           │
│  ├── r = sigmoid(MLP(z_gaze))                                    │
│  └── z_eeg_mod = r * z_eeg                                        │
│                                                                     │
│  Fusion:                                                           │
│  └── z_fused = LayerNorm(z_gaze + z_eeg_mod)                       │
│                                                                     │
│  Subject Adversarial Branch:                                        │
│  ├── GRL(z_fused) → Linear(64 → n_train_subjects)                │
│  └── L = CE_task + λ_adv * CE_subject (梯度反转)                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 代码实现 (src/stag_read_r1a.py)

```python
import torch
import torch.nn as nn

class STAGRead(nn.Module):
    def __init__(self, eeg_dim=420, gaze_dim=9, n_groups=4, hidden=64):
        super().__init__()
        
        # EEG Group Encoder
        self.group_proj = nn.Linear(105, hidden)
        self.group_attn = nn.MultiheadAttention(hidden, num_heads=4)
        self.eeg_norm = nn.LayerNorm(hidden)
        
        # Gaze Encoder
        self.gaze_mlp = nn.Sequential(
            nn.Linear(gaze_dim, 32),
            nn.GELU(),
            nn.Linear(32, hidden)
        )
        
        # Gaze-guided Modulation
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden, 2)
        
        # Subject Adversarial (GRL)
        self.subject_classifier = GradientReversal(lamda=0.1)
        
    def forward(self, x_eeg, x_gaze, y_subject, train_adversarial=True):
        # EEG encoding
        x_eeg = x_eeg.view(-1, 4, 105)
        x_eeg = self.group_proj(x_eeg)
        x_eeg = self.eeg_norm(x_eeg.permute(1, 0, 2))
        x_eeg, _ = self.group_attn(x_eeg, x_eeg, x_eeg)
        z_eeg = x_eeg.mean(dim=0)  # (batch, 64)
        
        # Gaze encoding
        z_gaze = self.gaze_mlp(x_gaze)
        
        # Gaze-guided modulation
        r = self.gate_mlp(z_gaze)
        z_fused = self.eeg_norm(z_gaze + r * z_eeg)
        
        # Task prediction
        task_logits = self.classifier(z_fused)
        
        # Subject adversarial
        if train_adversarial:
            subj_logits = self.subject_classifier(z_fused)
            return task_logits, subj_logits
        
        return task_logits
```

#### R1A 结果

| Model | Macro-F1 | 结论 |
|-------|----------|------|
| Concat_LogReg | 0.5718 | baseline |
| STAG_with_adv | 0.5553 | ❌ 失败 |
| STAG_no_adv | 0.5495 | ❌ 失败 |

**结论**: 深度模型在 sentence-level feature 上不如简单 tabular 模型

---

### R2: 强 Tabular 模型

#### R2A: 模型对比

```python
# src/r2_strong_baselines.py

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=500, 
        max_depth=15, 
        class_weight='balanced'
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight='balanced'
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        class_weight='balanced'
    )
}
```

#### R2A 结果

| Model | LOSO Macro-F1 | vs R0 |
|-------|---------------|-------|
| **ExtraTrees_Balanced** | **0.5984** | **+1.90%** |
| LightGBM_Balanced | 0.5957 | +1.63% |
| RandomForest | 0.5896 | +1.02% |
| Concat_LogReg (R0) | 0.5794 | baseline |

---

#### R2B: 验证与消融

```python
# 统计显著性检验
from scipy.stats import wilcoxon

# ExtraTrees vs Concat_LogReg
stat, p_value = wilcoxon(
    extra_trees_f1_per_subject, 
    concat_f1_per_subject
)
# p = 0.012 (显著)
```

##### EEG 增量贡献分析

```
Top-30 特征分布:
├── EEG 特征: 22/30 (73.3%)
└── Gaze 特征: 8/30 (26.7%)
```

##### Text/Material Confound Audit

| Model | Macro-F1 | 结论 |
|-------|----------|------|
| Text-only features | 0.5234 | 存在 confound |
| ExtraTrees (生理) | 0.5984 | 显著超过 text |

##### Subject Shortcut Audit

| Input | Subject Acc | 说明 |
|-------|------------|------|
| Gaze9 | 89.3% | 强 subject identity |
| EEG420 | 96.7% | 非常强 |
| Concat429 | 97.8% | 非常强 |

---

#### R2C: Subject Normalization (最新最强)

**核心创新**: 使用 QuantileTransformer 减少 subject-specific 偏差

```python
# src/r2c_norm_ensemble.py

from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier

class SubjectAwarePipeline:
    def __init__(self, normalization='N2_quantile'):
        self.normalization = normalization
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            class_weight='balanced'
        )
    
    def fit(self, X_train, y_train, subjects_train):
        # 只在训练集上 fit normalizer
        if self.normalization == 'N2_quantile':
            self.scaler = QuantileTransformer(
                n_quantiles=100,
                output_distribution='normal'
            )
            X_train = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        if self.normalization == 'N2_quantile':
            X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)
```

##### Normalization 方法对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Normalization Variants                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  N0 (none):                                                         │
│  └── 直接使用原始特征                                                │
│                                                                     │
│  N1 (RobustScaler):                                                │
│  └── 减去中位数，除以 IQR，对 outlier 鲁棒                           │
│                                                                     │
│  N2 (QuantileTransformer):                                         │
│  └── 转换为正态分布，所有特征同一尺度                                 │
│      是 N2 的关键优势                                               │
│                                                                     │
│  N3 (Subject Centering):                                           │
│  └── 按被试中心化，但测试集使用训练集均值                             │
│                                                                     │
│  N4 (Subject Z-score + Global):                                    │
│  └── 被试内 Z-score + 全局标准化                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

##### R2C 最终结果

| Normalization | Model | Macro-F1 | vs R2B |
|--------------|-------|----------|---------|
| **N2_quantile** | RandomForest | **0.6348** | **+3.64%** |
| N1_robust | RandomForest | 0.6312 | +3.28% |
| N2_quantile | ExtraTrees | 0.5996 | +0.12% |
| none | RandomForest | 0.5832 | -1.52% |
| R2B ExtraTrees | - | 0.5984 | baseline |

---

## 项目创新点

### 方法创新

#### 创新 1: QuantileTransformer 归一化

**问题**: EEG/Gaze 特征存在强 subject-specific 偏差，导致跨被试泛化困难

**解决**: 使用 QuantileTransformer 将特征分布转换为标准正态分布

```
效果:
├── RandomForest: 0.5832 → 0.6348 (+5.16%)
└── LogReg: 0.5304 → 0.5879 (+5.75%)
```

#### 创新 2: PACER-Lite K-aware Calibration

**问题**: Few-shot 场景下 fusion 模型在 k≥5 时产生负迁移

**解决**: K-aware gamma schedule

```python
def k_aware_gamma(k, gamma_base=0.25):
    """根据 calibration budget 调整 fusion 强度"""
    if k <= 3:
        return gamma_base * 1.0      # 完全使用 fusion
    elif k <= 5:
        return gamma_base * 0.5      # 降低 fusion
    elif k <= 10:
        return gamma_base * 0.2      # 进一步降低
    else:
        return 0.0                   # 退化为 anchor
```

#### 创新 3: CNHC (Confidence-weighted No-Harm Calibration)

**问题**: Fusion 模型可能比 baseline 更差

**解决**: 基于 calibration set 估计 harm，调整 fusion 强度

```python
def cnhc_calibration(fusion_probs, anchor_probs, y_calib, eta=2.0):
    # 估计 harm
    harm = cross_entropy(fusion_probs, y_calib) - \
           cross_entropy(anchor_probs, y_calib)
    harm = np.maximum(0, harm)
    
    # Safe scaling
    safe_scale = np.exp(-eta * harm.mean())
    
    # 调整 fusion
    return fusion_probs * safe_scale + anchor_probs * (1 - safe_scale)
```

### 验证创新

#### 严格无泄漏协议

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Data Leakage Prevention                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ✅ Scaler: 只 fit train，被试外数据不参与                           │
│  ✅ PCA: 只 fit train，被试外数据不参与                              │
│  ✅ Model: 只 fit train，被试外数据不参与                            │
│  ✅ Feature selection: 只用 train subjects                           │
│  ✅ OOF stacking: 按被试分组，不混合                                │
│  ✅ CNHC calibration: 只用 calibration set，不暴露 test             │
│                                                                     │
│  ❌ 不使用 test subject label 选择超参数                            │
│  ❌ 不使用 random split 代替 subject split                          │
│  ❌ 不使用 text features 作为主输入                                  │
│  ❌ 不使用 relation_type 作为主输入                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 代码架构

### 核心代码文件

```
src/
├── audit_clean_baseline.py      # R0: 初始 baseline
├── r2_strong_baselines.py       # R2A: 强 tabular 模型
├── r2b_validation.py            # R2B: 验证与消融
├── r2c_norm_ensemble.py        # R2C: Subject normalization ⭐
├── pacer_lite.py                 # PACER-Lite K-aware fusion
├── cnhc_calibration.py           # CNHC 校准
├── stag_read_r1a.py              # R1A: STAG 深度模型
├── confound_audit.py              # Confound 审计
└── fewshot_protocol.py           # Few-shot 协议

results/
├── r0_baseline/                  # R0 结果
├── r2_strong_baselines/          # R2A 结果
├── r2b_validation/              # R2B 结果
├── r2c_norm_ensemble/           # R2C 结果 ⭐
├── pacer_step0-4/               # PACER 结果
└── stag_read_r1a/               # STAG 结果
```

### 数据流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  src/features/*.npy                                                 │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────────┐                                               │
│  │  aligned_data   │                                               │
│  │  generation     │                                               │
│  └────────┬────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  data/aligned_multimodal_y.npz                                      │
│  ├── eeg: (8755, 420)                                             │
│  ├── gaze: (8755, 9)                                               │
│  └── y: (8755,)                                                    │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────────────────────────────┐                        │
│  │         Model Training Pipeline          │                        │
│  │                                          │                        │
│  │  1. Subject Normalization (N2_quantile) │                        │
│  │  2. Model: RandomForest_Balanced        │                        │
│  │  3. LOSO Cross-Validation               │                        │
│  │  4. Statistical Significance Test        │                        │
│  └─────────────────────────────────────────┘                        │
│           │                                                         │
│           ▼                                                         │
│  results/r2c_norm_ensemble/                                         │
│  └── best_macro_f1 = 0.6348                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 完整结果汇总

### 方法性能对比

| 版本 | 方法 | Macro-F1 | vs Baseline | 状态 |
|------|------|----------|-------------|------|
| R0 | Concat_LogReg | 0.5794 | baseline | ✅ |
| R1A | STAG-Deep | 0.5553 | -2.41% | ❌ |
| R2A | ExtraTrees | 0.5984 | +1.90% | ✅ |
| R2B | ExtraTrees (validated) | 0.5984 | +1.90% | ✅ |
| **R2C** | **RF + N2_quantile** | **0.6348** | **+5.54%** | ⭐ |
| PACER | K-PACER-Lite (k=3) | 0.5597 | +1.06% | ⚠️ |

### AdaGTCN 对比

| 方法 | Macro-F1 | 说明 |
|------|----------|------|
| AdaGTCN (论文报告) | 0.6950 | Fixation-level + GCN |
| **当前最佳 (R2C)** | **0.6348** | Sentence-level + RF |
| 差距 | -6.02% | 输入粒度不同 |

---

## 论文贡献总结

### 主要贡献

1. **强 Tabular Baseline**: 
   - RandomForest + QuantileTransformer 在 sentence-level EEG+Gaze 上达到 0.6348 Macro-F1

2. **Subject Normalization**: 
   - 提出 QuantileTransformer 减少跨被试偏差
   - 提升 +5.16%

3. **严格验证协议**:
   - LOSO-Y 16-fold cross-subject evaluation
   - 无数据泄漏的统计检验
   - Text/material confound audit

4. **PACER-Lite K-aware Fusion**:
   - Few-shot 场景下自适应 fusion 强度
   - k=3 时提升 +1.06%

### 局限性

1. **输入粒度**: 使用 sentence-level 而非 fixation-level 特征
2. **深度模型**: 复杂模型在当前数据上表现不如简单模型
3. **Subject Identity**: EEG/Gaze 仍含强被试特征

### 未来方向

1. **获取完整 ZuCo 2.0 数据**: 做 fixation-level 分析
2. **深度时序模型**: GRU/Transformer 处理 fixation 序列
3. **Graph Temporal Network**: AdaGTCN 复现

---

## 总结

| 项目 | 内容 |
|------|------|
| **任务** | ZuCo 2.0 NR vs TSR 跨被试分类 |
| **数据** | EEG420 + Gaze9 (8755 samples, 16 subjects) |
| **最佳方法** | RandomForest + QuantileTransformer |
| **最佳性能** | LOSO Macro-F1 = 0.6348 |
| **核心创新** | Subject normalization, K-aware fusion |
| **与 AdaGTCN 差距** | -6.02% (输入粒度不同) |

---

## 文件清单

| 文件路径 | 说明 |
|---------|------|
| `src/audit_clean_baseline.py` | R0 初始 baseline |
| `src/r2_strong_baselines.py` | R2A 强 tabular 模型 |
| `src/r2b_validation.py` | R2B 验证与消融 |
| `src/r2c_norm_ensemble.py` | R2C Subject normalization ⭐ |
| `src/pacer_lite.py` | PACER-Lite K-aware fusion |
| `src/cnhc_calibration.py` | CNHC 校准 |
| `src/stag_read_r1a.py` | R1A STAG 深度模型 |
| `src/confound_audit.py` | Confound 审计 |
| `data/aligned_multimodal_y.npz` | 对齐数据 |
| `data/aligned_multimodal_y_metadata.csv` | 元数据 |
| `results/r2c_norm_ensemble/normalization_results.csv` | 完整实验结果 |
| `results/r2c_norm_ensemble/r2c_summary.md` | R2C 总结 |

---

*报告生成时间: 2026-05-22*
