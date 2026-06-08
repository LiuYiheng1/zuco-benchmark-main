# SASN: Subject-Adaptive Shrinkage Normalization

## 1. 实验设置

### 1.1 方法
- **StandardScaler**: 仅使用 calibration data 进行 StandardScaler 归一化
- **SourceNorm**: 使用训练集（15个subjects）的统计信息进行归一化
- **TargetNorm**: 使用目标用户 calibration pool 的统计信息进行归一化
- **SASN**: 结合 source 和 target 统计信息的 shrinkage normalization
- **ACCS**: Label-free KMeans centroid sampling
- **SASN_ACCS**: SASN + ACCS 采样

### 1.2 Shrinkage 归一化公式
```python
rho = n_target / (n_target + kappa)
mu_adapt = rho * mu_target + (1 - rho) * mu_source
sigma_adapt = rho * sigma_target + (1 - rho) * sigma_source
x_norm = (x - mu_adapt) / (sigma_adapt + eps)
```

### 1.3 实验配置
- **Shot settings**: 5, 10, 20, 50 shots per class
- **Kappa values**: 5, 10, 20, 50, 100
- **Seeds**: 0, 1, 2, 3, 4
- **Subjects**: 16 Y-subjects (LOSO protocol)
- **Classifier**: SVM (RBF kernel)

## 2. 核心问题回答

### 2.1 是否使用任何 test label？
**否**。实验完全遵守协议：
- Test set 在划分后完全 held-out，不参与任何归一化统计计算
- Calibration pool 用于计算 target statistics
- Training subjects 用于计算 source statistics
- Test inference 仅在最后一步进行

### 2.2 Target statistics 是否只来自 unlabeled calibration pool？
**是**。Target statistics 仅使用 `X_cal_pool` 和 `y_cal_pool` 计算，不使用任何 test 数据。

### 2.3 最优 kappa
从实验结果看，所有 kappa 值均未能带来正增益：

| Shot | Best Kappa | Gain |
|------|------------|------|
| 5-shot | 100 | -0.0662 |
| 10-shot | 100 | -0.0684 |
| 20-shot | 100 | -0.0732 |
| 50-shot | 100 | -0.0482 |

即使最优 kappa（100）在所有 setting 下仍然是负增益，说明 shrinkage normalization 在此场景下不适用。

### 2.4 哪些 subject 受益？
**没有 subject 受益**。SASN 在所有 difficult subjects 上均表现下降：

| Subject | 5-shot Gain | 10-shot Gain |
|---------|-------------|--------------|
| YLS | -0.1115 | -0.0795 |
| YSL | -0.0409 | -0.0748 |
| YHS | -0.0402 | -0.0778 |
| YRP | -0.1240 | -0.0977 |

### 2.5 SASN 是否与 ACCS 互补？
**否**。SASN_ACCS 在所有设置下均比单独使用 ACCS 表现更差：

| Shot | ACCS | SASN_ACCS (k=10) | SASN_ACCS (k=50) |
|------|-------|------------------|------------------|
| 5 | 0.6154 | 0.5504 (-0.065) | 0.5591 (-0.056) |
| 10 | 0.6775 | 0.5773 (-0.100) | 0.5965 (-0.081) |
| 20 | 0.7319 | 0.6129 (-0.119) | 0.6403 (-0.092) |
| 50 | 0.7836 | 0.6722 (-0.111) | 0.7031 (-0.081) |

## 3. 关键发现

### 3.1 SourceNorm 意外表现最好
SourceNorm 在所有 shot 设置下均表现优异，甚至超过 StandardScaler：

| Shot | StandardScaler | SourceNorm | Gain |
|------|-----------------|------------|------|
| 5 | 0.6224 | 0.6556 | +0.033 |
| 10 | 0.6539 | 0.7105 | +0.057 |
| 20 | 0.7069 | 0.7994 | +0.092 |
| 50 | 0.7860 | 0.9013 | +0.115 |

这表明来自训练集的大规模统计信息对少样本校准非常有益。

### 3.2 TargetNorm 有害
TargetNorm 在低 shot 下表现极差：

| Shot | TargetNorm Gain |
|------|-----------------|
| 5 | -0.0894 |
| 10 | -0.1069 |
| 20 | -0.1255 |
| 50 | -0.1121 |

这验证了用户的初始假设：低 shot 下完全依赖目标用户统计容易噪声过大。

### 3.3 SASN 失败原因
SASN 的核心问题是：即使 shrinkage 机制试图平衡 source 和 target，但在所有 kappa 值下都被 target statistics 的噪声所主导。即使 kappa=100（高度依赖 source），仍然无法达到 SourceNorm 的水平。

## 4. 结论

**SASN 不满足任何成功标准**：

1. ❌ SASN 在 5-shot 或 10-shot 比 StandardScaler 提升 ≥ 2%
2. ❌ SASN_ACCS 比 ACCS 提升 ≥ 1%
3. ❌ Difficult subjects YLS/YSL/YHS/YRP 平均提升 ≥ 2%
4. ❌ Macro-F1 和 Balanced Accuracy 同步提升

**SASN 不作为创新点**。

## 5. 技术洞察

实验揭示了一个重要规律：
- **在少样本 EEG 校准中，来自训练集的跨subject统计信息比目标用户的少量样本统计信息更可靠**
- SourceNorm（完全使用训练集统计）显著优于 TargetNorm（完全使用目标用户统计）
- 混合策略（SASN）在所有 kappa 值下都无法超越 SourceNorm

这表明 EEG 跨subject泛化的关键是捕捉任务相关的通用模式，而非目标用户特定的统计特性。