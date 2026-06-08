# SAN Report

## 1. 实验设置

### 1.1 归一化方法
- **StandardScaler**: 仅使用 calibration data
- **SourceNorm (SAN)**: 使用训练集统计信息作为锚点
- **TargetNorm**: 使用 calibration pool 统计信息
- **Shot settings**: 3, 5, 10, 20, 50 shots per class

### 1.2 Source Statistics 计算
```python
mu_source = mean(X_train_subjects[y_train == class_label])
sigma_source = std(X_train_subjects[y_train == class_label])
x_norm = (x - mu_source) / sigma_source
```

## 2. 核心结果

| Shot | StandardScaler | SourceNorm | Gain | TargetNorm | ACCS |
|------|---------------|------------|------|------------|------|
| 3 | 0.4346 | 0.4123 | -2.2% | 0.4746 | 0.4708 |
| 5 | 0.4161 | 0.4008 | -1.5% | 0.4759 | 0.4849 |
| **10** | 0.5764 | **0.6287** | **+5.2%** | 0.4815 | 0.5097 |
| **20** | 0.5964 | **0.7453** | **+14.9%** | 0.4552 | 0.5986 |
| **50** | 0.7623 | **0.8889** | **+12.7%** | 0.5707 | 0.7596 |

## 3. 成功标准验证

| 标准 | 结果 |
|------|------|
| C1: SourceNorm 10-shot ≥ baseline + 2% | **PASS** (+5.2%) |
| C2: SAN_ACCS ≥ ACCS + 1% (10/20/50) | **PASS** |
| C3: Difficult subjects 10-shot ≥ +2% | **PASS** (+6.6%) |
| C4: Macro-F1 & BAcc 同步提升 | **PASS** |

## 4. 机制分析

### 4.1 为什么 SAN 在 10+ shot 有效？
1. **跨 subject 模式捕获**: EEG 信号在跨 subject 间共享任务相关模式
2. **统计稳定性**: 15 个 subjects 的统计估计更稳健
3. **类分离改善**: Balanced Accuracy 显著提升

### 4.2 为什么 TargetNorm 失败？
- **统计噪声**: 低样本下均值/标准差估计噪声大
- **Subject-specific 偏置**: 个人 EEG 模式引入方差

### 4.3 SAN 与 ACCS 关系
**不是互补关系，而是替代关系**:
- SourceNorm 本身已经足够强
- ACCS 采样在 SourceNorm 框架下增益被削弱

## 5. Difficult Subjects 分析

| Subject | 5-shot Gain | 10-shot Gain |
|---------|-------------|--------------|
| YLS | -3.3% | **+4.1%** |
| YSL | -1.7% | **+3.3%** |
| YHS | -3.2% | **+20.1%** |
| YRP | +0.5% | -1.1% |

## 6. 论文表述

### 6.1 正确表述
> "SAN stabilizes personalized EEG calibration from moderate shot settings onward (10+ shots), using source-domain statistics as a stable normalization anchor."

### 6.2 避免表述
> ~~"Target-user normalization improves performance."~~ (TargetNorm 明显失败)
> ~~"SAN improves all low-shot settings."~~ (5-shot 以下效果不明显)

## 7. 结论

SAN 通过使用跨 subject 的源域统计信息作为稳定锚点，在 10-shot 及以上设置下显著优于传统归一化方法。该方法简单有效，无需调参，可作为 EEG 少样本用户校准的强 baseline。