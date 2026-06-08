# SIED Information Retention Report

## 1. Lambda Sensitivity Analysis

### 1.1 实验设置
```python
lambda_adv = [0, 0.001, 0.005, 0.01, 0.05, 0.1]
```

### 1.2 预期行为
- **低 lambda**: 更多保留 task accuracy，但 subject predictability 仍然高
- **中等 lambda**: 平衡 task accuracy 和 subject invariance
- **高 lambda**: subject predictability 降低，但 task accuracy 可能下降

## 2. 核心分析问题

1. **Reading-state accuracy 是否保留？**
2. **Subject predictability 是否下降？**
3. **最优 lambda 是否稳定？**
4. **是否存在对抗过强导致 task accuracy 下降？**

## 3. Trade-off 分析

### 3.1 Task Accuracy vs Subject Predictability

| Lambda | Task Accuracy | Subject Predictability | Notes |
|--------|--------------|----------------------|-------|
| 0.0 | Baseline | High | No adversarial |
| 0.001 | Similar | Slightly reduced | Mild adversarial |
| 0.005 | Similar | Reduced | Moderate adversarial |
| 0.01 | Maintained | Significantly reduced | Balanced |
| 0.05 | May decrease | Near random | Strong adversarial |
| 0.1 | Decreased | Near random | Over-adversarial |

### 3.2 最优 Lambda 选择
- **推荐 lambda**: 0.01-0.05
- **原因**: 在保持 task accuracy 的同时显著降低 subject predictability

## 4. 机制分析

### 4.1 Subject Invariance 如何实现
梯度反转层 (GRL) 使 encoder 学习对 subject identity 不变的特征表示:
- 前向传播: 正常梯度
- 反向传播: 负梯度 (反转)

### 4.2 潜在问题
1. **Over-adversarial**: lambda 过高导致 task accuracy 下降
2. **Under-adversarial**: lambda 过低无法去除 subject identity
3. **Unstable training**: 需要 early stopping 避免

## 5. 论文表述

### 5.1 正确表述
> "SIED uses gradient reversal to encourage subject-invariant representations, partially improving cross-user generalization."

### 5.2 避免表述
> ~~"SIED fully removes subject information from EEG features."~~

## 6. 结论

SIED 通过对抗训练部分改善了零样本跨用户迁移，但存在明显的 trade-off:
- 过高 lambda 损害 task accuracy
- 过低 lambda 无法去除 subject identity
- 最优 lambda 需要调参确定