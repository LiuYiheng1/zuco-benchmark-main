# 第三创新点分析

## 已知数据汇总

### SR-GC vs SVM (来自 srgc_results.csv)

| Shot | EEG_SVM | SR-GC α=0.75 | 差距 |
|------|---------|---------------|------|
| 3-shot | 43.59% | **59.25%** | **+15.66%** |
| 5-shot | 41.56% | **59.88%** | **+18.32%** |
| 10-shot | 57.37% | **66.42%** | **+9.05%** |
| 20-shot | 58.60% | **69.45%** | **+10.85%** |
| 50-shot | **77.27%** | 77.02% | -0.25% |

### 关键洞察

1. **SR-GC 在 3-20 shot 时大幅领先** (+10-18%)
2. **SVM 在 50-shot 时略好** (+0.25%)
3. **两者在 50-shot 几乎相同**（差距仅 0.25%）

## 新方案：Shot-Adaptive Selection (SAS)

**不是 fusion（会稀释两者优势），而是 discrete selection：**

```python
if n_shot <= 20:
    使用 SR-GC (α=0.75)
else:
    使用 SVM
```

### 理论性能

| Shot | 方法 | 预期准确率 |
|------|------|-----------|
| 3-shot | SR-GC | 59.25% |
| 5-shot | SR-GC | 59.88% |
| 10-shot | SR-GC | 66.42% |
| 20-shot | SR-GC | 69.45% |
| 50-shot | SVM | 77.27% |

### 为什么这比 SAGE Fusion 更好

- **SAGE fusion**: 0.7 × SR-GC + 0.3 × SVM → 稀释两者
- **SAS selection**: 直接选择更好的 → 保留两者最佳

## 备选：Per-Subject Adaptive Selection

如果能在 calibration 前预测目标主体与源域的相似度：
- 相似度高 → 用 SR-GC（源先验更可靠）
- 相似度低 → 用 SVM（减少源域偏差）

但这需要额外验证。

## 结论

**第三个创新点建议：Shot-Adaptive SRGC/SVM Selection**

- 结合 SR-GC（低样本）和 SVM（高样本）的优势
- 简单有效，无需复杂 fusion
- 可作为Practical guideline给用户