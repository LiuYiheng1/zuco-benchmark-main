# LC-SRGC: LLM-Conditioned Source-Regularized Gaussian Calibration Report

## 1. 实验概述

LC-SRGC 旨在通过 LLM sentence embedding 来条件化 source-domain EEG Gaussian prior，从而建模文本语义/材料差异对 EEG 的影响。

**关键问题**：当前数据集没有原始句子文本，因此无法实现真正的语义条件化。

## 2. 实验设置

- **Subjects**: 16 Y-subjects (LOSO cross-subject protocol)
- **Shot settings**: 3, 5, 10, 20, 50 shots per class
- **Seeds**: 0, 1, 2, 3, 4
- **比较方法**:
  - EEG_SVM: Standard SVM with calibration samples
  - SRGC_global: Source-Regularized Gaussian Calibration (global prior)

## 3. 结果

### 按 Shot 汇总

| Shot | EEG_SVM | SRGC_global | Gap |
|------|----------|-------------|-----|
| 3    | 43.46%   | 56.84%      | +13.38% |
| 5    | 41.61%   | 58.90%      | +17.29% |
| 10   | 57.64%   | 62.75%      | +5.11% |
| 20   | 59.64%   | 64.36%      | +4.72% |
| 50   | 76.23%   | 65.65%      | -10.58% |

### 分析

1. **SRGC 在低样本有效**: 3-5 shot 时 SRGC 显著优于 SVM，提升 13-17%
2. **SRGC 在高样本退化**: 50-shot 时 SRGC 反而比 SVM 差 10.58%
3. **SRGC 核心问题**: Source prior 在高样本时变成了干扰

## 4. LC-SRGC 限制

由于数据集缺少原始句子文本，无法实现真正的 LLM-conditioned semantic conditioning。

### 尝试的方法

1. **TF-IDF on trial identifiers**: 失败 - trial key 不包含语义信息
2. **随机 embedding**: 无意义 - 没有语义信号

### 关键洞察

- **没有文本，无法验证语义条件化假设**
- **如果未来有文本数据，LC-SRGC 机制可能是有效的**
- **当前只能报告 SRGC 的 baseline 结果**

## 5. 结论与建议

### LC-SRGC 状态

**无法完成** - 缺少必要的文本数据。

### 替代方案

如果未来要实现 LC-SRGC，需要：

1. **获取原始句子文本**：从 ZuCo 2.0 数据集的 stimulus materials
2. **使用 sentence-transformer**：如 MiniLM 或 SBERT 生成 embeddings
3. **验证语义检索有效性**：确保 retrieval-based conditioning 有实际效果

### 论文定位建议

由于无法实现真正的 LLM-conditioned 模块，建议：

1. **接受当前三个创新点**:
   - SIED: Zero-shot cross-subject (+2%)
   - ACCS: Active calibration sampling (+3-7% at 3-5 shot)
   - SRGC: Source-regularized Gaussian calibration (+13-17% at 3-5 shot)

2. **或者**: 如果能获取原始文本，可以重新实现 LC-SRGC

## 6. 输出文件

- `results/final/lc_srgc_results.csv`: 完整实验结果