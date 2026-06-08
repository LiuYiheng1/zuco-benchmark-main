# UC-DAR 最终论文实验总结

## 1. 论文框架

**UC-DAR: User-Calibrated Domain Adaptation for EEG-aware Reading**

三个核心模块：
1. **SIED**: Subject-Invariant EEG Disentanglement
2. **ACCS**: Active Cognitive Calibration Sampling
3. **SAN**: Source-Anchored Normalization

## 2. 主要贡献

### 2.1 创新点 1: SIED (Zero-shot Cross-user Transfer)
- **方法**: 对抗梯度反转减少 EEG 中的 subject identity 信息
- **结果**: Raw EEG 50.82% → SIED 52.86% (+2.04%)
- **局限**: 部分 subject 提升显著，部分反而下降

### 2.2 创新点 2: ACCS (Calibration Sample Efficiency)
- **方法**: Label-free KMeans centroid sampling
- **结果**: 5-shot 下比 Random 提升 +6.9%
- **局限**: 10-shot 以上效果减弱

### 2.3 创新点 3: SAN (Stable Normalization Anchor)
- **方法**: 使用跨 subject 源域统计信息作为归一化锚点
- **结果**: 10-shot +5.2%, 20-shot +14.9%, 50-shot +12.7%
- **局限**: 5-shot 以下效果不明显

## 3. 最终结果汇总

### 3.1 Zero-shot Results (16 Y-subjects LOSO)

| Model | Accuracy | Macro-F1 | Balanced Accuracy |
|-------|----------|----------|------------------|
| Raw_EEG | 0.5082 | 0.4226 | 0.5164 |
| SIED | 0.5286 | 0.4263 | 0.5237 |

### 3.2 Personalized Results (50-shot per class)

| Method | Accuracy | Macro-F1 | Balanced Accuracy |
|--------|----------|----------|------------------|
| StandardScaler | 0.7623 | - | - |
| TargetNorm | 0.5707 | - | - |
| ACCS | 0.7596 | - | - |
| **SAN** | **0.8889** | - | - |

### 3.3 Shot Curve Summary

| Shot | StandardScaler | ACCS | SAN |
|------|---------------|------|-----|
| 3 | 0.4346 | 0.4708 | 0.4123 |
| 5 | 0.4161 | 0.4849 | 0.4008 |
| 10 | 0.5764 | 0.5097 | **0.6287** |
| 20 | 0.5964 | 0.5986 | **0.7453** |
| 50 | 0.7623 | 0.7596 | **0.8889** |

## 4. 论文写作边界

### 4.1 可以写
- "UC-DAR is a plug-and-play user-calibrated domain adaptation pipeline for EEG-aware reading state recognition."
- "SIED partially improves zero-shot cross-user EEG transfer."
- "ACCS improves calibration sample efficiency."
- "SAN stabilizes personalized EEG calibration from moderate shot settings onward."

### 4.2 不能写
- ~~"EEG decodes pure cognitive state."~~
- ~~"SIED fully solves cross-user generalization."~~
- ~~"ACCES reduces manual annotation cost."~~
- ~~"SAN improves all low-shot settings."~~
- ~~"NR/TSR classification is free from text/material confounds."~~

## 5. 任务定义

> **Protocol-conditioned reading state recognition**: 在给定阅读协议(NR vs TSR)条件下识别阅读状态

本文不声称纯粹的 stimulus-invariant cognitive decoding，因为我们无法排除 text/material confounds。

## 6. 实验清单

| 实验 | 状态 | 输出文件 |
|------|------|----------|
| Zero-shot SIED | ✅ 完成 | zero_shot_loso_results.csv |
| SIED Lambda 敏感性 | ✅ 完成 | sied_lambda_sensitivity.csv |
| ACCS 主实验 | ✅ 完成 | accs_results.csv |
| SAN 主实验 | ✅ 完成 | san_results.csv |
| Text confound | ⏳ 待补充 | - |

## 7. 核心结论

1. **EEG cross-user 迁移困难**: Raw EEG zero-shot accuracy 仅 50.82%
2. **SIED 部分有效**: +2.04% 改善，但不解决根本问题
3. **SAN 是最强方法**: 10+ shot 下显著优于所有其他方法
4. **TargetNorm 失败**: 验证了"低样本下目标用户统计噪声大"的假设
5. **SAN + ACCS 无互补**: SourceNorm 本身已足够强

## 8. 局限性和未来方向

### 8.1 局限性
1. SIED 在部分 subject 上反而降低性能
2. SAN 在 5-shot 以下效果不明显
3. 无法完全排除 text/material confounds

### 8.2 未来方向
1. 结合 SIED 和 SAN
2. Subject-adaptive lambda
3. Text/material confound 控制