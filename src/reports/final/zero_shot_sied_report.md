# Zero-shot SIED Report

## 1. 实验设置

### 1.1 Protocol
- **Subjects**: 16 Y-subjects (LOSO)
- **Seeds**: 0, 1, 2, 3, 4
- **Metrics**: Accuracy, Macro-F1, Balanced Accuracy, AUROC

### 1.2 Models
- **Raw_EEG**: StandardScaler + SGDClassifier (no adaptation)
- **SIED**: Subject-Invariant EEG Disentanglement with gradient reversal

## 2. 核心结果

| Model | Accuracy | Macro-F1 | Balanced Accuracy | AUROC |
|-------|----------|----------|-------------------|-------|
| Raw_EEG | 0.5082 ± 0.0632 | 0.4226 | 0.5164 | - |
| SIED | 0.5286 ± 0.0481 | 0.4263 | 0.5237 | - |

**SIED vs Raw_EEG**: +2.04% accuracy improvement

## 3. Subject-level Analysis

| Subject | Raw_EEG | SIED | Gain |
|---------|---------|------|------|
| YLS | 0.4085 | 0.6277 | +21.9% |
| YRK | 0.5128 | 0.6239 | +11.1% |
| YDG | 0.6065 | 0.5418 | -6.5% |
| YFR | 0.5057 | 0.5029 | -0.3% |
| YAC | 0.4389 | 0.5417 | +10.3% |

## 4. 关键发现

1. **SIED 部分改善零样本跨用户迁移**: +2.04% accuracy improvement
2. **Subject-level 效果差异大**: YLS 提升 21.9%，但部分 subject 反而下降
3. **Subject predictability 需要进一步分析**: 需要验证 SIED 是否真的减少了 subject identity encoding

## 5. 结论

> SIED partially improves zero-shot cross-user EEG transfer, demonstrating that subject-adversarial training can help EEG generalize across users. However, the improvement is modest (+2.04%) and varies significantly across subjects, suggesting that SIED alone does not fully solve cross-user generalization.

## 6. 论文表述

### 6.1 可以写
> "SIED partially improves zero-shot cross-user EEG transfer, achieving +2.04% accuracy improvement over raw EEG features."

### 6.2 不能写
> ~~"SIED fully solves cross-user generalization."~~
> ~~"EEG decodes pure cognitive state."~~