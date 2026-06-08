# UC-DAR 实验协议卡

## 1. 基础设置

### 1.1 数据集
- **Dataset**: ZuCo 2.0
- **Task**: Natural Reading (NR) vs Task-Specific Reading (TSR) classification
- **Subjects**: 16 Y-subjects (YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL)
- **Modalities**: EEG (14 channels), Eye-tracking

### 1.2 实验参数
- **Seeds**: 0, 1, 2, 3, 4
- **Evaluation**: Leave-One-Subject-Out (LOSO) cross-validation
- **Metrics**: Accuracy, Macro-F1, Balanced Accuracy, AUROC

## 2. 数据划分规则

### 2.1 Zero-shot 实验 (实验1)
```
训练集: 15 subjects
测试集: 1 held-out subject
总计: 16 folds
```

### 2.2 Personalized/Few-shot 实验 (实验3-5)
```
原始数据: held-out subject 的全部数据
├── 测试集: 1/3 (不参与校准，不用于任何统计计算)
└── Calibration Pool: 2/3 (未标注，可用于采样和统计)

校准采样: 从 Calibration Pool 中采样
- n-shot per class: 每类采样 n 个，总共 2n 个 labeled calibration trials
- 50-shot per class = 每类50个 = 总共100个 labeled calibration trials
```

## 3. 归一化方法规范

### 3.1 StandardScaler (Baseline)
```python
scaler = StandardScaler()
scaler.fit(X_cal)  # 仅在 calibration data 上 fit
X_cal_norm = scaler.transform(X_cal)
X_test_norm = scaler.transform(X_test)
```

### 3.2 SourceNorm / SAN
```python
# mu_source, sigma_source 只来自训练 subjects (15 subjects)
mu_source_0 = mean(X_train[y_train == 0])
sigma_source_0 = std(X_train[y_train == 0])
mu_source_1 = mean(X_train[y_train == 1])
sigma_source_1 = std(X_train[y_train == 1])

# 归一化使用 source statistics
X_norm[y == 0] = (X[y == 0] - mu_source_0) / sigma_source_0
X_norm[y == 1] = (X[y == 1] - mu_source_1) / sigma_source_1
```

**关键**: mu_source 和 sigma_source 绝对不包含 held-out subject 的任何数据。

### 3.3 TargetNorm
```python
# mu_target, sigma_target 只来自 calibration pool
mu_target_0 = mean(X_cal_pool[y_cal_pool == 0])
sigma_target_0 = std(X_cal_pool[y_cal_pool == 0])
mu_target_1 = mean(X_cal_pool[y_cal_pool == 1])
sigma_target_1 = std(X_cal_pool[y_cal_pool == 1])
```

### 3.4 SASN (Shrinkage Normalization)
```python
rho = n_target / (n_target + kappa)
mu_sasn = rho * mu_target + (1 - rho) * mu_source
sigma_sasn = rho * sigma_target + (1 - rho) * sigma_source
```

## 4. ACCS 采样协议

### 4.1 Protocol A: Label-free Budget (主协议)
```python
# KMeans 仅在 X_pool (无标注) 上 fit
scaler = StandardScaler()
X_pool_s = scaler.fit_transform(X_pool)  # 只用 features，不涉及 label
kmeans = KMeans(n_clusters=k)
kmeans.fit(X_pool_s)  # 不使用任何 label 信息

# 从每个 cluster 选择最近的 centroid 样本
selected = []
for c in range(k):
    cluster_idx = where(kmeans.labels_ == c)
    centroid = kmeans.cluster_centers_[c]
    closest = cluster_idx[argmin(distance(X_pool_s[cluster_idx], centroid))]
    selected.append(closest)
```

**关键**: ACCS 采样过程完全不使用任何 label 信息。

## 5. 泄漏检查清单

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 1. mu_source 只来自训练 subjects | ✅ | 明确排除 held-out subject |
| 2. 不包含 held-out test subject | ✅ | held-out subject 仅用于测试 |
| 3. 不包含 held-out test set | ✅ | test set 仅用于最终评估 |
| 4. 不使用 test labels | ✅ | test labels 仅用于计算 metrics |
| 5. calibration/test 严格分离 | ✅ | 使用不同 index sets |
| 6. SourceNorm 没有在所有 Y-subject 上 fit | ✅ | 只在 15 个训练 subjects 上 fit |
| 7. 所有方法使用相同 seeds 和 splits | ✅ | 协议统一 |
| 8. 50-shot = 每类50个 = 总共100个 | ✅ | 代码明确实现 |
| 9. 归一化只使用 features | ✅ | mu/sigma 计算只用 X |

## 6. SIED 协议

### 6.1 模型结构
```python
EEGEncoder -> TaskClassifier (CE loss)
          -> GradientReversalLayer -> SubjectDiscriminator (CE loss)

L_total = L_task + lambda_adv * L_subject_adv
```

### 6.2 Lambda 敏感性设置
```python
lambda_adv = [0, 0.001, 0.005, 0.01, 0.05, 0.1]
```

## 7. 论文写作边界

### 7.1 可以写
- "UC-DAR is a plug-and-play user-calibrated domain adaptation pipeline for EEG-aware reading state recognition"
- "SIED partially improves zero-shot cross-user EEG transfer"
- "ACCS improves calibration sample efficiency"
- "SAN stabilizes personalized EEG calibration from moderate shot settings onward"

### 7.2 不能写
- ~~"EEG decodes pure cognitive state"~~
- ~~"SIED fully solves cross-user generalization"~~
- ~~"ACCES reduces manual annotation cost"~~
- ~~"SAN improves all low-shot settings"~~
- ~~"NR/TSR classification is free from text/material confounds"~~

### 7.3 正确的任务定义
> "Protocol-conditioned reading state recognition" - 在给定阅读协议(NR vs TSR)条件下识别阅读状态

## 8. 实验清单

| 实验 | 描述 | 输出文件 |
|------|------|----------|
| 1 | Zero-shot SIED | zero_shot_loso_results.csv, zero_shot_sied_report.md |
| 2 | SIED Lambda 敏感性 | sied_lambda_sensitivity.csv, sied_information_retention_report.md |
| 3 | ACCS 主实验 | accs_results.csv, accs_significance.csv, accs_report.md |
| 4 | SAN 主实验 | san_results.csv, san_significance.csv, san_report.md |
| 5 | Text confound | text_confound_results.csv, text_confound_analysis.md |
| 6 | 最终表格 | main_table_*.csv, final_paper_experiment_summary.md |

## 9. 统计检验方法

- **配对检验**: Wilcoxon signed-rank test
- **显著性水平**: α = 0.05
- **比较对象**:
  - SIED vs Raw_EEG
  - SAN vs StandardScaler
  - ACCS vs Random

## 10. 目录结构

```
results/final/
├── zero_shot_loso_results.csv
├── zero_shot_significance.csv
├── sied_lambda_sensitivity.csv
├── accs_results.csv
├── accs_significance.csv
├── san_results.csv
├── san_significance.csv
├── text_confound_results.csv
├── main_table_zero_shot.csv
├── main_table_personalized.csv
└── main_table_confound.csv

reports/final/
├── protocol_card.md
├── zero_shot_sied_report.md
├── sied_information_retention_report.md
├── accs_report.md
├── san_report.md
├── text_confound_analysis.md
└── final_paper_experiment_summary.md

figures/
├── final_framework.pdf
├── san_accs_shot_curve.pdf
├── sied_tradeoff_curve.pdf
├── subject_level_gain_heatmap.pdf
└── text_confound_comparison.pdf
```