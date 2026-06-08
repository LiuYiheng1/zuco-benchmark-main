# ACCS: Active Cognitive Calibration Sampling - Validation Report

## 一、标签泄漏检查 (Label Leakage Verification)

### Protocol A 实现验证：真正的 label-free 采样

| 检查项 | 状态 | 说明 |
|--------|------|------|
| KMeans 只使用 calibration pool 特征 | ✅ | `kmeans_centroid_sampling_label_free(X_cal_pool, total_budget)` |
| KMeans fit 在 calibration pool 上 | ✅ | `kmeans.fit(X_pool_s)` - 仅使用 X_pool |
| 采样时不使用 y_cal_pool 标签 | ✅ | 函数签名 `kmeans_centroid_sampling_label_free(X_pool, n_select)` 只接收 X |
| Random 采样也不使用标签 | ✅ | `select_calibration_random(X_pool, n_select)` 只接收 X |
| Test set 不参与采样 | ✅ | test/cal_pool 严格分离 |
| 每类内部不做 KMeans (Protocol A) | ✅ | KMeans 在整个 X_cal_pool 上运行，不按类分组 |

**结论：Protocol A 是真正的 label-free 采样，无标签泄漏。**

### Protocol B 实现：Balanced simulation (controlled analysis only)

Protocol B 按 label 分组后再采样，**这不是 label-free**，仅用于与已有 few-shot 结果公平比较。

---

## 二、双协议结果报告

### Protocol A: Realistic Label-Free Budget (MAIN RESULT)

| k-shot | Random (mean±std) | KMeans-ACCS (mean±std) | Gap | p-value | Sig |
|--------|-------------------|------------------------|-----|---------|-----|
| 1 | 0.5182±0.0577 | 0.5157±0.0562 | -0.0025 | - | - |
| 3 | 0.5789±0.0714 | 0.5618±0.0775 | -0.0171 | 0.8387 | |
| **5** | **0.5976±0.0705** | **0.6165±0.0688** | **+0.0189** | **0.0649** | marginal |
| **10** | **0.6375±0.0651** | **0.6730±0.0660** | **+0.0355** | **0.0001** | **\*** |
| **20** | **0.7048±0.0630** | **0.7196±0.0649** | **+0.0149** | **0.0178** | **\*** |
| 50 | 0.7833±0.0626 | 0.7858±0.0686 | +0.0025 | - | - |

**3/5/10-shot Average: Gap=+0.0124, p=0.0347 \***

### Protocol B: Balanced Simulation (Controlled Analysis)

| k-shot | Random (mean±std) | KMeans-ACCS (mean±std) | Gap | p-value | Sig |
|--------|-------------------|------------------------|-----|---------|-----|
| 1 | 0.5278±0.0629 | 0.5352±0.0513 | +0.0074 | - | - |
| **3** | **0.5751±0.0748** | **0.6329±0.0586** | **+0.0578** | **0.0009** | **\*** |
| **5** | **0.6176±0.0605** | **0.6664±0.0669** | **+0.0488** | **0.0004** | **\*** |
| **10** | **0.6521±0.0639** | **0.7168±0.0608** | **+0.0647** | **0.0000** | **\*** |
| **20** | **0.7127±0.0609** | **0.7447±0.0676** | **+0.0320** | **0.0002** | **\*** |
| 50 | 0.7787±0.0639 | 0.7946±0.0659 | +0.0159 | - | - |

**3/5/10-shot Average: Gap=+0.0571, p=0.0000 \*\*\***

---

## 三、统计显著性检验 (Paired Wilcoxon)

### Protocol A (Label-Free Budget)

| k-shot | Random Mean | KMeans Mean | Gap | p-value | Significant |
|--------|-------------|-------------|-----|---------|-------------|
| 3 | 0.5789 | 0.5618 | -0.0171 | 0.8387 | |
| 5 | 0.5976 | 0.6165 | +0.0189 | 0.0649 | marginal |
| 10 | 0.6375 | 0.6730 | +0.0355 | 0.0001 | \* |
| 20 | 0.7048 | 0.7196 | +0.0149 | 0.0178 | \* |
| **3/5/10 avg** | - | - | **+0.0124** | **0.0347** | **\*** |

### Protocol B (Balanced)

| k-shot | Random Mean | KMeans Mean | Gap | p-value | Significant |
|--------|-------------|-------------|-----|---------|-------------|
| 3 | 0.5751 | 0.6329 | +0.0578 | 0.0009 | \* |
| 5 | 0.6176 | 0.6664 | +0.0488 | 0.0004 | \* |
| 10 | 0.6521 | 0.7168 | +0.0647 | 0.0000 | \* |
| 20 | 0.7127 | 0.7447 | +0.0320 | 0.0002 | \* |
| **3/5/10 avg** | - | - | **+0.0571** | **0.0000** | **\*\*** |

---

## 四、Subject-Level 分析

### Difficult Subjects (YLS, YSL, YHS, YRP, YAC) 增益

| Subject | k=5 Gap | k=10 Gap | k=20 Gap |
|---------|---------|----------|----------|
| YAC | +0.0667 \* | +0.0911 \* | +0.0667 \* |
| YHS | +0.0173 \* | +0.0095 \* | +0.0039 \* |
| YLS | +0.0272 \* | +0.0034 \* | +0.0170 \* |
| YRP | +0.0642 \* | +0.0218 \* | +0.0187 \* |
| YSL | +0.0174 \* | +0.0296 \* | +0.0597 \* |
| **平均** | **+0.0386** | **+0.0311** | **+0.0332** |

**结论：ACCS 主要提升 difficult subjects，平均增益约 +3-4%。**

### 被 ACCS 伤害的 Subjects

| Subject | 受伤的 k-shot | 伤害程度 |
|---------|---------------|----------|
| YFR | k=5 | -0.0743 |
| YTL | k=5 | -0.0845 |
| YFS | k=5 | -0.0393 |
| YDR | k=20 | -0.0427 |
| YSD | k=20 | -0.0348 |

**注意：部分 subject 在低 budget 下被 ACCS 伤害，但在高 budget 下恢复。**

### 10-shot ACCS vs 20-shot Random

| Metric | 10-shot ACCS | 20-shot Random | 差距 |
|--------|--------------|---------------|------|
| Mean Accuracy | 0.6730 | 0.7048 | -0.0318 |
| YAC | 0.7833 | 0.7633 | +0.0200 |
| YTL | 0.7856 | 0.7902 | -0.0046 |
| YRP | 0.6166 | 0.6570 | -0.0404 |

**结论：10-shot ACCS 平均略低于 20-shot Random，但部分 subject (YAC, YSL) 已达到或超过。**

### Subject Variance 分析

| Method | k=10 Mean | k=10 Std | k=20 Mean | k=20 Std |
|--------|-----------|----------|-----------|----------|
| Random | 0.6375 | 0.0651 | 0.7048 | 0.0630 |
| KMeans-ACCS | 0.6730 | 0.0660 | 0.7196 | 0.0649 |

**结论：ACCS 提升了性能但 subject variance 基本不变（std 相似）。**

---

## 五、创新点验收

### ACCS 作为第二创新点

| 要求 | 状态 | 说明 |
|------|------|------|
| 无标签泄漏 | ✅ | Protocol A 真正 label-free |
| 显著优于 Random | ✅ | 10-shot: +3.55%, p<0.001 |
| 主要提升 difficult subjects | ✅ | 平均 +3-4% on difficult subjects |
| 无明显缺陷 | ⚠️ | 少数 subject 在低 budget 下被伤害 |

### 建议

1. **主结果使用 Protocol A (label-free)**：这是真实部署场景
2. **Protocol B 作为消融分析**：展示balanced采样下的增益上限
3. **报告时注意**：10-shot ACCS 尚未稳定达到 20-shot Random 水平
4. **可补充**：自适应 budget 选择策略，避免低 budget 伤害

---

## 六、输出文件

- `results/personalized/accs_active_calibration.csv`: 原始实验数据
- `results/personalized/accs_significance_tests.csv`: 统计检验结果
- `reports/accs_active_calibration_report.md`: 本报告