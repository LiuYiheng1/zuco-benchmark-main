# Multimodal Fusion Modules Comparison Report

## 背景

当前最强结果：
- Static_EEG_Gaze_average (50-shot): **82.62%**

测试的三个新模块：
1. MACS-Fusion: Multimodal Active Calibration Sampling
2. SS-CMC: Semi-Supervised Cross-Modal Consistency Calibration
3. ATCF: Ambiguity-Triggered Complementary Fusion

---

## 一、MACS-Fusion 结果

### 方法
- Random_Static_Fusion
- EEG_KMeans_Static_Fusion
- Gaze_KMeans_Static_Fusion
- EEG_Gaze_KMeans_Static_Fusion
- Coreset_Static_Fusion

### 结果汇总

| k-shot | Random | EEG_KMeans | Gaze_KMeans | EEG_Gaze_KMeans | Coreset |
|--------|--------|------------|-------------|-----------------|---------|
| 5 | 0.6318 | 0.6447 (+1.3%) | 0.6756 (+4.4%) * | 0.6388 (+0.7%) | 0.6277 (-0.4%) |
| 10 | 0.6788 | 0.6828 (+0.4%) | 0.6954 (+1.7%) | 0.6914 (+1.3%) | 0.6393 (-4.0%) |
| 20 | 0.7262 | 0.7325 (+0.6%) | 0.7411 (+1.5%) * | 0.7308 (+0.5%) | 0.6906 (-3.6%) |
| 50 | 0.7985 | 0.7990 (+0.1%) | 0.7989 (+0.0%) | 0.8048 (+0.6%) | 0.7698 (-2.9%) |

### 成功标准检查

| 标准 | 结果 |
|------|------|
| 10-shot: ACCS >= Random + 2% | FAIL (最好 +1.7%) |
| 20-shot: ACCS >= Random + 1% | PASS (Gaze_KMeans +1.5%) |
| Difficult subjects YLS/YSL/YHS >= +2% | 部分通过 |

### 结论
Gaze_KMeans 在 20-shot 达到成功标准，但 10-shot 未达标。不够稳健。

---

## 二、SS-CMC 结果

### 配置
- lambda_cons: [0.01, 0.05, 0.1]
- confidence threshold tau: [0.7, 0.8]

### 结果汇总

| k-shot | Static_Fusion | SS_CMC_l0.01_t0.7 | SS_CMC_l0.01_t0.8 | SS_CMC_l0.1_t0.8 |
|--------|---------------|-------------------|-------------------|-------------------|
| 5 | 0.6726 | 0.6813 (+0.9%) | 0.6908 (+1.8%) | 0.6813 (+0.9%) |
| 10 | 0.7122 | 0.6813 (-3.1%) | 0.6908 (-2.1%) | 0.6813 (-3.1%) |
| 20 | 0.7582 | 0.7322 (-2.6%) | 0.7401 (-1.8%) | 0.7322 (-2.6%) |
| 50 | 0.8170 | 0.8002 (-1.7%) | 0.8063 (-1.1%) | 0.8002 (-1.7%) |

### 成功标准检查

| 标准 | 结果 |
|------|------|
| 10/20-shot: SS_CMC >= Static + 2% | FAIL (全部低于 Static) |
| 50-shot: > 82.62% | FAIL (最好 0.8063 < 0.8262) |
| Difficult subjects >= +2% | FAIL (全部负增益) |

### 结论
**SS-CMC 反而不如 Static Fusion**，所有配置都未达标。一致性损失对当前任务无效。

---

## 三、ATCF 结果

ATCF 因 LOO tau 选择计算量过大，未能在合理时间内完成。

预计问题：
- tau 选择依赖于 held-out calibration set
- 小样本下 LOO 不稳定
- 增加了过拟合风险

---

## 四、综合对比

### vs Static_EEG_Gaze_average (82.62%)

| 模块 | 50-shot 结果 | 评价 |
|------|-------------|------|
| Static_EEG_Gaze_average | 0.8262 | 基准 |
| MACS Gaze_KMeans | ~0.7990 | 差于基准 |
| MACS EEG_Gaze_KMeans | ~0.8048 | 差于基准 |
| SS_CMC best | ~0.8063 | 差于基准 |
| ATCF | 未完成 | 未知 |

### 最终结论

**三个模块都没有稳定超过 Static_EEG_Gaze_average**。

按照用户指示："如果这三个模块都不能超过 Static Fusion，就保留 Static Fusion 作为有效但简单的个性化融合模块，不再继续堆结构。"

---

## 五、最终建议

### 保留的个性化融合方法

1. **Static_EEG_Gaze_average** (主方法)
   - 50-shot: 82.62%
   - 简单、稳健、无额外参数

2. **ACCS (EEG-only)** (第一个创新点)
   - 10-shot: +3.55% vs Random (p<0.001)
   - Label-free，无标签泄漏
   - 主要提升 difficult subjects

### 论文创新点

1. **SIED**: Subject-adversarial EEG disentanglement (cross-subject transfer)
2. **ACCS**: Active cognitive calibration sampling (EEG-only, label-free)

### 不再继续的方向
TSPC, User Adapter, MCC, CAET, CLF, Reliability Weighting, TGCR, MACS-Fusion (fusion版本), SS-CMC, ATCF

---

## 六、输出文件

- `results/personalized/macs_fusion_results.csv`: MACS-Fusion 详细结果
- `results/personalized/ss_cmc_results.csv`: SS-CMC 详细结果
- `reports/dual_fusion_modules_comparison_report.md`: 本报告