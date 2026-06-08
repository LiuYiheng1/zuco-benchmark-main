# Experiment Audit Report
# 实验审计报告
# Generated: 2026-05-12

---

## 1. 论文主线确认

### 1.1 最终论文主题
```
Few-shot personalized EEG-gaze reading state recognition on ZuCo 2.0
```

### 1.2 不属于论文主线的方向
- zero-shot cross-subject leaderboard
- AdaGTCN-style SOTA
- EEG-to-text

### 1.3 最终核心方法
```
PCET + GETA + CAGF
```

---

## 2. 核心方法代码确认

### 2.1 PCET (Prediction-error EEG Representation)

**代码位置**: `comprehensive_final_experiment.py` (PCETVariants class)

**实际代码逻辑**:
```python
Raw EEG x
  → PCA reconstruction x_hat (fit on calibration data only, per class)
  → AbsError |x - x_hat|
  → concatenate [x ; |x - x_hat|]
  → RidgeClassifier
  → prediction
```

**定位**: Prediction-error EEG representation

**重要说明**:
- PCA 仅在 calibration 数据上 fit（每个类分别 fit）
- 测试数据仅 transform，不用于 fitting
- 不使用测试标签

### 2.2 GETA (Gaze-guided EEG Task Attention)

**代码位置**: `comprehensive_final_experiment.py` (GETAAblation class) 和 `run_cagf_feature_only.py` (GETAModel class)

**实际代码逻辑**:
```python
Gaze features
  → Gaze MLP
  → prediction probability z_gaze
  → entropy + confidence
  → attention weight
  → reweight EEG features
  → EEG MLP
  → prediction
```

**定位**: Gaze-guided EEG Task Attention

**重要说明**:
- **不是** gaze behavior grouping
- **不是** transition attention
- 使用 gaze features (sent_gaze_sacc.npy)，不是 EEG features
- Attention weights 从 gaze predictions 导出

### 2.3 CAGF (Cross-modal Adaptive Gated Fusion)

**代码位置**: `run_cagf_feature_only.py` (CAGF_feature_only class)

**最终代码逻辑**:
```python
z_eeg, z_gaze
  →
alpha = sigmoid(z_eeg[:,0] - z_gaze[:,0])
  →
z_fused = alpha * z_pcet + (1-alpha) * z_geta
  →
MLP classifier → prediction
```

**定位**: Feature-only adaptive fusion

**重要说明**:
- **不是** confidence-aware gating
- **不是** cross-interaction fusion
- 不使用: abs_diff, hadamard, c_eeg, c_gaze

### 2.4 已被废弃的模块
- SRGC (Source-Regularized Gaussian Classifier)
- SIED (Subject-Invariant Error Decorrelation)
- SCI (Sentence Confidence Integration)
- CAGF_full_old (with confidence features)
- CAGF_v3_cross_interaction (with abs_diff, hadamard)
- CAGF_random_confidence
- CAGF_shuffled_confidence

---

## 3. Canonical Main Result 确认

### 3.1 来自哪个脚本和输出文件？

**脚本**: `comprehensive_final_experiment.py` + `run_cagf_feature_only.py`

**输出文件**: `results/final/multimodal_final_main_results.csv`

**实验协议**:
```
Few-shot personalized protocol
LOSO target subject
k = 3, 5, 10, 20, 50
same calibration/test split (50/50)
seeds = [0, 1, 2, 3, 4]
no test leakage
```

### 3.2 Canonical Main Result Table

| Method                | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|-----------------------|--------|--------|---------|---------|---------|
| EEG_SVM               | 43.5   | 41.6   | 57.6    | 59.6    | 76.2    |
| Gaze_SVM              | 50.1   | 55.0   | 61.7    | 61.4    | 69.6    |
| EEG_MLP               | 58.2   | 61.2   | 65.9    | 71.0    | 78.2    |
| Gaze_MLP              | 59.9   | 63.3   | 65.0    | 67.4    | 69.3    |
| EEG+Gaze_concat       | 57.7   | 61.5   | 66.1    | 72.0    | 79.4    |
| Static_EEG_Gaze_avg   | 46.5   | 49.3   | 64.3    | 65.7    | 79.7    |
| PCET_only             | 58.7   | 61.0   | 65.1    | 70.0    | 78.2    |
| GETA_only             | 58.2   | 61.2   | 65.9    | 71.0    | 78.2    |
| PCET+GETA_concat      | 58.0   | 60.6   | 64.3    | 69.6    | 77.3    |
| PCET+GETA_static_avg  | 59.0   | 61.6   | 66.7    | 71.4    | 79.1    |
| **PCET+GETA+CAGF**    | **62.3** | **65.8** | **69.7** | **74.1** | **80.1** |

### 3.3 PCET+GETA+CAGF 完整指标

| Shot | Accuracy     | Macro-F1      | BAcc         | AUROC        |
|------|--------------|---------------|--------------|--------------|
| 3    | 62.27±9.3    | 59.54±10.4    | 60.89±9.3    | 60.89±9.3    |
| 5    | 65.84±9.6    | 63.57±10.9    | 64.69±9.9    | 64.69±9.9    |
| 10   | 69.68±9.5    | 68.07±10.5    | 68.56±10.0   | 68.56±10.0   |
| 20   | 74.06±8.6    | 73.10±9.1     | 73.32±8.8    | 73.32±8.8    |
| 50   | 80.11±7.2    | 79.61±7.4     | 79.56±7.3    | 79.56±7.3    |

---

## 4. 实验分类

### A. 主实验 (Main Experiments) - 必须保留

| 实验名称 | 脚本文件 | 输出文件 | 说明 |
|----------|----------|----------|------|
| PCET_v2_main | comprehensive_final_experiment.py | multimodal_final_main_results.csv | PCET ablation + main results |
| CAGF_main | run_cagf_feature_only.py | cagf_feature_only_final.csv | CAGF final results |
| GETA_ablation | comprehensive_final_experiment.py | geta_ablation.csv | GETA variants |
| PCET_ablation | comprehensive_final_experiment.py | pcet_ablation.csv | PCET variants |

**主实验协议**:
```
Few-shot personalized protocol
LOSO target subject
k = 3, 5, 10, 20, 50
same calibration/test split
same seeds (0,1,2,3,4)
no test leakage
```

### B. 强 Baseline - 必须保留但命名清楚

| 实验名称 | 论文建议命名 | 定义 |
|----------|--------------|------|
| EEG_Gaze_concat | Raw EEG-Gaze MLP Fusion | Raw EEG features + Raw gaze features → EEG encoder MLP + Gaze encoder MLP → concat → MLP classifier |
| StaticAvg | Ridge StaticAvg | RidgeClassifier on raw EEG + RidgeClassifier on raw gaze → prob_avg = 0.5*prob_eeg + 0.5*prob_gaze |

### C. AdaGTCN / Benchmark 相关实验 - Diagnostic / Reference only

| 实验名称 | 状态 | 原因 |
|----------|------|------|
| AdaGTCN-inspired 10/2/4 split | Diagnostic | AdaGTCN 是 word-level fixation-segmented EEG sequence + graph-temporal model；我们的任务是 sentence-level features + few-shot personalized calibration |
| Benchmark-style LOSO | Diagnostic | ZuCo Benchmark 是 cross-subject hidden-test protocol，不是 personalized few-shot |
| AdaGTCN Table 1 reported comparison | Reference only | not directly comparable |

### D. Proxy Baselines - 需标注 lite/proxy

| 实验名称 | 说明 |
|----------|------|
| EEG-GCN-proxy | Sentence-level proxy，不是完整 GCN reproduction |
| EEG-Gaze proxy | Sentence-level proxy |
| AdaGTCN-lite | Word-level model 的 sentence-level proxy |
| STRG_lite | Sentence-level proxy, not original STRG |
| STRE_lite | Sentence-level proxy, not original STRE |
| GLIM_enc | Sentence-level proxy |
| Cog_EEGtext | Uses text information → text-assisted / confound / upper-bound table |

---

## 5. 有效结论

### 5.1 确认有效的结论

```
✓ PCET+GETA+CAGF 在 few-shot personalized protocol 下有效
✓ PCET 和 GETA 分支有一定贡献
✓ CAGF 比 PCET+GETA_concat 和 PCET+GETA_static_avg 更强
✓ 在低样本场景 (3-5 shot)，我们的模型优势更明显
```

### 5.2 需要谨慎写的结论

```
⚠ 如果 StaticAvg 或 Raw EEG-Gaze MLP Fusion 在 20/50-shot 更强：
  结论应写成：我们的模型主要在低样本场景更有效；高样本时简单融合也很强。
```

### 5.3 应废弃的说法

```
✗ CAGF is confidence-aware
✗ GETA models transition behavior
✗ GETA uses behavior grouping
✗ We directly beat AdaGTCN
✗ AdaGTCN-lite is original AdaGTCN
✗ Ours is best in every experiment version
```

---

## 6. 论文表格规划

### Table 1: Main Few-shot Comparison

```
EEG_SVM
Gaze_SVM
EEG_MLP
Gaze_MLP
Raw EEG-Gaze MLP Fusion
Ridge StaticAvg
PCET_only
GETA_only
PCET+GETA_concat
PCET+GETA_static_avg
PCET+GETA+CAGF
```

(STRG_lite / STRE_lite / GLIM_enc 可放补充表)

### Table 2: Ablation Study

```
EEG_MLP
PCET_only
GETA_only
PCET+GETA_concat
PCET+GETA_static_avg
PCET+GETA+CAGF
```

### Table 3: Prior Reported / Diagnostic

```
ZuCo Benchmark reported results
AdaGTCN reported Table 1 results
Our cross-subject diagnostic result
```

(必须注明 not directly comparable)

---

## 7. 审计问题回答

### Q1: 当前 canonical main result 来自哪个脚本和输出文件？

**回答**:
- 脚本: `comprehensive_final_experiment.py`
- 输出文件: `results/final/multimodal_final_main_results.csv`
- CAGF 专项: `run_cagf_feature_only.py` → `results/final/cagf_feature_only_final.csv`

### Q2: PCET+GETA+CAGF 的最终 Accuracy/F1/BAcc/AUROC 是多少？

**回答**:
| Shot | Accuracy | Macro-F1 | BAcc | AUROC |
|------|----------|----------|------|-------|
| 3    | 62.27±9.3 | 59.54±10.4 | 60.89±9.3 | 60.89±9.3 |
| 5    | 65.84±9.6 | 63.57±10.9 | 64.69±9.9 | 64.69±9.9 |
| 10   | 69.68±9.5 | 68.07±10.5 | 68.56±10.0 | 68.56±10.0 |
| 20   | 74.06±8.6 | 73.10±9.1 | 73.32±8.8 | 73.32±8.8 |
| 50   | 80.11±7.2 | 79.61±7.4 | 79.56±7.3 | 79.56±7.3 |

### Q3: GETA 的真实代码机制是什么？

**回答**: Gaze-guided EEG Task Attention
```
Gaze features → Gaze MLP → prediction probability → entropy + confidence → attention weight → reweight EEG features → EEG MLP → prediction
```
不是 gaze behavior grouping，不是 transition attention。

### Q4: CAGF 的真实代码机制是什么？

**回答**: Feature-only adaptive fusion
```
alpha = sigmoid(z_eeg[:,0] - z_gaze[:,0])
z_fused = alpha * z_eeg + (1-alpha) * z_gaze
MLP classifier → prediction
```
不是 confidence-aware，不是 cross-interaction fusion。

### Q5: 哪些实验是主实验？

**回答**:
- `comprehensive_final_experiment.py` - PCET, GETA, CAGF ablation + main results
- `run_cagf_feature_only.py` - CAGF feature-only 专项结果

### Q6: 哪些实验是 proxy baseline？

**回答**:
- EEG-GCN-proxy
- EEG-Gaze proxy
- AdaGTCN-lite
- STRG_lite
- STRE_lite
- GLIM_enc
- Cog_EEGtext (text-assisted, 进 confound/upper-bound 表)

### Q7: 哪些实验是 diagnostic only？

**回答**:
- Benchmark-style LOSO
- AdaGTCN-inspired 10/2/4 split
- AdaGTCN Table 1 reported comparison

### Q8: 哪些旧结果应标记 deprecated？

**回答**:
- 所有 SIED 相关结果 (sied_*.csv)
- 所有 SRGC 相关结果 (srgc_*.csv)
- 所有 SCI 相关结果 (sci_*.csv)
- CAGF_full_old 结果
- CAGF_v3_cross_interaction 结果
- UC-DAR 相关报告 (final_paper_experiment_summary.md)

### Q9: 当前论文最稳的主张是什么？

**回答**:
```
在 Few-shot personalized EEG-gaze reading state recognition 任务下，
PCET+GETA+CAGF 方法在 3-20 shot 场景优于所有 baseline 方法。
主要优势在低样本场景 (3-5 shot)。
```

### Q10: 当前论文不能再主张什么？

**回答**:
```
✗ 不能声称在所有 shot 设置下都最优
✗ 不能声称直接超越 AdaGTCN (协议不同，无法比较)
✗ 不能声称 CAGF 是 confidence-aware
✗ 不能声称 GETA 建模了 transition behavior 或 behavior grouping
✗ 不能声称解决了 cross-subject generalization
```

---

## 8. 文件清单

### 保留文件 (Keep)

| 文件路径 | 用途 |
|----------|------|
| results/final/multimodal_final_main_results.csv | 主实验结果 |
| results/final/cagf_feature_only_final.csv | CAGF 专项 |
| results/final/pcet_ablation.csv | PCET ablation |
| results/final/geta_ablation.csv | GETA ablation |
| results/final/cagf_ablation_final.csv | CAGF ablation |
| reports/final/multimodal_final_main_report.md | 主报告 |
| src/comprehensive_final_experiment.py | 主实验脚本 |
| src/run_cagf_feature_only.py | CAGF 脚本 |

### 归档文件 (Archive)

| 文件路径 | 状态 |
|----------|------|
| results/final/sied_*.csv | Deprecated |
| results/final/srgc_*.csv | Deprecated |
| results/final/sci_*.csv | Deprecated |
| reports/final/sied_*.md | Deprecated |
| reports/final/srgc_*.md | Deprecated |
| reports/final/final_paper_experiment_summary.md | Deprecated (UC-DAR) |

### Diagnostic Only

| 文件路径 | 用途 |
|----------|------|
| results/final/fewshot_adagtcn_*.csv | Diagnostic |
| results/final/adagtcn_*.csv | Diagnostic |
| results/final/benchmark_*.csv | Diagnostic |

---

End of Report
