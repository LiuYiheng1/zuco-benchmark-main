# Model Provenance Audit Report
# 模型溯源审计报告
# Generated: 2026-05-12

---

## Executive Summary

| 状态 | 结果文件数量 | 说明 |
|------|-------------|------|
| ✅ 正确 | 2 | `multimodal_final_main_results.csv`, `cagf_feature_only_final.csv` |
| ❌ 错误命名 | 3 | `eeg_gaze_pilot_results.csv`, `cagf_v2_ablation.csv`, `cagf_v3_cross_interaction.csv` 中的部分列 |
| ⚠️ Diagnostic | 2 | `fewshot_adagtcn_proxy_*` (proxy baseline) |

---

## 1. 关键发现

### 1.1 eeg_gaze_pilot_results.csv 中的 PCET+GETA+CAGF 是 INVALID

**问题**：`eeg_gaze_pilot_v2.py` 中的 `CAGFModel` 类：
- 使用 **普通 EEG_MLP**（不是 PCET）
- 使用 **普通 Gaze_MLP**（不是 GETA）
- 使用 **MLP(16,) fusion**（不是 CAGF gate）

```python
# eeg_gaze_pilot_v2.py 第 167-210行 - 错误实现
class CAGFModel:
    def fit_predict(self, X_eeg_cal, y_cal, X_gaze_cal, X_eeg_test, X_gaze_test):
        # ❌ 使用普通 EEG_MLP，不是 PCET
        eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), ...)
        eeg_mlp.fit(X_eeg_cal_s, y_cal)
        z_eeg = eeg_mlp.predict_proba(...)

        # ❌ 使用普通 Gaze_MLP，不是 GETA
        gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), ...)
        gaze_mlp.fit(X_gaze_cal_s, y_cal)
        z_gaze = gaze_mlp.predict_proba(...)

        # ❌ 用 MLP(16,) fusion，不是 CAGF gate
        clf_final = MLPClassifier(hidden_layer_sizes=(16,), ...)
```

**应该命名为**: `Raw EEG-Gaze MLP Fusion` 或 `EEG_MLP + Gaze_MLP + MLP Fusion`

### 1.2 cagf_v3_cross_interaction.py 同样存在问题

`CAGFv3Variants.cagf_feature_only` 也是同样的问题：
- 使用普通 EEG_MLP，不是 PCET
- 使用普通 Gaze_MLP，不是 GETA
- 使用 MLP(16,) fusion，不是 CAGF

---

## 2. 唯一正确的结果来源

### ✅ multimodal_final_main_results.csv

这是**唯一正确**的 PCET+GETA+CAGF 结果来源：

| 模块 | 实现 | 代码位置 |
|------|------|----------|
| PCET | PCA reconstruction + AbsError + RidgeClassifier | `comprehensive_final_experiment.py` `PCETVariants.raw_plus_abserror()` |
| GETA | Gaze MLP → entropy + confidence → attention → reweight EEG → EEG MLP | `comprehensive_final_experiment.py` `GETAAblation.geta_confidence_entropy()` |
| CAGF | z_pcet + z_geta → alpha = sigmoid(z_pcet[:,0] - z_geta[:,0]) → fused | `comprehensive_final_experiment.py` inline code |

### ✅ cagf_feature_only_final.csv

同样是正确的实现（来自 `run_cagf_feature_only.py`）。

---

## 3. 硬性验收标准检查

### PCET 必须满足的条件

| 检查项 | 要求 | multimodal_final_main | cagf_feature_only | eeg_gaze_pilot |
|--------|------|---------------------|-------------------|-----------------|
| PCA fit on calibration | 是 | ✅ | ✅ | ✅ |
| PCA per class | 是 | ✅ | ✅ | ✅ |
| AbsError 计算 | 是 | ✅ | ✅ | ✅ |
| concat [x; abs_error] | 是 | ✅ | ✅ | ✅ |
| 不是 EEG_MLP | 是 | ✅ | ✅ | ❌ (用EEG_MLP) |

### GETA 必须满足的条件

| 检查项 | 要求 | multimodal_final_main | cagf_feature_only | eeg_gaze_pilot |
|--------|------|---------------------|-------------------|-----------------|
| Gaze features | 是 | ✅ | ✅ | ✅ |
| Gaze MLP | 是 | ✅ | ✅ | ✅ |
| entropy 计算 | 是 | ✅ | ✅ | ✅ |
| confidence 计算 | 是 | ✅ | ✅ | ✅ |
| attention reweight EEG | 是 | ✅ | ✅ | ✅ |
| 不是 Gaze_MLP alone | 是 | ✅ | ✅ | ❌ (就是Gaze_MLP) |

### CAGF 必须满足的条件

| 检查项 | 要求 | multimodal_final_main | cagf_feature_only | eeg_gaze_pilot |
|--------|------|---------------------|-------------------|-----------------|
| 输入来自 PCET | 是 | ✅ | ✅ | ❌ (用EEG_MLP) |
| 输入来自 GETA | 是 | ✅ | ✅ | ❌ (用Gaze_MLP) |
| alpha = sigmoid(diff) | 是 | ✅ | ✅ | ❌ (用MLP) |
| 不是 MLP(16,) fusion | 是 | ✅ | ✅ | ❌ (用MLP) |
| 无 confidence features | 是 | ✅ | ✅ | N/A |
| 无 abs_diff/hadamard | 是 | ✅ | ✅ | N/A |

---

## 4. 结果文件分类

### A. 可用于论文 (Approved)

| 文件 | 方法 | 状态 |
|------|------|------|
| `multimodal_final_main_results.csv` | 全部正确 | ✅ |
| `cagf_feature_only_final.csv` | 全部正确 | ✅ |

### B. 错误命名 (Misnamed - DO NOT USE)

| 文件 | 列名 | 实际实现 | 正确命名 |
|------|------|----------|----------|
| `eeg_gaze_pilot_results.csv` | `PCET+GETA+CAGF` | EEG_MLP + Gaze_MLP + MLP(16,) | `Raw EEG-Gaze MLP Fusion` |
| `cagf_v2_ablation.csv` | `CAGF_feature_only` | EEG_MLP + Gaze_MLP + MLP(16,) | `Raw EEG-Gaze MLP Fusion` |
| `cagf_v2_ablation.csv` | `CAGF_full_old` | EEG_MLP + Gaze_MLP + MLP(confidence) | Deprecated |
| `cagf_v3_cross_interaction.csv` | `CAGF_v3_cross_interaction` | EEG_MLP + Gaze_MLP + abs_diff/hadamard | Deprecated |

### C. Diagnostic / Reference Only

| 文件 | 用途 |
|------|------|
| `fewshot_adagtcn_proxy_*.csv` | Diagnostic - AdaGTCN-lite proxy |
| `adagtcn_table1_comparison_*.csv` | Reference - 不可直接比较 |

---

## 5. 核心结论

### 唯一合法的 PCET+GETA+CAGF 结果

```
文件: results/final/multimodal_final_main_results.csv
脚本: comprehensive_final_experiment.py
协议: Few-shot personalized LOSO (k=3,5,10,20,50)
```

### 哪一版结果才是真正的 PCET+GETA+CAGF？

**Answer**: `multimodal_final_main_results.csv` 和 `cagf_feature_only_final.csv`

### 哪一版只是 EEG_MLP+Gaze_MLP+fusion？

**Answer**: `eeg_gaze_pilot_results.csv` 中的 `PCET+GETA+CAGF` 列

### 哪一版应该废弃？

**Answer**:
1. `eeg_gaze_pilot_results.csv` 中名为 `PCET+GETA+CAGF` 的列 - 错误命名
2. `cagf_v2_ablation.csv` 中名为 `CAGF_feature_only` 的列 - 错误命名
3. `cagf_v3_cross_interaction.csv` 中所有 CAGF 变体 - 使用错误的基础模块

---

End of Report
