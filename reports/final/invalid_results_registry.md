# Invalid Results Registry
# 错误结果注册表
# Generated: 2026-05-12

---

## 严重问题：结果命名与实现不符

以下结果文件的命名与实际实现不符，**绝对不能**用于论文。

---

## 1. eeg_gaze_pilot_results.csv - CRITICAL INVALID

### 问题描述

`eeg_gaze_pilot_v2.py` 中的 `CAGFModel` 类（第167-210行）**不是真正的 PCET+GETA+CAGF**：

**错误实现（CAGFModel）**:
```python
# 使用普通 EEG_MLP，不是 PCET
eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), ...)
eeg_mlp.fit(X_eeg_cal_s, y_cal)
z_eeg_cal = eeg_mlp.predict_proba(X_eeg_cal_s)

# 使用普通 Gaze_MLP，不是 GETA
gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), ...)
gaze_mlp.fit(X_gaze_cal_s, y_cal)
z_gaze_cal = gaze_mlp.predict_proba(X_gaze_cal_s)

# 用 MLP(16,) fusion，不是 CAGF gate
clf_final = MLPClassifier(hidden_layer_sizes=(16,), ...)
```

**实际这个类应该叫**: `EEG_Gaze_MLP_Fusion` 或 `Raw_MLP_Fusion`

### CSV 中错误命名的列

| CSV 列名 | 实际实现 | 正确命名 | 能否用于论文 |
|----------|----------|----------|--------------|
| `PCET+GETA+CAGF_acc` | EEG_MLP + Gaze_MLP + MLP(16,) | **EEG-Gaze MLP Fusion** | **NO** |
| `PCET_only_acc` | PCA reconstruction + AbsError + RidgeClassifier | PCET_only | NO (pilot, different protocol) |
| `GETA_only_acc` | Gaze attention reweight + EEG MLP | GETA_only | NO (pilot, different protocol) |

### 结论

**`eeg_gaze_pilot_results.csv` 中的 `PCET+GETA+CAGF` 列是 INVALID**，必须废弃。

正确的对应关系应该是：
- 如果要报告真正 PCET+GETA+CAGF 的结果，应该用 `run_cagf_feature_only.py` 的输出
- 如果要报告 EEG_MLP + Gaze_MLP + MLP fusion 的结果，应该命名为 `Raw EEG-Gaze MLP Fusion`

---

## 2. eeg_gaze_pilot_results.csv 协议问题

除了命名问题外，`eeg_gaze_pilot_v2.py` 还使用了**不同的协议**：

- 使用 1/3 test split
- 使用其他 subjects 作为训练数据（但实际没有用到）

这与主实验协议不符（50/50 split，LOSO target subject）。

---

## 3. 已确认正确的结果文件

| 文件 | PCET | GETA | CAGF | 可用于论文 |
|------|------|------|------|------------|
| `multimodal_final_main_results.csv` | ✅ 正确 | ✅ 正确 | ✅ 正确 | YES |
| `cagf_feature_only_final.csv` | ✅ 正确 | ✅ 正确 | ✅ 正确 | YES |
| `eeg_gaze_pilot_results.csv` PCET_only | ✅ 正确 | N/A | N/A | NO (protocol diff) |
| `eeg_gaze_pilot_results.csv` GETA_only | N/A | ✅ 正确 | N/A | NO (protocol diff) |
| `eeg_gaze_pilot_results.csv` PCET+GETA+CAGF | ❌ **INVALID** | ❌ **INVALID** | ❌ **INVALID** | **NO** |

---

## 4. 正确实现 vs 错误实现对比

### PCET 正确实现 (comprehensive_final_experiment.py)

```python
# ✅ CORRECT: 使用 PCA fit 在 calibration 数据上
for c in [0, 1]:
    X_c = X_eeg_cal[y_cal == c]
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_c)  # 只在 calibration 数据上 fit
    pca_models[c] = pca

# ✅ CORRECT: 计算 AbsError
X_rec = pca.inverse_transform(pca.transform(X))
e = X - X_rec
abs_error = np.abs(e)

# ✅ CORRECT: concat [x ; abs_error]
X_combined = np.hstack([scaler.fit_transform(X_eeg_cal), abs_error])
```

### CAGFModel 错误实现 (eeg_gaze_pilot_v2.py)

```python
# ❌ WRONG: 使用普通 EEG_MLP，不是 PCET
eeg_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), ...)
eeg_mlp.fit(X_eeg_cal_s, y_cal)
z_eeg = eeg_mlp.predict_proba(X_eeg_cal_s)

# ❌ WRONG: 使用普通 Gaze_MLP，不是 GETA
gaze_mlp = MLPClassifier(hidden_layer_sizes=(32,), ...)
gaze_mlp.fit(X_gaze_cal_s, y_cal)
z_gaze = gaze_mlp.predict_proba(X_gaze_cal_s)

# ❌ WRONG: 用 MLP(16,) fusion，不是 CAGF gate
clf_final = MLPClassifier(hidden_layer_sizes=(16,), ...)
clf_final.fit(z_fused_cal, y_cal)
```

---

## 5. 审计结论

### 需要立即废弃的结果

1. **`eeg_gaze_pilot_results.csv` 中的 `PCET+GETA+CAGF` 列** - 错误命名，实际是 EEG_MLP + Gaze_MLP + MLP(16,) fusion

### 可保留但不用于主论文的结果

2. `eeg_gaze_pilot_results.csv` 中的 `PCET_only` - 正确实现，但协议不同
3. `eeg_gaze_pilot_results.csv` 中的 `GETA_only` - 正确实现，但协议不同

### 可用于论文的正确结果

4. `multimodal_final_main_results.csv` - 全部正确
5. `cagf_feature_only_final.csv` - 全部正确

---

## 6. 正确的 PCET+GETA+CAGF 流程

### Step 1: PCET (EEG branch)
```
Raw EEG x
  → PCA reconstruction x_hat (per class, on calibration data only)
  → AbsError |x - x_hat|
  → concat [x ; |x - x_hat|]
  → classifier → z_pcet (probability)
```

### Step 2: GETA (Gaze branch)
```
Gaze features
  → Gaze MLP
  → prediction probability z_gaze
  → entropy + confidence
  → attention weight = entropy*0.01 + confidence
  → reweight EEG features: X_eeg * attention
  → EEG MLP → z_geta (probability)
```

### Step 3: CAGF (Fusion)
```
input: z_pcet, z_geta (NOT raw EEG_MLP, Gaze_MLP outputs)

alpha = sigmoid(z_pcet[:,0] - z_geta[:,0])
z_fused = alpha * z_pcet + (1-alpha) * z_geta

final MLP classifier on z_fused → prediction
```

---

End of Registry
