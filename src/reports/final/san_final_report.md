# SAN Final Report - After Leakage Fix

## 1. 关键发现：原始 SAN 存在 Test Label Leakage

### 1.1 问题代码
```python
# 原始 SAN 使用 class-conditional source statistics
# 但在 test set 上使用真实 y 来选择 class-specific normalization
X_test_norm[y_test == 0] = (X_test[y_test == 0] - mu_source_0) / sigma_source_0
X_test_norm[y_test == 1] = (X_test[y_test == 1] - mu_source_1) / sigma_source_1
#                         ^^^ 这是 test label leakage!
```

### 1.2 影响
- **原始报告的 SAN 数字不可信**
- 所有使用 class-conditional source statistics 的结果可能都是 leaky 的

## 2. 修复后重新验证

### 2.1 Global SourceNorm (不使用任何 label)
```python
mu_source = mean(X_train_all)  # 不区分 class
sigma_source = std(X_train_all)
X_norm = (X - mu_source) / sigma_source
```

**结果**:
| Shot | StandardScaler | SAN_global | Gap |
|------|---------------|-------------|-----|
| 3 | 0.4346 | 0.4346 | 0.0% |
| 5 | 0.4161 | 0.4161 | 0.0% |
| 10 | 0.5764 | 0.5764 | 0.0% |
| 20 | 0.5964 | 0.5964 | 0.0% |
| 50 | 0.7623 | 0.7623 | 0.0% |

**结论**: Global SourceNorm = StandardScaler，**无任何提升**

### 2.2 Pseudo-label SAN
使用 pseudo-label 来选择 class-specific source statistics:

```python
# Step 1: 训练初始分类器
clf = train_SVM(X_cal, y_cal)

# Step 2: 获取 test set pseudo-labels
pseudo_labels = clf.predict(X_test)

# Step 3: 使用 pseudo-labels 选择 class-specific statistics
X_test_norm[pseudo_labels == 0] = (X_test[pseudo_labels == 0] - mu_source_0) / sigma_source_0
X_test_norm[pseudo_labels == 1] = (X_test[pseudo_labels == 1] - mu_source_1) / sigma_source_1
```

**结果**:
| Shot | StandardScaler | ACCS | SAN_pseudo_label | pseudo_acc |
|------|---------------|------|------------------|------------|
| 3 | 0.4346 | 0.4708 | 0.4626 | 54.0% |
| 5 | 0.4161 | **0.4889** | **0.4889** | 54.8% |
| 10 | 0.5764 | 0.5097 | 0.5247 | 61.1% |
| 20 | 0.5964 | 0.5986 | **0.6400** | 68.8% |
| 50 | 0.7623 | 0.7596 | 0.7631 | 75.8% |

**分析**:
- 5-shot: SAN_pseudo_label = ACCS (+7.28%)
- 20-shot: SAN_pseudo_label 超过 ACCS (+4.37%)
- 但结果**不稳定** - 有时还不如 StandardScaler

## 3. 论文写作决策

### 3.1 不能作为主方法的版本
- ❌ ~~原始 SAN~~ - test label leakage
- ❌ ~~SAN_global~~ - 无提升 (= StandardScaler)
- ❌ ~~SAN_oracle_class_conditional~~ - 使用 test labels

### 3.2 可考虑的方法
- ⏳ **SAN_pseudo_label** - 在 5-shot 和 20-shot 有提升，但不稳定
- ✅ **ACCS** - 在 3/5-shot 稳定提升，且完全 label-free

### 3.3 建议

**SAN 不作为创新点**，原因：
1. 原始 SAN 存在 test label leakage，论文中不可用
2. Global SAN 无提升
3. Pseudo-label SAN 不稳定，效果不可靠

**Focus on ACCS** 作为 calibration sample efficiency 的贡献。

## 4. 最终结论

| 方法 | 状态 | 原因 |
|------|------|------|
| 原始 SAN | ❌ 排除 | test label leakage |
| Global SAN | ❌ 排除 | 无提升 (= StandardScaler) |
| Oracle SAN | ❌ 排除 | 使用 test labels |
| Pseudo-label SAN | ⚠️ 可疑 | 不稳定，不可靠 |
| **ACCS** | ✅ 可用 | 3-5 shot 稳定提升 |

## 5. 论文表述更新

### 5.1 可以写
> "ACCS improves calibration sample efficiency in the 3-5 shot range, achieving +6.9% improvement over random sampling."

### 5.2 不能写
> ~~"SAN stabilizes personalized EEG calibration"~~ (不可信)
> ~~"SourceNorm improves performance"~~ (已验证无效)

## 6. 数据完整性确认

✅ Test labels 仅用于最终 metric 计算
✅ Normalization 不使用 test y (Global version)
✅ SAN_global 不使用任何 target labels
✅ Calibration/test split 不变
✅ 所有方法使用相同 split

## 7. 教训

1. **Class-conditional normalization 需要 label 信息**
2. **Global normalization 不提供类别判别性**
3. **Pseudo-label 方法引入噪声，效果不稳定**
4. **必须区分 oracle 和 label-free 方法**

## 8. 下一步

**SAN 相关实验已完成，结论：SAN 不能作为创新点。**

**建议聚焦于**:
1. **SIED**: zero-shot transfer (+2.04%)
2. **ACCS**: calibration sample efficiency (3-5 shot +6.9%)
3. **Personalized baseline**: StandardScaler (50-shot 76.2%)