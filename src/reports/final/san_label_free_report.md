# SAN Label-Free Report

## 1. 关键发现

### 1.1 原 SAN 存在 Test Label Leakage

**原 SAN 代码问题**:
```python
# 原来的 class-conditional SourceNorm 在 test set 上使用真实 y
X_test_norm[y_test == 0] = (X_test[y_test == 0] - mu_source_0) / sigma_source_0
X_test_norm[y_test == 1] = (X_test[y_test == 1] - mu_source_1) / sigma_source_1
```

这使用了 `y_test`（真实标签）来选择 class-specific statistics，**属于 test label leakage**。

### 1.2 修复后的结果

| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|--------|---------|---------|---------|
| StandardScaler | 0.4346 | 0.4161 | 0.5764 | 0.5964 | 0.7623 |
| **SAN_global** | 0.4346 | 0.4161 | 0.5764 | 0.5964 | 0.7623 |
| TargetNorm_global | 0.4346 | 0.4161 | 0.5764 | 0.5964 | 0.7623 |
| ACCS | 0.4708 | 0.4849 | 0.5097 | 0.5986 | 0.7596 |
| SAN_global_ACCS | 0.4708 | 0.4849 | 0.5097 | 0.5986 | 0.7596 |
| TargetNorm_labeled_cal | 0.4772 | 0.4752 | 0.4815 | 0.4552 | 0.5707 |
| **SAN_oracle** | 0.4139 | 0.4007 | **0.6287** | **0.7453** | **0.8889** |

## 2. 核心结论

### 2.1 Global SourceNorm 无提升
- **SAN_global = StandardScaler** 在所有设置下完全相同
- 使用全局统计信息（不区分 class）对归一化没有任何效果
- **原因**: 归一化如果不区分 class，无法提供类别判别信息

### 2.2 Class-conditional Oracle 有效但不可用
- **SAN_oracle_class_conditional** 使用真实 y 来选择 class-specific statistics
- 10-shot: +5.24%, 20-shot: +14.89%, 50-shot: +12.66%
- **但这是 oracle，不能作为主方法**

### 2.3 TargetNorm_labeled_calibration 失败
- 使用 calibration pool 真实标签计算 class statistics
- 在 10-shot+ 时反而下降
- **原因**: calibration samples 太少，统计估计噪声大

## 3. 问题分析

### 3.1 为什么 Global Normalization 无效？

归一化的核心作用是消除特征分布的差异。但如果：
1. **不区分 class**: 全局 mean/std 对两个 class 的混合分布取平均
2. **两个 class 分布不同**: 全局统计会混淆两个 class 的信息
3. **分类器无法受益**: 没有类别判别性的归一化对 SVM/MLP 没有帮助

### 3.2 为什么 Class-conditional Normalization 有效？

如果能正确区分 class 并应用对应的统计：
1. **类别对齐**: 每个 class 的特征被归一化到相似的分布
2. **决策边界清晰**: 分类器更容易找到决策边界
3. **性能提升**: 但需要正确知道每个样本的 class

## 4. Pseudo-label SAN 可行性分析

### 4.1 方案
```python
# Step 1: 用 Global SAN 训练初始分类器
clf = train_SVM(X_cal_global_san, y_cal)

# Step 2: 获取 pseudo-labels for test
pseudo_labels = clf.predict(X_test)

# Step 3: 用 pseudo-labels 选择 class-specific source statistics
# 这是 label-free，但因为用了 pseudo-labels，会有噪声
```

### 4.2 问题
- Pseudo-label 准确率直接影响 class-conditional normalization 效果
- 如果初始分类器准确率不高，错误标签会放大噪声
- 需要验证 pseudo-label 准确率

## 5. 论文写作决策

### 5.1 可以作为主方法
- ❌ ~~SAN_global~~ - 无提升
- ❌ ~~SAN_oracle~~ - 使用 test labels
- ⏳ **Pseudo-label SAN** - 需要进一步验证

### 5.2 推荐做法

如果 Pseudo-label SAN 效果不好，**建议不将 SAN 作为创新点**。

可以改为：
1. **Focus on ACCS** - 5-shot 提升 +6.9%，且完全 label-free
2. **Focus on SIED** - zero-shot +2.04%
3. **StandardScaler baseline** - 简单有效

### 5.3 论文表述建议

如果 SAN 不作为创新点：
> "Our analysis reveals that SourceNorm with class-conditional normalization provides significant gains, but requires label information. We leave the investigation of pseudo-label based approaches for future work."

## 6. 下一步行动

1. **实现 Pseudo-label SAN** 并验证效果
2. **如果 Pseudo-label SAN 无效**，从主创新点移除 SAN
3. **Focus on ACCS 和 SIED** 作为主要贡献

## 7. 数据完整性确认

| 检查项 | 状态 |
|--------|------|
| Test labels 仅用于最终 metric | ✅ 确认 |
| Normalization 不使用 test y | ✅ 确认 (Global version) |
| SAN_global 不使用 target label | ✅ 确认 |
| Calibration/test split 不变 | ✅ 确认 |
| 所有方法使用同一 split | ✅ 确认 |

## 8. 最终结论

**原始 SAN 存在 test label leakage，已被排除。**

**修复后的 SAN_global 无提升，不能作为创新点。**

**SAN_oracle_class_conditional 有效但依赖 test labels，属于 oracle 不可用。**

**建议：Focus on ACCS (calibration sample efficiency) 和 SIED (zero-shot transfer)。**