# PCET Leakage Check Report

## 1. 代码分析

### 1.1 PCET 核心流程

```python
def pcet_predict(X_cal, y_cal, X_test, n_pca_components=20, lambda_reg=0.1):
    # Step 1: 在 calibration data 上训练 class-conditional PCA
    pca_models = train_predictive_coding_model(X_cal, y_cal, n_pca_components)

    # Step 2: 计算预测误差作为特征
    error_cal = compute_prediction_errors(X_cal, pca_models)
    error_test = compute_prediction_errors(X_test, pca_models)

    # Step 3: 组合原始特征和误差特征，训练分类器
    X_cal_combined = np.hstack([scaler.fit_transform(X_cal), error_cal])
    X_test_combined = np.hstack([scaler.transform(X_test), error_test])

    clf = RidgeClassifier(alpha=lambda_reg)
    clf.fit(X_cal_combined, y_cal)
    preds = clf.predict(X_test_combined)
```

### 1.2 关键检查点

| 检查项 | 状态 | 说明 |
|--------|------|------|
| PCA 只在 calibration data 上训练 | ✅ | `train_predictive_coding_model(X_cal, y_cal)` |
| Test data 不参与 PCA 训练 | ✅ | PCA 只用 X_cal |
| Test labels 不用于预测 | ✅ | `compute_prediction_errors` 不需要 y 参数 |
| Hyperparameters 固定 | ✅ | n_pca_components=20, lambda_reg=0.1 |
| Calibration/test split 一致 | ✅ | 与其他实验相同 |

## 2. 潜在风险分析

### 2.1 Class-conditional PCA 的使用

**风险**：PCA 是 class-conditional 的，即对 class 0 和 class 1 分别训练不同的 PCA。

**实际情况**：
- `compute_prediction_errors(X, pca_models)` 对每个样本计算两个误差：
  - 用 PCA_0 重构的误差
  - 用 PCA_1 重构的误差
- **不依赖真实标签**：两个误差都被计算

**结论**：✅ 无泄漏 - 测试时不需要真实标签

### 2.2 Hyperparameter 选择

**检查**：n_pca_components=20 是否通过 test performance 选择的？

**实际情况**：
- 当前值是手动设定
- 未使用 test data 进行调参
- 未来如果调参，需要在 calibration split 内部做 cross-validation

**结论**：✅ 当前无泄漏，但建议记录

## 3. Protocol 一致性检查

| 项目 | PCET | 其他实验 |
|------|------|---------|
| Calibration pool / Test split | 2/3 / 1/3 | 一致 |
| Balanced random sampling | 每类 n shots | 一致 |
| Seeds | 0,1,2,3,4 | 一致 |
| Subjects | 16 Y-subjects | 一致 |
| 50-shot 定义 | 每类 50，总共 100 | 一致 |

## 4. 泄漏验证实验

为了进一步确认，运行以下检查：

### 4.1 Label Permutation Test

如果 PCET 泄漏了标签信息，随机打乱 calibration labels 应该仍然有效（因为模型可能直接记住了标签）。

**预期**：打乱标签后性能应该显著下降。

### 4.2 Feature Isolation Test

单独使用 prediction error 特征 vs 单独使用原始特征，检查哪部分起主要作用。

**已在消融实验中设计**。

## 5. 结论

| 检查项 | 结果 |
|--------|------|
| PCET predictor 未使用 target test samples | ✅ PASS |
| 未使用 test labels | ✅ PASS |
| Class-conditional PCA 在测试时不依赖真实 y | ✅ PASS |
| Hyperparameters 固定，未通过 test 调参 | ✅ PASS |
| Calibration/test split 与其他实验一致 | ✅ PASS |
| 50-shot per class 定义正确 | ✅ PASS |

**总体结论**：✅ **PCET 无明显数据泄漏**

## 6. 建议改进

1. **Hyperparameter tuning**：如果需要调参，应在 calibration split 内部做 CV
2. **更多消融**：验证 prediction error 单独作用的效果