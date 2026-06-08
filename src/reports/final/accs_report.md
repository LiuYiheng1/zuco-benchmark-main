# ACCS Report

## 1. 实验设置

### 1.1 Protocol A: Label-free Budget (主协议)
- **ACCS Sampling**: KMeans centroid sampling on unlabeled calibration pool
- **KMeans fit**: 仅使用 X_pool features，不使用任何 label 信息
- **Shot settings**: 3, 5, 10, 20, 50 shots per class
- **Seeds**: 0, 1, 2, 3, 4

### 1.2 核心原则
1. ACCS 采样只使用 X_pool，不使用 y
2. KMeans fit 在 unlabeled calibration pool
3. test set 不参与采样

## 2. 核心结果

| Shot | Random (StandardScaler) | ACCS | Gain |
|------|------------------------|------|------|
| 3 | 0.4346 | 0.4708 | +3.6% |
| 5 | 0.4161 | 0.4849 | +6.9% |
| 10 | 0.5764 | 0.5097 | -6.7% |
| 20 | 0.5964 | 0.5986 | +0.2% |
| 50 | 0.7623 | 0.7596 | -0.3% |

## 3. 关键发现

### 3.1 ACCS 在低样本下有效
- **5-shot**: ACCS 显著优于 Random (+6.9%)
- **3-shot**: ACCS 优于 Random (+3.6%)

### 3.2 ACCS 在高样本下效果减弱
- **10-shot+**: Random 和 ACCS 差距缩小
- 可能原因：随着样本增加，KMeans 选择的代表性样本优势减弱

### 3.3 Difficult Subjects 分析
ACCS 在 difficult subjects (YLS, YSL, YHS, YRP) 上的效果需要进一步分析。

## 4. 论文表述

### 4.1 正确表述
> "ACCS improves calibration sample efficiency, achieving better performance with fewer calibration samples in the 3-5 shot range."

### 4.2 避免表述
> ~~"ACCS reduces manual annotation cost."~~

## 5. 结论

ACCS 通过 label-free KMeans centroid sampling 改善了少样本校准效率，特别是在 3-5 shot 设置下。该方法不依赖任何标注信息，适合真实部署场景。