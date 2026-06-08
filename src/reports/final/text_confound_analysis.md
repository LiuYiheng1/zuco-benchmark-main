# Text/Material Confound Analysis

## 1. 实验背景

本文任务为 NR (Normal Reading) vs TSR (Task-Specific Reading) 分类。
需要分析 text/material 是否构成 confound，即 text proxy 是否能区分 NR/TSR。

## 2. Text Proxy Baseline 结果

| Method | 1-shot | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|--------|--------|---------|---------|---------|
| Text_Proxy | 0.536 | 0.600 | 0.622 | 0.647 | 0.677 | 0.699 |
| Gaze_SVM | 0.549 | 0.611 | 0.633 | 0.661 | 0.683 | 0.709 |
| EEG_MLP | 0.527 | 0.590 | 0.621 | 0.664 | 0.713 | 0.786 |
| EEG_SVM | 0.527 | 0.585 | 0.608 | 0.657 | 0.711 | 0.786 |

## 3. 关键发现

### 3.1 Text Proxy 确实能区分 NR/TSR
- Text proxy 在 50-shot 时达到 69.9% accuracy
- 这表明 text/material 特征与阅读状态标签相关

### 3.2 EEG 与 Text Proxy 比较
- **EEG_MLP 50-shot**: 78.6%
- **Text_Proxy 50-shot**: 69.9%
- **EEG > Text Proxy**: EEG 超过了 text proxy，说明 EEG 编码了超越 text material 的信息

### 3.3 EEG + Text Proxy 组合
- **Combined_SVM 50-shot**: 73.3%
- Combined 方法没有超过 EEG alone，说明 EEG 已经包含了主要信息

## 4. Confound 评估

### 4.1 Text/material 是否影响分类？
**是**。Text proxy 能达到 ~70% 准确率，说明 text material 与阅读状态存在相关性。

### 4.2 EEG 是否纯粹解码 cognitive state？
**不是**。EEG 性能部分可能来自 text/material confound。

### 4.3 如何正确表述？
> "NR/TSR classification in this study is protocol-conditioned reading state recognition, which may be influenced by text/material confounds."

## 5. 论文写作边界

### 5.1 不能写
> ~~"EEG decodes pure cognitive state invariant to stimulus."~~
> ~~"NR/TSR classification is free from text/material confounds."~~

### 5.2 正确表述
> "This study focuses on protocol-conditioned reading state recognition under NR vs TSR paradigms. While EEG features provide strong signals, we acknowledge that text/material confounds may partially contribute to classification performance."

## 6. 结论

1. Text proxy 能区分 NR/TSR (~70% at 50-shot)
2. EEG 超过 text proxy，说明 EEG 编码了超越 text 的信息
3. 本文任务定义为 protocol-conditioned reading state recognition
4. 不声称 stimulus-invariant cognitive decoding