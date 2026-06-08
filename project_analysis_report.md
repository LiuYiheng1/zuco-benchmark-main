# ZuCo Benchmark 项目分析报告

## 一、项目概述

**项目名称**：ZuCo Benchmark on Reading Task Classification

**研究目标**：基于 EEG（脑电图）和眼动追踪数据，进行跨受试者分类任务，区分**正常阅读（NR）** 和**任务特定信息搜索（TSR）** 两种阅读行为。

**数据来源**：[Zurich Cognitive Language Processing Corpus (ZuCo 2.0)](https://osf.io/2urht/)

**论文引用**：[Frontiers in Psychology 2022](https://www.frontiersin.org/articles/10.3389/fpsyg.2022.1028824/full)

---

## 二、项目结构

```
zuco-benchmark-main/
├── src/                      # 源代码目录
│   ├── benchmark.py          # 基准测试评估脚本（有真实标签）
│   ├── benchmark_baseline.py # 基线分类脚本（无真实标签，用于提交）
│   ├── classifier.py         # SVM分类器实现
│   ├── config.py             # 配置文件
│   ├── data_helpers.py       # 数据处理辅助函数
│   ├── data_loading_helpers.py # MATLAB数据加载工具
│   ├── extract_features.py   # 特征提取模块
│   ├── feature_cleaner.py    # 特征清洗
│   ├── features/             # 预提取的特征文件（.npy格式）
│   └── task_materials/       # 任务材料（CSV文件）
├── matlab/                   # MATLAB脚本（绘图、拓扑图等）
├── README.md
├── requirements.txt
└── get_data.sh
```

---

## 三、核心配置（config.py）

### 3.1 数据划分

| 集合 | 受试者ID | 数量 | 用途 |
|------|----------|------|------|
| **训练集** | YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL | 16人 | 模型训练 |
| **测试集** | XBB, XDT, XLS, XPB, XSE, XTR, XWS, XAH, XBD, XSS | 10人 | 模型评估/提交 |

### 3.2 特征集类型

```python
# 默认特征集配置
feature_sets = ["electrode_features_all", "sent_gaze_sacc", "sent_gaze_sacc_eeg_means"]
```

| 特征类别 | 特征名称 | 说明 |
|----------|----------|------|
| **电极特征** | `electrode_features_{theta/alpha/beta/gamma/all}` | 各频段原始电极数据（105通道） |
| **EEG均值** | `{theta/alpha/beta/gamma}_mean`, `eeg_means` | 各频段均值特征 |
| **眼动追踪** | `fixation_number`, `omission_rate`, `reading_speed`, `sent_gaze`, `sent_saccade`, `mean_sacc_dur`, `max_sacc_velocity` 等 | 注视、扫视相关特征 |
| **组合特征** | `sent_gaze_eeg_means`, `sent_gaze_sacc_eeg_means` | 眼动+EEG组合特征 |

### 3.3 模型配置

```python
seed = 1
runs = 1
kernel = 'linear'  # 线性核用于系数分析
pca_preprocessing = False
explained_variance = 0.95
class_task = 'tasks-cross-subj'  # 跨受试者任务分类
```

---

## 四、特征文件结构（src/features/）

### 文件命名规则

```
{subject_id}_{feature_set}.npy
```

### 数据格式

每个 `.npy` 文件存储一个字典对象：

```python
# 键格式
{subject}_{label}_{idx}_{full_idx}

# 值结构（最后一个元素是标签）
[feature_vector..., label]
```

**字段说明**：
- `subject`：受试者ID（如 XAH、YAC）
- `label`：任务标签（NR=正常阅读，TSR=任务搜索）
- `idx`：句子在原始数据中的索引
- `full_idx`：全局索引

### 特征维度统计

| 特征集 | 维度 | 数据来源 |
|--------|------|----------|
| `theta_mean` | 1 | EEG |
| `alpha_mean` | 1 | EEG |
| `beta_mean` | 1 | EEG |
| `gamma_mean` | 1 | EEG |
| `eeg_means` | 4 | EEG（theta+alpha+beta+gamma） |
| `fixation_number` | 1 | 眼动 |
| `omission_rate` | 1 | 眼动 |
| `reading_speed` | 1 | 眼动 |
| `sent_gaze` | 4 | 眼动（omr, nFix, speed, sacc_dur） |
| `sent_saccade` | 6 | 眼动（扫视参数） |
| `sent_gaze_sacc` | 9 | 眼动组合 |
| `sent_gaze_eeg_means` | 11 | 眼动+EEG |
| `sent_gaze_sacc_eeg_means` | 13 | 眼动+扫视+EEG |
| `electrode_features_theta` | 105 | EEG电极 |
| `electrode_features_alpha` | 105 | EEG电极 |
| `electrode_features_beta` | 105 | EEG电极 |
| `electrode_features_gamma` | 105 | EEG电极 |
| `electrode_features_all` | 420 | 4频段×105电极 |

---

## 五、任务材料（src/task_materials/）

### 文件分类

| 文件类型 | 数量 | 内容 |
|----------|------|------|
| `nr_1.csv` ~ `nr_7.csv` | 7个 | 正常阅读任务句子 |
| `tsr_1.csv` ~ `tsr_7.csv` | 7个 | 任务搜索句子（带标注类型） |
| `nr_*_control_questions.csv` | 7个 | 控制问题 |

### CSV格式

```
{sentence_id};{block_id};{sentence_text};{optional_label}
```

**NR示例（nr_1.csv）**：
```
1;1;Henry Ford (July 30, 1863 - April 7, 1947) was the founder of the Henry Ford Motor Company...;
2;1;Henry Ford, with eleven other investors and $28,000 in capital...;CONTROL
```

**TSR示例（tsr_1.csv）**：
```
22;1;Jonathan Aitken (born August 30, 1942) is a former Conservative minister...;POLITICAL_AFFILIATION
```

### 标注类型（TSR任务）

- `POLITICAL_AFFILIATION`：政治派别标注
- `CONTROL`：控制句（无特定标注）

---

## 六、核心工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     数据加载/特征提取                           │
│  data_helpers.get_or_extract_features(subjects, rootdir)       │
│  ├── 检查 features/{subj}_{feat}.npy 是否存在                  │
│  ├── 存在 → 直接加载 .npy 文件                                  │
│  └── 不存在 → 从 .mat 文件提取（extract_features.py）           │
└─────────────────────────┬─────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                     可选PCA预处理                              │
│  data_helpers.apply_pca_preprocessing(train, test)             │
│  └── 仅对 electrode_features 应用PCA降维                       │
└─────────────────────────┬─────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                     模型训练与预测                             │
│  classifier.benchmark_baseline(X, y, test_X, test_y)          │
│  ├── build_data() → 构建训练/测试数据                          │
│  ├── MinMaxScaler 归一化到 [0,1]                              │
│  ├── SVM训练（线性核, random_state=seed）                      │
│  └── 对每个测试受试者独立预测                                   │
└─────────────────────────┬─────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                     结果输出                                   │
│  ├── predictions/ 目录保存每个受试者的预测结果                  │
│  └── submissions/ 目录生成符合EvalAI格式的JSON提交文件          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 七、关键代码模块说明

### 7.1 classifier.py

**核心函数**：

| 函数 | 功能 |
|------|------|
| `build_data(data, labels)` | 将受试者字典转换为特征矩阵 X 和标签向量 y |
| `benchmark_baseline(X, y, test_X, test_y)` | 基线分类流程（训练+预测） |
| `predict_subject_baseline(clf, test_X, index, subj)` | 对单个受试者进行预测 |

**标签编码逻辑**：
```python
# classifier.py line 20-23
if label == "NR":    # Normal Reading → 正常阅读
    y.append(1)
else:                # TSR → 任务搜索
    y.append(0)
```

### 7.2 extract_features.py

**核心函数**：

| 函数 | 功能 |
|------|------|
| `extract_sentence_features()` | 从MATLAB数据中提取句子级特征 |
| `extract_fixation_features()` | 提取注视级别的EEG特征 |
| `flesch_reading_ease()` | 计算Flesch阅读难度分数 |

**特征提取示例**（sent_gaze_sacc_eeg_means）：
```python
# 眼动特征
weighted_nFix = np.array(af['duration']).shape[0] / lnorm
weighted_speed = (np.sum(np.array(af['duration'])) * 2 / 100) / lnorm

# 组合特征 = [眼动(9) + EEG均值(4) + 标签]
[omr, weighted_nFix, weighted_speed, smeand, smaxv, smeanv, smaxd, smeana, smaxa, 
 t_mean, a_mean, b_mean, g_mean, label]
```

### 7.3 data_helpers.py

**核心函数**：

| 函数 | 功能 |
|------|------|
| `get_or_extract_features()` | 特征加载/提取入口 |
| `read_mat_file()` | 读取MATLAB .mat文件 |
| `apply_pca_preprocessing()` | 应用PCA降维 |
| `create_submission()` | 生成EvalAI提交文件 |
| `log_results()` | 记录评估结果 |

---

## 八、数据处理细节

### 8.1 数据预处理流程

```python
# 1. 构建数据
train_X, train_y = build_data(X, y)

# 2. 数据洗牌
train_X, train_y = shuffle(train_X, train_y)

# 3. 归一化（仅使用训练数据拟合）
scaling = MinMaxScaler(feature_range=(0, 1)).fit(train_X)
train_X = scaling.transform(train_X)
test_X = [scaling.transform(subj) for subj in test_X]

# 4. SVM训练
clf = SVC(random_state=config.seed, kernel=config.kernel, gamma='scale', cache_size=1000)
clf.fit(train_X, train_y)
```

### 8.2 跨受试者验证策略

该项目采用**跨受试者（cross-subject）** 验证策略：
- 训练集和测试集来自不同受试者
- 模型需要学习泛化到未见过的受试者数据
- 这是更具挑战性的评估方式，模拟真实场景

---

## 九、提交格式

提交文件为JSON格式，保存在 `submissions/` 目录：

```json
{
  "XBB": {
    "0": 1,
    "1": 0,
    "2": 1,
    "3": 0,
    ...
  },
  "XDT": {
    "0": 0,
    "1": 1,
    ...
  },
  "XLS": { ... },
  ...
}
```

**字段说明**：
- 顶层键：受试者ID
- 内层键：句子索引
- 值：预测标签（0=TSR，1=NR）

---

## 十、评估指标

项目使用以下指标评估模型性能：

| 指标 | 计算方式 |
|------|----------|
| **Accuracy** | 正确预测数 / 总样本数 |
| **F1 Score** | 2 × Precision × Recall / (Precision + Recall) |
| **Precision** | 真阳性 / (真阳性 + 假阳性) |
| **Recall** | 真阳性 / (真阳性 + 假阴性) |

---

## 十一、总结

| 维度 | 描述 |
|------|------|
| **任务类型** | 二分类（NR vs TSR） |
| **数据模态** | EEG（105通道）+ 眼动追踪 |
| **受试者数** | 训练16人，测试10人 |
| **特征维度** | 1~420维 |
| **核心模型** | SVM（线性核） |
| **评估指标** | Accuracy, F1, Precision, Recall |
| **提交平台** | EvalAI |

---

## 十二、文件清单

### 源代码文件

| 文件 | 路径 | 说明 |
|------|------|------|
| benchmark.py | [src/benchmark.py](file:///d:/pycharmproject/zuco-benchmark-main/src/benchmark.py) | 基准测试（有真实标签） |
| benchmark_baseline.py | [src/benchmark_baseline.py](file:///d:/pycharmproject/zuco-benchmark-main/src/benchmark_baseline.py) | 基线提交脚本 |
| classifier.py | [src/classifier.py](file:///d:/pycharmproject/zuco-benchmark-main/src/classifier.py) | SVM分类器 |
| config.py | [src/config.py](file:///d:/pycharmproject/zuco-benchmark-main/src/config.py) | 配置文件 |
| data_helpers.py | [src/data_helpers.py](file:///d:/pycharmproject/zuco-benchmark-main/src/data_helpers.py) | 数据辅助函数 |
| data_loading_helpers.py | [src/data_loading_helpers.py](file:///d:/pycharmproject/zuco-benchmark-main/src/data_loading_helpers.py) | MATLAB数据加载 |
| extract_features.py | [src/extract_features.py](file:///d:/pycharmproject/zuco-benchmark-main/src/extract_features.py) | 特征提取 |

### 数据文件

| 目录 | 说明 |
|------|------|
| [src/features/](file:///d:/pycharmproject/zuco-benchmark-main/src/features/) | 预提取特征（约260+个.npy文件） |
| [src/task_materials/](file:///d:/pycharmproject/zuco-benchmark-main/src/task_materials/) | 任务材料（21个CSV文件） |

---

**报告生成时间**：2026年5月  
**项目版本**：ZuCo Benchmark v1.0