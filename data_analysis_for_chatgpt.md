# ZuCo 2.0 数据集分析报告

## 一、数据概览

### 1.1 数据集来源
- 数据集名称：ZuCo 2.0 (Zurich Cognitive Language Processing Corpus)
- 任务：阅读困难识别（NR vs TSR 二分类）

### 1.2 被试信息
| 组别 | 前缀 | 人数 | 被试列表 |
|------|------|------|----------|
| 年轻成人对照组 | Y | 16 | YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YRK, YRP, YSD, YSL, YTL |
| 发展性阅读障碍组 | X | 10 | XAH, XBB, XBD, XDT, XLS, XPB, XSE, XSS, XTR, XWS |

### 1.3 数据类型
- EEG 电极特征
- Gaze 眼动特征  
- 文本句子数据

---

## 二、EEG 特征

### 2.1 文件格式
- 文件：`{subject}_electrode_features_all.npy`
- 数据结构：字典（dict）
- Key 格式：`{subject}_{condition}_{sentence_id}_{index}`

### 2.2 特征维度
- 每个样本：421 维
  - 前 420 维：420 个电极的特征值（float）
  - 最后 1 维：标签（'NR' 或 'TSR'）

### 2.3 标签定义
- NR (Normal Reading)：正常阅读条件
- TSR (Semantic Violation)：语义违背条件

### 2.4 示例数据
```python
# YAC 被试示例
key: 'YAC_NR_2_0'
value: [2.568, 1.537, 1.255, ..., 'NR']
```

---

## 三、Gaze 眼动特征

### 3.1 文件格式
- 文件：`{subject}_sent_gaze_sacc.npy`
- 数据结构：字典（dict）

### 3.2 特征维度
- 每个样本：10 维
  - [0] fixation_number: 注视点数量
  - [1] omission_rate: 遗漏率
  - [2] reading_speed: 阅读速度
  - [3] mean_sacc_amp: 平均眼跳幅度
  - [4] mean_sacc_dur: 平均眼跳持续时间
  - [5] mean_sacc_velocity: 平均眼跳速度
  - [6] max_sacc_amp: 最大眼跳幅度
  - [7] max_sacc_dur: 最大眼跳持续时间
  - [8] max_sacc_velocity: 最大眼跳速度
  - [9] label: 'NR' 或 'TSR'

### 3.3 关键注意事项
⚠️ **数据对齐问题**：gaze 文件中同一个句子有两个条目（NR 和 TSR 条件），必须使用 `label + sentence_id` 同时匹配才能正确对齐。

---

## 四、文本数据

### 4.1 文件格式
- NR 条件：`nr_1.csv` ~ `nr_7.csv`
- TSR 条件：`tsr_1.csv` ~ `tsr_7.csv`

### 4.2 数据格式
```
sentence_id;index;text;[label]
```

### 4.3 示例
```csv
# NR 示例
1;1;Henry Ford (July 30, 1863 - April 7, 1947) was the founder of...;

# TSR 示例  
22;1;Jonathan Aitken (born August 30, 1942) is a former Conservative minister...;POLITICAL_AFFILIATION
```

### 4.4 主题内容
- NR：传记文本（Henry Ford、Alexander Baldwin、Rosemary Clooney 等）
- TSR：政治人物及其政治立场声明

---

## 五、数据对齐方式

```python
def load_aligned_eeg_gaze(subject):
    # 1. 加载 EEG 和 Gaze 数据
    eeg_data = np.load(f'features/{subject}_electrode_features_all.npy', allow_pickle=True).item()
    gaze_data = np.load(f'features/{subject}_sent_gaze_sacc.npy', allow_pickle=True).item()
    
    # 2. 构建 gaze 索引：(label, sentence_id) -> gaze_key
    gaze_by_label_sent = {}
    for key in gaze_data.keys():
        parts = key.split('_')
        label = parts[1]
        sent_idx = int(parts[2])
        gaze_by_label_sent[(label, sent_idx)] = key
    
    # 3. 按 EEG key 对齐
    X_eeg, X_gaze, y = [], [], []
    for eeg_key in eeg_data.keys():
        parts = eeg_key.split('_')
        label = parts[1]
        sent_idx = int(parts[2])
        
        gaze_key = gaze_by_label_sent.get((label, sent_idx))
        if gaze_key:
            # 提取特征（去掉最后一个标签位）
            eeg_feat = np.array(eeg_data[eeg_key][:-1], dtype=float)
            gaze_feat = np.array(gaze_data[gaze_key][:-1], dtype=float)
            
            X_eeg.append(eeg_feat)
            X_gaze.append(gaze_feat)
            y.append(0 if label == 'NR' else 1)
    
    return np.array(X_eeg), np.array(X_gaze), np.array(y)
```

---

## 六、样本统计

以 YAC 被试为例：
- EEG 样本数：360
- Gaze 样本数：521（未对齐）
- 对齐后样本数：360
- NR 样本：158 (43.9%)
- TSR 样本：202 (56.1%)

---

## 七、典型实验协议

### 7.1 Few-shot Personalized Protocol
- k = 3, 5, 10, 20, 50（每个类别的标定样本数）
- 16 个 Y 组被试
- 5 个随机种子
- 50% 标定 / 50% 测试
- 无测试数据泄露

### 7.2 常用方法对比
| 方法 | 描述 |
|------|------|
| EEG_SVM | SVM on EEG features |
| Gaze_MLP | MLP on gaze features |
| Raw_Fusion | Concatenate EEG + Gaze + MLP |
| PCET_only | PCA reconstruction error enhanced EEG |
| PCET+GBE+CAGF | PCET + Gaze BE + Cross-modal Adaptive Gated Fusion |

---

## 八、文件目录结构

```
zuco-benchmark-main/
├── src/
│   ├── features/              # EEG/Gaze 特征文件
│   │   ├── YAC_electrode_features_all.npy
│   │   ├── YAC_sent_gaze_sacc.npy
│   │   └── ...
│   ├── task_materials/        # 文本数据
│   │   ├── nr_1.csv ~ nr_7.csv
│   │   └── tsr_1.csv ~ tsr_7.csv
│   ├── results/               # 实验结果
│   └── reports/               # 分析报告
└── matlab/                    # MATLAB 脚本
```

---

## 九、后续分析建议

你可以让 ChatGPT 帮你：
1. **特征工程**：提取更有效的 EEG/Gaze/文本特征
2. **模型设计**：设计多模态融合架构
3. **数据分析**：探索数据模式和相关性
4. **论文写作**：帮助撰写实验方法和结果部分

如需进一步分析，请提供具体需求！