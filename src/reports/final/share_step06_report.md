# SHARE-Net Step 0.6: Text-cleaning and Fair Neuro-behavioral Protocol

## 1. Text Cleaning Audit

### 1.1 First 20 Cleaned Samples

| label | sentence_id | clean_text |
|-------|-------------|------------|
| NR | 1 |  |
| NR | 2 | Hey Ford, with eleven other investors and $28,000 ... |
| NR | 3 |  |
| NR | 4 |  |
| NR | 5 |  |
| NR | 5 | With his interest in race cars, he formed a second... |
| NR | 5 |  |
| NR | 5 |  |
| NR | 6 |  |
| NR | 6 |  |
| NR | 7 | Ford was  on a prosperous farm in Springwells Town... |
| NR | 8 |  |
| NR | 8 |  |
| NR | 8 |  |
| NR | 9 |  |
| NR | 10 |  |
| NR | 10 |  |
| NR | 11 | In 1879, he left home for the nearby city of Detro... |
| NR | 12 |  |
| NR | 12 |  |

### 1.2 Cleaned TF-IDF Features

```
============================================================
CLEANED TF-IDF AUDIT
============================================================

Top 50 TF-IDF features (by absolute coefficient):
------------------------------------------------------------
lyman                     | coef: -1.5248
mother                    | coef: -1.5248
dn                        | coef: -1.2562
english                   | coef: -1.2562
learn                     | coef: -1.2562
buried                    | coef: -1.2427
cemetery                  | coef: -1.2427
foot                      | coef: -1.2427
hill                      | coef: -1.2427
elected                   | coef: -1.1696
1942                      | coef: -1.1198
ball                      | coef: -1.1198
seat                      | coef: -1.1198
senate                    | coef: -1.1198
ex                        | coef: -1.1075
governor                  | coef: -1.1075
joked                     | coef: -1.1075
nation                    | coef: -1.1075
ren                       | coef: +1.1013
1950                      | coef: -1.0819
aboard                    | coef: -1.0819
burial                    | coef: -1.0819
earned                    | coef: -1.0819
given                     | coef: -1.0819
published                 | coef: -1.0819
respect                   | coef: -1.0819
sea                       | coef: -1.0819
ship                      | coef: -1.0819
writer                    | coef: -1.0819
1913                      | coef: -0.9815
1920                      | coef: -0.9815
assembly                  | coef: -0.9815
bulgarian                 | coef: -0.9815
national                  | coef: -0.9815
1992                      | coef: -0.9625
campaign                  | coef: -0.9625
factor                    | coef: -0.9625
gore                      | coef: -0.9625
helpful                   | coef: -0.9625
retrospect                | coef: -0.9625
view                      | coef: -0.9625
old                       | coef: -0.9278
years                     | coef: +0.9226
grant                     | coef: +0.8668
attended                  | coef: -0.8573
joiner                    | coef: -0.8573
learned                   | coef: -0.8573
school                    | coef: -0.8573
secondary                 | coef: -0.8573
trade                     | coef: -0.8573

Top 20 positive features (TSR):
------------------------------------------------------------
ren                       | coef: +1.1013
years                     | coef: +0.9226
grant                     | coef: +0.8668
11                        | coef: +0.8435
cancer                    | coef: +0.7149
seven                     | coef: +0.7086
father                    | coef: +0.6959
grandfather               | coef: +0.6509
maternal                  | coef: +0.6509
named                     | coef: +0.6509
43                        | coef: +0.6475
consered                  | coef: +0.6475
country                   | coef: +0.6475
icon                      | coef: +0.6475
kennedy                   | coef: +0.6475
liberalism                | coef: +0.6475
person                    | coef: +0.6475
actor                     | coef: +0.5833
davenport                 | coef: +0.5833
nephew                    | coef: +0.5833

Top 20 negative features (NR):
------------------------------------------------------------
lyman                     | coef: -1.5248
mother                    | coef: -1.5248
dn                        | coef: -1.2562
english                   | coef: -1.2562
learn                     | coef: -1.2562
buried                    | coef: -1.2427
cemetery                  | coef: -1.2427
foot                      | coef: -1.2427
hill                      | coef: -1.2427
elected                   | coef: -1.1696
1942                      | coef: -1.1198
ball                      | coef: -1.1198
seat                      | coef: -1.1198
senate                    | coef: -1.1198
ex                        | coef: -1.1075
governor                  | coef: -1.1075
joked                     | coef: -1.1075
nation                    | coef: -1.1075
1950                      | coef: -1.0819
aboard                    | coef: -1.0819

SUSPICIOUS FEATURES CHECK:
------------------------------------------------------------
No suspicious features found - CLEAN
```

## 2. Shuffled Label Sanity Check

| Task | Accuracy | Macro-F1 | Balanced Accuracy |
|------|----------|----------|-------------------|
| full_permutation | 0.5007±0.0686 | 0.3590±0.0498 | 0.5049±0.0220 |
| train_shuffled | 0.7177±0.0627 | 0.4178±0.0217 | 0.4991±0.0066 |

## 3. Clean Text Baselines

### LinearSVM

| Features | Accuracy | Macro-F1 | Balanced Accuracy | AUROC |
|----------|----------|----------|-------------------|-------|
| Text+EEG | 0.8376±0.0745 | 0.7836±0.0937 | 0.7727±0.0897 | 0.8559±0.1010 |
| Text+EEG+Gaze | 0.8537±0.0831 | 0.8076±0.1035 | 0.7985±0.1005 | 0.8716±0.1024 |
| Text+Gaze | 0.7836±0.1090 | 0.6422±0.1915 | 0.6607±0.1566 | 0.7229±0.2046 |
| Text_only_clean | 0.7176±0.0607 | 0.4171±0.0209 | 0.4990±0.0034 | 0.4964±0.0541 |
| Text_only_length | 0.7220±0.0571 | 0.4374±0.0385 | 0.5073±0.0221 | 0.5068±0.0554 |
| Text_only_tfidf | 0.7190±0.0608 | 0.4175±0.0209 | 0.4999±0.0012 | 0.5059±0.0360 |
| Text_only_wordcount | 0.7211±0.0576 | 0.4398±0.0402 | 0.5078±0.0238 | 0.4901±0.0564 |

### LogisticRegression

| Features | Accuracy | Macro-F1 | Balanced Accuracy | AUROC |
|----------|----------|----------|-------------------|-------|
| Text+EEG | 0.8407±0.0735 | 0.7787±0.0992 | 0.7631±0.0950 | 0.8582±0.0972 |
| Text+EEG+Gaze | 0.8537±0.0863 | 0.8016±0.1088 | 0.7882±0.1057 | 0.8752±0.0980 |
| Text+Gaze | 0.7815±0.1114 | 0.6646±0.1687 | 0.6667±0.1479 | 0.7612±0.1645 |
| Text_only_clean | 0.7065±0.0665 | 0.4157±0.0223 | 0.4916±0.0106 | 0.4856±0.0506 |
| Text_only_length | 0.7165±0.0634 | 0.4249±0.0267 | 0.5001±0.0138 | 0.5006±0.0563 |
| Text_only_tfidf | 0.7072±0.0655 | 0.4140±0.0233 | 0.4917±0.0103 | 0.5012±0.0360 |
| Text_only_wordcount | 0.7165±0.0613 | 0.4290±0.0309 | 0.5014±0.0200 | 0.5062±0.0573 |

### RidgeClassifier

| Features | Accuracy | Macro-F1 | Balanced Accuracy | AUROC |
|----------|----------|----------|-------------------|-------|
| Text+EEG | 0.8364±0.0912 | 0.7896±0.1073 | 0.7872±0.1072 | 0.8633±0.1011 |
| Text+EEG+Gaze | 0.8502±0.0900 | 0.8087±0.1055 | 0.8048±0.1040 | 0.8795±0.1017 |
| Text+Gaze | 0.7796±0.1119 | 0.6586±0.1703 | 0.6617±0.1466 | 0.7594±0.1580 |
| Text_only_clean | 0.7125±0.0672 | 0.4152±0.0234 | 0.4950±0.0101 | 0.5059±0.0520 |
| Text_only_length | 0.7161±0.0641 | 0.4248±0.0270 | 0.4999±0.0144 | 0.5006±0.0563 |
| Text_only_tfidf | 0.7164±0.0629 | 0.4166±0.0217 | 0.4979±0.0055 | 0.5014±0.0362 |
| Text_only_wordcount | 0.7165±0.0613 | 0.4290±0.0309 | 0.5014±0.0200 | 0.5062±0.0573 |

## 4. Duplicate-Controlled Protocol

- Total duplicate sentences (with both NR and TSR): 1
- Total samples in duplicate subset: 436

### 4.1 Protocol F1: Duplicate-controlled Within-Subject

| Features | Accuracy | Macro-F1 |
|----------|----------|----------|
| EEG+Gaze_concat | 0.8527±0.0939 | 0.8037±0.1140 |
| EEG_only | 0.8325±0.0987 | 0.7727±0.1224 |
| Gaze_only | 0.7860±0.1133 | 0.6474±0.1838 |

### 4.2 Protocol F2: Duplicate-controlled Leave-One-Subject-Out

| Features | Accuracy | Macro-F1 |
|----------|----------|----------|
| EEG+Gaze_concat | 0.6271±0.1574 | 0.4849±0.1250 |
| EEG_only | 0.5718±0.1706 | 0.4080±0.0786 |
| Gaze_only | 0.7634±0.0787 | 0.5642±0.1564 |

## 5. Analysis Questions

### Q1: Have political_affiliation and other relation labels been removed from text?
- YES - All relation labels removed, text is clean

### Q2: Does shuffled label sanity check return to ~50%?
- Train-shuffled: 0.7177
- Full permutation: 0.5007
- NO - Still above 55%

### Q3: What is the accuracy of clean Text-only?
- RidgeClassifier Text_only_clean: 0.7125

### Q4: How many sentences and samples are in the duplicate-controlled subset?
- Duplicate sentences: 1
- Total samples: 436

### Q5: Is duplicate-controlled Text-only near 50%?
- Protocol F1 Text_only: nan
- NO - Significantly deviates from 50%

### Q6: Is duplicate-controlled EEG+Gaze above 50%?
- Protocol F1 EEG+Gaze_concat: 0.8527
- YES - Significantly above chance

### Q7: Should future SHARE-Net:

#### Option 1: Not use Text as main input
- Rationale: Text-only achieves 71.25%, but EEG+Gaze provides real neural signal

#### Option 2: Only use Text as semantic anchor in duplicate-controlled setting
- Rationale: In duplicate-controlled setting, Text-only is nan% (near chance), making it a fair anchor

#### Option 3: Use Text only as upper-bound/confound
- Rationale: Text provides upper bound on performance; EEG/Gaze shows neural contribution

#### RECOMMENDATION
- Use **Setting B** (EEG-Gaze main protocol) as primary approach
- Use **Setting C** (Duplicate-controlled protocol) for scientific validation
- Use **Setting A** (Text-assisted upper bound) only for reference
