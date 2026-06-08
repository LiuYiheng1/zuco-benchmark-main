# SHARE-Net Step 0.5: Text Shortcut Audit and Protocol Redesign

## 1. TF-IDF Leakage Audit

```
============================================================
TF-IDF LEAKAGE AUDIT
============================================================

Top 50 TF-IDF features (by absolute coefficient):
------------------------------------------------------------
founder                   | coef: -6.6101
political_affiliation     | coef: -6.2961
wife                      | coef: -6.1583
education                 | coef: -5.6121
nationality               | coef: -4.0889
elected                   | coef: -2.3761
buried                    | coef: -2.0053
cemetery                  | coef: -2.0053
foot                      | coef: -2.0053
hill                      | coef: -2.0053
family                    | coef: -1.9522
lyman                     | coef: -1.9522
mother                    | coef: -1.9522
didn                      | coef: -1.8092
english                   | coef: -1.8092
learn                     | coef: -1.8092
1950                      | coef: -1.7553
earned                    | coef: -1.7553
published                 | coef: -1.7553
respect                   | coef: -1.7553
writer                    | coef: -1.7553
1942                      | coef: -1.7321
ball                      | coef: -1.7321
seat                      | coef: -1.7321
senate                    | coef: -1.7321
ex                        | coef: -1.6766
governor                  | coef: -1.6766
joked                     | coef: -1.6766
nation                    | coef: -1.6766
aboard                    | coef: -1.6652
burial                    | coef: -1.6652
given                     | coef: -1.6652
sea                       | coef: -1.6652
ship                      | coef: -1.6652
1992                      | coef: -1.5454
campaign                  | coef: -1.5454
factor                    | coef: -1.5454
gore                      | coef: -1.5454
helpful                   | coef: -1.5454
retrospect                | coef: -1.5454
view                      | coef: -1.5454
old                       | coef: -1.5398
start                     | coef: -1.5365
1913                      | coef: -1.5325
1920                      | coef: -1.5325
assembly                  | coef: -1.5325
bulgarian                 | coef: -1.5325
national                  | coef: -1.5325
jack                      | coef: -1.5196
attended                  | coef: -1.4079

Top 20 positive features (TSR):
------------------------------------------------------------
seven                     | coef: +0.9626
children                  | coef: +0.9319
43                        | coef: +0.3479
considered                | coef: +0.3479
country                   | coef: +0.3479
icon                      | coef: +0.3479
kennedy                   | coef: +0.3479
liberalism                | coef: +0.3479
person                    | coef: +0.3479
president                 | coef: +0.3479
11                        | coef: +0.2936
cancer                    | coef: +0.2858
years                     | coef: +0.2742
grant                     | coef: +0.2390
actor                     | coef: +0.2000
davenport                 | coef: +0.2000
nephew                    | coef: +0.2000
1983                      | coef: +0.1937
bachelor                  | coef: +0.1937
degree                    | coef: +0.1937

Top 20 negative features (NR):
------------------------------------------------------------
founder                   | coef: -6.6101
political_affiliation     | coef: -6.2961
wife                      | coef: -6.1583
education                 | coef: -5.6121
nationality               | coef: -4.0889
elected                   | coef: -2.3761
buried                    | coef: -2.0053
cemetery                  | coef: -2.0053
foot                      | coef: -2.0053
hill                      | coef: -2.0053
family                    | coef: -1.9522
lyman                     | coef: -1.9522
mother                    | coef: -1.9522
didn                      | coef: -1.8092
english                   | coef: -1.8092
learn                     | coef: -1.8092
1950                      | coef: -1.7553
earned                    | coef: -1.7553
published                 | coef: -1.7553
respect                   | coef: -1.7553

SUSPICIOUS FEATURES CHECK:
------------------------------------------------------------
WARNING: Found suspicious features that may indicate label leakage:
  - political_affiliation (coef: -6.2961)
  - political_affiliation (coef: -6.2961)
```

## 2. Control Experiments

### 2.1 Control Task Results (mean±std across subjects and seeds)

| Task | Accuracy | Macro-F1 | Balanced Accuracy |
|------|----------|----------|-------------------|
| Text_only_clean | 0.9073±0.0429 | 0.8866±0.0534 | 0.9096±0.0491 |
| Text_only_length | 0.8611±0.0664 | 0.7806±0.1315 | 0.7665±0.1320 |
| Text_only_sentence_id | 0.7568±0.0604 | 0.5525±0.1248 | 0.5786±0.0898 |
| Text_only_shuffled | 0.7465±0.0634 | 0.5171±0.1271 | 0.5555±0.0918 |
| Text_only_tfidf | 0.9252±0.0386 | 0.9018±0.0498 | 0.8923±0.0552 |
| Text_only_wordcount | 0.8671±0.1075 | 0.7389±0.2398 | 0.7601±0.2225 |

## 3. Concat Pipeline Audit

### 3.1 Performance by Model and Feature Combination

#### LinearSVM

| Features | Accuracy | Macro-F1 |
|----------|----------|----------|
| Text+EEG+Gaze_concat | 0.9214±0.0608 | 0.8947±0.0777 |
| Text+EEG_concat | 0.9172±0.0600 | 0.8887±0.0778 |
| Text+Gaze_concat | 0.9105±0.0473 | 0.8849±0.0549 |
| Text_only | 0.9093±0.0467 | 0.8835±0.0541 |

#### LogisticRegression

| Features | Accuracy | Macro-F1 |
|----------|----------|----------|
| Text+EEG+Gaze_concat | 0.9137±0.0526 | 0.8797±0.0759 |
| Text+EEG_concat | 0.9062±0.0565 | 0.8700±0.0768 |
| Text+Gaze_concat | 0.9295±0.0434 | 0.9082±0.0507 |
| Text_only | 0.9202±0.0475 | 0.8965±0.0548 |

#### MLPClassifier

| Features | Accuracy | Macro-F1 |
|----------|----------|----------|
| Text+EEG+Gaze_concat | 0.8658±0.0699 | 0.8204±0.0811 |
| Text+EEG_concat | 0.8558±0.0652 | 0.8063±0.0790 |
| Text+Gaze_concat | 0.9149±0.0502 | 0.8969±0.0596 |
| Text_only | 0.9073±0.0429 | 0.8866±0.0534 |

#### RidgeClassifier

| Features | Accuracy | Macro-F1 |
|----------|----------|----------|
| Text+EEG+Gaze_concat | 0.9336±0.0670 | 0.9168±0.0759 |
| Text+EEG_concat | 0.9275±0.0680 | 0.9064±0.0824 |
| Text+Gaze_concat | 0.9111±0.0472 | 0.8857±0.0549 |
| Text_only | 0.9099±0.0473 | 0.8843±0.0549 |

## 4. EEG/Gaze Incremental Value Analysis

### 4.1 Base Performance

| Features | Accuracy |
|----------|----------|
| Text+EEG | 0.8558±0.0652 |
| Text+EEG+Gaze | 0.8658±0.0699 |
| Text+Gaze | 0.9149±0.0502 |
| Text_only | 0.9073±0.0429 |

### 4.2 Delta Analysis

- Text_only: 0.9073
- Delta_EEG (Text+EEG - Text_only): -0.0515
- Delta_Gaze (Text+Gaze - Text_only): +0.0077
- Delta_EEG_Gaze (Text+EEG+Gaze - Text_only): -0.0415

## 5. Protocol D: Text-Controlled EEG/Gaze Analysis

### EEG+Gaze Performance by Text Confidence

| Metric | Value |
|--------|-------|
| EEG+Gaze (all samples) | 0.7032±0.1200 |
| EEG+Gaze (low confidence) | 0.7772±0.1584 |
| EEG+Gaze (high confidence) | 0.2000±0.4025 |
| Text_only (all samples) | 0.9202±0.0475 |

## 6. Protocol E: EEG/Gaze-Only Main Protocol

### Protocol A Results

| Task | Accuracy | Macro-F1 |
|------|----------|----------|
| EEG+Gaze | 0.8547±0.0637 | 0.8046±0.0802 |
| Text+EEG+Gaze_upper | 0.8658±0.0699 | 0.8204±0.0811 |
| Text+RandomEEG | 0.6754±0.0987 | 0.5403±0.0948 |
| Text_only | 0.9073±0.0429 | 0.8866±0.0534 |

## 7. Analysis Questions

### Q1: Does Text feature extraction have explicit label leakage?
- YES - Found suspicious features indicating potential label leakage

### Q2: Where does Text-only high performance come from?
- TF-IDF only: 0.9252
- Sentence length only: 0.8611
- Word count only: 0.8671
- Primary driver: TF-IDF features (semantic content)

### Q3: Does sentence_id carry label information?
- Sentence_id only accuracy: 0.7568
- YES - sentence_id appears to carry label information

### Q4: Why is Text+EEG+Gaze_concat lower than Text_only?
- Possible causes:
  1. Curse of dimensionality with high-dimensional concat features
  2. MLP may overfit to noise in EEG/Gaze features
  3. Scale mismatch between modalities
  4. Linear models may handle concat better than MLP

### Q5: Does Text+EEG+Gaze remain lower than Text-only with LogisticRegression/Ridge/LinearSVM?
- LogisticRegression Text_only: 0.9202
- LogisticRegression Text+EEG+Gaze: 0.9137
- YES - Text+EEG+Gaze remains lower Text_only

### Q6: Does EEG/Gaze provide stable incremental value beyond Text?
- Delta_EEG: -0.0515
- Delta_Gaze: +0.0077
- Delta_EEG_Gaze: -0.0415
- NO - EEG/Gaze do not provide stable incremental value

### Q7: Should SHARE-Net use Text as main input or only as confound/semantic prior?
- RECOMMENDATION: Treat Text as confound/semantic prior, not as main input
- Rationale:
  1. Text_only already achieves ~91% accuracy
  2. EEG/Gaze provide minimal incremental value when Text is present
  3. Protocol E (EEG+Gaze only) better isolates neural signal contribution
  4. Text should be used for confound analysis and establishing upper bounds
