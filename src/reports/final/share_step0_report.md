# SHARE-Net Step 0: Data Alignment and Baseline Reproduction

## 1. Data Alignment Audit

### Subject YAC
- EEG samples: 360
- Gaze samples: 521
- Text samples: 509
- Aligned samples: 125
- NR count: 102
- TSR count: 23
- Label consistency rate: 100.0%

### Subject YAG
- EEG samples: 658
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 196
- NR count: 142
- TSR count: 54
- Label consistency rate: 100.0%

### Subject YAK
- EEG samples: 577
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 143
- NR count: 94
- TSR count: 49
- Label consistency rate: 100.0%

### Subject YDG
- EEG samples: 526
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 149
- NR count: 108
- TSR count: 41
- Label consistency rate: 100.0%

### Subject YDR
- EEG samples: 618
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 175
- NR count: 123
- TSR count: 52
- Label consistency rate: 100.0%

### Subject YFR
- EEG samples: 350
- Gaze samples: 602
- Text samples: 509
- Aligned samples: 128
- NR count: 104
- TSR count: 24
- Label consistency rate: 100.0%

### Subject YFS
- EEG samples: 488
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 123
- NR count: 75
- TSR count: 48
- Label consistency rate: 100.0%

### Subject YHS
- EEG samples: 717
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 218
- NR count: 163
- TSR count: 55
- Label consistency rate: 100.0%

### Subject YIS
- EEG samples: 729
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 215
- NR count: 158
- TSR count: 57
- Label consistency rate: 100.0%

### Subject YLS
- EEG samples: 470
- Gaze samples: 594
- Text samples: 509
- Aligned samples: 116
- NR count: 70
- TSR count: 46
- Label consistency rate: 100.0%

### Subject YMD
- EEG samples: 540
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 163
- NR count: 128
- TSR count: 35
- Label consistency rate: 100.0%

### Subject YRK
- EEG samples: 234
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 66
- NR count: 43
- TSR count: 23
- Label consistency rate: 100.0%

### Subject YRP
- EEG samples: 387
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 121
- NR count: 88
- TSR count: 33
- Label consistency rate: 100.0%

### Subject YSD
- EEG samples: 713
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 206
- NR count: 150
- TSR count: 56
- Label consistency rate: 100.0%

### Subject YSL
- EEG samples: 691
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 199
- NR count: 146
- TSR count: 53
- Label consistency rate: 100.0%

### Subject YTL
- EEG samples: 697
- Gaze samples: 739
- Text samples: 509
- Aligned samples: 206
- NR count: 154
- TSR count: 52
- Label consistency rate: 100.0%

## 2. Label Consistency Check
- All subjects have 100% label consistency: YES

## 3. Baseline Results

### Protocol A

| Method | Accuracy (mean±std) | Macro-F1 (mean±std) | Balanced Accuracy (mean±std) | AUROC (mean±std) |
|--------|---------------------|---------------------|-------------------------------|------------------|
| EEG+Gaze_concat | 0.8428±0.0761 | 0.7859±0.1035 | 0.7716±0.0943 | 0.8707±0.1100 |
| EEG_only | 0.8355±0.0778 | 0.7745±0.1018 | 0.7590±0.0931 | 0.8623±0.1071 |
| Gaze_only | 0.7772±0.1118 | 0.7018±0.1458 | 0.6994±0.1439 | 0.7586±0.1489 |
| Text+EEG+Gaze_concat | 0.8631±0.0703 | 0.8114±0.0935 | 0.7932±0.0901 | 0.8975±0.0924 |
| Text+EEG_concat | 0.8452±0.0822 | 0.7851±0.1072 | 0.7679±0.1022 | 0.8788±0.0987 |
| Text+Gaze_concat | 0.9249±0.0433 | 0.9078±0.0519 | 0.9249±0.0493 | 0.9633±0.0398 |
| Text_only | 0.9145±0.0375 | 0.8943±0.0473 | 0.9114±0.0423 | 0.9672±0.0299 |

### Protocol C

| Method | Accuracy (mean±std) | Macro-F1 (mean±std) | Balanced Accuracy (mean±std) | AUROC (mean±std) |
|--------|---------------------|---------------------|-------------------------------|------------------|
| EEG+Gaze_concat | 0.8715±0.0183 | 0.8281±0.0209 | 0.8102±0.0209 | 0.9156±0.0250 |
| EEG_only | 0.8543±0.0254 | 0.8010±0.0354 | 0.7820±0.0394 | 0.8959±0.0264 |
| Gaze_only | 0.7789±0.0333 | 0.6750±0.0197 | 0.6585±0.0142 | 0.7581±0.0289 |
| Text+EEG+Gaze_concat | 0.9157±0.0273 | 0.8964±0.0362 | 0.9112±0.0333 | 0.9609±0.0184 |
| Text+EEG_concat | 0.9117±0.0269 | 0.8927±0.0353 | 0.9111±0.0289 | 0.9535±0.0236 |
| Text+Gaze_concat | 0.9099±0.0290 | 0.8903±0.0382 | 0.9099±0.0385 | 0.9601±0.0196 |
| Text_only | 0.9024±0.0183 | 0.8827±0.0263 | 0.9055±0.0177 | 0.9636±0.0267 |

## 4. Analysis Questions

### Q1: Are EEG/Gaze/Text all correctly aligned?
- Total aligned samples across all subjects: 2549
- All modalities are correctly aligned: YES

### Q2: Is label consistency 100%?
- YES

### Q3: What are the baseline results under Protocol A and Protocol C?
- See Section 3 for detailed results.

### Q4: Is Text+EEG+Gaze_concat the strongest baseline?
- Protocol A strongest: Text+Gaze_concat
- Protocol C strongest: Text+EEG+Gaze_concat
- Text+EEG+Gaze_concat is strongest: NO

### Q5: Is Text_only strong, indicating potential text shortcut?
- Text_only accuracy (Protocol A): 0.9145
- Text_only accuracy (Protocol C): 0.9024
- Potential text shortcut: YES

### Q6: Is performance significantly lower under Protocol C?
- Text+EEG+Gaze_concat (Protocol A): 0.8631
- Text+EEG+Gaze_concat (Protocol C): 0.9157
- Difference: -0.0525
- Performance drop in Protocol C: NO

### Q7: Which baseline should be used as the strong lower bound for future SHARE-Net?
- Recommendation: Text+EEG+Gaze_concat with Protocol A accuracy of 0.8631
