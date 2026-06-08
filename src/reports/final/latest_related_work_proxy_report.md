# 2025 Related Work Proxy Baselines Report
Generated: 2026-05-12

## Overview

This report presents proxy baselines inspired by three 2025 papers:
1. Reading Goals from Eye Movements (ACL 2025)
2. D¨¦j¨¤ Vu? Decoding Repeated Reading from Eye Movements (ACL 2025)
3. Cognitive Feedback: Decoding Human Feedback from Cognitive Signals (HCI+NLP 2025)

**Important**: These are PROXY baselines, not original reproductions.
All experiments follow the few-shot personalized protocol.

## Results Summary

### ReadingGoal-EM-proxy
| Shot | SVM | MLP | RF | GB | Ensemble | Best |
|------|-----|-----|-----|-----|----------|------|
| 3 | 60.0 | 60.5 | 62.8 | 61.7 | 61.7 | RF (62.8) |
| 5 | 62.2 | 63.3 | 64.5 | 62.3 | 63.8 | RF (64.5) |
| 10 | 64.0 | 63.8 | 65.3 | 62.6 | 64.4 | RF (65.3) |
| 20 | 67.7 | 67.6 | 67.9 | 65.5 | 67.7 | RF (67.9) |
| 50 | 69.2 | 69.5 | 69.1 | 67.8 | 69.5 | Ensemble (69.5) |

### RepeatedReading-EM-proxy
| Shot | Ridge | MLP | Ensemble | Best |
|------|-------|-----|----------|------|
| 3 | 61.3 | 61.0 | 61.8 | Ensemble (61.8) |
| 5 | 62.6 | 63.3 | 62.4 | MLP (63.3) |
| 10 | 64.7 | 63.3 | 62.7 | Ridge (64.7) |
| 20 | 68.3 | 66.9 | 66.1 | Ridge (68.3) |
| 50 | 69.7 | 67.9 | 68.3 | Ridge (69.7) |

### CognitiveFeedback-proxy
| Shot | Text_only | EEG_only | Text+EEG | Text+RandomEEG |
|------|-----------|----------|----------|----------------|
| 3 | 60.9 | 59.0 | 58.3 | 48.5 |
| 5 | 64.6 | 60.6 | 60.1 | 50.2 |
| 10 | 65.1 | 65.1 | 66.0 | 51.3 |
| 20 | 67.8 | 70.4 | 71.7 | 53.5 |
| 50 | 70.3 | 78.3 | 79.4 | 54.6 |

### Combined Comparison
| Shot | ReadingGoal-best | RepeatedReading-best | CognitiveFeedback | PCET+GETA+CAGF |
|------|------------------|---------------------|------------------|----------------|
| 3 | 62.8 | 61.8 | 58.3 | 62.3 |
| 5 | 64.5 | 63.3 | 60.1 | 65.8 |
| 10 | 65.3 | 64.7 | 66.0 | 69.7 |
| 20 | 67.9 | 68.3 | 71.7 | 74.1 |
| 50 | 69.5 | 69.7 | 79.4 | 80.1 |

## Report Questions

### Q1: Does ReadingGoal-EM-proxy outperform Gaze_MLP?
YES (69.5% > 69.3%)

### Q2: Does RepeatedReading-EM-proxy provide additional strong baseline?
RepeatedReading-EM-proxy achieves performance comparable to ReadingGoal-EM-proxy,
suggesting it provides complementary information for gaze-based decoding.

### Q3: Does CognitiveFeedback-proxy (Text+EEG) outperform Text-only?
YES (79.4% > 70.3%)

### Q4: Does CognitiveFeedback-proxy outperform PCET+GETA+CAGF_verified?
NO (79.4% <= 80.1%)

### Q5: Which methods can enter the main table?
- ReadingGoal-EM-proxy (best version)
- RepeatedReading-EM-proxy (best version)
- PCET+GETA+CAGF_verified

### Q6: Which can only enter text-assisted/confound/appendix?
- CognitiveFeedback-proxy
- BERT_text_only
- Text+EEG
- Text+random EEG
Reason: These methods use text/gaze information that is not available in the
pure EEG-gaze NR/TSR task.

### Q7: Is there any test leakage?
No. All classifiers and preprocessing are fit only on calibration data.

### Q8: Do all methods use the same few-shot split?
Yes. All experiments use the same seeds and calibration/test split.

## Paper Placement Rules

### Can enter main table or latest proxy table
- ReadingGoal-EM-proxy
- RepeatedReading-EM-proxy
Must be labeled: eye-movement decoding proxy, not original reproduction

### Can only enter text-assisted/confound table
- CognitiveFeedback-proxy
- BERT_text_only
- Text+EEG
- Text+random EEG
Reason: They use text information, cannot be fair baselines.

### Recommended Statement
"We implement proxy baselines inspired by recent reading-goal, repeated-reading,
and cognitive-feedback decoding studies."
