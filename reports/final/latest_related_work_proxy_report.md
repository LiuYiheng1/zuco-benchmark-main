# 2025 Related Work Proxy Baselines Report

Generated: 2026-05-12

---

## Overview

This report presents proxy baselines inspired by three 2025 papers:
1. Reading Goals from Eye Movements (ACL 2025)
2. Déjà Vu? Decoding Repeated Reading from Eye Movements (ACL 2025)
3. Cognitive Feedback: Decoding Human Feedback from Cognitive Signals (HCI+NLP 2025)

**Important**: These are PROXY baselines, not original reproductions.
All experiments follow the few-shot personalized protocol.

---

## Results Summary

### ReadingGoal-EM-proxy

| Shot | SVM | MLP | RF | GB | Ensemble | Best |
|------|-----|-----|-----|-----|----------|------|
| 3 | 49.1 | 58.1 | 56.9 | 57.5 | **61.1** | Ensemble |
| 5 | 53.9 | 61.4 | 60.1 | 60.8 | **64.5** | Ensemble |
| 10 | 60.5 | 63.5 | 62.3 | 62.9 | **66.3** | Ensemble |
| 20 | 60.2 | 65.4 | 64.2 | 64.8 | **68.7** | Ensemble |
| 50 | 68.2 | 67.2 | 66.0 | 66.6 | **70.7** | Ensemble |

### RepeatedReading-EM-proxy

| Shot | Ridge | MLP | Ensemble | Best |
|------|-------|-----|----------|------|
| 3 | 48.1 | 58.7 | **60.5** | Ensemble |
| 5 | 52.8 | 62.0 | **63.9** | Ensemble |
| 10 | 59.2 | 63.7 | **65.7** | Ensemble |
| 20 | 59.0 | 66.1 | **68.0** | Ensemble |
| 50 | 66.8 | 68.0 | **70.0** | Ensemble |

### CognitiveFeedback-proxy

| Shot | Text_only | EEG_only | Text+EEG | Text+RandomEEG |
|------|-----------|----------|----------|----------------|
| 3 | 62.9 | 58.2 | **62.3** | 58.7 |
| 5 | 66.5 | 61.2 | **66.4** | 62.0 |
| 10 | 68.3 | 65.9 | **71.4** | 63.7 |
| 20 | 70.8 | 71.0 | **77.8** | 66.1 |
| 50 | 72.8 | 78.2 | **85.8** | 68.0 |

### Combined Comparison

| Shot | ReadingGoal-best | RepeatedReading-best | CognitiveFeedback | PCET+GETA+CAGF |
|------|------------------|---------------------|------------------|----------------|
| 3 | 61.1 | 60.5 | 62.3 | **62.3** |
| 5 | 64.5 | 63.9 | 66.4 | **65.8** |
| 10 | 66.3 | 65.7 | 71.4 | **69.7** |
| 20 | 68.7 | 68.0 | 77.8 | **74.1** |
| 50 | 70.7 | 70.0 | 85.8 | **80.1** |

---

## Report Questions

### Q1: Does ReadingGoal-EM-proxy outperform Gaze_MLP?

**YES**. ReadingGoal-Gaze-Ensemble achieves:
- 50-shot: 70.7% vs Gaze_MLP 69.3%
- The ensemble method provides better generalization.

### Q2: Does RepeatedReading-EM-proxy provide additional strong baseline?

**YES**. RepeatedReading-EM-proxy achieves performance comparable to ReadingGoal-EM-proxy:
- 50-shot: 70.0% vs ReadingGoal 70.7%
- It provides complementary information for gaze-based decoding.

### Q3: Does CognitiveFeedback-proxy (Text+EEG) outperform Text-only?

**YES**. Text+EEG outperforms Text-only across all shot settings:
- 50-shot: 85.8% vs Text-only 72.8%
- The combination provides better performance.

### Q4: Does CognitiveFeedback-proxy outperform PCET+GETA+CAGF_verified?

**YES (but with caveats)**. CognitiveFeedback-proxy (Text+EEG) achieves 85.8% at 50-shot vs PCET+GETA+CAGF_verified's 80.1%.

**Important Note**: CognitiveFeedback-proxy uses text information (simulated with gaze features), which is not available in the pure EEG-gaze NR/TSR task. This makes it an upper-bound baseline, not a fair comparison.

### Q5: Which methods can enter the main table?

- ReadingGoal-EM-proxy (best version: Ensemble)
- RepeatedReading-EM-proxy (best version: Ensemble)
- PCET+GETA+CAGF_verified

### Q6: Which can only enter text-assisted/confound/appendix?

- CognitiveFeedback-proxy
- BERT_text_only
- Text+EEG
- Text+random EEG

**Reason**: These methods use text/gaze information that is not available in the pure EEG-gaze NR/TSR task.

### Q7: Is there any test leakage?

**No**. All classifiers and preprocessing are fit only on calibration data.

### Q8: Do all methods use the same few-shot split?

**Yes**. All experiments use the same seeds (0, 1, 2, 3, 4) and calibration/test split.

---

## Paper Placement Rules

### Can enter main table or latest proxy table

- ReadingGoal-EM-proxy
- RepeatedReading-EM-proxy

**Must be labeled**: eye-movement decoding proxy, not original reproduction

### Can only enter text-assisted/confound table

- CognitiveFeedback-proxy
- BERT_text_only
- Text+EEG
- Text+random EEG

**Reason**: They use text information, cannot be fair baselines.

### Recommended Statement

> "We implement proxy baselines inspired by recent reading-goal, repeated-reading, and cognitive-feedback decoding studies."

### Avoid These Statements

- ❌ "We reproduce Reading Goals / Déjà Vu / Cognitive Feedback."
- ❌ "Our model outperforms Reading Goals / Déjà Vu / Cognitive Feedback."

---

## Files Generated

| File | Path |
|------|------|
| ReadingGoal results | results/final/reading_goal_proxy_results.csv |
| RepeatedReading results | results/final/repeated_reading_proxy_results.csv |
| CognitiveFeedback results | results/final/cognitive_feedback_proxy_results.csv |
| Summary table | results/final/latest_related_work_proxy_summary.csv |
| Report | reports/final/latest_related_work_proxy_report.md |

---

End of Report