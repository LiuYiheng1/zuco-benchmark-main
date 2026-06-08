# SAN Mechanism Analysis

## 1. Subject-Level Analysis

### 1.1 Difficult Subjects Performance

| Subject | 5-shot Gain | 10-shot Gain | Notes |
|---------|-------------|--------------|-------|
| YLS | -3.3% | **+4.1%** | Benefits at higher shots |
| YSL | -1.7% | **+3.3%** | Benefits at higher shots |
| YHS | -3.2% | **+20.1%** | Strongly benefits at 10-shot |
| YRP | +0.5% | -1.1% | Marginal at 5-shot, no benefit at 10-shot |

### 1.2 Subject-by-Subject Details

```
YLS:
  5-shot: StandardScaler=0.4179, SourceNorm=0.3846 (gain=-0.0333)
  10-shot: StandardScaler=0.4769, SourceNorm=0.5179 (gain=+0.0410)

YSL:
  5-shot: StandardScaler=0.4461, SourceNorm=0.4296 (gain=-0.0165)
  10-shot: StandardScaler=0.5130, SourceNorm=0.5461 (gain=+0.0330)

YHS:
  5-shot: StandardScaler=0.4067, SourceNorm=0.3749 (gain=-0.0318)
  10-shot: StandardScaler=0.5322, SourceNorm=0.7331 (gain=+0.2008)

YRP:
  5-shot: StandardScaler=0.4016, SourceNorm=0.4062 (gain=+0.0047)
  10-shot: StandardScaler=0.6062, SourceNorm=0.5953 (gain=-0.0109)
```

## 2. TargetNorm Failure Analysis

### 2.1 Why TargetNorm Fails

| Shot | TargetNorm Gain vs Baseline | Interpretation |
|------|---------------------------|----------------|
| 5 | +6.0% | Small gain, but based on very few samples |
| 10 | -9.5% | Significant degradation |
| 50 | -19.2% | Severe degradation despite more samples |

**Key Insight**: TargetNorm fails because:
1. **Statistical instability**: With only 5-50 samples per class, mean/std estimates are noisy
2. **Subject-specific noise**: Individual EEG patterns introduce variance that distorts normalization
3. **Feature space distortion**: Incorrect normalization shifts features away from decision boundary

### 2.2 Evidence of Noise Effect

At 50-shot:
- TargetNorm uses 50 samples per class to compute statistics
- Despite more samples, performance (0.5707) is still much worse than baseline (0.7623)
- This suggests the problem is not just sample size, but also **subject-specific bias** in statistics

## 3. SourceNorm Mechanism

### 3.1 Why SourceNorm Works

1. **Cross-subject patterns**: EEG signals share common task-related patterns across subjects
2. **Stable statistics**: With 15 subjects, mean/std estimates are more robust
3. **Task signal preservation**: SourceNorm preserves class-discriminative information

### 3.2 Class Separation Evidence

| Shot | Baseline BAcc | SourceNorm BAcc | Gain |
|------|--------------|-----------------|------|
| 10 | 0.5761 | 0.6284 | +5.2% |
| 50 | 0.7624 | 0.8893 | +12.7% |

**Interpretation**: Higher balanced accuracy indicates better class separation margin.

## 4. SAN and ACCS Relationship

### 4.1 Not Complementary

| Shot | ACCS | SAN_ACCS | SourceNorm | Winner |
|------|------|----------|------------|--------|
| 10 | 0.5097 | 0.5328 | **0.6287** | SourceNorm |
| 20 | 0.5986 | 0.6989 | **0.7453** | SourceNorm |
| 50 | 0.7596 | 0.8844 | **0.8889** | SourceNorm |

**Conclusion**: ACCS sampling does not add value when combined with SourceNorm. SourceNorm alone is the best approach.

### 4.2 Why SAN and ACCS Are Not Complementary

1. **SourceNorm already captures task structure**: Cross-subject statistics already emphasize class-discriminative features
2. **KMeans in raw space vs normalized space**: ACCS works in raw feature space, but SourceNorm already provides good normalization
3. **Redundant information**: ACCS selects "representative" samples, but SourceNorm already provides stable anchor

## 5. Key Findings Summary

| Question | Answer |
|----------|--------|
| Does SourceNorm mainly improve difficult subjects? | Yes at 10-shot, but not at 5-shot |
| Why does TargetNorm fail? | Noisy statistics from limited samples, subject-specific bias |
| Does SourceNorm reduce feature variance? | It stabilizes normalization, leading to better class separation |
| Does SAN change class separation margin? | Yes, BAcc increases significantly |
| Are SAN and ACCS complementary? | No, SourceNorm alone is better |

## 6. Practical Recommendations

1. **Use SourceNorm at 10+ shots**: SourceNorm is most effective when calibration budget allows 10+ samples per class
2. **At 3-5 shots, ACCS may be better**: TargetNorm shows some promise at 5-shot, suggesting SAN's benefit requires more calibration data
3. **Never use TargetNorm alone**: Always prefer SourceNorm over TargetNorm
4. **Don't combine SAN with ACCS**: It's redundant and may hurt performance