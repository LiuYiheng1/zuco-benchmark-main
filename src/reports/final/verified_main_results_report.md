# Verified Main Results Report
## PCET+GETA+CAGF_verified Code Path

```
PCET: VerifiedPCET class
  - PCA fit on calibration data only (per class)
  - AbsError computed: |x - x_hat|
  - Input: [x ; abs_error], dimension doubled

GETA: VerifiedGETA class
  - Gaze MLP to predict gaze probability
  - Entropy computed from gaze predictions
  - Confidence computed from gaze predictions
  - Attention = entropy*0.01 + confidence
  - EEG features reweighted by attention

CAGF: VerifiedCAGF class
  - Input from PCET: z_pcet
  - Input from GETA: z_geta
  - alpha = sigmoid(z_pcet[:,0] - z_geta[:,0])
  - z_fused = alpha*z_pcet + (1-alpha)*z_geta
  - Final MLP classifier
```

## Verification Checklist

| Check | PCET | GETA | CAGF |
|-------|------|------|------|
| PCA on calibration only | YES | N/A | N/A |
| AbsError computed | YES | N/A | N/A |
| Input dim doubled | YES | N/A | N/A |
| Gaze entropy computed | N/A | YES | N/A |
| Gaze confidence computed | N/A | YES | N/A |
| EEG attention reweight | N/A | YES | N/A |
| Input from PCET | N/A | N/A | YES |
| Input from GETA | N/A | N/A | YES |
| alpha = sigmoid(diff) | N/A | N/A | YES |
| No confidence features | N/A | N/A | YES |
| No abs_diff/hadamard | N/A | N/A | YES |
| No test leakage | YES | YES | YES |
