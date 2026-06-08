# EEG-Gaze Multimodal Framework Pilot Results


## Main Accuracy Comparison


| Method | 3-shot | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|--------|---------|---------|--------|
| EEG_SVM | 43.5¡À8.7 | 41.6¡À10.6 | 57.6¡À15.4 | 59.6¡À18.2 | 76.2¡À6.7 |
| Gaze_SVM | 50.1¡À14.7 | 55.0¡À16.2 | 61.7¡À15.3 | 61.4¡À17.2 | 69.6¡À11.8 |
| EEG_MLP | 58.2¡À8.1 | 61.2¡À7.6 | 65.9¡À7.2 | 71.0¡À6.8 | 78.2¡À6.2 |
| Gaze_MLP | 59.9¡À11.8 | 63.3¡À12.7 | 65.0¡À12.3 | 67.4¡À12.2 | 69.3¡À12.3 |
| EEG+Gaze_concat | 57.7¡À7.9 | 61.5¡À7.3 | 66.1¡À7.2 | 72.0¡À7.0 | 79.4¡À6.1 |
| Static_EEG_Gaze_avg | 46.5¡À14.0 | 49.3¡À15.8 | 64.3¡À15.1 | 65.7¡À16.5 | 79.7¡À7.0 |
| PCET_only | 58.7¡À8.3 | 61.0¡À7.8 | 65.1¡À7.8 | 70.0¡À6.7 | 78.2¡À8.2 |
| GETA_only | 58.2¡À8.1 | 61.2¡À7.4 | 65.9¡À7.1 | 71.0¡À6.6 | 78.2¡À6.3 |
| PCET+GETA_concat | 58.0¡À8.2 | 60.6¡À7.5 | 64.3¡À7.1 | 69.6¡À6.4 | 77.3¡À7.6 |
| PCET+GETA_static_avg | 59.0¡À8.2 | 61.6¡À7.5 | 66.7¡À7.5 | 71.4¡À6.8 | 79.1¡À6.7 |
| PCET+GETA+CAGF | 62.3¡À9.3 | 65.8¡À9.6 | 69.7¡À9.5 | 74.1¡À8.6 | 80.1¡À7.2 |

## Success Criteria Check


### GETA Success (GETA > Gaze_MLP)

- 3-shot: GETA=58.18%, Gaze_MLP=59.92%, diff=-1.74% [FAIL]
- 5-shot: GETA=61.22%, Gaze_MLP=63.26%, diff=-2.04% [FAIL]
- 10-shot: GETA=65.91%, Gaze_MLP=65.05%, diff=0.87% [FAIL]
- 20-shot: GETA=71.02%, Gaze_MLP=67.36%, diff=3.66% [PASS]
- 50-shot: GETA=78.18%, Gaze_MLP=69.31%, diff=8.87% [PASS]

### CAGF Success (CAGF > concat AND CAGF > static_avg)

- 3-shot: CAGF=62.27%, concat=58.04%, static=59.04% [PASS,PASS]
- 5-shot: CAGF=65.84%, concat=60.62%, static=61.63% [PASS,PASS]
- 10-shot: CAGF=69.68%, concat=64.33%, static=66.72% [PASS,PASS]
- 20-shot: CAGF=74.06%, concat=69.64%, static=71.37% [PASS,PASS]
- 50-shot: CAGF=80.11%, concat=77.28%, static=79.06% [PASS,PASS]

### Full Framework Success

- 3-shot: Full=62.27%, PCET=58.75%, GETA=58.18%, concat=57.67% [PASS,PASS,PASS]
- 5-shot: Full=65.84%, PCET=60.98%, GETA=61.22%, concat=61.52% [PASS,PASS,PASS]
- 10-shot: Full=69.68%, PCET=65.08%, GETA=65.91%, concat=66.10% [PASS,PASS,PASS]
- 20-shot: Full=74.06%, PCET=69.99%, GETA=71.02%, concat=72.01% [PASS,PASS,PASS]
- 50-shot: Full=80.11%, PCET=78.16%, GETA=78.18%, concat=79.40% [PASS,PASS,PASS]