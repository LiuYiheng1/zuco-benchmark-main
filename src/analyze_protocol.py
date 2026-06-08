import pandas as pd
import numpy as np

accs = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/personalized/accs_active_calibration.csv')

print('='*70)
print('PROTOCOL ANALYSIS')
print('='*70)

# Check Protocol A vs Protocol B
print('\nProtocol A (Label-free) - should NOT use labels:')
proto_a = accs[accs['protocol'] == 'ProtocolA']
print(f'  Data count: {len(proto_a)}')
for n in [3, 5, 10]:
    d = proto_a[proto_a['n_cal_per_class'] == n]
    if len(d) > 0:
        for m in d['method'].unique():
            print(f'  {n}-shot {m}: {d[d["method"]==m]["accuracy"].mean():.4f}')

print('\nProtocol B (Balanced) - uses labels for balanced sampling:')
proto_b = accs[accs['protocol'] == 'ProtocolB']
print(f'  Data count: {len(proto_b)}')
for n in [3, 5, 10]:
    d = proto_b[proto_b['n_cal_per_class'] == n]
    if len(d) > 0:
        for m in d['method'].unique():
            print(f'  {n}-shot {m}: {d[d["method"]==m]["accuracy"].mean():.4f}')

# The key question: Is KMeans_balanced using labels?
print('\n' + '='*70)
print('KEY QUESTION: Does KMeans_balanced use test labels?')
print('='*70)

print('''
Answer:
- KMeans_balanced = Protocol B
- In Protocol B, we FIRST use true labels to ensure balanced sampling
- Then we do KMeans within each class
- This means the "calibration set selection" IS using true labels

Therefore:
- KMeans_balanced is NOT truly label-free
- It can only be used in controlled experiments, NOT real deployment
- For real deployment, we must use Protocol A (label-free)

But Protocol A has NO DATA!
''')

# Check if there's any KMeans_label_free data
print('\n' + '='*70)
print('KMeans_label_free at each shot:')
print('='*70)
for n in [1, 3, 5, 10, 20, 50]:
    d = accs[(accs['n_cal_per_class'] == n) & (accs['method'] == 'KMeans_label_free')]
    if len(d) > 0:
        print(f'{n}-shot: {d["accuracy"].mean():.4f}')

print('\nKMeans_balanced at each shot:')
for n in [1, 3, 5, 10, 20, 50]:
    d = accs[(accs['n_cal_per_class'] == n) & (accs['method'] == 'KMeans_balanced')]
    if len(d) > 0:
        print(f'{n}-shot: {d["accuracy"].mean():.4f}')