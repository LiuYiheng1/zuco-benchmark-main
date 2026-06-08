import pandas as pd
accs = pd.read_csv('d:/pycharmproject/zuco-benchmark-main/src/results/personalized/accs_active_calibration.csv')
print('n_cal_per_class values:', sorted(accs['n_cal_per_class'].unique()))
print('Methods:', accs['method'].unique())
print()
for n in [1, 3, 5, 10, 20, 50]:
    print(f'{n}-shot:')
    for m in accs['method'].unique():
        d = accs[(accs['n_cal_per_class'] == n) & (accs['method'] == m)]
        if len(d) > 0:
            acc = d['accuracy'].mean()
            print(f'  {m}: {acc:.4f}')
    print()