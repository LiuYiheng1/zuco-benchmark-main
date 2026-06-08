import pandas as pd

df = pd.read_csv('results/domain_generalization/tcd_results.csv')

print('TCD Results Analysis')
print('='*60)

for model in ['SIED', 'TCD_full', 'TCD_full_plus_SupCon']:
    data = df[df['model'] == model]
    if len(data) > 0:
        acc = data['accuracy'].mean()
        std = data['accuracy'].std()
        f1 = data['macro_f1'].mean()
        bacc = data['balanced_accuracy'].mean()
        print(model + ':')
        print('  acc=' + str(round(acc, 4)) + '+-' + str(round(std, 4)))
        print('  f1=' + str(round(f1, 4)))
        print('  bacc=' + str(round(bacc, 4)))
        print()

# Compare SIED vs TCD_full
sied_data = df[df['model'] == 'SIED']
tcd_data = df[df['model'] == 'TCD_full']

sied_acc = sied_data['accuracy'].mean()
tcd_acc = tcd_data['accuracy'].mean()
gap = tcd_acc - sied_acc

print('Comparison:')
print('  SIED: ' + str(round(sied_acc, 4)))
print('  TCD_full: ' + str(round(tcd_acc, 4)))
print('  Gap: ' + str(round(gap, 4)))
print()
print('Target: SIED + 1.5% = ' + str(round(sied_acc + 0.015, 4)))
print()

if gap >= 0.015:
    print('SUCCESS: TCD exceeds SIED by >= 1.5%')
elif gap >= 0.005:
    print('MARGINAL: TCD exceeds SIED by < 1.5%')
else:
    print('FAILED: TCD does NOT exceed SIED by 1.5%')