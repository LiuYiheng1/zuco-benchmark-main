import pandas as pd

df_main = pd.read_csv('results/final/multimodal_final_main_results.csv')

methods = [
    ('EEG_SVM', 'EEG_SVM_acc', 'EEG_SVM_f1', 'EEG_SVM_bacc', 'EEG_SVM_auroc'),
    ('Gaze_SVM', 'Gaze_SVM_acc', 'Gaze_SVM_f1', 'Gaze_SVM_bacc', 'Gaze_SVM_auroc'),
    ('EEG_MLP', 'EEG_MLP_acc', 'EEG_MLP_f1', 'EEG_MLP_bacc', 'EEG_MLP_auroc'),
    ('Gaze_MLP', 'Gaze_MLP_acc', 'Gaze_MLP_f1', 'Gaze_MLP_bacc', 'Gaze_MLP_auroc'),
    ('Raw EEG-Gaze MLP Fusion', 'EEG+Gaze_concat_acc', 'EEG+Gaze_concat_f1', 'EEG+Gaze_concat_bacc', 'EEG+Gaze_concat_auroc'),
    ('Ridge StaticAvg', 'Static_EEG_Gaze_avg_acc', 'Static_EEG_Gaze_avg_f1', 'Static_EEG_Gaze_avg_bacc', 'Static_EEG_Gaze_avg_auroc'),
    ('PCET_only', 'PCET_only_acc', 'PCET_only_f1', 'PCET_only_bacc', 'PCET_only_auroc'),
    ('GETA_only', 'GETA_only_acc', 'GETA_only_f1', 'GETA_only_bacc', 'GETA_only_auroc'),
    ('PCET+GETA_concat', 'PCET+GETA_concat_acc', 'PCET+GETA_concat_f1', 'PCET+GETA_concat_bacc', 'PCET+GETA_concat_auroc'),
    ('PCET+GETA_static_avg', 'PCET+GETA_static_avg_acc', 'PCET+GETA_static_avg_f1', 'PCET+GETA_static_avg_bacc', 'PCET+GETA_static_avg_auroc'),
    ('PCET+GETA+CAGF_verified', 'PCET+GETA+CAGF_acc', 'PCET+GETA+CAGF_f1', 'PCET+GETA+CAGF_bacc', 'PCET+GETA+CAGF_auroc')
]

print('=' * 120)
print('FEW-SHOT MAIN COMPARISON - COMPLETE RESULTS')
print('=' * 120)

best_summary = []

for n_cal in [3, 5, 10, 20, 50]:
    print(f'\n--- {n_cal}-SHOT ---')
    print('{:<35} {:<10} {:<10} {:<10} {:<10}'.format('Method', 'Accuracy', 'Macro-F1', 'BAcc', 'AUROC'))
    print('-' * 80)
    
    best_acc = 0
    best_method = ''
    
    for name, acc_col, f1_col, bacc_col, auroc_col in methods:
        sub = df_main[df_main['n_cal'] == n_cal]
        if len(sub) == 0 or acc_col not in sub.columns:
            continue
        
        acc = sub[acc_col].mean() * 100
        f1_val = sub[f1_col].mean() * 100
        bacc_val = sub[bacc_col].mean() * 100
        auroc_val = sub[auroc_col].mean() * 100
        
        if name == 'PCET+GETA+CAGF_verified':
            print('{:<35} {:>8.1f}%    {:>8.1f}%    {:>8.1f}%    {:>8.1f}%  *'.format(name, acc, f1_val, bacc_val, auroc_val))
        else:
            print('{:<35} {:>8.1f}%    {:>8.1f}%    {:>8.1f}%    {:>8.1f}%'.format(name, acc, f1_val, bacc_val, auroc_val))
        
        if acc > best_acc:
            best_acc = acc
            best_method = name
    
    best_summary.append((n_cal, best_method, best_acc))
    print(f'\nBest method at {n_cal}-shot: {best_method} ({best_acc:.1f}%)')

print('\n' + '=' * 120)
print('SUMMARY ANALYSIS')
print('=' * 120)

print('\n1. Best baseline per shot:')
for n_cal, method, acc in best_summary:
    print(f'   {n_cal}-shot: {method} ({acc:.1f}%)')

print('\n2. Is PCET+GETA+CAGF_verified the best at every shot?')
is_all_best = all(method == 'PCET+GETA+CAGF_verified' for _, method, _ in best_summary)
if is_all_best:
    print('   YES - PCET+GETA+CAGF_verified is the best at all shot settings')
else:
    print('   NO - PCET+GETA+CAGF_verified is NOT the best at all shots')
    print('\n3. Which shots were exceeded by other baselines:')
    for n_cal, method, acc in best_summary:
        if method != 'PCET+GETA+CAGF_verified':
            cagf_acc = df_main[df_main['n_cal'] == n_cal]['PCET+GETA+CAGF_acc'].mean() * 100
            print(f'   {n_cal}-shot: {method} ({acc:.1f}%) > PCET+GETA+CAGF_verified ({cagf_acc:.1f}%)')

print('\n4. Recommended methods for main paper table:')
print('   - EEG_SVM (EEG-only baseline)')
print('   - Gaze_SVM (gaze-only baseline)')
print('   - EEG_MLP (MLP baseline)')
print('   - Gaze_MLP (MLP baseline)')
print('   - Raw EEG-Gaze MLP Fusion (simple fusion baseline)')
print('   - Ridge StaticAvg (probability averaging baseline)')
print('   - PCET_only (ablation component)')
print('   - GETA_only (ablation component)')
print('   - PCET+GETA_concat (ablation component)')
print('   - PCET+GETA_static_avg (ablation component)')
print('   - PCET+GETA+CAGF_verified (OUR METHOD - HIGHLIGHTED)')