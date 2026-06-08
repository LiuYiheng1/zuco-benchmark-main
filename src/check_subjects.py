import os
import numpy as np

FEATURES_DIR = 'features'

eeg_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('_electrode_features_all.npy')]
gaze_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith('_sent_gaze_sacc.npy')]

print('EEG files:', len(eeg_files))
print('Gaze files:', len(gaze_files))

eeg_subjs = set(f.split('_')[0] for f in eeg_files)
gaze_subjs = set(f.split('_')[0] for f in gaze_files)
both = eeg_subjs & gaze_subjs
only_eeg = eeg_subjs - gaze_subjs
only_gaze = gaze_subjs - eeg_subjs

print(f'\nSubjects with BOTH EEG and Gaze: {len(both)}')
print(f'Subjects with only EEG: {len(only_eeg)} - {only_eeg}')
print(f'Subjects with only Gaze: {len(only_gaze)} - {only_gaze}')

x_subjs = sorted([s for s in both if s.startswith('X')])
y_subjs = sorted([s for s in both if s.startswith('Y')])
print(f'\nX subjects ({len(x_subjs)}): {x_subjs}')
print(f'Y subjects ({len(y_subjs)}): {y_subjs}')

code_y = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
print(f'\nY subjects in code ({len(code_y)}): {code_y}')
print(f'Missing from code: {set(y_subjs) - set(code_y)}')
print(f'Extra in code: {set(code_y) - set(y_subjs)}')

# AdaGTCN original uses 18 subjects: 12 train + 2 val + 4 test
# According to ZuCo, there are 26 subjects total with both EEG and gaze
# But the code is hardcoded to use only 16 Y subjects
print('\n' + '='*50)
print('PROBLEM DIAGNOSIS:')
print('='*50)
print(f'AdaGTCN paper uses: 18 subjects (12 train + 2 val + 4 test)')
print(f'Our code uses: 16 Y subjects (cannot do 12/2/4 split)')
print(f'Available Y subjects: {len(y_subjs)}')
print(f'Available X subjects: {len(x_subjs)}')
print(f'Total with both EEG+gaze: {len(both)}')
print('\nAdaGTCN paper Table 1 used 18 subjects from ZuCo 2.0')
print('Our features only have 16 Y subjects with both modalities')