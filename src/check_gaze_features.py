import numpy as np
import os

FEATURES_DIR = 'features'
subject = 'YAC'

# Load EEG
path_eeg = os.path.join(FEATURES_DIR, f'{subject}_electrode_features_all.npy')
data_eeg = np.load(path_eeg, allow_pickle=True).item()
first_key = list(data_eeg.keys())[0]
first_val_eeg = data_eeg[first_key]
print(f'EEG feature vector: {len(first_val_eeg)} values, last is label: {first_val_eeg[-1]}')

# Load gaze
path_gaze = os.path.join(FEATURES_DIR, f'{subject}_sent_gaze_sacc.npy')
data_gaze = np.load(path_gaze, allow_pickle=True).item()
first_key_gaze = list(data_gaze.keys())[0]
first_val_gaze = data_gaze[first_key_gaze]
print(f'Gaze feature vector: {len(first_val_gaze)} values, last is label: {first_val_gaze[-1]}')

# Check alignment
print('\nSample keys:')
print('EEG:', list(data_eeg.keys())[:3])
print('Gaze:', list(data_gaze.keys())[:3])

# Check if they have same trial IDs
eeg_trials = set('_'.join(k.split('_')[:3]) for k in data_eeg.keys())
gaze_trials = set('_'.join(k.split('_')[:3]) for k in data_gaze.keys())
print(f'\nEEG trials: {len(eeg_trials)}')
print(f'Gaze trials: {len(gaze_trials)}')
print(f'Common trials: {len(eeg_trials & gaze_trials)}')

# Check sample counts per subject
for subj in ['YAC', 'YAG', 'YAK']:
    path = os.path.join(FEATURES_DIR, f'{subj}_electrode_features_all.npy')
    data = np.load(path, allow_pickle=True).item()
    print(f'{subj}: {len(data)} trials')

# 420 features: could be 7 bands x 60 channels or similar
# Let's check if it's standard 10-20 system with 140 channels x 3 features
print('\n420 = 7 x 60, 6 x 70, 5 x 84, 3 x 140, 140 x 3...')