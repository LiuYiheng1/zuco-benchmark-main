import numpy as np
import os

FEATURES_DIR = 'features'
subject = 'YAC'

path = os.path.join(FEATURES_DIR, f'{subject}_electrode_features_all.npy')
data = np.load(path, allow_pickle=True).item()
print('EEG data keys (first 5):', list(data.keys())[:5])

first_key = list(data.keys())[0]
first_val = data[first_key]
print(f'Feature vector length: {len(first_val)}')
print(f'Last value (likely label): {first_val[-1]}')

n_features = len(first_val) - 1
print(f'Number of features: {n_features}')

for n_channels in [16, 32, 64, 128]:
    if n_features % n_channels == 0:
        features_per_channel = n_features // n_channels
        print(f'Possible: {n_channels} channels x {features_per_channel} features per channel')