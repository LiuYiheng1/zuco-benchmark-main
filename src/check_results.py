import pandas as pd
import os

results_dir = "results/eeg_adaptation"
files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
for f in files:
    path = os.path.join(results_dir, f)
    df = pd.read_csv(path)
    print(f"{f}: {len(df)} rows")