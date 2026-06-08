import numpy as np
gaze = np.load('features/YAC_sent_gaze_sacc.npy', allow_pickle=True).item()
for k, v in list(gaze.items())[:3]:
    print(f'key={k}')
    arr = np.array(v)
    print(f'  values type={type(v)}, shape={arr.shape}, dtype={arr.dtype}')
    print(f'  values content={arr}')
    print()