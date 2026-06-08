import os, sys
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print('SAGE Ultra Fast Test', flush=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
FEATURES_DIR = 'features'

def load_eeg(subject):
    path = os.path.join(FEATURES_DIR, f'{subject}_electrode_features_all.npy')
    if not os.path.exists(path): return None, None
    data = np.load(path, allow_pickle=True).item()
    X, y = [], []
    for k, v in data.items():
        p = k.split('_')
        if len(p) >= 2 and p[1] == 'NR': lbl = 1
        elif len(p) >= 2 and p[1] == 'TSR': lbl = 0
        else: continue
        X.append(np.array(v[:-1], dtype=np.float64))
        y.append(lbl)
    return np.array(X), np.array(y)

def stats(X, y, c):
    Xc = X[y==c]
    if len(Xc) == 0: return None, None
    return np.mean(Xc, axis=0), np.std(Xc, axis=0) + 1e-8

def sr_pred(Xc, yc, Xt, ms0, ss0, ms1, ss1, a=0.25):
    mc0, sc0 = stats(Xc, yc, 0); mc1, sc1 = stats(Xc, yc, 1)
    if mc0 is None: return np.zeros(len(Xt))
    mb0, mb1 = a*mc0+(1-a)*ms0, a*mc1+(1-a)*ms1
    sb0, sb1 = a*sc0+(1-a)*ss0, a*sc1+(1-a)*ss1
    cov0 = np.linalg.inv(np.diag(sb0**2+1e-8))
    cov1 = np.linalg.inv(np.diag(sb1**2+1e-8))
    s0 = np.array([np.sqrt(np.dot(np.dot(x-mb0, cov0), x-mb0)) for x in Xt])
    s1 = np.array([np.sqrt(np.dot(np.dot(x-mb1, cov1), x-mb1)) for x in Xt])
    return (s1 < s0).astype(int), s0, s1

def lr_pred(Xc, yc, Xt):
    if len(np.unique(yc)) < 2: return np.zeros(len(Xt)), np.zeros(len(Xt)), np.zeros(len(Xt))
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(sc.fit_transform(Xc), yc)
    probs = clf.predict_proba(Xt)
    return clf.predict(sc.transform(Xt)), probs[:, 0], probs[:, 1]

def sample(y_pool, n):
    i0 = np.where(y_pool==0)[0]; i1 = np.where(y_pool==1)[0]
    np.random.shuffle(i0); np.random.shuffle(i1)
    sel = np.concatenate([i0[:min(n,len(i0))], i1[:min(n,len(i1))]])
    np.random.shuffle(sel); return sel

cache = {s: load_eeg(s) for s in Y_SUBJECTS}
print('Data loaded', flush=True)

results = []
SAGE_LAMBDA = {3: 0.9, 5: 0.9, 10: 0.7, 20: 0.5, 50: 0.1}

for seed in [0,1,2,3,4]:
    print(f'Seed {seed}', end=' ', flush=True)
    np.random.seed(seed)
    for ho in Y_SUBJECTS:
        Xt, yt = cache[ho]
        if Xt is None: continue
        Xtr = []; ytr = []
        for s in Y_SUBJECTS:
            if s != ho and cache[s][0] is not None:
                Xtr.append(cache[s][0]); ytr.append(cache[s][1])
        if not Xtr: continue
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        ms0, ss0 = stats(Xtr, ytr, 0); ms1, ss1 = stats(Xtr, ytr, 1)
        idx = np.random.permutation(len(yt))
        ts = len(yt)//3
        X_test, y_test = Xt[idx[:ts]], yt[idx[:ts]]
        Xcp, ycp = Xt[idx[ts:]], yt[idx[ts:]]
        for n_cal in [3,5,10,20,50]:
            if n_cal*2 > len(Xcp): continue
            ci = sample(ycp, n_cal)
            Xc, yc = Xcp[ci], ycp[ci]
            if len(np.unique(yc)) < 2: continue
            _, lr0, lr1 = lr_pred(Xc, yc, X_test)
            _, sr0, sr1 = sr_pred(Xc, yc, X_test, ms0, ss0, ms1, ss1, 0.25)
            for m, s0, s1 in [('LR', lr0, lr1), ('SR-GC', sr0, sr1)]:
                diff = s1 - s0
                preds = (diff > 0).astype(int)
                results.append({'seed':seed,'subject':ho,'n_cal':n_cal,'method':m,'accuracy':accuracy_score(y_test, preds)})
            lam = SAGE_LAMBDA[n_cal]
            fused_diff = lam * (sr1 - sr0) + (1 - lam) * (lr1 - lr0)
            preds_fused = (fused_diff > 0).astype(int)
            results.append({'seed':seed,'subject':ho,'n_cal':n_cal,'method':'SAGE_rule','accuracy':accuracy_score(y_test, preds_fused)})
        print('.', end='', flush=True)
    print('')

df = pd.DataFrame(results)
os.makedirs('results/final', exist_ok=True)
df.to_csv('results/final/sage_results.csv', index=False)
print('Saved!', flush=True)

for n in [3,5,10,20,50]:
    lr_a = df[(df.method=='LR') & (df.n_cal==n)].accuracy.mean()
    sr_a = df[(df.method=='SR-GC') & (df.n_cal==n)].accuracy.mean()
    sage_a = df[(df.method=='SAGE_rule') & (df.n_cal==n)].accuracy.mean()
    print(f'{n}-shot: LR={lr_a:.4f} SR-GC={sr_a:.4f} (gap={sr_a-lr_a:+.4f}) SAGE_rule={sage_a:.4f}')
print('Done!')