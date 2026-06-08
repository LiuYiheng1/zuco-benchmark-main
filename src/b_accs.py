"""B-ACCS: Balanced-ACC with Pseudo-label Teacher"""
import os
os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.cluster import KMeans

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
os.makedirs(RESULTS_DIR, exist_ok=True)

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return None, None
    data = np.load(path, allow_pickle=True).item()
    X, y = [], []
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 2 and parts[1] == "NR":
            label = 1
        elif len(parts) >= 2 and parts[1] == "TSR":
            label = 0
        else:
            continue
        features = np.array(values[:-1], dtype=np.float64)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def kmeans_sampling_with_pseudo(X_pool, pseudo_labels, n_select):
    class_0_indices = np.where(pseudo_labels == 0)[0]
    class_1_indices = np.where(pseudo_labels == 1)[0]

    selected = []
    n_per_class = n_select // 2

    for c_indices, c_label in [(class_0_indices, 0), (class_1_indices, 1)]:
        if len(c_indices) == 0:
            continue
        X_c = X_pool[c_indices]
        k = min(n_per_class, len(c_indices))
        scaler = StandardScaler()
        X_c_s = scaler.fit_transform(X_c)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        kmeans.fit(X_c_s)
        for c in range(k):
            idx_in_cluster = np.where(kmeans.labels_ == c)[0]
            if len(idx_in_cluster) > 0:
                centroid = kmeans.cluster_centers_[c]
                dists = np.linalg.norm(X_c_s[idx_in_cluster] - centroid, axis=1)
                closest_local = idx_in_cluster[np.argmin(dists)]
                closest_global = c_indices[closest_local]
                selected.append(closest_global)

    while len(selected) < n_select and len(selected) < len(X_pool):
        remaining = [i for i in range(len(X_pool)) if i not in selected]
        if not remaining:
            break
        selected.append(remaining[0])

    return np.array(selected[:n_select])

def kmeans_sampling_label_free(X_pool, n_select):
    n_pool = len(X_pool)
    k = min(n_select, n_pool)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_pool)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    kmeans.fit(X_s)
    selected = []
    for c in range(k):
        idx = np.where(kmeans.labels_ == c)[0]
        if len(idx) > 0:
            centroid = kmeans.cluster_centers_[c]
            dists = np.linalg.norm(X_s[idx] - centroid, axis=1)
            selected.append(idx[np.argmin(dists)])
    while len(selected) < n_select and len(selected) < n_pool:
        for c in range(k):
            idx = np.where(kmeans.labels_ == c)[0]
            for i in idx:
                if i not in selected:
                    selected.append(i)
                    if len(selected) >= n_select:
                        break
            if len(selected) >= n_select:
                break
    return np.array(selected[:n_select])

def balanced_random_sampling(y_pool, n_per_class):
    class_0_idx = np.where(y_pool == 0)[0]
    class_1_idx = np.where(y_pool == 1)[0]
    np.random.shuffle(class_0_idx)
    np.random.shuffle(class_1_idx)
    n0 = min(n_per_class, len(class_0_idx))
    n1 = min(n_per_class, len(class_1_idx))
    selected = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])
    np.random.shuffle(selected)
    return selected

def train_teacher(X_train, y_train):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y_train)
    return clf, scaler

def train_and_evaluate(X_cal, y_cal, X_test, y_test):
    if len(np.unique(y_cal)) < 2:
        return 0.0, 0.0, 0.0, 0.5
    scaler = StandardScaler()
    X_cal_s = scaler.fit_transform(X_cal)
    X_test_s = scaler.transform(X_test)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_cal_s, y_cal)
    probs = clf.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, probs)
    except:
        auroc = 0.5
    return acc, f1, bacc, auroc

print('B-ACCS Experiments', flush=True)
print('='*60, flush=True)

results = []
shot_settings = [3, 5, 10, 20, 50]
tau_values = [0.6, 0.7, 0.8]
seeds = [0, 1, 2, 3, 4]

for seed in seeds:
    print(f'\nSeed {seed}:', flush=True)
    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test_orig, y_test_orig = load_eeg_data(held_out)
        if len(X_train_all) == 0 or X_test_orig is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        teacher, teacher_scaler = train_teacher(X_train_all, y_train_all)

        n_samples = len(y_test_orig)
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        test_size = n_samples // 3
        test_indices = indices[:test_size]
        cal_pool_indices = indices[test_size:]

        X_test = X_test_orig[test_indices]
        y_test = y_test_orig[test_indices]
        X_cal_pool = X_test_orig[cal_pool_indices]
        y_cal_pool = y_test_orig[cal_pool_indices]

        X_pool_s = teacher_scaler.transform(X_cal_pool)
        pool_probs = teacher.predict_proba(X_pool_s)[:, 1]
        pseudo_labels = (pool_probs >= 0.5).astype(int)
        pseudo_acc = accuracy_score(y_cal_pool, pseudo_labels)

        print(f'  {held_out}', end='', flush=True)

        for n_cal in shot_settings:
            if n_cal * 2 > len(cal_pool_indices):
                continue

            cal_idx = balanced_random_sampling(y_cal_pool, n_cal)
            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            acc, f1, bacc, auroc = train_and_evaluate(X_cal, y_cal, X_test, y_test)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'Random', 'tau': 0, 'pseudo_acc': np.nan,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

            cal_idx_accs = kmeans_sampling_label_free(X_cal_pool, n_cal * 2)
            X_cal_accs = X_cal_pool[cal_idx_accs]
            y_cal_accs = y_cal_pool[cal_idx_accs]
            if len(np.unique(y_cal_accs)) < 2:
                continue
            acc, f1, bacc, auroc = train_and_evaluate(X_cal_accs, y_cal_accs, X_test, y_test)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'ACCS', 'tau': 0, 'pseudo_acc': np.nan,
                'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
            })

            for tau in tau_values:
                high_conf_mask = (pool_probs >= tau) | (pool_probs <= (1 - tau))
                X_high_conf = X_cal_pool[high_conf_mask]
                pseudo_high_conf = pseudo_labels[high_conf_mask]

                if len(X_high_conf) < n_cal * 2:
                    acc, f1, bacc, auroc = 0.0, 0.0, 0.0, 0.5
                else:
                    cal_idx_b = kmeans_sampling_with_pseudo(X_high_conf, pseudo_high_conf, n_cal * 2)
                    X_cal_b = X_high_conf[cal_idx_b]
                    y_cal_b = pseudo_high_conf[cal_idx_b]

                    if len(np.unique(y_cal_b)) < 2:
                        acc, f1, bacc, auroc = 0.0, 0.0, 0.0, 0.5
                    else:
                        acc, f1, bacc, auroc = train_and_evaluate(X_cal_b, y_cal_b, X_test, y_test)

                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                    'method': f'B-ACCS_tau{tau}', 'tau': tau, 'pseudo_acc': pseudo_acc,
                    'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
                })

        print(f'.', end='', flush=True)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR + '/b_accs_results.csv', index=False)

print('', flush=True)
print('\n' + '='*60, flush=True)
print('B-ACCS Results Summary', flush=True)
print('='*60, flush=True)

for n_cal in shot_settings:
    print(f'\n{n_cal}-shot per class:', flush=True)
    random_acc = df[(df['method'] == 'Random') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    accs_acc = df[(df['method'] == 'ACCS') & (df['n_cal'] == n_cal)]['accuracy'].mean()
    print(f'  Random: {random_acc:.4f}', flush=True)
    print(f'  ACCS: {accs_acc:.4f} (gap={accs_acc-random_acc:+.4f})', flush=True)
    for tau in tau_values:
        b_accs = df[(df['method'] == f'B-ACCS_tau{tau}') & (df['n_cal'] == n_cal)]['accuracy'].mean()
        print(f'  B-ACCS (tau={tau}): {b_accs:.4f} (gap vs ACCS={b_accs-accs_acc:+.4f})', flush=True)

avg_3_5_10 = df[df['n_cal'].isin([3, 5, 10])].groupby(['method'])['accuracy'].mean()
print('\n3/5/10-shot average:', flush=True)
for method in ['Random', 'ACCS'] + [f'B-ACCS_tau{tau}' for tau in tau_values]:
    if method in avg_3_5_10.index:
        print(f'  {method}: {avg_3_5_10[method]:.4f}', flush=True)

print('\nDone!', flush=True)