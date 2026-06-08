"""
EEG Adaptation - Step 2: CORAL
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

FEATURES_DIR = "features"
RESULTS_DIR = "results/eeg_adaptation"
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

def coral_align(X_source, X_target):
    cov_src = np.cov(X_source.T)
    cov_tgt = np.cov(X_target.T)
    eigval_src, eigvec_src = np.linalg.eigh(cov_src)
    order_src = eigval_src.argsort()[::-1]
    eigval_src = eigval_src[order_src]
    eigvec_src = eigvec_src[:, order_src]
    d_src = np.diag(1.0 / np.sqrt(np.maximum(eigval_src, 1e-8)))
    whitening_src = eigvec_src @ d_src @ eigvec_src.T
    eigval_tgt, eigvec_tgt = np.linalg.eigh(cov_tgt)
    order_tgt = eigval_tgt.argsort()[::-1]
    eigval_tgt = eigval_tgt[order_tgt]
    eigvec_tgt = eigvec_tgt[:, order_tgt]
    d_tgt = np.diag(1.0 / np.sqrt(np.maximum(eigval_tgt, 1e-8)))
    whitening_tgt = eigvec_tgt @ d_tgt @ eigvec_tgt.T
    X_src_aligned = X_source @ whitening_src
    X_tgt_aligned = X_target @ whitening_src
    return X_src_aligned, X_tgt_aligned

def run_coral(seed):
    print(f"CORAL (seed={seed})...", flush=True)
    results = []

    for held_out in Y_SUBJECTS:
        train_subjs = [s for s in Y_SUBJECTS if s != held_out]
        X_train_all, y_train_all = [], []
        for subj in train_subjs:
            X, y = load_eeg_data(subj)
            if X is not None:
                X_train_all.append(X)
                y_train_all.append(y)

        X_test, y_test = load_eeg_data(held_out)

        if len(X_train_all) == 0 or X_test is None:
            continue

        X_train_all = np.vstack(X_train_all)
        y_train_all = np.concatenate(y_train_all)

        np.random.seed(seed)
        indices = np.random.permutation(len(y_train_all))
        val_size = int(len(y_train_all) * 0.1)
        train_idx = indices[val_size:]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_all[train_idx])
        X_test_s = scaler.transform(X_test)

        X_tr_aligned, X_test_aligned = coral_align(X_tr, X_test_s)

        clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
        clf.fit(X_tr_aligned, y_train_all[train_idx])
        y_pred = clf.predict(X_test_aligned)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bacc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', warn_for=[])
        cm = confusion_matrix(y_test, y_pred)

        try:
            auroc = roc_auc_score(y_test, clf.decision_function(X_test_aligned))
        except:
            auroc = 0.5

        results.append({
            'model': 'EEG_CORAL',
            'seed': seed,
            'held_out': held_out,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'precision_macro': prec,
            'recall_macro': rec,
            'auroc': auroc,
            'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])
        })
        print(f"  {held_out}: {acc:.4f}", flush=True)

    return results

if __name__ == '__main__':
    all_results = []
    for seed in [0, 1, 2, 3, 4]:
        all_results.extend(run_coral(seed))

    df = pd.DataFrame(all_results)
    output_path = os.path.join(RESULTS_DIR, "coral_5seeds.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(df.groupby('model')['accuracy'].agg(['mean', 'std']))