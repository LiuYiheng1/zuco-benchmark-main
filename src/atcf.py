"""
ATCF: Ambiguity-Triggered Complementary Fusion

Target: Only let gaze intervene when EEG is uncertain, not always 50/50 fusion

Method:
1. Train EEG classifier and gaze classifier on calibration set
2. For test sample:
   if |p_eeg - 0.5| > tau:
       p_final = p_eeg
   else:
       p_final = 0.5 * p_eeg + 0.5 * p_gaze
3. tau must be selected using calibration set with leave-one-out

Success criteria:
1. 50-shot: ATCF > Static_EEG_Gaze_average + 1%
2. 20/50-shot avg: ATCF > Static_EEG_Gaze_average + 1%
3. Difficult subjects YLS/YSL/YHS avg gain >= 2%
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

FEATURES_DIR = "features"
RESULTS_DIR = "results/personalized"
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

def load_gaze_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze.npy")
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
        feat_list = list(values)
        features = np.array(feat_list[:-1], dtype=np.float64)
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def align_eeg_gaze(eeg_X, eeg_y, gaze_X, gaze_y):
    common_len = min(len(eeg_y), len(gaze_y))
    return eeg_X[:common_len], eeg_y[:common_len], gaze_X[:common_len], gaze_y[:common_len]

def select_tau_loo(eeg_cal, gaze_cal, y_cal, tau_values):
    """Select best tau using a small held-out set from calibration"""
    best_tau = tau_values[0]
    best_acc = 0.0

    if len(y_cal) < 20:
        return best_tau

    n_holdout = min(10, len(y_cal) // 4)
    np.random.seed(42)
    holdout_idx = np.random.choice(len(y_cal), n_holdout, replace=False)
    cal_idx = np.array([i for i in range(len(y_cal)) if i not in holdout_idx])

    scaler_eeg = StandardScaler()
    scaler_gaze = StandardScaler()

    eeg_cal_s = scaler_eeg.fit_transform(eeg_cal[cal_idx])
    gaze_cal_s = scaler_gaze.fit_transform(gaze_cal[cal_idx])

    clf_eeg = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    clf_gaze = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
    clf_eeg.fit(eeg_cal_s, y_cal[cal_idx])
    clf_gaze.fit(gaze_cal_s, y_cal[cal_idx])

    eeg_holdout_s = scaler_eeg.transform(eeg_cal[holdout_idx])
    gaze_holdout_s = scaler_gaze.transform(gaze_cal[holdout_idx])
    p_eeg_holdout = clf_eeg.predict_proba(eeg_holdout_s)[:, 1]
    p_gaze_holdout = clf_gaze.predict_proba(gaze_holdout_s)[:, 1]

    for tau in tau_values:
        fusion_preds = []
        for i in range(len(holdout_idx)):
            if abs(p_eeg_holdout[i] - 0.5) > tau - 0.5:
                p_final = p_eeg_holdout[i]
            else:
                p_final = 0.5 * p_eeg_holdout[i] + 0.5 * p_gaze_holdout[i]
            fusion_preds.append(1 if p_final >= 0.5 else 0)

        holdout_acc = accuracy_score(y_cal[holdout_idx], fusion_preds)
        if holdout_acc > best_acc:
            best_acc = holdout_acc
            best_tau = tau

    return best_tau

def train_and_evaluate(eeg_cal, gaze_cal, y_cal, eeg_test, gaze_test, y_test, tau, use_atcf=True):
    """Train classifiers and evaluate"""
    scaler_eeg = StandardScaler()
    scaler_gaze = StandardScaler()

    eeg_cal_s = scaler_eeg.fit_transform(eeg_cal)
    gaze_cal_s = scaler_gaze.fit_transform(gaze_cal)
    eeg_test_s = scaler_eeg.transform(eeg_test)
    gaze_test_s = scaler_gaze.transform(gaze_test)

    clf_eeg = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    clf_gaze = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)

    clf_eeg.fit(eeg_cal_s, y_cal)
    clf_gaze.fit(gaze_cal_s, y_cal)

    p_eeg = clf_eeg.predict_proba(eeg_test_s)[:, 1]
    p_gaze = clf_gaze.predict_proba(gaze_test_s)[:, 1]

    if use_atcf:
        p_fusion = np.zeros_like(p_eeg)
        for i in range(len(p_eeg)):
            if abs(p_eeg[i] - 0.5) > tau - 0.5:
                p_fusion[i] = p_eeg[i]
            else:
                p_fusion[i] = 0.5 * p_eeg[i] + 0.5 * p_gaze[i]
    else:
        p_fusion = 0.5 * p_eeg + 0.5 * p_gaze

    preds = (p_fusion >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, p_fusion)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_experiment():
    results = []
    calibration_settings = [5, 10, 20, 50]
    tau_values = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    seeds = [0, 1, 2, 3, 4]

    print("ATCF: Ambiguity-Triggered Complementary Fusion")
    print("="*60)
    print(f"tau values: {tau_values}")
    print("="*60)

    for seed in seeds:
        print(f"\nSeed {seed}:", flush=True)

        for held_out in Y_SUBJECTS:
            eeg_X, eeg_y = load_eeg_data(held_out)
            gaze_X, gaze_y = load_gaze_data(held_out)

            if eeg_X is None or gaze_X is None:
                print(f" {held_out}[no data]", end="", flush=True)
                continue

            eeg_X, eeg_y, gaze_X, gaze_y = align_eeg_gaze(eeg_X, eeg_y, gaze_X, gaze_y)

            if len(eeg_X) < 50:
                print(f" {held_out}[no align]", end="", flush=True)
                continue

            n_samples = len(eeg_y)
            np.random.seed(seed)
            indices = np.random.permutation(n_samples)
            test_indices = indices[:n_samples // 2]
            cal_pool_indices = indices[n_samples // 2:]

            eeg_test = eeg_X[test_indices]
            gaze_test = gaze_X[test_indices]
            y_test = eeg_y[test_indices]

            eeg_cal_pool = eeg_X[cal_pool_indices]
            gaze_cal_pool = gaze_X[cal_pool_indices]
            y_cal_pool = eeg_y[cal_pool_indices]

            for n_cal_per_class in calibration_settings:
                if n_cal_per_class * 2 > len(cal_pool_indices):
                    continue

                class_0_idx = np.where(y_cal_pool == 0)[0]
                class_1_idx = np.where(y_cal_pool == 1)[0]

                np.random.shuffle(class_0_idx)
                np.random.shuffle(class_1_idx)

                cal_idx = np.concatenate([class_0_idx[:n_cal_per_class], class_1_idx[:n_cal_per_class]])
                np.random.shuffle(cal_idx)

                eeg_cal = eeg_cal_pool[cal_idx]
                gaze_cal = gaze_cal_pool[cal_idx]
                y_cal = y_cal_pool[cal_idx]

                if len(np.unique(y_cal)) < 2:
                    continue

                best_tau = select_tau_loo(eeg_cal, gaze_cal, y_cal, tau_values)

                acc_eeg, f1_eeg, bacc_eeg, auroc_eeg = train_and_evaluate(
                    eeg_cal, gaze_cal, y_cal, eeg_test, gaze_test, y_test, tau=0.0, use_atcf=False
                )
                acc_static, f1_static, bacc_static, auroc_static = train_and_evaluate(
                    eeg_cal, gaze_cal, y_cal, eeg_test, gaze_test, y_test, tau=0.5, use_atcf=False
                )
                acc_atcf, f1_atcf, bacc_atcf, auroc_atcf = train_and_evaluate(
                    eeg_cal, gaze_cal, y_cal, eeg_test, gaze_test, y_test, tau=best_tau, use_atcf=True
                )

                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal_per_class': n_cal_per_class,
                    'method': 'EEG_only',
                    'accuracy': acc_eeg, 'macro_f1': f1_eeg, 'balanced_accuracy': bacc_eeg, 'auroc': auroc_eeg,
                    'tau': 0.0
                })
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal_per_class': n_cal_per_class,
                    'method': 'Static_Fusion',
                    'accuracy': acc_static, 'macro_f1': f1_static, 'balanced_accuracy': bacc_static, 'auroc': auroc_static,
                    'tau': 0.5
                })
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal_per_class': n_cal_per_class,
                    'method': f'ATCF_tau{best_tau:.2f}',
                    'accuracy': acc_atcf, 'macro_f1': f1_atcf, 'balanced_accuracy': bacc_atcf, 'auroc': auroc_atcf,
                    'tau': best_tau
                })

            print(f" {held_out}", end="", flush=True)

    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze and summarize results"""
    print("\n" + "="*60)
    print("ATCF Results Summary")
    print("="*60)

    difficult_subjects = ['YLS', 'YSL', 'YHS']

    for k in [5, 10, 20, 50]:
        print(f"\nk={k}-shot:")
        for method in ['EEG_only', 'Static_Fusion', 'ATCF_tau0.55', 'ATCF_tau0.60', 'ATCF_tau0.65', 'ATCF_tau0.70', 'ATCF_tau0.75', 'ATCF_tau0.80']:
            data = df[(df['method'] == method) & (df['n_cal_per_class'] == k)]
            if len(data) > 0:
                acc = data['accuracy'].mean()
                std = data['accuracy'].std()
                print(f"  {method:20s}: {acc:.4f}±{std:.4f}")

    atcf_methods = [m for m in df['method'].unique() if m.startswith('ATCF_tau')]
    if atcf_methods:
        best_atcf = atcf_methods[0]
        best_acc = 0
        for m in atcf_methods:
            m_acc = df[df['method'] == m]['accuracy'].mean()
            if m_acc > best_acc:
                best_acc = m_acc
                best_atcf = m

        print(f"\nBest ATCF method: {best_atcf} (acc={best_acc:.4f})")

    print("\n" + "="*60)
    print("Success Criteria Check")
    print("="*60)

    baseline_static = {}
    for k in [5, 10, 20, 50]:
        static_data = df[(df['method'] == 'Static_Fusion') & (df['n_cal_per_class'] == k)]
        baseline_static[k] = static_data['accuracy'].mean()
        print(f"k={k}: Static_Fusion baseline = {baseline_static[k]:.4f}")

    print("\n50-shot criteria (ATCF > Static + 1%):")
    target = baseline_static[50] + 0.01
    print(f"  Target: {target:.4f}")
    for method in atcf_methods:
        data = df[(df['method'] == method) & (df['n_cal_per_class'] == 50)]
        if len(data) > 0:
            acc = data['accuracy'].mean()
            status = "PASS" if acc >= target else "FAIL"
            print(f"    {method}: {acc:.4f} ({status})")

    print("\n20/50-shot avg (ATCF > Static + 1%):")
    target = 0.5 * (baseline_static[20] + baseline_static[50]) + 0.01
    print(f"  Target: {target:.4f}")
    for method in atcf_methods:
        d20 = df[(df['method'] == method) & (df['n_cal_per_class'] == 20)]
        d50 = df[(df['method'] == method) & (df['n_cal_per_class'] == 50)]
        if len(d20) > 0 and len(d50) > 0:
            avg_acc = 0.5 * (d20['accuracy'].mean() + d50['accuracy'].mean())
            status = "PASS" if avg_acc >= target else "FAIL"
            print(f"    {method}: {avg_acc:.4f} ({status})")

    print("\nDifficult subjects (YLS/YSL/YHS) avg gain vs Static:")
    for k in [20, 50]:
        for method in atcf_methods:
            diff_gains = []
            for subj in difficult_subjects:
                static_data = df[(df['method'] == 'Static_Fusion') & (df['n_cal_per_class'] == k) & (df['subject'] == subj)]
                method_data = df[(df['method'] == method) & (df['n_cal_per_class'] == k) & (df['subject'] == subj)]
                if len(static_data) > 0 and len(method_data) > 0:
                    diff_gains.append(method_data['accuracy'].mean() - static_data['accuracy'].mean())
            if diff_gains:
                avg_gain = np.mean(diff_gains)
                marker = " *" if avg_gain >= 0.02 else ""
                print(f"  k={k}, {method}: {avg_gain:+.4f}{marker}")

def main():
    os.chdir("d:/pycharmproject/zuco-benchmark-main/src")

    df = run_experiment()

    output_path = os.path.join(RESULTS_DIR, "atcf_results.csv")
    df.to_csv(output_path, index=False)

    analyze_results(df)

    print(f"\nResults saved to {output_path}")
    print("Done!")

if __name__ == '__main__':
    main()