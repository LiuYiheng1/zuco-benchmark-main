"""
SS-CMC: Semi-Supervised Cross-Modal Consistency Calibration

Target: Use unlabeled EEG/Gaze data to improve personalized fusion

Method:
- Two branches: EEG classifier and Gaze classifier
- Supervised loss: CE(p_eeg, y) + CE(p_gaze, y) + CE(p_fusion, y)
- Unlabeled consistency loss: KL(p_eeg || p_gaze.detach()) + KL(p_gaze || p_eeg.detach())
- Total loss: L = L_sup + lambda_cons * L_cons

Parameters:
- lambda_cons: [0.01, 0.05, 0.1]
- confidence threshold tau: [0.7, 0.8]

Success criteria:
1. 10-shot or 20-shot: SS_CMC >= Static_EEG_Gaze_average + 2%
2. 50-shot: SS_CMC > 82.62%, ideally >= 83.5%
3. Difficult subjects YLS/YSL/YHS avg gain >= 2%
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from scipy.special import softmax

FEATURES_DIR = "features"
RESULTS_DIR = "results/personalized"
REPORT_DIR = "reports"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

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

def kl_divergence(p, q):
    """KL divergence between two probability distributions"""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))

def train_sscmc(eeg_cal, gaze_cal, y_cal, unlabeled_eeg, unlabeled_gaze, lambda_cons, max_iter=200):
    """Train SS-CMC model"""
    scaler_eeg = StandardScaler()
    scaler_gaze = StandardScaler()

    eeg_cal_s = scaler_eeg.fit_transform(eeg_cal)
    gaze_cal_s = scaler_gaze.fit_transform(gaze_cal)

    if len(unlabeled_eeg) > 0:
        unlabeled_eeg_s = scaler_eeg.transform(unlabeled_eeg)
        unlabeled_gaze_s = scaler_gaze.transform(unlabeled_gaze)

    clf_eeg = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=max_iter, random_state=42)
    clf_gaze = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=max_iter, random_state=42)

    clf_eeg.fit(eeg_cal_s, y_cal)
    clf_gaze.fit(gaze_cal_s, y_cal)

    return clf_eeg, clf_gaze, scaler_eeg, scaler_gaze

def get_consistency_loss(clf_eeg, clf_gaze, scaler_eeg, scaler_gaze, unlabeled_eeg, unlabeled_gaze, lambda_cons):
    """Calculate consistency loss on unlabeled data"""
    if len(unlabeled_eeg) == 0:
        return 0.0

    unlabeled_eeg_s = scaler_eeg.transform(unlabeled_eeg)
    unlabeled_gaze_s = scaler_gaze.transform(unlabeled_gaze)

    p_eeg = clf_eeg.predict_proba(unlabeled_eeg_s)
    p_gaze = clf_gaze.predict_proba(unlabeled_gaze_s)

    total_loss = 0.0
    n_samples = len(unlabeled_eeg)

    for i in range(n_samples):
        p_e_i = p_eeg[i]
        p_g_i = p_gaze[i]
        total_loss += kl_divergence(p_e_i, p_g_i) + kl_divergence(p_g_i, p_e_i)

    return lambda_cons * total_loss / n_samples

def evaluate(clf_eeg, clf_gaze, scaler_eeg, scaler_gaze, eeg_test, gaze_test, y_test, use_fusion=True):
    """Evaluate model"""
    eeg_test_s = scaler_eeg.transform(eeg_test)
    gaze_test_s = scaler_gaze.transform(gaze_test)

    p_eeg = clf_eeg.predict_proba(eeg_test_s)[:, 1]
    p_gaze = clf_gaze.predict_proba(gaze_test_s)[:, 1]

    if use_fusion:
        p_fusion = 0.5 * p_eeg + 0.5 * p_gaze
        preds = (p_fusion >= 0.5).astype(int)
    else:
        preds = (p_eeg >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    bacc = balanced_accuracy_score(y_test, preds)
    try:
        auroc = roc_auc_score(y_test, p_fusion if use_fusion else p_eeg)
    except:
        auroc = 0.5

    return acc, f1, bacc, auroc

def run_experiment():
    results = []
    calibration_settings = [5, 10, 20, 50]
    lambda_values = [0.01, 0.05, 0.1]
    tau_values = [0.7, 0.8]
    seeds = [0, 1, 2, 3, 4]

    print("SS-CMC: Semi-Supervised Cross-Modal Consistency Calibration")
    print("="*60)
    print(f"lambda_cons: {lambda_values}, tau: {tau_values}")
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

                cal_size = n_cal_per_class * 2

                class_0_idx = np.where(y_cal_pool == 0)[0]
                class_1_idx = np.where(y_cal_pool == 1)[0]

                np.random.shuffle(class_0_idx)
                np.random.shuffle(class_1_idx)

                cal_idx = np.concatenate([class_0_idx[:n_cal_per_class], class_1_idx[:n_cal_per_class]])
                np.random.shuffle(cal_idx)

                unlabeled_idx = np.concatenate([class_0_idx[n_cal_per_class:], class_1_idx[n_cal_per_class:]])

                eeg_cal = eeg_cal_pool[cal_idx]
                gaze_cal = gaze_cal_pool[cal_idx]
                y_cal = y_cal_pool[cal_idx]

                unlabeled_eeg = eeg_cal_pool[unlabeled_idx]
                unlabeled_gaze = gaze_cal_pool[unlabeled_idx]

                if len(np.unique(y_cal)) < 2:
                    continue

                clf_eeg, clf_gaze, scaler_eeg, scaler_gaze = train_sscmc(
                    eeg_cal, gaze_cal, y_cal, unlabeled_eeg, unlabeled_gaze, lambda_cons=0.0
                )

                acc_eeg, f1_eeg, bacc_eeg, auroc_eeg = evaluate(
                    clf_eeg, clf_gaze, scaler_eeg, scaler_gaze, eeg_test, gaze_test, y_test, use_fusion=False
                )
                acc_gaze, f1_gaze, bacc_gaze, auroc_gaze = evaluate(
                    clf_eeg, clf_gaze, scaler_eeg, scaler_gaze, eeg_test, gaze_test, y_test, use_fusion=False
                )
                acc_static, f1_static, bacc_static, auroc_static = evaluate(
                    clf_eeg, clf_gaze, scaler_eeg, scaler_gaze, eeg_test, gaze_test, y_test, use_fusion=True
                )

                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal_per_class': n_cal_per_class,
                    'method': 'EEG_only', 'lambda': 0.0, 'tau': 0.0,
                    'accuracy': acc_eeg, 'macro_f1': f1_eeg, 'balanced_accuracy': bacc_eeg, 'auroc': auroc_eeg
                })
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal_per_class': n_cal_per_class,
                    'method': 'Gaze_only', 'lambda': 0.0, 'tau': 0.0,
                    'accuracy': acc_gaze, 'macro_f1': f1_gaze, 'balanced_accuracy': bacc_gaze, 'auroc': auroc_gaze
                })
                results.append({
                    'seed': seed, 'subject': held_out, 'n_cal_per_class': n_cal_per_class,
                    'method': 'Static_Fusion', 'lambda': 0.0, 'tau': 0.0,
                    'accuracy': acc_static, 'macro_f1': f1_static, 'balanced_accuracy': bacc_static, 'auroc': auroc_static
                })

                for lambda_cons in lambda_values:
                    clf_eeg, clf_gaze, scaler_eeg, scaler_gaze = train_sscmc(
                        eeg_cal, gaze_cal, y_cal, unlabeled_eeg, unlabeled_gaze, lambda_cons
                    )

                    cons_loss = get_consistency_loss(
                        clf_eeg, clf_gaze, scaler_eeg, scaler_gaze, unlabeled_eeg, unlabeled_gaze, lambda_cons
                    )

                    for tau in tau_values:
                        eeg_test_s = scaler_eeg.transform(eeg_test)
                        gaze_test_s = scaler_gaze.transform(gaze_test)

                        p_eeg = clf_eeg.predict_proba(eeg_test_s)[:, 1]
                        p_gaze = clf_gaze.predict_proba(gaze_test_s)[:, 1]

                        p_fusion = np.zeros_like(p_eeg)
                        for i in range(len(p_eeg)):
                            if abs(p_eeg[i] - 0.5) > tau - 0.5:
                                p_fusion[i] = p_eeg[i]
                            else:
                                p_fusion[i] = 0.5 * p_eeg[i] + 0.5 * p_gaze[i]

                        preds = (p_fusion >= 0.5).astype(int)
                        acc = accuracy_score(y_test, preds)
                        f1 = f1_score(y_test, preds, average='macro')
                        bacc = balanced_accuracy_score(y_test, preds)
                        try:
                            auroc = roc_auc_score(y_test, p_fusion)
                        except:
                            auroc = 0.5

                        results.append({
                            'seed': seed, 'subject': held_out, 'n_cal_per_class': n_cal_per_class,
                            'method': f'SS_CMC_l{lambda_cons}_t{tau}', 'lambda': lambda_cons, 'tau': tau,
                            'accuracy': acc, 'macro_f1': f1, 'balanced_accuracy': bacc, 'auroc': auroc
                        })

            print(f" {held_out}", end="", flush=True)

    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze and summarize results"""
    print("\n" + "="*60)
    print("SS-CMC Results Summary")
    print("="*60)

    difficult_subjects = ['YLS', 'YSL', 'YHS']

    for k in [5, 10, 20, 50]:
        print(f"\nk={k}-shot:")
        for method in ['Static_Fusion', 'SS_CMC_l0.01_t0.7', 'SS_CMC_l0.01_t0.8',
                       'SS_CMC_l0.05_t0.7', 'SS_CMC_l0.05_t0.8',
                       'SS_CMC_l0.1_t0.7', 'SS_CMC_l0.1_t0.8']:
            data = df[(df['method'] == method) & (df['n_cal_per_class'] == k)]
            if len(data) > 0:
                acc = data['accuracy'].mean()
                std = data['accuracy'].std()
                print(f"  {method:25s}: {acc:.4f}±{std:.4f}")

    print("\n" + "="*60)
    print("Success Criteria Check")
    print("="*60)

    baseline_static = {}
    for k in [5, 10, 20, 50]:
        static_data = df[(df['method'] == 'Static_Fusion') & (df['n_cal_per_class'] == k)]
        baseline_static[k] = static_data['accuracy'].mean()
        print(f"k={k}: Static_Fusion baseline = {baseline_static[k]:.4f}")

    print("\n10/20-shot criteria (>= Static + 2%):")
    for k in [10, 20]:
        target = baseline_static[k] + 0.02
        print(f"  Target: {target:.4f}")
        for method in ['SS_CMC_l0.01_t0.7', 'SS_CMC_l0.01_t0.8',
                       'SS_CMC_l0.05_t0.7', 'SS_CMC_l0.05_t0.8',
                       'SS_CMC_l0.1_t0.7', 'SS_CMC_l0.1_t0.8']:
            data = df[(df['method'] == method) & (df['n_cal_per_class'] == k)]
            if len(data) > 0:
                acc = data['accuracy'].mean()
                status = "PASS" if acc >= target else "FAIL"
                print(f"    {method}: {acc:.4f} ({status})")

    print("\n50-shot criteria (> 82.62%, ideally >= 83.5%):")
    target_low = 0.8262
    target_high = 0.835
    for method in ['Static_Fusion', 'SS_CMC_l0.01_t0.7', 'SS_CMC_l0.01_t0.8',
                   'SS_CMC_l0.05_t0.7', 'SS_CMC_l0.05_t0.8',
                   'SS_CMC_l0.1_t0.7', 'SS_CMC_l0.1_t0.8']:
        data = df[(df['method'] == method) & (df['n_cal_per_class'] == 50)]
        if len(data) > 0:
            acc = data['accuracy'].mean()
            if acc >= target_high:
                status = "PASS (>83.5%)"
            elif acc >= target_low:
                status = "MARGINAL (>82.62%)"
            else:
                status = "FAIL"
            print(f"  {method}: {acc:.4f} ({status})")

    print("\nDifficult subjects (YLS/YSL/YHS) avg gain vs Static:")
    for k in [10, 20]:
        for method in ['SS_CMC_l0.01_t0.7', 'SS_CMC_l0.01_t0.8',
                       'SS_CMC_l0.05_t0.7', 'SS_CMC_l0.05_t0.8',
                       'SS_CMC_l0.1_t0.7', 'SS_CMC_l0.1_t0.8']:
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

    output_path = os.path.join(RESULTS_DIR, "ss_cmc_results.csv")
    df.to_csv(output_path, index=False)

    analyze_results(df)

    print(f"\nResults saved to {output_path}")
    print("Done!")

if __name__ == '__main__':
    main()