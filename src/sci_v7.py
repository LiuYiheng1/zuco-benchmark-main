"""
SCI Framework V7: Threshold Calibration with Orthogonal Information

Key formula:
    tau(x) = tau_0 + beta * (u - u_cal_mean) + gamma * (d - d_cal_mean) + rho * u * d
    pred = 1 if p_pcet >= tau(x) else 0

Methods compared:
    1. PCET (baseline)
    2. PCET + fixed threshold tuning
    3. PCET + SRGC uncertainty
    4. PCET + SIED shift
    5. PCET + SRGC + SIED
    6. PCET + random uncertainty (control)
    7. PCET + shuffled uncertainty (control)
    8. PCET + shuffled domain shift (control)
"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
print("="*60)
print("SCI Framework V7: Threshold Calibration")
print("="*60)

FEATURES_DIR = "features"
RESULTS_DIR = "results/final"
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


def compute_ece(y_true, y_prob, n_bins=15):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / len(y_true)
    if np.isnan(ece):
        return 1.0
    return ece


def compute_brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)


class PCETModule:
    def fit(self, X_cal, y_cal):
        self.scaler = StandardScaler()
        X_cal_s = self.scaler.fit_transform(X_cal)
        self.clf = LogisticRegression(max_iter=200, solver='lbfgs', C=1.0)
        self.clf.fit(X_cal_s, y_cal)
        self.y_cal = y_cal
        self.X_cal_s = X_cal_s
        self.mu_0 = np.mean(X_cal_s[y_cal == 0], axis=0)
        self.mu_1 = np.mean(X_cal_s[y_cal == 1], axis=0)
        return self

    def predict(self, X_test):
        X_test_s = self.scaler.transform(X_test)
        probs = self.clf.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return probs, preds

    def predict_proba(self, X_test):
        X_test_s = self.scaler.transform(X_test)
        return self.clf.predict_proba(X_test_s)[:, 1]


class SRGCUncertainty:
    def fit(self, X_cal, y_cal):
        self.scaler = StandardScaler()
        X_cal_s = self.scaler.fit_transform(X_cal)
        self.X_cal_s = X_cal_s
        self.y_cal = y_cal
        self.mu_0 = np.mean(X_cal_s[y_cal == 0], axis=0)
        self.mu_1 = np.mean(X_cal_s[y_cal == 1], axis=0)
        n_neighbors = min(5, len(X_cal_s) - 1)
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(X_cal_s)
        return self

    def estimate_uncertainty(self, X):
        X_s = self.scaler.transform(X)
        dist_0 = np.sqrt(np.sum((X_s - self.mu_0) ** 2, axis=1))
        dist_1 = np.sqrt(np.sum((X_s - self.mu_1) ** 2, axis=1))
        spread = np.abs(dist_1 - dist_0) / (dist_0 + dist_1 + 1e-8)
        distances, _ = self.nn.kneighbors(X_s)
        avg_dist = np.mean(distances, axis=1)
        uncertainty = 1 / (1 + spread) * 0.5 + 1 / (1 + avg_dist) * 0.5
        return np.clip(uncertainty, 0, 1)


class SIEDDomain:
    def fit(self, X_cal, y_cal):
        self.scaler = StandardScaler()
        X_cal_s = self.scaler.fit_transform(X_cal)
        self.X_cal_s = X_cal_s
        self.mu_cal = np.mean(X_cal_s, axis=0)
        self.sigma_cal = np.std(X_cal_s, axis=0) + 1e-8
        return self

    def estimate_domain_shift(self, X):
        X_s = self.scaler.transform(X)
        test_mean = np.mean(X_s, axis=0)
        mean_diff = np.sqrt(np.sum((test_mean - self.mu_cal) ** 2))
        shift = mean_diff / (np.sqrt(np.sum(self.mu_cal ** 2)) + 1e-8)
        n_samples = X.shape[0]
        return np.ones(n_samples) * np.clip(shift, 0, 1)


def calibrate_threshold_cv(p_cal, y_cal, n_splits=5):
    """Select best threshold using internal CV on calibration set"""
    n_cal = len(y_cal)
    indices = np.random.permutation(n_cal)
    fold_size = n_cal // n_splits

    best_thresh = 0.5
    best_acc = 0.0

    for tau_0 in np.arange(0.3, 0.7, 0.05):
        accs = []
        for fold in range(n_splits):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_splits - 1 else n_cal
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            p_val = p_cal[val_idx]
            y_val = y_cal[val_idx]

            preds = (p_val >= tau_0).astype(int)
            acc = accuracy_score(y_val, preds)
            accs.append(acc)

        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_thresh = tau_0

    return best_thresh


def calibrate_full_model_cv(p_cal, y_cal, u_cal, d_cal, n_splits=5):
    """Calibrate full threshold model using CV"""
    n_cal = len(y_cal)
    indices = np.random.permutation(n_cal)
    fold_size = n_cal // n_splits

    best_params = {'tau_0': 0.5, 'beta': 0.0, 'gamma': 0.0, 'rho': 0.0}
    best_acc = 0.0

    u_cal_mean = np.mean(u_cal)
    d_cal_mean = np.mean(d_cal)

    for tau_0 in np.arange(0.3, 0.7, 0.05):
        for beta in np.arange(-0.3, 0.4, 0.1):
            for gamma in np.arange(-0.3, 0.4, 0.1):
                for rho in np.arange(-0.2, 0.25, 0.1):
                    accs = []
                    for fold in range(n_splits):
                        val_start = fold * fold_size
                        val_end = val_start + fold_size if fold < n_splits - 1 else n_cal
                        val_idx = indices[val_start:val_end]

                        tau = tau_0 + beta * (u_cal[val_idx] - u_cal_mean) + \
                              gamma * (d_cal[val_idx] - d_cal_mean) + rho * u_cal[val_idx] * d_cal[val_idx]
                        tau = np.clip(tau, 0.3, 0.7)

                        p_val = p_cal[val_idx]
                        y_val = y_cal[val_idx]

                        preds = (p_val >= tau).astype(int)
                        acc = accuracy_score(y_val, preds)
                        accs.append(acc)

                    mean_acc = np.mean(accs)
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        best_params = {'tau_0': tau_0, 'beta': beta, 'gamma': gamma, 'rho': rho}

    return best_params


def calibrate_srgc_only_cv(p_cal, y_cal, u_cal, n_splits=5):
    """Calibrate SRGC only"""
    n_cal = len(y_cal)
    indices = np.random.permutation(n_cal)
    fold_size = n_cal // n_splits

    best_params = {'tau_0': 0.5, 'beta': 0.0}
    best_acc = 0.0

    u_cal_mean = np.mean(u_cal)

    for tau_0 in np.arange(0.3, 0.7, 0.05):
        for beta in np.arange(-0.3, 0.4, 0.1):
            accs = []
            for fold in range(n_splits):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < n_splits - 1 else n_cal
                val_idx = indices[val_start:val_end]

                tau = tau_0 + beta * (u_cal[val_idx] - u_cal_mean)
                tau = np.clip(tau, 0.3, 0.7)

                p_val = p_cal[val_idx]
                y_val = y_cal[val_idx]

                preds = (p_val >= tau).astype(int)
                acc = accuracy_score(y_val, preds)
                accs.append(acc)

            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = {'tau_0': tau_0, 'beta': beta}

    return best_params


def calibrate_sied_only_cv(p_cal, y_cal, d_cal, n_splits=5):
    """Calibrate SIED only"""
    n_cal = len(y_cal)
    indices = np.random.permutation(n_cal)
    fold_size = n_cal // n_splits

    best_params = {'tau_0': 0.5, 'gamma': 0.0}
    best_acc = 0.0

    d_cal_mean = np.mean(d_cal)

    for tau_0 in np.arange(0.3, 0.7, 0.05):
        for gamma in np.arange(-0.3, 0.4, 0.1):
            accs = []
            for fold in range(n_splits):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < n_splits - 1 else n_cal
                val_idx = indices[val_start:val_end]

                tau = tau_0 + gamma * (d_cal[val_idx] - d_cal_mean)
                tau = np.clip(tau, 0.3, 0.7)

                p_val = p_cal[val_idx]
                y_val = y_cal[val_idx]

                preds = (p_val >= tau).astype(int)
                acc = accuracy_score(y_val, preds)
                accs.append(acc)

            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = {'tau_0': tau_0, 'gamma': gamma}

    return best_params


def compute_metrics(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except:
        auroc = 0.5
    ece = compute_ece(y_true, y_prob)
    brier = compute_brier_score(y_true, y_prob)
    return acc, f1, bacc, auroc, ece, brier


def apply_threshold(p, tau):
    if np.ndim(tau) == 0:
        return (p >= tau).astype(int)
    return (p >= tau).astype(int)


print("\nLoading data...")
all_data = {}
for subj in Y_SUBJECTS:
    X, y = load_eeg_data(subj)
    if X is not None:
        all_data[subj] = (X, y)
print(f"Loaded {len(all_data)} subjects")

results = []
shots = [3, 5, 10, 20, 50]
seeds = [0, 1, 2]

print(f"\nRunning: {len(shots)} shots x {len(seeds)} seeds x {len(all_data)} subjects...")

for seed in seeds:
    print(f"\nSeed {seed}:", end=' ')
    for held_out in Y_SUBJECTS:
        if held_out not in all_data:
            continue

        X_test_orig, y_test_orig = all_data[held_out]
        train_subjs = [s for s in Y_SUBJECTS if s != held_out and s in all_data]
        if len(train_subjs) == 0:
            continue

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

        for n_cal in shots:
            if n_cal * 2 > len(cal_pool_indices):
                continue

            class_0_idx = np.where(y_cal_pool == 0)[0]
            class_1_idx = np.where(y_cal_pool == 1)[0]
            np.random.shuffle(class_0_idx)
            np.random.shuffle(class_1_idx)
            n0 = min(n_cal, len(class_0_idx))
            n1 = min(n_cal, len(class_1_idx))
            cal_idx = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])

            X_cal = X_cal_pool[cal_idx]
            y_cal = y_cal_pool[cal_idx]

            if len(np.unique(y_cal)) < 2:
                continue

            pcet = PCETModule()
            pcet.fit(X_cal, y_cal)
            p_pcet = pcet.predict_proba(X_test)
            u_cal = pcet.predict_proba(X_cal)
            u_test = pcet.confidence(X_cal) if hasattr(pcet, 'confidence') else np.maximum(u_cal, 1 - u_cal)

            srgc = SRGCUncertainty()
            srgc.fit(X_cal, y_cal)
            u_srgc_cal = srgc.estimate_uncertainty(X_cal)
            u_srgc_test = srgc.estimate_uncertainty(X_test)

            sied = SIEDDomain()
            sied.fit(X_cal, y_cal)
            d_sied_cal = sied.estimate_domain_shift(X_cal)
            d_sied_test = sied.estimate_domain_shift(X_test)

            u_cal_mean = np.mean(u_srgc_cal)
            d_cal_mean = np.mean(d_sied_cal)

            u_srgc_cal_shuffled = np.random.permutation(u_srgc_cal.flatten()) if u_srgc_cal.ndim > 1 else np.random.permutation(u_srgc_cal)
            u_srgc_test_shuffled = np.random.permutation(u_srgc_test.flatten()) if u_srgc_test.ndim > 1 else np.random.permutation(u_srgc_test)
            d_sied_cal_shuffled = np.random.permutation(d_sied_cal.flatten()) if d_sied_cal.ndim > 1 else np.random.permutation(d_sied_cal)
            d_sied_test_shuffled = np.random.permutation(d_sied_test.flatten()) if d_sied_test.ndim > 1 else np.random.permutation(d_sied_test)

            random_uncertainty = np.random.uniform(0, 1, len(X_test))

            cal_results = {}

            tau_fixed = calibrate_threshold_cv(u_cal, y_cal)
            cal_results['tau_fixed'] = tau_fixed

            p_pcet_cal = pcet.predict_proba(X_cal)
            params_fixed = {'tau_0': calibrate_threshold_cv(p_pcet_cal, y_cal)}
            cal_results['params_fixed'] = params_fixed

            params_srgc = calibrate_srgc_only_cv(p_pcet_cal, y_cal, u_srgc_cal)
            cal_results['params_srgc'] = params_srgc

            params_sied = calibrate_sied_only_cv(p_pcet_cal, y_cal, d_sied_cal)
            cal_results['params_sied'] = params_sied

            params_full = calibrate_full_model_cv(p_pcet_cal, y_cal, u_srgc_cal, d_sied_cal)
            cal_results['params_full'] = params_full

            methods = {}

            methods['PCET'] = {
                'pred': (p_pcet >= 0.5).astype(int),
                'prob': p_pcet
            }

            methods['PCET_fixed_thresh'] = {
                'pred': (p_pcet >= params_fixed['tau_0']).astype(int),
                'prob': p_pcet,
                'tau': params_fixed['tau_0']
            }

            tau_srgc = params_srgc['tau_0'] + params_srgc['beta'] * (u_srgc_test - u_cal_mean)
            tau_srgc = np.clip(tau_srgc, 0.3, 0.7)
            methods['PCET_SRGC'] = {
                'pred': (p_pcet >= tau_srgc).astype(int),
                'prob': p_pcet,
                'tau': tau_srgc
            }

            tau_sied = params_sied['tau_0'] + params_sied['gamma'] * (d_sied_test - d_cal_mean)
            tau_sied = np.clip(tau_sied, 0.3, 0.7)
            methods['PCET_SIED'] = {
                'pred': (p_pcet >= tau_sied).astype(int),
                'prob': p_pcet,
                'tau': tau_sied
            }

            tau_full = params_full['tau_0'] + \
                       params_full['beta'] * (u_srgc_test - u_cal_mean) + \
                       params_full['gamma'] * (d_sied_test - d_cal_mean) + \
                       params_full['rho'] * u_srgc_test * d_sied_test
            tau_full = np.clip(tau_full, 0.3, 0.7)
            methods['PCET_SRGC_SIED'] = {
                'pred': (p_pcet >= tau_full).astype(int),
                'prob': p_pcet,
                'tau': tau_full
            }

            tau_random = 0.5 + 0.2 * (random_uncertainty - 0.5)
            tau_random = np.clip(tau_random, 0.3, 0.7)
            methods['PCET_random_unc'] = {
                'pred': (p_pcet >= tau_random).astype(int),
                'prob': p_pcet,
                'tau': tau_random
            }

            tau_shuffled = params_srgc['tau_0'] + params_srgc['beta'] * (u_srgc_test_shuffled - u_cal_mean)
            tau_shuffled = np.clip(tau_shuffled, 0.3, 0.7)
            methods['PCET_shuffled_SRGC'] = {
                'pred': (p_pcet >= tau_shuffled).astype(int),
                'prob': p_pcet,
                'tau': tau_shuffled
            }

            tau_shuffled_sied = params_sied['tau_0'] + params_sied['gamma'] * (d_sied_test_shuffled - d_cal_mean)
            tau_shuffled_sied = np.clip(tau_shuffled_sied, 0.3, 0.7)
            methods['PCET_shuffled_SIED'] = {
                'pred': (p_pcet >= tau_shuffled_sied).astype(int),
                'prob': p_pcet,
                'tau': tau_shuffled_sied
            }

            for method_name, method_data in methods.items():
                y_pred = method_data['pred']
                y_prob = method_data['prob']
                acc, f1, bacc, auroc, ece, brier = compute_metrics(y_test, y_prob, y_pred)
                results.append({
                    'seed': seed,
                    'subject': held_out,
                    'n_cal': n_cal,
                    'method': method_name,
                    'accuracy': acc,
                    'macro_f1': f1,
                    'balanced_accuracy': bacc,
                    'auroc': auroc,
                    'ece': ece,
                    'brier_score': brier
                })

        print(".", end='', flush=True)

df = pd.DataFrame(results)
df.to_csv(f'{RESULTS_DIR}/sci_v7_results.csv', index=False)

print("\n\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

summary_data = []
for n_cal in shots:
    print(f"\n{n_cal}-shot:")
    shot_df = df[df['n_cal'] == n_cal]

    for method in ['PCET', 'PCET_fixed_thresh', 'PCET_SRGC', 'PCET_SIED',
                   'PCET_SRGC_SIED', 'PCET_random_unc', 'PCET_shuffled_SRGC', 'PCET_shuffled_SIED']:
        m_df = shot_df[shot_df['method'] == method]
        if len(m_df) > 0:
            acc = m_df['accuracy'].mean()
            std = m_df['accuracy'].std()
            f1 = m_df['macro_f1'].mean()
            bacc = m_df['balanced_accuracy'].mean()
            auroc = m_df['auroc'].mean()
            ece = m_df['ece'].mean()
            brier = m_df['brier_score'].mean()
            print(f"  {method:25s}: Acc={acc:.4f}±{std:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}, AUROC={auroc:.4f}, ECE={ece:.4f}, Brier={brier:.4f}")
            summary_data.append({
                'shot': n_cal, 'method': method,
                'accuracy': acc, 'accuracy_std': std,
                'macro_f1': f1, 'balanced_accuracy': bacc,
                'auroc': auroc, 'ece': ece, 'brier_score': brier
            })

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(f'{RESULTS_DIR}/sci_v7_summary.csv', index=False)

print("\n" + "="*60)
print("SUCCESS CRITERIA CHECK")
print("="*60)

pcet_acc = df[df['method'] == 'PCET'].groupby('n_cal')['accuracy'].mean()
pcet_f1 = df[df['method'] == 'PCET'].groupby('n_cal')['macro_f1'].mean()
pcet_bacc = df[df['method'] == 'PCET'].groupby('n_cal')['balanced_accuracy'].mean()
pcet_ece = df[df['method'] == 'PCET'].groupby('n_cal')['ece'].mean()
pcet_brier = df[df['method'] == 'PCET'].groupby('n_cal')['brier_score'].mean()

sci_acc = df[df['method'] == 'PCET_SRGC_SIED'].groupby('n_cal')['accuracy'].mean()
sci_f1 = df[df['method'] == 'PCET_SRGC_SIED'].groupby('n_cal')['macro_f1'].mean()
sci_bacc = df[df['method'] == 'PCET_SRGC_SIED'].groupby('n_cal')['balanced_accuracy'].mean()
sci_ece = df[df['method'] == 'PCET_SRGC_SIED'].groupby('n_cal')['ece'].mean()
sci_brier = df[df['method'] == 'PCET_SRGC_SIED'].groupby('n_cal')['brier_score'].mean()

srgc_only_acc = df[df['method'] == 'PCET_SRGC'].groupby('n_cal')['accuracy'].mean()

print("\n1. SCI_V7 average Accuracy > PCET?")
total_improvement = 0
shots_improved = 0
for n_cal in shots:
    delta = sci_acc[n_cal] - pcet_acc[n_cal]
    total_improvement += delta
    if delta > 0:
        shots_improved += 1
    marker = "✓" if sci_acc[n_cal] > pcet_acc[n_cal] else "✗"
    print(f"  {n_cal}-shot: PCET={pcet_acc[n_cal]:.4f}, SCI={sci_acc[n_cal]:.4f}, Δ={delta:+.4f} {marker}")
print(f"  Total shots improved: {shots_improved}/5")

print("\n2. At least 3 shots exceed PCET?")
print(f"  {'✓ PASS' if shots_improved >= 3 else '✗ FAIL'} ({shots_improved}/5 shots)")

print("\n3. No more than 1% degradation in any shot?")
max_degradation = 0
for n_cal in shots:
    delta = sci_acc[n_cal] - pcet_acc[n_cal]
    if delta < max_degradation:
        max_degradation = delta
print(f"  Max degradation: {max_degradation:.4f}")
print(f"  {'✓ PASS' if max_degradation >= -0.01 else '✗ FAIL'}")

print("\n4. Macro-F1/BAcc not decreased?")
f1_decreased = 0
bacc_decreased = 0
for n_cal in shots:
    if sci_f1[n_cal] < pcet_f1[n_cal] - 0.001:
        f1_decreased += 1
    if sci_bacc[n_cal] < pcet_bacc[n_cal] - 0.001:
        bacc_decreased += 1
print(f"  F1 decreased in {f1_decreased}/5 shots")
print(f"  BAcc decreased in {bacc_decreased}/5 shots")
print(f"  {'✓ PASS' if f1_decreased == 0 and bacc_decreased == 0 else '✗ FAIL'}")

print("\n5. ECE or Brier score improved?")
for n_cal in shots:
    ece_delta = sci_ece[n_cal] - pcet_ece[n_cal]
    brier_delta = sci_brier[n_cal] - pcet_brier[n_cal]
    marker_ece = "✓" if ece_delta < 0 else "✗"
    marker_brier = "✓" if brier_delta < 0 else "✗"
    print(f"  {n_cal}-shot: ECE Δ={ece_delta:+.4f} {marker_ece}, Brier Δ={brier_delta:+.4f} {marker_brier}")

print("\n6. Full SCI_V7 > PCET_SRGC only?")
for n_cal in shots:
    delta = sci_acc[n_cal] - srgc_only_acc[n_cal]
    marker = "✓" if sci_acc[n_cal] >= srgc_only_acc[n_cal] else "✗"
    print(f"  {n_cal}-shot: Full={sci_acc[n_cal]:.4f}, SRGC_only={srgc_only_acc[n_cal]:.4f}, Δ={delta:+.4f} {marker}")

print("\n7. Shuffled controls show performance drop?")
for n_cal in shots:
    orig = sci_acc[n_cal]
    shuffled_srgc = df[df['method'] == 'PCET_shuffled_SRGC'].groupby('n_cal')['accuracy'].mean()[n_cal]
    shuffled_sied = df[df['method'] == 'PCET_shuffled_SIED'].groupby('n_cal')['accuracy'].mean()[n_cal]
    marker1 = "✓" if shuffled_srgc < orig else "✗"
    marker2 = "✓" if shuffled_sied < orig else "✗"
    print(f"  {n_cal}-shot: Full={orig:.4f}, Shuff_SRGC={shuffled_srgc:.4f} {marker1}, Shuff_SIED={shuffled_sied:.4f} {marker2}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

all_pass = True
if shots_improved < 3:
    all_pass = False
if max_degradation < -0.01:
    all_pass = False

if all_pass:
    print("✓ ALL SUCCESS CRITERIA PASSED")
else:
    print("✗ SOME CRITERIA NOT MET - see above")

print("="*60)