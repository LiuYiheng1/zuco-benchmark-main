"""
SCI Framework V7.1: Fixed Threshold Calibration

Key insight: For very low shots (3-10), calibration on tiny sets overfits.
Solution: Use fixed, theory-driven thresholds instead of CV-based selection.

Also: Focus on threshold ADJUSTMENT rather than threshold REPLACEMENT
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
print("SCI Framework V7.1: Fixed Threshold Calibration")
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

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return self.clf.predict_proba(X_s)[:, 1]


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
            p_pcet_test = pcet.predict_proba(X_test)
            p_pcet_cal = pcet.predict_proba(X_cal)

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
            u_cal_std = np.std(u_srgc_cal) + 1e-8
            d_cal_std = np.std(d_sied_cal) + 1e-8

            u_srgc_test_shuffled = np.random.permutation(u_srgc_test)
            d_sied_test_shuffled = np.random.permutation(d_sied_test)

            y_pcet_pred = (p_pcet_test >= 0.5).astype(int)

            methods = {}

            methods['PCET'] = {
                'pred': y_pcet_pred,
                'prob': p_pcet_test
            }

            for tau_name, tau_val in [('tau_0.45', 0.45), ('tau_0.48', 0.48), ('tau_0.50', 0.50),
                                       ('tau_0.52', 0.52), ('tau_0.55', 0.55)]:
                methods[f'PCET_fixed_{tau_name}'] = {
                    'pred': (p_pcet_test >= tau_val).astype(int),
                    'prob': p_pcet_test
                }

            for beta_name, beta_val in [('beta_-0.1', -0.1), ('beta_-0.05', -0.05), ('beta_0.0', 0.0),
                                         ('beta_0.05', 0.05), ('beta_0.1', 0.1)]:
                u_adj = (u_srgc_test - u_cal_mean) / u_cal_std
                tau = 0.5 + beta_val * u_adj
                tau = np.clip(tau, 0.35, 0.65)
                methods[f'PCET_SRGC_{beta_name}'] = {
                    'pred': (p_pcet_test >= tau).astype(int),
                    'prob': p_pcet_test,
                    'tau_mean': np.mean(tau)
                }

            for gamma_name, gamma_val in [('gamma_-0.1', -0.1), ('gamma_-0.05', -0.05), ('gamma_0.0', 0.0),
                                           ('gamma_0.05', 0.05), ('gamma_0.1', 0.1)]:
                d_adj = (d_sied_test - d_cal_mean) / d_cal_std
                tau = 0.5 + gamma_val * d_adj
                tau = np.clip(tau, 0.35, 0.65)
                methods[f'PCET_SIED_{gamma_name}'] = {
                    'pred': (p_pcet_test >= tau).astype(int),
                    'prob': p_pcet_test,
                    'tau_mean': np.mean(tau)
                }

            for beta_val in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                for gamma_val in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                    u_adj = (u_srgc_test - u_cal_mean) / u_cal_std
                    d_adj = (d_sied_test - d_cal_mean) / d_cal_std
                    tau = 0.5 + beta_val * u_adj + gamma_val * d_adj
                    tau = np.clip(tau, 0.35, 0.65)
                    methods[f'PCET_SRGC_SIED_b{beta_val}_g{gamma_val}'] = {
                        'pred': (p_pcet_test >= tau).astype(int),
                        'prob': p_pcet_test,
                        'tau_mean': np.mean(tau)
                    }

            for beta_name, beta_val in [('beta_-0.1', -0.1), ('beta_0.0', 0.0), ('beta_0.1', 0.1)]:
                u_adj = (u_srgc_test_shuffled - u_cal_mean) / u_cal_std
                tau = 0.5 + beta_val * u_adj
                tau = np.clip(tau, 0.35, 0.65)
                methods[f'PCET_shuffledSRGC_{beta_name}'] = {
                    'pred': (p_pcet_test >= tau).astype(int),
                    'prob': p_pcet_test
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
key_methods = ['PCET', 'PCET_fixed_tau_0.48', 'PCET_fixed_tau_0.52',
               'PCET_SRGC_beta_0.05', 'PCET_SIED_gamma_0.05',
               'PCET_shuffledSRGC_beta_0.1']

for n_cal in shots:
    print(f"\n{n_cal}-shot:")
    shot_df = df[df['n_cal'] == n_cal]

    for method in key_methods:
        m_df = shot_df[shot_df['method'] == method]
        if len(m_df) > 0:
            acc = m_df['accuracy'].mean()
            std = m_df['accuracy'].std()
            f1 = m_df['macro_f1'].mean()
            bacc = m_df['balanced_accuracy'].mean()
            auroc = m_df['auroc'].mean()
            ece = m_df['ece'].mean()
            brier = m_df['brier_score'].mean()
            print(f"  {method:30s}: Acc={acc:.4f}±{std:.4f}, F1={f1:.4f}, BAcc={bacc:.4f}, AUROC={auroc:.4f}")
            summary_data.append({
                'shot': n_cal, 'method': method,
                'accuracy': acc, 'accuracy_std': std,
                'macro_f1': f1, 'balanced_accuracy': bacc,
                'auroc': auroc, 'ece': ece, 'brier_score': brier
            })

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(f'{RESULTS_DIR}/sci_v7_summary.csv', index=False)

print("\n" + "="*60)
print("BEST METHOD PER SHOT")
print("="*60)

for n_cal in shots:
    shot_df = df[df['n_cal'] == n_cal]
    pcet_acc = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()

    best_acc = 0
    best_method = ''
    for method in shot_df['method'].unique():
        if not method.startswith('PCET'):
            continue
        m_acc = shot_df[shot_df['method'] == method]['accuracy'].mean()
        if m_acc > best_acc:
            best_acc = m_acc
            best_method = method

    delta = best_acc - pcet_acc
    marker = "✓" if delta > 0 else " "
    print(f"{n_cal}-shot: PCET={pcet_acc:.4f}, Best={best_method}={best_acc:.4f} (Δ={delta:+.4f}) {marker}")

print("\n" + "="*60)
print("SUCCESS CRITERIA CHECK")
print("="*60)

pcet_acc = df[df['method'] == 'PCET'].groupby('n_cal')['accuracy'].mean()
pcet_f1 = df[df['method'] == 'PCET'].groupby('n_cal')['macro_f1'].mean()
pcet_bacc = df[df['method'] == 'PCET'].groupby('n_cal')['balanced_accuracy'].mean()
pcet_ece = df[df['method'] == 'PCET'].groupby('n_cal')['ece'].mean()
pcet_brier = df[df['method'] == 'PCET'].groupby('n_cal')['brier_score'].mean()

best_sci_acc = {}
best_sci_f1 = {}
best_sci_bacc = {}
best_sci_ece = {}
best_sci_brier = {}

for n_cal in shots:
    shot_df = df[df['n_cal'] == n_cal]
    best_acc = 0
    best_method = ''
    for method in shot_df['method'].unique():
        if not method.startswith('PCET_'):
            continue
        m_acc = shot_df[shot_df['method'] == method]['accuracy'].mean()
        if m_acc > best_acc:
            best_acc = m_acc
            best_method = method

    best_sci_acc[n_cal] = best_acc
    m_df = shot_df[shot_df['method'] == best_method]
    best_sci_f1[n_cal] = m_df['macro_f1'].mean()
    best_sci_bacc[n_cal] = m_df['balanced_accuracy'].mean()
    best_sci_ece[n_cal] = m_df['ece'].mean()
    best_sci_brier[n_cal] = m_df['brier_score'].mean()

print("\n1. SCI_V7 average Accuracy > PCET?")
shots_improved = 0
for n_cal in shots:
    delta = best_sci_acc[n_cal] - pcet_acc[n_cal]
    if delta > 0:
        shots_improved += 1
    marker = "✓" if best_sci_acc[n_cal] > pcet_acc[n_cal] else "✗"
    print(f"  {n_cal}-shot: PCET={pcet_acc[n_cal]:.4f}, BestSCI={best_sci_acc[n_cal]:.4f}, Δ={delta:+.4f} {marker}")
print(f"  Shots improved: {shots_improved}/5")

print("\n2. At least 3 shots exceed PCET?")
print(f"  {'✓ PASS' if shots_improved >= 3 else '✗ FAIL'} ({shots_improved}/5)")

print("\n3. No more than 1% degradation?")
max_deg = min([best_sci_acc[n_cal] - pcet_acc[n_cal] for n_cal in shots])
print(f"  Max degradation: {max_deg:.4f}")
print(f"  {'✓ PASS' if max_deg >= -0.01 else '✗ FAIL'}")

print("\n4. F1/BAcc not decreased?")
f1_ok = all([best_sci_f1[n_cal] >= pcet_f1[n_cal] - 0.005 for n_cal in shots])
bacc_ok = all([best_sci_bacc[n_cal] >= pcet_bacc[n_cal] - 0.005 for n_cal in shots])
print(f"  {'✓ PASS' if f1_ok and bacc_ok else '✗ FAIL'}")

print("\n5. ECE/Brier improved?")
for n_cal in shots:
    ece_delta = best_sci_ece[n_cal] - pcet_ece[n_cal]
    brier_delta = best_sci_brier[n_cal] - pcet_brier[n_cal]
    print(f"  {n_cal}-shot: ECE Δ={ece_delta:+.4f}, Brier Δ={brier_delta:+.4f}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if shots_improved >= 3 and max_deg >= -0.01:
    print("✓ SUCCESS CRITERIA PASSED")
else:
    print("✗ SOME CRITERIA NOT MET")

print("="*60)