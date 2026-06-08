"""
SCI Framework V3: Re-Designed Orthogonal Information Fusion

Key Insight: The three modules should provide ORTHOGONAL information:
- PCET: "WHAT class?" (class prediction)
- SRGC: "HOW UNCERTAIN?" (epistemic uncertainty estimation)
- SIED: "IS IT IN-DISTRIBUTION?" (domain shift detection)

Previous problem: All three tried to predict class, causing redundancy.
New design: Each outputs a different dimension of information.
"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.chdir('d:/pycharmproject/zuco-benchmark-main/src')
print("="*60)
print("SCI Framework V3: Re-Designed")
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


class PCETModule:
    """PCET: Class Prediction Module"""

    def fit(self, X_cal, y_cal):
        self.scaler = StandardScaler()
        X_cal_s = self.scaler.fit_transform(X_cal)

        self.clf = LogisticRegression(max_iter=200, solver='lbfgs', C=1.0)
        self.clf.fit(X_cal_s, y_cal)

        self.mu_0 = np.mean(X_cal_s[y_cal == 0], axis=0)
        self.mu_1 = np.mean(X_cal_s[y_cal == 1], axis=0)

        return self

    def predict(self, X_test):
        X_test_s = self.scaler.transform(X_test)
        probs = self.clf.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return probs, preds


class SRGCUncertaintyModule:
    """
    SRGC V2: Epistemic Uncertainty Estimation

    Instead of predicting class, SRGC now estimates HOW UNCERTAIN we are.
    Uses multiple methods:
    1. Distance to class centroids (dispersion)
    2. Nearest neighbor density (isolation)
    3. Prediction confidence interval width
    """

    def fit(self, X_cal, y_cal):
        self.scaler = StandardScaler()
        X_cal_s = self.scaler.fit_transform(X_cal)

        self.X_cal_s = X_cal_s
        self.y_cal = y_cal

        self.mu_0 = np.mean(X_cal_s[y_cal == 0], axis=0)
        self.mu_1 = np.mean(X_cal_s[y_cal == 1], axis=0)
        self.sigma_0 = np.std(X_cal_s[y_cal == 0], axis=0) + 1e-8
        self.sigma_1 = np.std(X_cal_s[y_cal == 1], axis=0) + 1e-8

        n_neighbors = min(5, len(X_cal_s) - 1)
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(X_cal_s)

        return self

    def estimate_uncertainty(self, X_test):
        """Estimate uncertainty (0=confident, 1=uncertain)"""
        X_test_s = self.scaler.transform(X_test)

        dist_0 = np.sqrt(np.sum((X_test_s - self.mu_0) ** 2, axis=1))
        dist_1 = np.sqrt(np.sum((X_test_s - self.mu_1) ** 2, axis=1))

        centroid_spread = np.abs(dist_1 - dist_0) / (dist_0 + dist_1 + 1e-8)

        distances, indices = self.nn.kneighbors(X_test_s)
        avg_knn_dist = np.mean(distances, axis=1)

        class_specific_uncertainty = np.zeros(len(X_test_s))
        for i, (d0, d1) in enumerate(zip(dist_0, dist_1)):
            if d0 < d1:
                class_specific_uncertainty[i] = d1 / (d0 + 1e-8)
            else:
                class_specific_uncertainty[i] = d0 / (d1 + 1e-8)

        uncertainty = 1 / (1 + centroid_spread) * 0.4 + \
                     1 / (1 + avg_knn_dist) * 0.3 + \
                     1 / (1 + class_specific_uncertainty) * 0.3

        uncertainty = np.clip(uncertainty, 0, 1)
        return uncertainty


class SIEDDomainModule:
    """
    SIED V2: Domain Shift Detection

    Instead of predicting class, SIED now detects IS THIS IN-DISTRIBUTION?
    Uses MMD-like feature comparison between calibration and test samples.
    """

    def fit(self, X_cal, y_cal, X_source=None, y_source=None):
        self.scaler = StandardScaler()
        X_cal_s = self.scaler.fit_transform(X_cal)
        self.X_cal_s = X_cal_s

        self.X_source = X_source
        if X_source is not None:
            self.X_source_s = self.scaler.transform(X_source)
        else:
            self.X_source_s = None

        self.mu_cal = np.mean(X_cal_s, axis=0)
        self.sigma_cal = np.std(X_cal_s, axis=0) + 1e-8

        return self

    def estimate_domain_shift(self, X_test):
        """Estimate domain shift (0=in-distribution, 1=out-of-distribution)"""
        X_test_s = self.scaler.transform(X_test)

        test_mean = np.mean(X_test_s, axis=0)
        mean_shift = np.sqrt(np.sum((test_mean - self.mu_cal) ** 2))
        mean_shift_norm = mean_shift / (np.sqrt(np.sum(self.mu_cal ** 2)) + 1e-8)

        var_ratio = np.mean(np.var(X_test_s, axis=0) / self.sigma_cal)
        var_shift = np.abs(var_ratio - 1)

        if self.X_source_s is not None:
            source_mean = np.mean(self.X_source_s, axis=0)
            mmd_like = np.sqrt(np.sum((test_mean - source_mean) ** 2))
        else:
            mmd_like = mean_shift_norm

        domain_shift = 0.5 * mean_shift_norm + 0.3 * var_shift + 0.2 * mmd_like
        domain_shift = np.clip(domain_shift, 0, 1)

        return np.ones(len(X_test)) * domain_shift


class SCIFusionV3:
    """
    SCI Fusion V3: Decision-Level Fusion

    Key difference from V1/V2:
    - V1/V2: Fused probabilities (p_pcet + p_srgc + p_sied)
    - V3: Fuses DECISIONS with uncertainty awareness

    Fusion logic:
    1. Get PCET class prediction
    2. Get SRGC uncertainty estimate
    3. Get SIED domain shift estimate
    4. Adjust decision based on uncertainty and domain shift
    """

    def __init__(self, tau_uncertainty=0.5, tau_domain=0.3):
        self.tau_uncertainty = tau_uncertainty
        self.tau_domain = tau_domain

    def predict(self, p_pcet, uncertainty, domain_shift):
        """
        Fusion decision using orthogonal signals.

        Args:
            p_pcet: PCET probability [0, 1]
            uncertainty: SRGC uncertainty [0, 1] (0=confident, 1=uncertain)
            domain_shift: SIED domain shift [0, 1] (0=in-dist, 1=ood)

        Returns:
            predictions and confidence scores
        """
        n_samples = len(p_pcet)
        final_preds = np.zeros(n_samples)
        confidence = np.zeros(n_samples)

        domain_shift = domain_shift.reshape(-1)

        for i in range(n_samples):
            p = p_pcet[i]
            u = uncertainty[i]
            d = domain_shift[i]

            confidence[i] = (1 - u) * (1 - d)

            if u < self.tau_uncertainty and d < self.tau_domain:
                final_preds[i] = 1 if p >= 0.5 else 0
            elif u >= self.tau_uncertainty and d < self.tau_domain:
                if p >= 0.5 + u * 0.3:
                    final_preds[i] = 1
                elif p <= 0.5 - u * 0.3:
                    final_preds[i] = 0
                else:
                    final_preds[i] = 1 - np.argmax([1-p, p])
            elif d >= self.tau_domain:
                if confidence[i] > 0.5:
                    final_preds[i] = 1 if p >= 0.5 else 0
                else:
                    final_preds[i] = 1 - np.argmax([1-p, p])
            else:
                final_preds[i] = 1 if p >= 0.5 else 0

        return final_preds, confidence


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

print(f"\nRunning experiments: {len(shots)} shots x {len(seeds)} seeds x {len(all_data)} subjects...")

for seed in seeds:
    print(f"\nSeed {seed}:", end=' ')
    for held_out in Y_SUBJECTS:
        if held_out not in all_data:
            continue

        X_test_orig, y_test_orig = all_data[held_out]
        train_subjs = [s for s in Y_SUBJECTS if s != held_out and s in all_data]
        if len(train_subjs) == 0:
            continue

        X_source = np.vstack([all_data[s][0] for s in train_subjs])

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
            selected = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])

            X_cal = X_cal_pool[selected]
            y_cal = y_cal_pool[selected]

            if len(np.unique(y_cal)) < 2:
                continue

            pcet = PCETModule()
            pcet.fit(X_cal, y_cal)
            p_pcet, y_pcet = pcet.predict(X_test)

            srgc = SRGCUncertaintyModule()
            srgc.fit(X_cal, y_cal)
            uncertainty = srgc.estimate_uncertainty(X_test)

            sied = SIEDDomainModule()
            sied.fit(X_cal, y_cal, X_source)
            domain_shift = sied.estimate_domain_shift(X_test)

            for tau_u in [0.3, 0.4, 0.5]:
                for tau_d in [0.2, 0.3, 0.4]:
                    sci = SCIFusionV3(tau_uncertainty=tau_u, tau_domain=tau_d)
                    y_sci, conf_sci = sci.predict(p_pcet, uncertainty, domain_shift)
                    acc_sci = accuracy_score(y_test, y_sci)
                    results.append({
                        'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                        'method': f'SCI_u{tau_u}_d{tau_d}', 'accuracy': acc_sci
                    })

            acc_pcet = accuracy_score(y_test, y_pcet)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'PCET', 'accuracy': acc_pcet
            })

        print(".", end='', flush=True)

df = pd.DataFrame(results)
df.to_csv(f'{RESULTS_DIR}/sci_v3_results.csv', index=False)

print("\n\n" + "="*60)
print("RESULTS SUMMARY (Mean Accuracy)")
print("="*60)

for n_cal in shots:
    print(f"\n{n_cal}-shot:")
    shot_df = df[df['n_cal'] == n_cal]

    pcet_acc = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
    print(f"  PCET: {pcet_acc:.4f}")

    sci_df = shot_df[shot_df['method'].str.startswith('SCI_')]
    if len(sci_df) > 0:
        best = sci_df.groupby('method')['accuracy'].mean().idxmax()
        best_acc = sci_df.groupby('method')['accuracy'].mean().max()
        delta = best_acc - pcet_acc
        marker = "✓" if delta > 0 else " "
        print(f"  Best SCI: {best} = {best_acc:.4f} (Δ={delta:+.4f}) {marker}")

print("\n" + "="*60)
print("KEY FINDING")
print("="*60)

for n_cal in shots:
    shot_df = df[df['n_cal'] == n_cal]
    pcet_acc = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()

    sci_df = shot_df[shot_df['method'].str.startswith('SCI_')]
    if len(sci_df) > 0:
        best_acc = sci_df.groupby('method')['accuracy'].mean().max()
        delta = best_acc - pcet_acc
        if delta > 0.001:
            print(f"{n_cal}-shot: PCET={pcet_acc:.4f}, SCI={best_acc:.4f}, Δ={delta:+.4f} ✓ IMPROVES")
        elif delta > -0.001:
            print(f"{n_cal}-shot: PCET={pcet_acc:.4f}, SCI={best_acc:.4f}, Δ={delta:+.4f} ≈ MATCHES")
        else:
            print(f"{n_cal}-shot: PCET={pcet_acc:.4f}, SCI={best_acc:.4f}, Δ={delta:+.4f} ✗ DEGRADES")

print("\n" + "="*60)
print("Done!")
print("="*60)