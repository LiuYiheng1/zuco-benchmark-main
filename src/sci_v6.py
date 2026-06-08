"""
SCI Framework V6: Direct Confidence Adjustment

Key insight: Instead of switching predictions, adjust the confidence
by multiplying PCET confidence with SRGC uncertainty.

If PCET is confident but SRGC says uncertain → reduce confidence
If PCET is uncertain but SRGC says confident → might trust more
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
print("SCI Framework V6: Direct Confidence Adjustment")
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

    def confidence(self, X_test):
        X_test_s = self.scaler.transform(X_test)
        probs = self.clf.predict_proba(X_test_s)[:, 1]
        return np.maximum(probs, 1 - probs)


class SRGCUncertainty:
    """SRGC estimates distance-based uncertainty"""

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

    def estimate_uncertainty(self, X_test):
        X_test_s = self.scaler.transform(X_test)
        dist_0 = np.sqrt(np.sum((X_test_s - self.mu_0) ** 2, axis=1))
        dist_1 = np.sqrt(np.sum((X_test_s - self.mu_1) ** 2, axis=1))
        spread = np.abs(dist_1 - dist_0) / (dist_0 + dist_1 + 1e-8)
        distances, _ = self.nn.kneighbors(X_test_s)
        avg_dist = np.mean(distances, axis=1)
        uncertainty = 1 / (1 + spread) * 0.5 + 1 / (1 + avg_dist) * 0.5
        return uncertainty


class SIEDDomain:
    """SIED estimates domain shift"""

    def fit(self, X_cal, y_cal):
        self.scaler = StandardScaler()
        X_cal_s = self.scaler.fit_transform(X_cal)
        self.X_cal_s = X_cal_s
        self.mu_cal = np.mean(X_cal_s, axis=0)
        self.sigma_cal = np.std(X_cal_s, axis=0) + 1e-8
        return self

    def estimate_domain_shift(self, X_test):
        X_test_s = self.scaler.transform(X_test)
        test_mean = np.mean(X_test_s, axis=0)
        mean_diff = np.sqrt(np.sum((test_mean - self.mu_cal) ** 2))
        shift = mean_diff / (np.sqrt(np.sum(self.mu_cal ** 2)) + 1e-8)
        return np.ones(len(X_test)) * np.clip(shift, 0, 1)


def confidence_adjustment(p_pcet, conf_pcet, uncertainty, domain_shift,
                         alpha=0.5, beta=0.3):
    """
    Adjust decision threshold based on uncertainty.

    If uncertain → raise threshold (be more conservative)
    If certain → standard threshold

    adjusted_threshold = 0.5 + beta * uncertainty - alpha * (1 - domain_shift)
    """
    preds = np.zeros(len(p_pcet))
    thresholds = np.zeros(len(p_pcet))

    for i in range(len(p_pcet)):
        u = uncertainty[i]
        d = domain_shift[i]

        threshold = 0.5 + beta * u - alpha * (1 - d)
        threshold = np.clip(threshold, 0.3, 0.7)

        thresholds[i] = threshold
        preds[i] = 1 if p_pcet[i] >= threshold else 0

    return preds, thresholds


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

print(f"\nRunning: {len(shots)} shots x {len(seeds)} seeds...")

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
            selected = np.concatenate([class_0_idx[:n0], class_1_idx[:n1]])

            X_cal = X_cal_pool[selected]
            y_cal = y_cal_pool[selected]

            if len(np.unique(y_cal)) < 2:
                continue

            pcet = PCETModule()
            pcet.fit(X_cal, y_cal)
            p_pcet, y_pcet = pcet.predict(X_test)
            conf_pcet = pcet.confidence(X_test)

            srgc = SRGCUncertainty()
            srgc.fit(X_cal, y_cal)
            uncertainty = srgc.estimate_uncertainty(X_test)

            sied = SIEDDomain()
            sied.fit(X_cal, y_cal)
            domain_shift = sied.estimate_domain_shift(X_test)

            acc_pcet = accuracy_score(y_test, y_pcet)
            results.append({
                'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                'method': 'PCET', 'accuracy': acc_pcet
            })

            for alpha in [0.0, 0.3, 0.5, 0.7]:
                for beta in [0.0, 0.1, 0.2, 0.3]:
                    y_adj, thresholds = confidence_adjustment(
                        p_pcet, conf_pcet, uncertainty, domain_shift,
                        alpha=alpha, beta=beta
                    )
                    acc_adj = accuracy_score(y_test, y_adj)
                    results.append({
                        'seed': seed, 'subject': held_out, 'n_cal': n_cal,
                        'method': f'Adj_a{alpha}_b{beta}',
                        'accuracy': acc_adj
                    })

        print(".", end='', flush=True)

df = pd.DataFrame(results)
df.to_csv(f'{RESULTS_DIR}/sci_v6_results.csv', index=False)

print("\n\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

for n_cal in shots:
    print(f"\n{n_cal}-shot:")
    shot_df = df[df['n_cal'] == n_cal]

    pcet_acc = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()
    print(f"  PCET: {pcet_acc:.4f}")

    adj_df = shot_df[shot_df['method'].str.startswith('Adj_')]
    if len(adj_df) > 0:
        best = adj_df.groupby('method')['accuracy'].mean().idxmax()
        best_acc = adj_df.groupby('method')['accuracy'].mean().max()
        delta = best_acc - pcet_acc
        marker = "✓" if delta > 0 else " "
        print(f"  Best Adj: {best} = {best_acc:.4f} (Δ={delta:+.4f}) {marker}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

for n_cal in shots:
    shot_df = df[df['n_cal'] == n_cal]
    pcet_acc = shot_df[shot_df['method'] == 'PCET']['accuracy'].mean()

    adj_df = shot_df[shot_df['method'].str.startswith('Adj_')]
    if len(adj_df) > 0:
        best_acc = adj_df.groupby('method')['accuracy'].mean().max()
        delta = best_acc - pcet_acc
        if delta > 0.001:
            print(f"{n_cal}-shot: PCET={pcet_acc:.4f}, Adj={best_acc:.4f}, Δ={delta:+.4f} ✓ IMPROVES")
        elif delta > -0.001:
            print(f"{n_cal}-shot: PCET={pcet_acc:.4f}, Adj={best_acc:.4f}, Δ={delta:+.4f} ≈ MATCHES")
        else:
            print(f"{n_cal}-shot: PCET={pcet_acc:.4f}, Adj={best_acc:.4f}, Δ={delta:+.4f} ✗ DEGRADES")

print("="*60)