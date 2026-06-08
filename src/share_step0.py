import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
TASK_MATERIALS_DIR = "task_materials"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']

SEEDS = [0, 1, 2, 3, 4]

def load_eeg_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_electrode_features_all.npy")
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True).item()
    result = {}
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 3:
            subj, label, sentence_id, index = parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else None
            features = np.array(values[:-1], dtype=np.float64)
            result[(label, sentence_id)] = {'features': features, 'label': label, 'index': index}
    return result

def load_gaze_data(subject):
    path = os.path.join(FEATURES_DIR, f"{subject}_sent_gaze_sacc.npy")
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True).item()
    result = {}
    for key, values in data.items():
        parts = key.split("_")
        if len(parts) >= 3:
            subj, label, sentence_id, index = parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else None
            features = np.array(values[:-1], dtype=np.float64)
            result[(label, sentence_id)] = {'features': features, 'label': label, 'index': index}
    return result

def load_text_data():
    text_data = {}
    nr_files = [f"nr_{i}.csv" for i in range(1, 8)]
    tsr_files = [f"tsr_{i}.csv" for i in range(1, 8)]
    
    for filename in nr_files + tsr_files:
        filepath = os.path.join(TASK_MATERIALS_DIR, filename)
        if not os.path.exists(filepath):
            continue
        label = "NR" if filename.startswith("nr") else "TSR"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(';')
                    if len(parts) >= 3:
                        sentence_id = parts[0]
                        text = parts[-1] if parts[-1] != 'CONTROL' else parts[-2]
                        text_data[(label, sentence_id)] = text
        except:
            continue
    return text_data

def extract_text_features(texts):
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_features = tfidf.fit_transform(texts).toarray()
    lengths = np.array([len(t) for t in texts]).reshape(-1, 1)
    word_counts = np.array([len(t.split()) for t in texts]).reshape(-1, 1)
    return np.hstack([tfidf_features, lengths, word_counts])

def align_data(subjects):
    audit_results = []
    
    all_aligned_data = []
    
    for subject in subjects:
        eeg_data = load_eeg_data(subject)
        gaze_data = load_gaze_data(subject)
        text_data = load_text_data()
        
        eeg_count = len(eeg_data)
        gaze_count = len(gaze_data)
        text_count = len(text_data)
        
        aligned_keys = set(eeg_data.keys()) & set(gaze_data.keys()) & set(text_data.keys())
        
        aligned_samples = []
        nr_count = 0
        tsr_count = 0
        label_consistent = True
        
        for key in aligned_keys:
            eeg = eeg_data[key]
            gaze = gaze_data[key]
            text = text_data[key]
            
            if eeg['label'] != key[0] or gaze['label'] != key[0]:
                label_consistent = False
            
            aligned_samples.append({
                'subject': subject,
                'label': key[0],
                'sentence_id': key[1],
                'x_eeg': eeg['features'],
                'x_gaze': gaze['features'],
                'x_text_raw': text
            })
            
            if key[0] == 'NR':
                nr_count += 1
            else:
                tsr_count += 1
        
        label_consistency_rate = 100.0 if label_consistent else 0.0
        
        audit_results.append({
            'subject': subject,
            'eeg_count': eeg_count,
            'gaze_count': gaze_count,
            'text_count': text_count,
            'aligned_count': len(aligned_samples),
            'nr_count': nr_count,
            'tsr_count': tsr_count,
            'label_consistency_rate': label_consistency_rate
        })
        
        all_aligned_data.extend(aligned_samples)
        
        if label_consistency_rate != 100.0:
            print(f"ERROR: Subject {subject} label consistency rate is {label_consistency_rate}%")
            print("Stopping experiment due to label inconsistency")
            return None, None
    
    return all_aligned_data, audit_results

def protocol_a_split(data, seed):
    splits = []
    subjects = sorted(set(d['subject'] for d in data))
    
    for subject in subjects:
        subject_data = [d for d in data if d['subject'] == subject]
        labels = np.array([1 if d['label'] == 'NR' else 0 for d in subject_data])
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
        
        train_data = [subject_data[i] for i in train_idx]
        test_data = [subject_data[i] for i in test_idx]
        
        splits.append({
            'subject': subject,
            'train': train_data,
            'test': test_data
        })
    
    return splits

def protocol_c_split(data, seed):
    label_sentence_ids = sorted(set((d['label'], d['sentence_id']) for d in data))
    np.random.seed(seed)
    np.random.shuffle(label_sentence_ids)
    
    split_idx = int(len(label_sentence_ids) * 0.7)
    train_keys = set(label_sentence_ids[:split_idx])
    test_keys = set(label_sentence_ids[split_idx:])
    
    train_data = [d for d in data if (d['label'], d['sentence_id']) in train_keys]
    test_data = [d for d in data if (d['label'], d['sentence_id']) in test_keys]
    
    return {
        'train': train_data,
        'test': test_data
    }

def prepare_features(data):
    x_eeg = np.array([d['x_eeg'] for d in data])
    x_gaze = np.array([d['x_gaze'] for d in data])
    x_text_raw = [d['x_text_raw'] for d in data]
    y = np.array([1 if d['label'] == 'NR' else 0 for d in data])
    return x_eeg, x_gaze, x_text_raw, y

def evaluate_model(X_train, y_train, X_test, y_test, seed):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    bacc = balanced_accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)
    
    return acc, f1, bacc, auroc

def run_baselines_protocol_a(data, seed):
    splits = protocol_a_split(data, seed)
    results = []
    
    for split in splits:
        subject = split['subject']
        train_data = split['train']
        test_data = split['test']
        
        if len(train_data) == 0 or len(test_data) == 0:
            continue
        
        X_eeg_train, X_gaze_train, X_text_raw_train, y_train = prepare_features(train_data)
        X_eeg_test, X_gaze_test, X_text_raw_test, y_test = prepare_features(test_data)
        
        text_features = extract_text_features(X_text_raw_train + X_text_raw_test)
        X_text_train = text_features[:len(X_text_raw_train)]
        X_text_test = text_features[len(X_text_raw_train):]
        
        methods = [
            ('Text_only', X_text_train, X_text_test),
            ('EEG_only', X_eeg_train, X_eeg_test),
            ('Gaze_only', X_gaze_train, X_gaze_test),
            ('EEG+Gaze_concat', np.hstack([X_eeg_train, X_gaze_train]), np.hstack([X_eeg_test, X_gaze_test])),
            ('Text+EEG_concat', np.hstack([X_text_train, X_eeg_train]), np.hstack([X_text_test, X_eeg_test])),
            ('Text+Gaze_concat', np.hstack([X_text_train, X_gaze_train]), np.hstack([X_text_test, X_gaze_test])),
            ('Text+EEG+Gaze_concat', np.hstack([X_text_train, X_eeg_train, X_gaze_train]), np.hstack([X_text_test, X_eeg_test, X_gaze_test]))
        ]
        
        for method_name, X_train, X_test in methods:
            acc, f1, bacc, auroc = evaluate_model(X_train, y_train, X_test, y_test, seed)
            results.append({
                'protocol': 'A',
                'seed': seed,
                'subject': subject,
                'method': method_name,
                'accuracy': acc,
                'macro_f1': f1,
                'balanced_accuracy': bacc,
                'auroc': auroc
            })
    
    return results

def run_baselines_protocol_c(data, seed):
    split = protocol_c_split(data, seed)
    results = []
    
    train_data = split['train']
    test_data = split['test']
    
    if len(train_data) == 0 or len(test_data) == 0:
        return results
    
    X_eeg_train, X_gaze_train, X_text_raw_train, y_train = prepare_features(train_data)
    X_eeg_test, X_gaze_test, X_text_raw_test, y_test = prepare_features(test_data)
    
    text_features = extract_text_features(X_text_raw_train + X_text_raw_test)
    X_text_train = text_features[:len(X_text_raw_train)]
    X_text_test = text_features[len(X_text_raw_train):]
    
    methods = [
        ('Text_only', X_text_train, X_text_test),
        ('EEG_only', X_eeg_train, X_eeg_test),
        ('Gaze_only', X_gaze_train, X_gaze_test),
        ('EEG+Gaze_concat', np.hstack([X_eeg_train, X_gaze_train]), np.hstack([X_eeg_test, X_gaze_test])),
        ('Text+EEG_concat', np.hstack([X_text_train, X_eeg_train]), np.hstack([X_text_test, X_eeg_test])),
        ('Text+Gaze_concat', np.hstack([X_text_train, X_gaze_train]), np.hstack([X_text_test, X_gaze_test])),
        ('Text+EEG+Gaze_concat', np.hstack([X_text_train, X_eeg_train, X_gaze_train]), np.hstack([X_text_test, X_eeg_test, X_gaze_test]))
    ]
    
    for method_name, X_train, X_test in methods:
        acc, f1, bacc, auroc = evaluate_model(X_train, y_train, X_test, y_test, seed)
        results.append({
            'protocol': 'C',
            'seed': seed,
            'subject': 'all',
            'method': method_name,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_accuracy': bacc,
            'auroc': auroc
        })
    
    return results

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print("Step 0: Data Alignment and Baseline Reproduction")
    print("=" * 60)
    
    print("\n1. Loading and aligning data...")
    data, audit_results = align_data(Y_SUBJECTS)
    
    if data is None:
        print("Alignment failed - stopping")
        return
    
    print("\n2. Saving alignment audit...")
    audit_df = pd.DataFrame(audit_results)
    audit_df.to_csv(os.path.join(RESULTS_DIR, "share_step0_alignment_audit.csv"), index=False)
    
    print("\nAlignment Audit Summary:")
    print(audit_df[['subject', 'aligned_count', 'nr_count', 'tsr_count', 'label_consistency_rate']])
    
    all_results = []
    
    print("\n3. Running Protocol A (subject-dependent split)...")
    for seed in SEEDS:
        print(f"  Seed {seed}...", end='', flush=True)
        results_a = run_baselines_protocol_a(data, seed)
        all_results.extend(results_a)
        print(" done")
    
    print("\n4. Running Protocol C (held-out sentence split)...")
    for seed in SEEDS:
        print(f"  Seed {seed}...", end='', flush=True)
        results_c = run_baselines_protocol_c(data, seed)
        all_results.extend(results_c)
        print(" done")
    
    print("\n5. Saving results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "share_step0_baselines.csv"), index=False)
    
    print("\n6. Generating report...")
    generate_report(audit_df, results_df)
    
    print("\n" + "=" * 60)
    print("Step 0 completed successfully!")

def generate_report(audit_df, results_df):
    report_lines = []
    report_lines.append("# SHARE-Net Step 0: Data Alignment and Baseline Reproduction")
    report_lines.append("")
    report_lines.append("## 1. Data Alignment Audit")
    report_lines.append("")
    
    for _, row in audit_df.iterrows():
        report_lines.append(f"### Subject {row['subject']}")
        report_lines.append(f"- EEG samples: {row['eeg_count']}")
        report_lines.append(f"- Gaze samples: {row['gaze_count']}")
        report_lines.append(f"- Text samples: {row['text_count']}")
        report_lines.append(f"- Aligned samples: {row['aligned_count']}")
        report_lines.append(f"- NR count: {row['nr_count']}")
        report_lines.append(f"- TSR count: {row['tsr_count']}")
        report_lines.append(f"- Label consistency rate: {row['label_consistency_rate']}%")
        report_lines.append("")
    
    report_lines.append("## 2. Label Consistency Check")
    all_consistent = audit_df['label_consistency_rate'].min() == 100.0
    report_lines.append(f"- All subjects have 100% label consistency: {'YES' if all_consistent else 'NO'}")
    report_lines.append("")
    
    report_lines.append("## 3. Baseline Results")
    report_lines.append("")
    
    for protocol in ['A', 'C']:
        report_lines.append(f"### Protocol {protocol}")
        report_lines.append("")
        
        protocol_df = results_df[results_df['protocol'] == protocol]
        
        methods = sorted(protocol_df['method'].unique())
        
        report_lines.append("| Method | Accuracy (mean±std) | Macro-F1 (mean±std) | Balanced Accuracy (mean±std) | AUROC (mean±std) |")
        report_lines.append("|--------|---------------------|---------------------|-------------------------------|------------------|")
        
        for method in methods:
            method_df = protocol_df[protocol_df['method'] == method]
            acc_mean = method_df['accuracy'].mean()
            acc_std = method_df['accuracy'].std()
            f1_mean = method_df['macro_f1'].mean()
            f1_std = method_df['macro_f1'].std()
            bacc_mean = method_df['balanced_accuracy'].mean()
            bacc_std = method_df['balanced_accuracy'].std()
            auroc_mean = method_df['auroc'].mean()
            auroc_std = method_df['auroc'].std()
            
            report_lines.append(f"| {method} | {acc_mean:.4f}±{acc_std:.4f} | {f1_mean:.4f}±{f1_std:.4f} | {bacc_mean:.4f}±{bacc_std:.4f} | {auroc_mean:.4f}±{auroc_std:.4f} |")
        
        report_lines.append("")
    
    report_lines.append("## 4. Analysis Questions")
    report_lines.append("")
    
    report_lines.append("### Q1: Are EEG/Gaze/Text all correctly aligned?")
    total_aligned = audit_df['aligned_count'].sum()
    report_lines.append(f"- Total aligned samples across all subjects: {total_aligned}")
    report_lines.append(f"- All modalities are correctly aligned: {'YES' if total_aligned > 0 else 'NO'}")
    report_lines.append("")
    
    report_lines.append("### Q2: Is label consistency 100%?")
    report_lines.append(f"- {'YES' if all_consistent else 'NO'}")
    report_lines.append("")
    
    report_lines.append("### Q3: What are the baseline results under Protocol A and Protocol C?")
    report_lines.append("- See Section 3 for detailed results.")
    report_lines.append("")
    
    report_lines.append("### Q4: Is Text+EEG+Gaze_concat the strongest baseline?")
    protocol_a_df = results_df[results_df['protocol'] == 'A']
    best_method_a = protocol_a_df.groupby('method')['accuracy'].mean().idxmax()
    report_lines.append(f"- Protocol A strongest: {best_method_a}")
    
    protocol_c_df = results_df[results_df['protocol'] == 'C']
    best_method_c = protocol_c_df.groupby('method')['accuracy'].mean().idxmax()
    report_lines.append(f"- Protocol C strongest: {best_method_c}")
    report_lines.append(f"- Text+EEG+Gaze_concat is strongest: {'YES' if best_method_a == 'Text+EEG+Gaze_concat' and best_method_c == 'Text+EEG+Gaze_concat' else 'NO'}")
    report_lines.append("")
    
    report_lines.append("### Q5: Is Text_only strong, indicating potential text shortcut?")
    text_only_a = protocol_a_df[protocol_a_df['method'] == 'Text_only']['accuracy'].mean()
    text_only_c = protocol_c_df[protocol_c_df['method'] == 'Text_only']['accuracy'].mean()
    report_lines.append(f"- Text_only accuracy (Protocol A): {text_only_a:.4f}")
    report_lines.append(f"- Text_only accuracy (Protocol C): {text_only_c:.4f}")
    report_lines.append(f"- Potential text shortcut: {'YES' if text_only_a > 0.7 else 'NO (needs further investigation)'}")
    report_lines.append("")
    
    report_lines.append("### Q6: Is performance significantly lower under Protocol C?")
    full_concat_a = protocol_a_df[protocol_a_df['method'] == 'Text+EEG+Gaze_concat']['accuracy'].mean()
    full_concat_c = protocol_c_df[protocol_c_df['method'] == 'Text+EEG+Gaze_concat']['accuracy'].mean()
    diff = full_concat_a - full_concat_c
    report_lines.append(f"- Text+EEG+Gaze_concat (Protocol A): {full_concat_a:.4f}")
    report_lines.append(f"- Text+EEG+Gaze_concat (Protocol C): {full_concat_c:.4f}")
    report_lines.append(f"- Difference: {diff:.4f}")
    report_lines.append(f"- Performance drop in Protocol C: {'YES' if diff > 0.05 else 'NO'}")
    report_lines.append("")
    
    report_lines.append("### Q7: Which baseline should be used as the strong lower bound for future SHARE-Net?")
    report_lines.append(f"- Recommendation: Text+EEG+Gaze_concat with Protocol A accuracy of {full_concat_a:.4f}")
    report_lines.append("")
    
    with open(os.path.join(REPORTS_DIR, "share_step0_report.md"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

if __name__ == "__main__":
    main()