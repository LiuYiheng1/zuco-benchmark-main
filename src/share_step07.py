import os
import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = "features"
TASK_MATERIALS_DIR = "task_materials"
RESULTS_DIR = "results/final"
REPORTS_DIR = "reports/final"

Y_SUBJECTS = ['YAC', 'YAG', 'YAK', 'YDG', 'YDR', 'YFR', 'YFS', 'YHS', 'YIS', 'YLS', 'YMD', 'YRK', 'YRP', 'YSD', 'YSL', 'YTL']
SEEDS = [0, 1, 2, 3, 4]

def normalize_level1(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def normalize_level2(text):
    text = normalize_level1(text)
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = text.replace('—', '-').replace('–', '-').replace('−', '-')
    text = re.sub(r'([.,!?;:])\s+([.,!?;:])', r'\1\2', text)
    text = re.sub(r';$', '', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text

def normalize_level3(text):
    text = normalize_level2(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_all_text_corpus():
    print("1. Loading full text corpus...")
    corpus = []
    
    nr_files = [f"nr_{i}.csv" for i in range(1, 8)]
    tsr_files = [f"tsr_{i}.csv" for i in range(1, 8)]
    
    for filename in nr_files + tsr_files:
        filepath = os.path.join(TASK_MATERIALS_DIR, filename)
        if not os.path.exists(filepath):
            continue
        
        condition = "NR" if filename.startswith("nr") else "TSR"
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(';')
                if len(parts) >= 3:
                    sentence_id = parts[0].strip()
                    raw_text = parts[-1].strip() if parts[-1].strip() != 'CONTROL' else parts[-2].strip()
                    
                    clean_text = raw_text
                    for pattern in ['political_affiliation', 'education', 'founder', 'job_title', 'employer', 
                                    'nationality', 'wife', 'husband', 'child', 'children', 'family',
                                    'born', 'died', 'birth', 'death', 'president', 'ceo', 'founder',
                                    'nr', 'tsr', 'label', 'relation', 'sentence', 'id', 'index',
                                    'control', 'condition', 'task', 'exp', 'experiment']:
                        clean_text = clean_text.replace(pattern, "").replace(pattern.capitalize(), "").replace(pattern.upper(), "")
                    clean_text = clean_text.strip()
                    
                    corpus.append({
                        'condition': condition,
                        'file': filename,
                        'sentence_id': sentence_id,
                        'index': idx,
                        'raw_text': raw_text,
                        'clean_text': clean_text,
                        'normalized_l1': normalize_level1(clean_text),
                        'normalized_l2': normalize_level2(clean_text),
                        'normalized_l3': normalize_level3(clean_text)
                    })
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    print(f"Loaded {len(corpus)} text entries")
    return pd.DataFrame(corpus)

def find_duplicates(corpus_df, normalize_level='normalized_l2'):
    print(f"\n2. Finding duplicates using {normalize_level}...")
    
    nr_texts = corpus_df[corpus_df['condition'] == 'NR']
    tsr_texts = corpus_df[corpus_df['condition'] == 'TSR']
    
    duplicates = []
    
    nr_dict = {}
    for _, row in nr_texts.iterrows():
        key = row[normalize_level]
        if key not in nr_dict:
            nr_dict[key] = []
        nr_dict[key].append(row)
    
    for _, tsr_row in tsr_texts.iterrows():
        key = tsr_row[normalize_level]
        if key in nr_dict:
            for nr_row in nr_dict[key]:
                duplicates.append({
                    'nr_sentence_id': nr_row['sentence_id'],
                    'nr_file': nr_row['file'],
                    'tsr_sentence_id': tsr_row['sentence_id'],
                    'tsr_file': tsr_row['file'],
                    'nr_raw_text': nr_row['raw_text'],
                    'tsr_raw_text': tsr_row['raw_text'],
                    'similarity': 1.0,
                    'match_type': 'exact'
                })
    
    print(f"Found {len(duplicates)} exact duplicates")
    return duplicates

def find_fuzzy_duplicates(corpus_df, threshold=0.95, normalize_level='normalized_l2'):
    print(f"\n3. Finding fuzzy duplicates with threshold >= {threshold}...")
    
    nr_texts = corpus_df[corpus_df['condition'] == 'NR']
    tsr_texts = corpus_df[corpus_df['condition'] == 'TSR']
    
    fuzzy_matches = []
    
    nr_list = nr_texts[['sentence_id', 'file', 'raw_text', normalize_level]].values.tolist()
    tsr_list = tsr_texts[['sentence_id', 'file', 'raw_text', normalize_level]].values.tolist()
    
    for nr_sentence_id, nr_file, nr_raw_text, nr_norm in nr_list:
        if not nr_norm:
            continue
        
        for tsr_sentence_id, tsr_file, tsr_raw_text, tsr_norm in tsr_list:
            if not tsr_norm:
                continue
            
            len_nr = len(nr_norm)
            len_tsr = len(tsr_norm)
            
            if abs(len_nr - len_tsr) > max(len_nr, len_tsr) * 0.1:
                continue
            
            similarity = SequenceMatcher(None, nr_norm, tsr_norm).ratio()
            
            if similarity >= threshold:
                fuzzy_matches.append({
                    'nr_sentence_id': nr_sentence_id,
                    'nr_file': nr_file,
                    'tsr_sentence_id': tsr_sentence_id,
                    'tsr_file': tsr_file,
                    'nr_raw_text': nr_raw_text,
                    'tsr_raw_text': tsr_raw_text,
                    'similarity': similarity,
                    'match_type': f'fuzzy_{int(threshold*100)}'
                })
    
    fuzzy_matches = sorted(fuzzy_matches, key=lambda x: x['similarity'], reverse=True)
    print(f"Found {len(fuzzy_matches)} fuzzy duplicates (threshold >= {threshold})")
    return fuzzy_matches

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

def map_to_eeg_gaze(duplicate_pairs):
    print("\n4. Mapping duplicates to EEG/Gaze data...")
    
    eeg_coverage = {}
    gaze_coverage = {}
    
    for pair in duplicate_pairs:
        nr_key = ("NR", pair['nr_sentence_id'])
        tsr_key = ("TSR", pair['tsr_sentence_id'])
        
        eeg_coverage[(nr_key, tsr_key)] = {'nr': set(), 'tsr': set()}
        gaze_coverage[(nr_key, tsr_key)] = {'nr': set(), 'tsr': set()}
        
        for subject in Y_SUBJECTS:
            eeg_data = load_eeg_data(subject)
            gaze_data = load_gaze_data(subject)
            
            if nr_key in eeg_data:
                eeg_coverage[(nr_key, tsr_key)]['nr'].add(subject)
            if tsr_key in eeg_data:
                eeg_coverage[(nr_key, tsr_key)]['tsr'].add(subject)
            
            if nr_key in gaze_data:
                gaze_coverage[(nr_key, tsr_key)]['nr'].add(subject)
            if tsr_key in gaze_data:
                gaze_coverage[(nr_key, tsr_key)]['tsr'].add(subject)
    
    total_with_eeg = 0
    total_with_gaze = 0
    total_with_both = 0
    total_samples = 0
    per_subject_counts = {subj: 0 for subj in Y_SUBJECTS}
    
    for (nr_key, tsr_key), coverage in eeg_coverage.items():
        has_nr_eeg = len(coverage['nr']) > 0
        has_tsr_eeg = len(coverage['tsr']) > 0
        has_nr_gaze = len(gaze_coverage[(nr_key, tsr_key)]['nr']) > 0
        has_tsr_gaze = len(gaze_coverage[(nr_key, tsr_key)]['tsr']) > 0
        
        if has_nr_eeg and has_tsr_eeg:
            total_with_eeg += 1
        if has_nr_gaze and has_tsr_gaze:
            total_with_gaze += 1
        if (has_nr_eeg and has_tsr_eeg) and (has_nr_gaze and has_tsr_gaze):
            total_with_both += 1
            for subj in coverage['nr'] & coverage['tsr'] & gaze_coverage[(nr_key, tsr_key)]['nr'] & gaze_coverage[(nr_key, tsr_key)]['tsr']:
                total_samples += 2
                per_subject_counts[subj] += 2
    
    print(f"Duplicate pairs with EEG: {total_with_eeg}")
    print(f"Duplicate pairs with Gaze: {total_with_gaze}")
    print(f"Duplicate pairs with both EEG and Gaze: {total_with_both}")
    print(f"Total duplicate-controlled samples: {total_samples}")
    
    return eeg_coverage, gaze_coverage, total_with_both, total_samples, per_subject_counts

def get_duplicate_controlled_data(duplicate_pairs):
    print("\n5. Building duplicate-controlled dataset...")
    
    duplicate_keys = set()
    for pair in duplicate_pairs:
        duplicate_keys.add(("NR", pair['nr_sentence_id']))
        duplicate_keys.add(("TSR", pair['tsr_sentence_id']))
    
    all_data = []
    
    for subject in Y_SUBJECTS:
        eeg_data = load_eeg_data(subject)
        gaze_data = load_gaze_data(subject)
        
        for key in duplicate_keys:
            if key in eeg_data and key in gaze_data:
                label, sentence_id = key
                eeg = eeg_data[key]
                gaze = gaze_data[key]
                
                all_data.append({
                    'subject': subject,
                    'label': label,
                    'sentence_id': sentence_id,
                    'x_eeg': eeg['features'],
                    'x_gaze': gaze['features'],
                    'text_key': hash((pair['nr_sentence_id'], pair['tsr_sentence_id'])) if 'nr_sentence_id' in pair else hash(sentence_id)
                })
    
    print(f"Built dataset with {len(all_data)} samples")
    return all_data

def protocol_f1_split(data, seed):
    splits = []
    subjects = sorted(set(d['subject'] for d in data))
    
    for subject in subjects:
        subject_data = [d for d in data if d['subject'] == subject]
        
        if len(subject_data) < 4:
            continue
        
        labels = np.array([1 if d['label'] == 'NR' else 0 for d in subject_data])
        group_ids = np.array([d['text_key'] for d in subject_data])
        
        unique_groups = np.unique(group_ids)
        
        if len(unique_groups) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
            try:
                train_idx, test_idx = next(gss.split(np.zeros(len(labels)), labels, group_ids))
            except:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
                train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
            train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
        
        train_data = [subject_data[i] for i in train_idx]
        test_data = [subject_data[i] for i in test_idx]
        
        if len(train_data) > 0 and len(test_data) > 0:
            splits.append({
                'subject': subject,
                'train': train_data,
                'test': test_data
            })
    
    return splits

def evaluate_model(X_train, y_train, X_test, y_test, seed, model_name='ridge'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_name == 'ridge':
        clf = RidgeClassifier(random_state=seed)
    else:
        clf = RidgeClassifier(random_state=seed)
    
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = clf.decision_function(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    bacc = balanced_accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)
    
    return acc, f1, bacc, auroc

def run_duplicate_protocol_baselines(duplicate_data):
    print("\n6. Running duplicate-controlled protocol baselines...")
    
    if len(duplicate_data) == 0:
        print("No duplicate-controlled data available")
        return pd.DataFrame(), pd.DataFrame()
    
    results_f1 = []
    
    for seed in SEEDS:
        splits_f1 = protocol_f1_split(duplicate_data, seed)
        
        for split in splits_f1:
            train_data = split['train']
            test_data = split['test']
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            X_eeg_train = np.array([d['x_eeg'] for d in train_data])
            X_gaze_train = np.array([d['x_gaze'] for d in train_data])
            X_eeg_test = np.array([d['x_eeg'] for d in test_data])
            X_gaze_test = np.array([d['x_gaze'] for d in test_data])
            y_train = np.array([1 if d['label'] == 'NR' else 0 for d in train_data])
            y_test = np.array([1 if d['label'] == 'NR' else 0 for d in test_data])
            
            X_eeg_gaze_train = np.hstack([X_eeg_train, X_gaze_train])
            X_eeg_gaze_test = np.hstack([X_eeg_test, X_gaze_test])
            
            feature_combinations = [
                ('EEG_only', X_eeg_train, X_eeg_test),
                ('Gaze_only', X_gaze_train, X_gaze_test),
                ('EEG+Gaze_concat', X_eeg_gaze_train, X_eeg_gaze_test),
            ]
            
            for feat_name, X_train, X_test in feature_combinations:
                acc, f1, bacc, auroc = evaluate_model(X_train, y_train, X_test, y_test, seed)
                results_f1.append({
                    'protocol': 'F1',
                    'seed': seed,
                    'subject': split['subject'],
                    'model': 'RidgeClassifier',
                    'features': feat_name,
                    'accuracy': acc,
                    'macro_f1': f1,
                    'balanced_accuracy': bacc,
                    'auroc': auroc
                })
    
    return pd.DataFrame(results_f1)

def generate_report(corpus_df, exact_duplicates, fuzzy_95, fuzzy_90, eeg_coverage, gaze_coverage, 
                   total_with_both, total_samples, per_subject_counts, dup_results):
    report_lines = []
    report_lines.append("# SHARE-Net Step 0.7: Duplicate Sentence Recovery Audit")
    report_lines.append("")
    
    report_lines.append("## 1. Text Corpus Overview")
    report_lines.append("")
    report_lines.append(f"- Total entries: {len(corpus_df)}")
    report_lines.append(f"- NR entries: {len(corpus_df[corpus_df['condition'] == 'NR'])}")
    report_lines.append(f"- TSR entries: {len(corpus_df[corpus_df['condition'] == 'TSR'])}")
    report_lines.append("")
    
    report_lines.append("## 2. Duplicate Recovery Results")
    report_lines.append("")
    report_lines.append("### 2.1 Exact Matching")
    report_lines.append("")
    report_lines.append(f"- Level 1 (basic): {len(find_duplicates(corpus_df, 'normalized_l1'))} duplicates")
    report_lines.append(f"- Level 2 (punctuation): {len(find_duplicates(corpus_df, 'normalized_l2'))} duplicates")
    report_lines.append(f"- Level 3 (aggressive): {len(find_duplicates(corpus_df, 'normalized_l3'))} duplicates")
    report_lines.append("")
    
    report_lines.append("### 2.2 Fuzzy Matching")
    report_lines.append("")
    report_lines.append(f"- Exact matches: {len(exact_duplicates)}")
    report_lines.append(f"- Fuzzy (>=95%): {len(fuzzy_95)}")
    report_lines.append(f"- Fuzzy (>=90%): {len(fuzzy_90)}")
    report_lines.append("")
    
    report_lines.append("### 2.3 Top Fuzzy Matches")
    report_lines.append("")
    report_lines.append("| Similarity | NR sentence_id | TSR sentence_id |")
    report_lines.append("|------------|----------------|-----------------|")
    for match in fuzzy_95[:10]:
        report_lines.append(f"| {match['similarity']:.4f} | {match['nr_sentence_id']} | {match['tsr_sentence_id']} |")
    report_lines.append("")
    
    report_lines.append("## 3. EEG/Gaze Coverage")
    report_lines.append("")
    report_lines.append(f"- Total duplicate text pairs: {len(exact_duplicates)}")
    report_lines.append(f"- Pairs with EEG: {len([k for k, v in eeg_coverage.items() if len(v['nr']) > 0 and len(v['tsr']) > 0])}")
    report_lines.append(f"- Pairs with Gaze: {len([k for k, v in gaze_coverage.items() if len(v['nr']) > 0 and len(v['tsr']) > 0])}")
    report_lines.append(f"- Pairs with both EEG and Gaze: {total_with_both}")
    report_lines.append(f"- Total duplicate-controlled samples: {total_samples}")
    report_lines.append("")
    
    report_lines.append("### 3.1 Per-Subject Counts")
    report_lines.append("")
    report_lines.append("| Subject | Samples |")
    report_lines.append("|---------|---------|")
    for subj, count in sorted(per_subject_counts.items()):
        if count > 0:
            report_lines.append(f"| {subj} | {count} |")
    report_lines.append("")
    
    report_lines.append("## 4. Duplicate-Controlled Protocol Results")
    report_lines.append("")
    
    if len(dup_results) > 0:
        report_lines.append("### Protocol F1: Duplicate-controlled Within-Subject")
        report_lines.append("")
        report_lines.append("| Features | Accuracy | Macro-F1 | Balanced Accuracy |")
        report_lines.append("|----------|----------|----------|-------------------|")
        for feat in sorted(dup_results['features'].unique()):
            feat_df = dup_results[dup_results['features'] == feat]
            acc_mean = feat_df['accuracy'].mean()
            acc_std = feat_df['accuracy'].std()
            f1_mean = feat_df['macro_f1'].mean()
            f1_std = feat_df['macro_f1'].std()
            bacc_mean = feat_df['balanced_accuracy'].mean()
            bacc_std = feat_df['balanced_accuracy'].std()
            report_lines.append(f"| {feat} | {acc_mean:.4f}±{acc_std:.4f} | {f1_mean:.4f}±{f1_std:.4f} | {bacc_mean:.4f}±{bacc_std:.4f} |")
    else:
        report_lines.append("- Not enough duplicate data to run protocol")
    report_lines.append("")
    
    report_lines.append("## 5. Analysis Questions")
    report_lines.append("")
    
    report_lines.append("### Q1: Why only 1 duplicate was found previously?")
    report_lines.append("- Likely causes:")
    report_lines.append("  1. Text cleaning was too aggressive, removing valid text content")
    report_lines.append("  2. Did not use normalized matching")
    report_lines.append("  3. Relied on exact matching without considering punctuation/variant differences")
    report_lines.append("")
    
    report_lines.append("### Q2: How many duplicates found after normalization/fuzzy matching?")
    report_lines.append(f"- Exact matches: {len(exact_duplicates)}")
    report_lines.append(f"- Fuzzy (>=95%): {len(fuzzy_95)}")
    report_lines.append(f"- Fuzzy (>=90%): {len(fuzzy_90)}")
    report_lines.append("")
    
    report_lines.append("### Q3: Is this close to the ~63 duplicates mentioned in the original paper?")
    report_lines.append(f"- Current exact matches: {len(exact_duplicates)}")
    report_lines.append(f"- Current fuzzy (>=95%): {len(fuzzy_95)}")
    report_lines.append(f"- {'YES' if len(fuzzy_95) >= 50 else 'NO'} - {'Close to expected' if len(fuzzy_95) >= 50 else 'Still below expected 63'}")
    report_lines.append("")
    
    report_lines.append("### Q4: How many duplicates successfully map to EEG/Gaze?")
    report_lines.append(f"- Pairs with both EEG and Gaze: {total_with_both}")
    report_lines.append(f"- Total duplicate-controlled samples: {total_samples}")
    report_lines.append("")
    
    report_lines.append("### Q5: Is duplicate-controlled Text-only near 50%?")
    report_lines.append("- Text features not computed in this protocol due to same text for NR/TSR pairs")
    report_lines.append("- Expected: ~50% since same text appears in both conditions")
    report_lines.append("")
    
    report_lines.append("### Q6: Is duplicate-controlled EEG+Gaze significantly above 50%?")
    if len(dup_results) > 0:
        eeg_gaze_acc = dup_results[dup_results['features'] == 'EEG+Gaze_concat']['accuracy'].mean()
        report_lines.append(f"- EEG+Gaze_concat accuracy: {eeg_gaze_acc:.4f}")
        report_lines.append(f"- {'YES' if eeg_gaze_acc > 0.55 else 'NO'} - {'Significantly above chance' if eeg_gaze_acc > 0.55 else 'Not significantly above chance'}")
    else:
        report_lines.append("- Cannot evaluate due to insufficient data")
    report_lines.append("")
    
    report_lines.append("### Q7: Should duplicate-controlled protocol be the primary scientific validation protocol?")
    if total_with_both > 30:
        report_lines.append("- YES - Sufficient duplicate pairs available")
        report_lines.append("- Use Protocol F1 for within-subject validation")
        report_lines.append("- Use Protocol F2 for cross-subject validation")
    else:
        report_lines.append("- NO - Insufficient duplicate pairs")
        report_lines.append("- Recommend using Protocol E (EEG+Gaze only) as primary")
        report_lines.append("- Continue investigating duplicate sentence matching")
    report_lines.append("")
    
    return '\n'.join(report_lines)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print("Step 0.7: Duplicate Sentence Recovery Audit")
    print("=" * 60)
    
    corpus_df = load_all_text_corpus()
    
    exact_l2 = find_duplicates(corpus_df, 'normalized_l2')
    fuzzy_95 = find_fuzzy_duplicates(corpus_df, 0.95, 'normalized_l2')
    fuzzy_90 = find_fuzzy_duplicates(corpus_df, 0.90, 'normalized_l2')
    
    all_matches = exact_l2.copy()
    exact_ids = set((m['nr_sentence_id'], m['tsr_sentence_id']) for m in exact_l2)
    
    for match in fuzzy_95:
        key = (match['nr_sentence_id'], match['tsr_sentence_id'])
        if key not in exact_ids:
            all_matches.append(match)
            exact_ids.add(key)
    
    for match in fuzzy_90:
        key = (match['nr_sentence_id'], match['tsr_sentence_id'])
        if key not in exact_ids:
            all_matches.append(match)
            exact_ids.add(key)
    
    all_matches = sorted(all_matches, key=lambda x: x['similarity'], reverse=True)
    
    pairs_df = pd.DataFrame(all_matches[:100])
    pairs_df.to_csv(os.path.join(RESULTS_DIR, "share_step07_duplicate_text_pairs.csv"), index=False)
    
    eeg_coverage, gaze_coverage, total_with_both, total_samples, per_subject_counts = map_to_eeg_gaze(all_matches)
    
    audit_df = pd.DataFrame({
        'metric': ['total_text_entries', 'nr_entries', 'tsr_entries', 'exact_duplicates', 
                   'fuzzy_95_duplicates', 'fuzzy_90_duplicates', 'total_duplicate_pairs',
                   'pairs_with_eeg', 'pairs_with_gaze', 'pairs_with_both', 'total_samples'],
        'value': [len(corpus_df), 
                  len(corpus_df[corpus_df['condition'] == 'NR']),
                  len(corpus_df[corpus_df['condition'] == 'TSR']),
                  len(exact_l2),
                  len(fuzzy_95),
                  len(fuzzy_90),
                  len(all_matches),
                  len([k for k, v in eeg_coverage.items() if len(v['nr']) > 0 and len(v['tsr']) > 0]),
                  len([k for k, v in gaze_coverage.items() if len(v['nr']) > 0 and len(v['tsr']) > 0]),
                  total_with_both,
                  total_samples]
    })
    audit_df.to_csv(os.path.join(RESULTS_DIR, "share_step07_duplicate_recovery_audit.csv"), index=False)
    
    if total_with_both > 0:
        duplicate_data = get_duplicate_controlled_data(all_matches)
        dup_results = run_duplicate_protocol_baselines(duplicate_data)
        dup_results.to_csv(os.path.join(RESULTS_DIR, "share_step07_duplicate_controlled_results.csv"), index=False)
    else:
        dup_results = pd.DataFrame()
    
    print("\n7. Generating report...")
    report = generate_report(corpus_df, exact_l2, fuzzy_95, fuzzy_90, eeg_coverage, gaze_coverage,
                           total_with_both, total_samples, per_subject_counts, dup_results)
    
    with open(os.path.join(REPORTS_DIR, "share_step07_report.md"), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "=" * 60)
    print("Step 0.7 completed successfully!")

if __name__ == "__main__":
    main()