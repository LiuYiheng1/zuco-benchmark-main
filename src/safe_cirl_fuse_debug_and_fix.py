import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

def load_aligned_data(data_dir="data"):
    npz_path = os.path.join(data_dir, "aligned_multimodal_y.npz")
    metadata_path = os.path.join(data_dir, "aligned_multimodal_y_metadata.csv")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"aligned_multimodal_y.npz not found at {npz_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"aligned_multimodal_y_metadata.csv not found at {metadata_path}")
    
    npz_data = np.load(npz_path, allow_pickle=True)
    metadata = pd.read_csv(metadata_path)
    
    return npz_data, metadata

def audit_subject_fields(metadata, npz_data):
    output_dir = "results/safe_cirl_fuse_debug"
    os.makedirs(output_dir, exist_ok=True)
    
    if 'subject' not in metadata.columns:
        print("警告: metadata 中没有 'subject' 列")
        return None
    
    metadata['subject'] = metadata['subject'].apply(lambda x: str(x).strip() if pd.notna(x) else x)
    
    unique_subjects = metadata['subject'].dropna().unique()
    print(f"Metadata 中唯一 subjects ({len(unique_subjects)}): {sorted(unique_subjects)}")
    
    audit_notes = []
    
    audit_notes.append("# Subject Field Audit\n\n")
    
    audit_notes.append("## Subject Field Statistics\n")
    audit_notes.append(f"- Total samples: {len(metadata)}\n")
    audit_notes.append(f"- Unique subjects: {len(unique_subjects)}\n")
    audit_notes.append(f"- Missing subject: {metadata['subject'].isna().sum()}\n")
    
    subject_counts = metadata['subject'].value_counts()
    audit_notes.append("\n## Per-Subject Sample Counts\n")
    for subj, count in subject_counts.items():
        nr_count = len(metadata[(metadata['subject'] == subj) & (metadata['label'] == 0)])
        tsr_count = len(metadata[(metadata['subject'] == subj) & (metadata['label'] == 1)])
        audit_notes.append(f"- {subj}: total={count}, NR={nr_count}, TSR={tsr_count}\n")
    
    has_y_subjects = any(s.startswith('Y') for s in unique_subjects)
    has_x_subjects = any(s.startswith('X') for s in unique_subjects)
    
    audit_notes.append("\n## Subject Type Analysis\n")
    audit_notes.append(f"- Has Y subjects: {has_y_subjects}\n")
    audit_notes.append(f"- Has X subjects: {has_x_subjects}\n")
    
    if has_x_subjects:
        audit_notes.append("\n⚠️ WARNING: X subjects detected. These should NOT be used for fair evaluation.\n")
    
    target_subjects = ["YHS", "YIS", "YSD", "YRK", "YFR"]
    for subj in target_subjects:
        if subj in unique_subjects:
            audit_notes.append(f"\n✅ Target subject '{subj}' found with {subject_counts.get(subj, 0)} samples\n")
        else:
            audit_notes.append(f"\n❌ Target subject '{subj}' NOT FOUND\n")
    
    with open(os.path.join(output_dir, "subject_field_audit.md"), 'w', encoding='utf-8') as f:
        f.writelines(audit_notes)
    
    return metadata

def inventory_subjects(metadata, npz_data):
    output_dir = "results/safe_cirl_fuse_debug"
    os.makedirs(output_dir, exist_ok=True)
    
    if 'subject' not in metadata.columns:
        print("错误: metadata 中没有 'subject' 列")
        return None
    
    metadata['subject'] = metadata['subject'].apply(lambda x: str(x).strip() if pd.notna(x) else x)
    
    inventory = []
    for subj in sorted(metadata['subject'].dropna().unique()):
        subj_data = metadata[metadata['subject'] == subj]
        nr_count = len(subj_data[subj_data['label'] == 0])
        tsr_count = len(subj_data[subj_data['label'] == 1])
        
        inventory.append({
            'subject': subj,
            'sample_count': len(subj_data),
            'NR_count': nr_count,
            'TSR_count': tsr_count
        })
    
    inventory_df = pd.DataFrame(inventory)
    inventory_df.to_csv(os.path.join(output_dir, "aligned_subject_inventory.csv"), index=False)
    
    print("\nSubject Inventory:")
    print(inventory_df.to_string(index=False))
    
    return inventory_df

def split_dryrun(held_out_subjects, metadata, npz_data, all_y_subjects):
    output_dir = "results/safe_cirl_fuse_debug"
    os.makedirs(output_dir, exist_ok=True)
    
    train_subjects = [s for s in all_y_subjects if s not in held_out_subjects]
    
    split_data = []
    all_hashes = []
    
    for held_out in held_out_subjects:
        print(f"\n检查 held_out_subject: {held_out}")
        
        test_data = metadata[metadata['subject'] == held_out]
        train_data = metadata[metadata['subject'].isin(train_subjects)]
        
        if len(test_data) == 0:
            raise ValueError(f"CRITICAL ERROR: held_out_subject '{held_out}' has NO samples in metadata!")
        
        test_hash = hashlib.md5(str(test_data.index.tolist()).encode()).hexdigest()
        all_hashes.append(test_hash)
        
        row = {
            'held_out_subject': held_out,
            'train_subjects': ','.join(train_subjects),
            'val_subjects': 'N/A (no validation split)',
            'test_subjects': held_out,
            'train_N': len(train_data),
            'val_N': 0,
            'test_N': len(test_data),
            'test_NR_count': len(test_data[test_data['label'] == 0]),
            'test_TSR_count': len(test_data[test_data['label'] == 1]),
            'unique_subjects_in_test': held_out,
            'first_10_keys': ','.join(test_data['key'].head(10).tolist()) if 'key' in test_data.columns else 'N/A',
            'test_index_hash': test_hash
        }
        
        split_data.append(row)
        
        print(f"  - test_N: {row['test_N']}")
        print(f"  - test unique subjects: {row['unique_subjects_in_test']}")
        print(f"  - test_index_hash: {test_hash}")
    
    split_df = pd.DataFrame(split_data)
    split_df.to_csv(os.path.join(output_dir, "split_dryrun_stage_b.csv"), index=False)
    
    unique_hashes = set(all_hashes)
    print(f"\n\n=== Split Dry-run Validation ===")
    print(f"Total folds: {len(held_out_subjects)}")
    print(f"Unique test hashes: {len(unique_hashes)}")
    
    if len(unique_hashes) != len(held_out_subjects):
        print("\n❌ FAIL: Some folds share the same test set!")
        return False
    
    all_valid = True
    for row in split_data:
        if row['test_N'] == 0:
            print(f"\n❌ FAIL: {row['held_out_subject']} has 0 test samples")
            all_valid = False
        if row['unique_subjects_in_test'] != row['held_out_subject']:
            print(f"\n❌ FAIL: {row['held_out_subject']} test contains {row['unique_subjects_in_test']}")
            all_valid = False
    
    if all_valid:
        print("\n✅ PASS: All splits are valid and unique")
    
    return all_valid

def audit_stage_a_validity():
    output_dir = "results/safe_cirl_fuse_debug"
    os.makedirs(output_dir, exist_ok=True)
    
    notes = []
    notes.append("# Stage A Validity Audit\n\n")
    
    notes.append("## Question 1: Did Stage A use load_eeg_gaze_features?\n")
    notes.append("**Answer**: YES. Both SAFE-CIRL Stage A and SAFE-CIRL-Fuse Stage A scripts use `load_eeg_gaze_features()` function.\n\n")
    
    notes.append("## Question 2: Could Stage A fallback to random split?\n")
    notes.append("**Answer**: YES. The old code had:\n")
    notes.append("```python\n")
    notes.append("if len(test_data) == 0:\n")
    notes.append("    train_data, test_data = train_test_split(...)\n")
    notes.append("```\n")
    notes.append("This means if `held_out_subjects` didn't exist in features, it would fallback.\n\n")
    
    notes.append("## Question 3: Did held-out subjects actually work in Stage A?\n")
    notes.append("**Answer**: UNCLEAR. The held_out_subjects in Stage A were [\"YHS\", \"YRK\", \"YFR\"].\n")
    notes.append("If these subjects existed in `load_eeg_gaze_features` output, the splits were valid.\n")
    notes.append("However, the fallback mechanism was present and could have been triggered.\n\n")
    
    notes.append("## Question 4: Should Stage A results be invalidated?\n")
    notes.append("**Answer**: NEEDS VERIFICATION.\n")
    notes.append("- Stage A used held_out_subjects = [\"YHS\", \"YRK\", \"YFR\"]\n")
    notes.append("- Stage B used held_out_subjects = [\"YHS\", \"YIS\", \"YSD\", \"YRK\", \"YFR\"]\n")
    notes.append("- The difference is YIS and YSD\n")
    notes.append("- If YHS/YRK/YFR existed in features, Stage A splits were valid\n")
    notes.append("- If YHS/YRK/YFR did NOT exist, Stage A also used fallback (INVALID)\n\n")
    
    notes.append("## Recommendation\n")
    notes.append("Stage A results should be treated as **POTENTIALLY INVALID** until verified.\n")
    notes.append("The fix should re-run Stage A with the corrected data loading.\n")
    
    with open(os.path.join(output_dir, "stage_a_validity_audit.md"), 'w', encoding='utf-8') as f:
        f.writelines(notes)

def main():
    output_dir = "results/safe_cirl_fuse_debug"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SAFE-CIRL-Fuse Debug and Fix")
    print("=" * 60)
    
    print("\n任务 1-2: 加载 aligned_multimodal_y 数据...")
    npz_data, metadata = load_aligned_data()
    print(f"  - npz keys: {list(npz_data.keys())}")
    print(f"  - metadata shape: {metadata.shape}")
    print(f"  - metadata columns: {list(metadata.columns)}")
    
    print("\n任务 3: Subject 字段审计...")
    metadata = audit_subject_fields(metadata, npz_data)
    
    print("\n任务 2: 生成 subject inventory...")
    inventory_df = inventory_subjects(metadata, npz_data)
    
    all_y_subjects = [s for s in inventory_df['subject'].tolist() if s.startswith('Y')]
    print(f"\n所有 Y subjects: {sorted(all_y_subjects)}")
    
    print("\n任务 4: Split dry-run...")
    held_out_subjects = ["YHS", "YIS", "YSD", "YRK", "YFR"]
    
    audit_stage_a_validity()
    
    try:
        split_valid = split_dryrun(held_out_subjects, metadata, npz_data, all_y_subjects)
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\n无法继续进行 Stage B split dry-run。")
        return
    
    print("\n" + "=" * 60)
    print("Debug 完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
