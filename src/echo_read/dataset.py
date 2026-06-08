import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split


class EchoReadDataset(Dataset):
    def __init__(self, eeg, gaze, y, subjects, sample_ids):
        self.eeg = eeg
        self.gaze = gaze
        self.y = y
        self.subjects = subjects
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "eeg": self.eeg[idx],
            "gaze": self.gaze[idx],
            "y": self.y[idx],
            "subject": self.subjects[idx],
            "sample_id": self.sample_ids[idx]
        }


def load_aligned_data(data_dir="data"):
    data = np.load(os.path.join(data_dir, "aligned_multimodal_y.npz"))
    eeg = data["eeg"].astype(np.float32)
    gaze = data["gaze"].astype(np.float32)
    y = data["y"].astype(np.int64)
    
    df_meta = pd.read_csv(os.path.join(data_dir, "aligned_multimodal_y_metadata.csv"))
    subjects = df_meta["subject"].values
    sample_ids = df_meta["sample_id"].values
    
    return eeg, gaze, y, subjects, sample_ids


def make_smoke_split(held_out_subject="YHS", val_fraction=0.1, seed=1):
    eeg, gaze, y, subjects, sample_ids = load_aligned_data()
    
    test_mask = subjects == held_out_subject
    train_val_mask = ~test_mask
    
    eeg_train_val, eeg_test = eeg[train_val_mask], eeg[test_mask]
    gaze_train_val, gaze_test = gaze[train_val_mask], gaze[test_mask]
    y_train_val, y_test = y[train_val_mask], y[test_mask]
    subjects_train_val, subjects_test = subjects[train_val_mask], subjects[test_mask]
    sample_ids_train_val, sample_ids_test = sample_ids[train_val_mask], sample_ids[test_mask]
    
    scaler_eeg = StandardScaler()
    scaler_gaze = StandardScaler()
    
    eeg_train_val = scaler_eeg.fit_transform(eeg_train_val)
    gaze_train_val = scaler_gaze.fit_transform(gaze_train_val)
    
    eeg_test = scaler_eeg.transform(eeg_test)
    gaze_test = scaler_gaze.transform(gaze_test)
    
    n_val = int(len(y_train_val) * val_fraction)
    n_train = len(y_train_val) - n_val
    
    np.random.seed(seed)
    indices = np.random.permutation(len(y_train_val))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    eeg_train, eeg_val = eeg_train_val[train_indices], eeg_train_val[val_indices]
    gaze_train, gaze_val = gaze_train_val[train_indices], gaze_train_val[val_indices]
    y_train, y_val = y_train_val[train_indices], y_train_val[val_indices]
    subjects_train, subjects_val = subjects_train_val[train_indices], subjects_train_val[val_indices]
    sample_ids_train, sample_ids_val = sample_ids_train_val[train_indices], sample_ids_train_val[val_indices]
    
    train_dataset = EchoReadDataset(eeg_train, gaze_train, y_train, subjects_train, sample_ids_train)
    val_dataset = EchoReadDataset(eeg_val, gaze_val, y_val, subjects_val, sample_ids_val)
    test_dataset = EchoReadDataset(eeg_test, gaze_test, y_test, subjects_test, sample_ids_test)
    
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "scalers": {
            "eeg": scaler_eeg,
            "gaze": scaler_gaze
        }
    }


def get_dataloaders(split, batch_size=64, num_workers=0):
    train_loader = DataLoader(split["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(split["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(split["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def get_shapes(split):
    sample = split["train"][0]
    return {
        "eeg_shape": list(sample["eeg"].shape),
        "gaze_shape": list(sample["gaze"].shape),
        "y_shape": [],
        "num_train": len(split["train"]),
        "num_val": len(split["val"]),
        "num_test": len(split["test"]),
        "num_total": len(split["train"]) + len(split["val"]) + len(split["test"])
    }