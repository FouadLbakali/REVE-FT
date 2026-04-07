from functools import partial
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from scipy.signal import butter, lfilter
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

BCI_CHANNELS = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
                 "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                 "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]


class BCIDataset(Dataset):
    def __init__(self, X, y, subject_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subject_ids = torch.tensor(subject_ids, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return {"data": self.X[idx], "labels": self.y[idx], "subject_id": self.subject_ids[idx]}

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band")

def collate(batch, positions):
    x_data = torch.stack([x["data"] for x in batch])
    y_label = torch.tensor([x["labels"] for x in batch])
    subject_ids = torch.tensor([x["subject_id"] for x in batch])
    positions = positions.repeat(len(batch), 1, 1)
    return {"sample": x_data, "label": y_label.long(), "pos": positions, "subject_id": subject_ids}

def load_bciciv2a(pos_bank, batch_size, seed=None):
    positions = pos_bank(BCI_CHANNELS)
    paradigm = MotorImagery(n_classes=4, resample=250, fmin=8, fmax=30)
    bci_dataset = BNCI2014_001()
    X, y, metadata = paradigm.get_data(dataset=bci_dataset)

    b, a = butter_bandpass(8, 30, 250)
    X = lfilter(b, a, X, axis=-1)

    label_map = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
    y = np.array([label_map[label] for label in y])

    subjects = metadata["subject"].values.astype(int) - 1  # 1..9 → 0..8

    n = len(y)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    full_dataset = BCIDataset(X, y, subjects)
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])

    collate_fn = partial(collate, positions=positions)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return (train_loader, val_loader, test_loader)


def load_bciciv2a_per_subject(pos_bank, batch_size, seed=None):
    """Return per-subject data loaders for two-stage fine-tuning.

    Returns:
        pooled_loaders: (train, val, test) loaders on all subjects
        subject_loaders: dict mapping subject_id -> (train, val, test) loaders
    """
    positions = pos_bank(BCI_CHANNELS)
    paradigm = MotorImagery(n_classes=4, resample=250, fmin=8, fmax=30)
    bci_dataset = BNCI2014_001()
    X, y, metadata = paradigm.get_data(dataset=bci_dataset)

    b, a = butter_bandpass(8, 30, 250)
    X = lfilter(b, a, X, axis=-1)

    label_map = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
    y = np.array([label_map[label] for label in y])

    subjects_raw = metadata["subject"].values.astype(int)
    subject_ids = subjects_raw - 1  # 1..9 → 0..8

    collate_fn = partial(collate, positions=positions)

    # Pooled loaders (all subjects)
    n = len(y)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    full_dataset = BCIDataset(X, y, subject_ids)

    gen = torch.Generator().manual_seed(seed) if seed else torch.Generator()
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test], generator=gen)

    pooled_loaders = (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
    )

    # Per-subject loaders
    unique_subjects = sorted(np.unique(subjects_raw))
    subject_loaders = {}
    for subj in unique_subjects:
        mask = (subjects_raw == subj)
        subj_X, subj_y = X[mask], y[mask]
        subj_ids = subject_ids[mask]
        subj_dataset = BCIDataset(subj_X, subj_y, subj_ids)

        sn = len(subj_y)
        sn_train = int(0.7 * sn)
        sn_val = int(0.15 * sn)
        sn_test = sn - sn_train - sn_val

        gen_s = torch.Generator().manual_seed(seed) if seed else torch.Generator()
        s_train, s_val, s_test = random_split(subj_dataset, [sn_train, sn_val, sn_test], generator=gen_s)

        subject_loaders[subj] = (
            torch.utils.data.DataLoader(s_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
            torch.utils.data.DataLoader(s_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
            torch.utils.data.DataLoader(s_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        )

    return pooled_loaders, subject_loaders