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
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return {"data": self.X[idx], "labels": self.y[idx]}

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band")

def collate(batch, positions):
    x_data = torch.stack([x["data"] for x in batch])
    y_label = torch.tensor([x["labels"] for x in batch])
    positions = positions.repeat(len(batch), 1, 1)
    return {"sample": x_data,"label": y_label.long(),"pos": positions}

def load_bciciv2a(pos_bank, batch_size, seed=None):
    positions = pos_bank(BCI_CHANNELS)
    paradigm = MotorImagery(n_classes=4, resample=250, fmin=8, fmax=30)
    bci_dataset = BNCI2014_001()
    X, y, metadata = paradigm.get_data(dataset=bci_dataset)

    b, a = butter_bandpass(8, 30, 250)
    X = lfilter(b, a, X, axis=-1)

    label_map = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
    y = np.array([label_map[label] for label in y])

    n = len(y)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    full_dataset = BCIDataset(X, y)
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])

    collate_fn = partial(collate, positions=positions)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return (train_loader, val_loader, test_loader)