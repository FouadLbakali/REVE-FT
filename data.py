import os
import socket
import time
from functools import partial

_MNE_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mne_data")
os.makedirs(_MNE_DATA, exist_ok=True)
os.environ.setdefault("MNE_DATA", _MNE_DATA)
os.environ.setdefault("MNE_DATASETS_EEGBCI_PATH", _MNE_DATA)
os.environ.setdefault("MNE_DATASETS_PHYSIONET_PATH", _MNE_DATA)

from moabb.datasets import BNCI2014_001, PhysionetMI, Zuo2025
from moabb.paradigms import MotorImagery
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, random_split

BCI_CHANNELS = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
                 "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                 "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]

PHYSIONET_RESAMPLE = 200
PHYSIONET_PATCH_SIZE = 200


class BCIDataset(Dataset):
    def __init__(self, X, y, subject_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subject_ids = torch.tensor(subject_ids, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return {"data": self.X[idx], "labels": self.y[idx], "subject_id": self.subject_ids[idx]}

def collate(batch, positions):
    x_data = torch.stack([x["data"] for x in batch])
    y_label = torch.tensor([x["labels"] for x in batch])
    subject_ids = torch.tensor([x["subject_id"] for x in batch])
    positions = positions.repeat(len(batch), 1, 1)
    return {"sample": x_data, "label": y_label.long(), "pos": positions, "subject_id": subject_ids}


def _per_subject_from_pooled(full_dataset, splits, subjects_raw,
                             batch_size, collate_fn, seed):
    """Carve per-subject loaders out of the already-computed pooled splits.

    Each subject's train/val/test is the subset of the *matching* pooled split
    that belongs to that subject. This guarantees subj_test is a subset of
    pooled_test, so no trial used in a pooled-data stage (linear probing /
    global LoRA) can leak into a per-subject test set.
    """
    train_ds, val_ds, test_ds = splits
    subject_loaders = {}
    for subj in np.unique(subjects_raw):
        sub = {}
        for name, split in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            idx = [i for i in split.indices if subjects_raw[i] == subj]
            sub[name] = torch.utils.data.Subset(full_dataset, idx)
        gen_s = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
        subject_loaders[int(subj)] = (
            torch.utils.data.DataLoader(sub["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=gen_s),
            torch.utils.data.DataLoader(sub["val"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
            torch.utils.data.DataLoader(sub["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        )
    return subject_loaders


_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _load_pp(model_name):
    """Per-model input-preprocessing config from models/<model_name>.yaml.

    The yaml mirrors the upstream LaBraM `data.neuro` schema; this maps it to
    the filter/resample/normalize settings the loaders consume. `scaler: null`
    (LaBraM / LUNA) means no per-channel z-score plus the fixed `scale_factor`
    (given in V units while MOABB hands us microvolts, so it becomes
    scale_factor / 1e6) and a trim to whole-patch trial lengths; `scaler:
    StandardScaler` (REVE) means per-channel z-score on train stats. The
    optional `patch_size` field controls the trim divisor (defaults to
    `frequency`, i.e. one-second LaBraM patches); LUNA sets it to 40 samples.
    """
    with open(os.path.join(_MODELS_DIR, f"{model_name}.yaml")) as f:
        neuro = yaml.safe_load(f)["data"]["neuro"]
    fmin, fmax = neuro["filter"]
    notch = neuro["notch_filter"]
    freq = int(neuro["frequency"])
    is_labram = neuro["scaler"] is None
    patch_size = neuro.get("patch_size")
    if is_labram:
        patch_trim = int(patch_size) if patch_size is not None else freq
    else:
        patch_trim = None
    return {
        "fmin": fmin,
        "fmax": fmax,
        "resample": freq,
        "notch": list(notch) if notch else None,
        "normalize": "labram" if is_labram else "zscore",
        "patch_trim": patch_trim,
        "scale": (neuro["scale_factor"] or 1e6) / 1e6,
    }


def _standardize_per_channel(X, train_indices):
    """Per-channel z-score. Stats fit on the train split only to avoid leakage."""
    train_X = X[np.asarray(train_indices)]
    mean = train_X.mean(axis=(0, 2), keepdims=True)
    std = train_X.std(axis=(0, 2), keepdims=True) + 1e-7
    return (X - mean) / std


def _apply_notch(X, sfreq, freq):
    """In-band line-noise notch (LaBraM keeps up to 75 Hz, so 50 Hz must go)."""
    from mne.filter import notch_filter

    return notch_filter(X.astype(np.float64), sfreq, freqs=freq,
                        method="iir", verbose=False)


def _split_dataset(X, y, subject_ids, n_train, n_val, n_test, seed,
                   normalize="zscore", scale=1.0):
    """Split indices, normalize, then build the dataset. `normalize="zscore"`
    (REVE, default) z-scores per channel on train stats; `"labram"` applies no
    scaler and the fixed uV-input `scale` (see models/labram.yaml)."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
    n = n_train + n_val + n_test
    train_idx, val_idx, test_idx = random_split(range(n), [n_train, n_val, n_test], generator=gen)
    if normalize == "labram":
        X = X * scale
    else:
        X = _standardize_per_channel(X, train_idx.indices)
    full_dataset = BCIDataset(X, y, subject_ids)
    splits = tuple(torch.utils.data.Subset(full_dataset, s.indices)
                   for s in (train_idx, val_idx, test_idx))
    return full_dataset, splits, gen


def load_bciciv2a(pos_bank, batch_size, seed=None, num_subjects=None,
                  resample=250, patch_trim=None, fmin=8.0, fmax=30.0,
                  notch=None, normalize="zscore", scale=1.0):
    positions = pos_bank(BCI_CHANNELS)
    paradigm = MotorImagery(n_classes=4, resample=resample, fmin=fmin, fmax=fmax)
    bci_dataset = BNCI2014_001()
    subjects_arg = bci_dataset.subject_list[:num_subjects] if num_subjects is not None else None
    epochs, y, metadata = paradigm.get_data(
        dataset=bci_dataset, subjects=subjects_arg, return_epochs=True)
    X = epochs.get_data(units="uV")

    if notch:
        X = _apply_notch(X, resample, notch)
    if patch_trim:  # keep a whole number of temporal patches (LaBraM)
        X = X[:, :, : (X.shape[-1] // patch_trim) * patch_trim]

    label_map = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
    y = np.array([label_map[label] for label in y])

    subjects = metadata["subject"].values.astype(int) - 1  # 1..9 → 0..8

    n = len(y)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    _, (train_ds, val_ds, test_ds), gen = _split_dataset(
        X, y, subjects, n_train, n_val, n_test, seed, normalize=normalize, scale=scale)

    collate_fn = partial(collate, positions=positions)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=gen)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return (train_loader, val_loader, test_loader), BCI_CHANNELS


def load_bciciv2a_per_subject(pos_bank, batch_size, seed=None,
                              resample=250, patch_trim=None, fmin=8.0, fmax=30.0,
                              notch=None, normalize="zscore", scale=1.0):
    """Return per-subject data loaders for two-stage fine-tuning.

    Returns:
        pooled_loaders: (train, val, test) loaders on all subjects
        subject_loaders: dict mapping subject_id -> (train, val, test) loaders
        ch_names: channel names (for LaBraM montage construction)
    """
    positions = pos_bank(BCI_CHANNELS)
    paradigm = MotorImagery(n_classes=4, resample=resample, fmin=fmin, fmax=fmax)
    bci_dataset = BNCI2014_001()
    epochs, y, metadata = paradigm.get_data(dataset=bci_dataset, return_epochs=True)
    X = epochs.get_data(units="uV")

    if notch:
        X = _apply_notch(X, resample, notch)
    if patch_trim:  # keep a whole number of temporal patches (LaBraM)
        X = X[:, :, : (X.shape[-1] // patch_trim) * patch_trim]

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
    full_dataset, (train_ds, val_ds, test_ds), gen = _split_dataset(
        X, y, subject_ids, n_train, n_val, n_test, seed,
        normalize=normalize, scale=scale)

    pooled_loaders = (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=gen),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
    )

    # Per-subject loaders carved out of the pooled splits (no leakage).
    subject_loaders = _per_subject_from_pooled(
        full_dataset, (train_ds, val_ds, test_ds), subjects_raw,
        batch_size, collate_fn, seed,
    )

    return pooled_loaders, subject_loaders, BCI_CHANNELS


def _load_physionet_data(num_subjects, fmin=8.0, fmax=30.0):
    subjects_list = list(range(1, num_subjects + 1))

    prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(300)

    paradigm = MotorImagery(
        n_classes=4, resample=PHYSIONET_RESAMPLE, fmin=fmin, fmax=fmax,
        tmin=0.0, tmax=4.0,
    )
    mi_dataset = PhysionetMI()

    all_X, all_y, all_meta = [], [], []
    ch_names = None
    for subj in subjects_list:
        for attempt in range(3):
            try:
                epochs_i, yi, mi = paradigm.get_data(dataset=mi_dataset, subjects=[subj], return_epochs=True)
                if ch_names is None:
                    ch_names = epochs_i.ch_names
                Xi = epochs_i.get_data(units="uV")
                all_X.append(Xi)
                all_y.append(yi)
                all_meta.append(mi)
                print(f"  Subject {subj}: {len(yi)} trials")
                break
            except Exception as e:
                print(f"  Subject {subj} attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    raise

    min_len = min(Xi.shape[-1] for Xi in all_X)
    all_X = [Xi[:, :, :min_len] for Xi in all_X]
    X = np.concatenate(all_X, axis=0)
    y_raw = np.concatenate(all_y, axis=0)
    metadata = pd.concat(all_meta, ignore_index=True)

    socket.setdefaulttimeout(prev_timeout)

    # 4 s @ 200 Hz = 800 samples (MNE returns 801 with tmax inclusive); trim to
    # an exact multiple of the LaBraM patch.
    n_samples = X.shape[-1]
    usable = (n_samples // PHYSIONET_PATCH_SIZE) * PHYSIONET_PATCH_SIZE
    X = X[:, :, :usable]

    # PhysioNet's 4-class paradigm returns "rest" trials we must drop
    label_map = {"left_hand": 0, "right_hand": 1, "feet": 2, "hands": 3}
    keep = np.array([str(label) in label_map for label in y_raw])
    X = X[keep]
    y_raw = y_raw[keep]
    metadata = metadata[keep].reset_index(drop=True)
    y = np.array([label_map[str(label)] for label in y_raw])

    return X, y, metadata, ch_names


def load_physionet_mi(pos_bank, batch_size, seed=None, num_subjects=10,
                      fmin=8.0, fmax=30.0, notch=None, normalize="zscore", scale=1.0):
    X, y, metadata, ch_names = _load_physionet_data(num_subjects, fmin=fmin, fmax=fmax)
    if notch:
        X = _apply_notch(X, PHYSIONET_RESAMPLE, notch)
    positions = pos_bank(ch_names)
    subject_ids = metadata["subject"].values.astype(int) - 1

    n = len(y)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    _, (train_ds, val_ds, test_ds), gen = _split_dataset(
        X, y, subject_ids, n_train, n_val, n_test, seed, normalize=normalize, scale=scale)

    collate_fn = partial(collate, positions=positions)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=gen)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # PhysioNet is already 200 Hz, trimmed to a whole number of 200-sample
    # patches (see _load_physionet_data), so it is LaBraM-ready as-is.
    return (train_loader, val_loader, test_loader), ch_names


def load_physionet_mi_per_subject(pos_bank, batch_size, seed=None, num_subjects=10,
                                  fmin=8.0, fmax=30.0, notch=None,
                                  normalize="zscore", scale=1.0):
    X, y, metadata, ch_names = _load_physionet_data(num_subjects, fmin=fmin, fmax=fmax)
    if notch:
        X = _apply_notch(X, PHYSIONET_RESAMPLE, notch)
    positions = pos_bank(ch_names)
    subjects_raw = metadata["subject"].values.astype(int)
    subject_ids = subjects_raw - 1

    collate_fn = partial(collate, positions=positions)

    n = len(y)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    full_dataset, (train_ds, val_ds, test_ds), gen = _split_dataset(
        X, y, subject_ids, n_train, n_val, n_test, seed,
        normalize=normalize, scale=scale)

    pooled_loaders = (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=gen),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
    )

    subject_loaders = _per_subject_from_pooled(
        full_dataset, (train_ds, val_ds, test_ds), subjects_raw,
        batch_size, collate_fn, seed,
    )

    return pooled_loaders, subject_loaders, ch_names


def _load_zuo2025_data(num_subjects=None, resample=250, fmin=8.0, fmax=30.0):
    paradigm = MotorImagery(n_classes=2, resample=resample, fmin=fmin, fmax=fmax,
                            tmin=0.0, tmax=4.0)
    dataset = Zuo2025()

    subjects_list = dataset.subject_list
    if num_subjects is not None:
        subjects_list = subjects_list[:num_subjects]

    all_epochs, all_y, all_meta = [], [], []
    ch_names = None
    for subj in subjects_list:
        try:
            epochs_i, yi, mi = paradigm.get_data(dataset=dataset, subjects=[subj], return_epochs=True)
        except FileNotFoundError as e:
            print(f"  Zuo2025 subject {subj} skipped: {e}")
            continue
        if ch_names is None:
            ch_names = epochs_i.ch_names
        all_epochs.append(epochs_i.get_data(units="uV"))
        all_y.append(yi)
        all_meta.append(mi)

    X = np.concatenate(all_epochs, axis=0)
    y_raw = np.concatenate(all_y, axis=0)
    metadata = pd.concat(all_meta, ignore_index=True)

    label_map = {"left_leg": 0, "right_leg": 1}
    keep = np.array([str(label) in label_map for label in y_raw])
    X = X[keep]
    y_raw = y_raw[keep]
    metadata = metadata[keep].reset_index(drop=True)
    y = np.array([label_map[str(label)] for label in y_raw])

    return X, y, metadata, ch_names


def load_zuo2025(pos_bank, batch_size, seed=None, num_subjects=None,
                 resample=250, patch_trim=None, fmin=8.0, fmax=30.0,
                 notch=None, normalize="zscore", scale=1.0):
    X, y, metadata, ch_names = _load_zuo2025_data(
        num_subjects, resample=resample, fmin=fmin, fmax=fmax)
    if notch:
        X = _apply_notch(X, resample, notch)
    if patch_trim:  # keep a whole number of temporal patches (LaBraM)
        X = X[:, :, : (X.shape[-1] // patch_trim) * patch_trim]
    positions = pos_bank(ch_names)
    subject_ids = metadata["subject"].values.astype(int) - 1

    n = len(y)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    _, (train_ds, val_ds, test_ds), gen = _split_dataset(
        X, y, subject_ids, n_train, n_val, n_test, seed, normalize=normalize, scale=scale)

    collate_fn = partial(collate, positions=positions)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=gen)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return (train_loader, val_loader, test_loader), ch_names


def load_zuo2025_per_subject(pos_bank, batch_size, seed=None, num_subjects=None,
                             resample=250, patch_trim=None, fmin=8.0, fmax=30.0,
                             notch=None, normalize="zscore", scale=1.0):
    X, y, metadata, ch_names = _load_zuo2025_data(
        num_subjects, resample=resample, fmin=fmin, fmax=fmax)
    if notch:
        X = _apply_notch(X, resample, notch)
    if patch_trim:  # keep a whole number of temporal patches (LaBraM)
        X = X[:, :, : (X.shape[-1] // patch_trim) * patch_trim]
    positions = pos_bank(ch_names)
    subjects_raw = metadata["subject"].values.astype(int)
    subject_ids = subjects_raw - 1

    collate_fn = partial(collate, positions=positions)

    n = len(y)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    full_dataset, (train_ds, val_ds, test_ds), gen = _split_dataset(
        X, y, subject_ids, n_train, n_val, n_test, seed,
        normalize=normalize, scale=scale)

    pooled_loaders = (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=gen),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
    )

    subject_loaders = _per_subject_from_pooled(
        full_dataset, (train_ds, val_ds, test_ds), subjects_raw,
        batch_size, collate_fn, seed,
    )

    return pooled_loaders, subject_loaders, ch_names


def load_loaders(dataset, pos_bank, batch_size, seed=None, num_subjects=109,
                 model_name="reve"):
    """Returns ((train, val, test), ch_names). Per-model input preprocessing
    is read from models/<model_name>.yaml (labram: 0.1-75 Hz, 50 Hz notch,
    200 Hz, no scaler, uV->0.1 mV; luna: 0.1-75 Hz, 50 Hz notch, 256 Hz, raw
    uV in (per-trial z-score happens inside the model); reve: 8-30 Hz,
    z-scored on train stats)."""
    pp = _load_pp(model_name if model_name in ("labram", "luna") else "reve")
    flt = {"fmin": pp["fmin"], "fmax": pp["fmax"], "notch": pp["notch"],
           "normalize": pp["normalize"], "scale": pp["scale"]}
    # PhysioNet is fixed at 200 Hz internally (PHYSIONET_RESAMPLE) and already
    # trimmed to whole 200-sample patches, so it takes no resample/patch_trim.
    if dataset == "physionet":
        return load_physionet_mi(pos_bank, batch_size, seed=seed,
                                 num_subjects=num_subjects, **flt)
    resample = pp.get("resample", 250)
    if dataset == "zuo2025":
        return load_zuo2025(pos_bank, batch_size, seed=seed, num_subjects=num_subjects,
                            resample=resample, patch_trim=pp["patch_trim"], **flt)
    return load_bciciv2a(pos_bank, batch_size, seed=seed, num_subjects=num_subjects,
                         resample=resample, patch_trim=pp["patch_trim"], **flt)


def load_loaders_per_subject(dataset, pos_bank, batch_size, seed=None,
                             num_subjects=109, model_name="reve"):
    """Returns (pooled_loaders, subject_loaders, ch_names). Per-model input
    preprocessing is read from models/<model_name>.yaml, identically to
    load_loaders (labram: 0.1-75 Hz, 50 Hz notch, 200 Hz, no scaler,
    uV->0.1 mV; luna: 0.1-75 Hz, 50 Hz notch, 256 Hz, raw uV in (per-trial
    z-score happens inside the model); reve: 8-30 Hz, z-scored)."""
    pp = _load_pp(model_name if model_name in ("labram", "luna") else "reve")
    flt = {"fmin": pp["fmin"], "fmax": pp["fmax"], "notch": pp["notch"],
           "normalize": pp["normalize"], "scale": pp["scale"]}
    if dataset == "physionet":
        return load_physionet_mi_per_subject(
            pos_bank, batch_size, seed=seed, num_subjects=num_subjects, **flt)
    resample = pp.get("resample", 250)
    if dataset == "zuo2025":
        return load_zuo2025_per_subject(
            pos_bank, batch_size, seed=seed, num_subjects=num_subjects,
            resample=resample, patch_trim=pp["patch_trim"], **flt)
    return load_bciciv2a_per_subject(
        pos_bank, batch_size, seed=seed,
        resample=resample, patch_trim=pp["patch_trim"], **flt)