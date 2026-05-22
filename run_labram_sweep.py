"""Sweep LaBraM x {joint, joint_multilora, joint_multilora_global} x {seed} on
bciciv2a + zuo2025, loading each dataset's raw tensors only once per process
instead of once per run.

Equivalent to the chain of 18 `python main.py --model labram --bf16 ...`
invocations, but `paradigm.get_data` / `_load_zuo2025_data` (the slow MOABB
preprocessing pass) runs once per dataset; only the seed-dependent split is
rebuilt for each iteration.

Safe for LaBraM specifically because the model's yaml uses scaler=null
(normalize="labram"), so X is just multiplied by a constant scale_factor
inside `_split_dataset` — no train-stat-dependent z-score. Do not reuse this
script as-is with REVE (normalize="zscore" fits stats on the train split,
which changes with the seed).
"""

import gc
import os
import sys
from functools import partial

import numpy as np
import torch

import data as _data

_raw_cache = {}


def _make_pooled_subject_loaders(X, y, metadata, ch_names, pos_bank,
                                 batch_size, seed, pp):
    positions = pos_bank(ch_names)
    subjects_raw = metadata["subject"].values.astype(int)
    subject_ids = subjects_raw - 1

    collate_fn = partial(_data.collate, positions=positions)

    n = len(y)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    full_dataset, splits, gen = _data._split_dataset(
        X, y, subject_ids, n_train, n_val, n_test, seed,
        normalize=pp["normalize"], scale=pp["scale"])
    train_ds, val_ds, test_ds = splits

    pooled_loaders = (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    collate_fn=collate_fn, generator=gen),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    collate_fn=collate_fn),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                    collate_fn=collate_fn),
    )
    subject_loaders = _data._per_subject_from_pooled(
        full_dataset, splits, subjects_raw, batch_size, collate_fn, seed,
    )
    return pooled_loaders, subject_loaders, ch_names


def _load_raw(dataset, model_name, num_subjects):
    pp_key = model_name if model_name in ("labram", "luna") else "reve"
    pp = _data._load_pp(pp_key)
    resample = pp["resample"]

    if dataset == "bciciv2a":
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
        paradigm = MotorImagery(n_classes=4, resample=resample,
                                fmin=pp["fmin"], fmax=pp["fmax"])
        epochs, y_raw, metadata = paradigm.get_data(
            dataset=BNCI2014_001(), return_epochs=True)
        X = epochs.get_data(units="uV")
        if pp["notch"]:
            X = _data._apply_notch(X, resample, pp["notch"])
        if pp["patch_trim"]:
            X = X[:, :, : (X.shape[-1] // pp["patch_trim"]) * pp["patch_trim"]]
        label_map = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
        y = np.array([label_map[l] for l in y_raw])
        ch_names = _data.BCI_CHANNELS
    elif dataset == "zuo2025":
        X, y, metadata, ch_names = _data._load_zuo2025_data(
            num_subjects, resample=resample, fmin=pp["fmin"], fmax=pp["fmax"])
        if pp["notch"]:
            X = _data._apply_notch(X, resample, pp["notch"])
        if pp["patch_trim"]:
            X = X[:, :, : (X.shape[-1] // pp["patch_trim"]) * pp["patch_trim"]]
    else:
        raise NotImplementedError(f"caching for dataset={dataset!r}")

    return X, y, metadata, ch_names, pp


def _cached_load_loaders_per_subject(dataset, pos_bank, batch_size, seed=None,
                                     num_subjects=109, model_name="reve"):
    key = (dataset, model_name, num_subjects)
    if key not in _raw_cache:
        print(f"[cache] loading raw data once for dataset={dataset} "
              f"model={model_name} num_subjects={num_subjects}")
        _raw_cache[key] = _load_raw(dataset, model_name, num_subjects)
    X, y, metadata, ch_names, pp = _raw_cache[key]
    return _make_pooled_subject_loaders(
        X, y, metadata, ch_names, pos_bank, batch_size, seed, pp)


_data.load_loaders_per_subject = _cached_load_loaders_per_subject
import stages as _stages  # noqa: E402
_stages.load_loaders_per_subject = _cached_load_loaders_per_subject

import main as _main  # noqa: E402


SEEDS = (42, 67, 1331)
MODES = (
    ("joint",                  "global"),
    ("joint_multilora",        "multi"),
    ("joint_multilora_global", "stacked"),
)
DATASETS = ("bciciv2a", "zuo2025")
RESULTS_DIR = "results/new_labram"


def _build_combos():
    combos = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for mode, tag in MODES:
                out = os.path.join(RESULTS_DIR, f"{dataset}_{tag}_s{seed}.json")
                combos.append((dataset, seed, mode, out))
    return combos


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    combos = _build_combos()
    for i, (dataset, seed, mode, results_out) in enumerate(combos, 1):
        print("\n" + "#" * 72)
        print(f"# [{i}/{len(combos)}] dataset={dataset} seed={seed} mode={mode}")
        print(f"# -> {results_out}")
        print("#" * 72, flush=True)
        sys.argv = [
            "main.py",
            "--model", "labram",
            "--bf16",
            "--seed", str(seed),
            "--mode", mode,
            "--dataset", dataset,
            "--results-out", results_out,
        ]
        _main.main()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
