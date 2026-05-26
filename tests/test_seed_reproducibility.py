"""Seed reproducibility checks for data.py split + training DataLoader.

Runs offline: the moabb/network fetch (`_load_zuo2025_data`) is monkeypatched
with a fixed synthetic dataset, so this exercises only the seeding logic
(`torch.Generator().manual_seed(seed)` -> random_split -> shuffled DataLoader).

Run:  conda run -n venv python tests/test_seed_reproducibility.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import data

N_SUBJECTS = 4
TRIALS_PER_SUBJECT = 24
N_CHANNELS = 8
N_TIMES = 64


def _synthetic_zuo(num_subjects=None):
    """Deterministic stand-in for data._load_zuo2025_data (no network)."""
    rng = np.random.default_rng(0)
    n = N_SUBJECTS * TRIALS_PER_SUBJECT
    X = rng.standard_normal((n, N_CHANNELS, N_TIMES)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    subjects = np.repeat(np.arange(1, N_SUBJECTS + 1), TRIALS_PER_SUBJECT)
    metadata = pd.DataFrame({"subject": subjects})
    ch_names = [f"C{i}" for i in range(N_CHANNELS)]
    return X, y, metadata, ch_names


def _fake_pos_bank(ch_names):
    return torch.zeros(len(ch_names), 3)


def _split_indices(loaders):
    """(train, val, test) Subset.indices for a 3-tuple of loaders."""
    return tuple(sorted(ld.dataset.indices) for ld in loaders)


def _first_batch(loader):
    return next(iter(loader))


def _batches_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in ("sample", "label", "pos", "subject_id"))


def _run_flat(seed):
    return data.load_zuo2025(_fake_pos_bank, batch_size=8, seed=seed)


def test_flat_same_seed_reproducible():
    a = _run_flat(seed=123)
    b = _run_flat(seed=123)
    assert _split_indices(a) == _split_indices(b), "split indices differ at equal seed"
    assert _batches_equal(_first_batch(a[0]), _first_batch(b[0])), \
        "first training batch differs at equal seed"


def test_flat_different_seed_changes_split():
    a = _run_flat(seed=1)
    b = _run_flat(seed=2)
    assert _split_indices(a) != _split_indices(b), \
        "split identical for different seeds — seed not driving random_split"


def test_per_subject_same_seed_reproducible():
    pooled_a, subj_a = data.load_zuo2025_per_subject(_fake_pos_bank, batch_size=8, seed=7)
    pooled_b, subj_b = data.load_zuo2025_per_subject(_fake_pos_bank, batch_size=8, seed=7)

    assert _split_indices(pooled_a) == _split_indices(pooled_b), \
        "pooled split differs at equal seed"
    assert _batches_equal(_first_batch(pooled_a[0]), _first_batch(pooled_b[0])), \
        "pooled first training batch differs at equal seed"

    assert sorted(subj_a) == sorted(subj_b)
    for s in subj_a:
        assert _split_indices(subj_a[s]) == _split_indices(subj_b[s]), \
            f"subject {s} split differs at equal seed"
        assert _batches_equal(_first_batch(subj_a[s][0]), _first_batch(subj_b[s][0])), \
            f"subject {s} first training batch differs at equal seed"


def main():
    data._load_zuo2025_data = _synthetic_zuo
    tests = [
        test_flat_same_seed_reproducible,
        test_flat_different_seed_changes_split,
        test_per_subject_same_seed_reproducible,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
