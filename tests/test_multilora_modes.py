"""End-to-end correctness tests for the joint_multilora and
joint_multilora_global modes, parametrized over the reve and labram backbones.

Each test runs twice: once with a REVE-shaped fake backbone (to_qkv / to_out /
net.1 / net.3) and once with a LaBraM-shaped fake backbone (qkv / proj /
mlp.0 / mlp.2). The fake modules expose the same forward(data, pos) signature
as the real ones; `_materialize_lazy` is a passthrough for nn.Modules that
aren't LabramSpec/LunaSpec, so this also covers the labram dispatch in
`_LORA_TARGETS[args.model]`.

Verified:
  - LoRA banks are allocated for the right number of subjects (no off-by-one
    with the `len(subject_loaders)` fix),
  - backbone params stay frozen across training,
  - per-subject LoRA params move only for subjects present in the batches,
  - the shared head moves,
  - global LoRA params move in the `_global` mode (and don't exist otherwise),
  - the `_load_trainable_state` round-trip preserves the best val checkpoint,
  - the results dict has the expected per-stage block,
  - per-subject test metrics are emitted with 1-based ids,
  - the `_SubjectCtx` is cleared at the end of train / eval,
  - a tiny overfittable task drives the train accuracy above chance.

Runs fully offline (no model / dataset download).
Run:  uv run python tests/test_joint_multilora_modes.py
"""
import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import multilora
import stages
from multilora import MultiSubjectLoraLinear


# ----- Test fixtures -------------------------------------------------------- #

NUM_SUBJECTS = 4
TRIALS_PER_SUBJECT_TRAIN = 16
TRIALS_PER_SUBJECT_VAL = 4
TRIALS_PER_SUBJECT_TEST = 4
N_CLASSES = 3
D = 8
N_CH = 6
N_LORA_LAYERS = 8           # 2 blocks * 4 targeted Linears (same for both)


# REVE-shaped backbone: to_qkv / to_out / net.1 / net.3.

class _ReveAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.to_qkv = nn.Linear(d, d)
        self.to_out = nn.Linear(d, d)

    def forward(self, x):
        return self.to_out(torch.relu(self.to_qkv(x)))


class _ReveFF(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d),
                                 nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class TinyReve(nn.Module):
    def __init__(self, d=D, n_classes=N_CLASSES):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.ModuleList([_ReveAttn(d), _ReveFF(d)]) for _ in range(2)]
        )
        self.final_layer = nn.Linear(d, n_classes)

    def forward(self, data, pos):
        x = data
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return self.final_layer(x)


# LaBraM-shaped backbone: qkv / proj / mlp.0 / mlp.2.

class _LabramAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        return self.proj(torch.relu(self.qkv(x)))


class _LabramBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = _LabramAttn(d)
        # `mlp` must be indexable so mlp.0 and mlp.2 resolve as named submodules.
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return self.mlp(self.attn(x))


class TinyLabram(nn.Module):
    def __init__(self, d=D, n_classes=N_CLASSES):
        super().__init__()
        self.blocks = nn.ModuleList([_LabramBlock(d) for _ in range(2)])
        self.final_layer = nn.Linear(d, n_classes)

    def forward(self, data, pos):
        x = data
        for b in self.blocks:
            x = x + b(x)
        return self.final_layer(x)


BACKBONES = [("reve", TinyReve), ("labram", TinyLabram)]


# ----- Synthetic loaders ---------------------------------------------------- #

def _make_synthetic_dataset(seed=0):
    """Subject-correlated dataset: the label is a deterministic function of
    the subject id and a per-trial offset, so per-subject adapters can fit it."""
    g = torch.Generator().manual_seed(seed)
    X_train, y_train, s_train = [], [], []
    X_val, y_val, s_val = [], [], []
    X_test, y_test, s_test = [], [], []
    for s in range(NUM_SUBJECTS):
        sig = torch.randn(D, generator=g)

        def _emit(n, store_x, store_y, store_s):
            for k in range(n):
                cls = (s + k) % N_CLASSES
                noise = 0.05 * torch.randn(D, generator=g)
                store_x.append(sig * (cls + 1) + noise)
                store_y.append(cls)
                store_s.append(s)

        _emit(TRIALS_PER_SUBJECT_TRAIN, X_train, y_train, s_train)
        _emit(TRIALS_PER_SUBJECT_VAL,   X_val,   y_val,   s_val)
        _emit(TRIALS_PER_SUBJECT_TEST,  X_test,  y_test,  s_test)

    def _stack(X, y, s):
        return (torch.stack(X), torch.tensor(y, dtype=torch.long),
                torch.tensor(s, dtype=torch.long))
    return _stack(X_train, y_train, s_train), _stack(X_val, y_val, s_val), \
           _stack(X_test,  y_test,  s_test)


class _Set(torch.utils.data.Dataset):
    def __init__(self, X, y, s):
        self.X, self.y, self.s = X, y, s

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"data": self.X[i], "labels": self.y[i], "subject_id": self.s[i]}


def _collate(batch):
    return {
        "sample":     torch.stack([b["data"] for b in batch]),
        "label":      torch.tensor([int(b["labels"])     for b in batch]).long(),
        "pos":        torch.zeros(len(batch), N_CH, 3),
        "subject_id": torch.tensor([int(b["subject_id"]) for b in batch]).long(),
    }


def _loaders(batch_size=8, seed=0):
    (Xtr, ytr, str_), (Xv, yv, sv), (Xte, yte, ste) = _make_synthetic_dataset(seed)
    pooled = (
        torch.utils.data.DataLoader(_Set(Xtr, ytr, str_), batch_size=batch_size,
                                    shuffle=True, collate_fn=_collate,
                                    generator=torch.Generator().manual_seed(seed)),
        torch.utils.data.DataLoader(_Set(Xv, yv, sv), batch_size=batch_size,
                                    shuffle=False, collate_fn=_collate),
        torch.utils.data.DataLoader(_Set(Xte, yte, ste), batch_size=batch_size,
                                    shuffle=False, collate_fn=_collate),
    )

    # 1-based subject ids in the per-subject loaders, matching
    # data._per_subject_from_pooled.
    subject_loaders = {}
    for s in range(NUM_SUBJECTS):
        ds_tr = _Set(Xtr[str_ == s], ytr[str_ == s], str_[str_ == s])
        ds_v  = _Set(Xv [sv  == s], yv [sv  == s], sv [sv  == s])
        ds_te = _Set(Xte[ste == s], yte[ste == s], ste[ste == s])
        subject_loaders[s + 1] = (
            torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                                        collate_fn=_collate),
            torch.utils.data.DataLoader(ds_v,  batch_size=batch_size, shuffle=False,
                                        collate_fn=_collate),
            torch.utils.data.DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                                        collate_fn=_collate),
        )

    ch_names = [f"C{i}" for i in range(N_CH)]
    return pooled, subject_loaders, ch_names


def _patch_loaders():
    def _fake(dataset, pos_bank, batch_size, seed=None, num_subjects=None,
              model_name="reve"):
        return _loaders(batch_size=batch_size, seed=seed or 0)
    stages.load_loaders_per_subject = _fake


def _default_args(model="reve", **overrides):
    args = types.SimpleNamespace(
        dataset="bciciv2a",
        batch_size=8,
        seed=0,
        num_subjects=NUM_SUBJECTS,
        model=model,
        epochs=3,
        patience=0,             # disable early stopping for deterministic loops
        lr=1e-3,
        lora_rank=4,
        gl_rank=4,
        load_final_layer=None,
        load_global_lora=None,
        save_final_layer=None,
        save_global_lora=None,
        bf16=False,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


# ----- Tests (parametrized) ------------------------------------------------- #

def test_lora_bank_size_matches_num_subjects(make_model, model_name):
    """No off-by-one with the `len(subject_loaders)` fix."""
    _patch_loaders()
    args = _default_args(model=model_name)
    model = make_model()
    stages.run_joint_multilora(model, pos_bank=None, args=args,
                               device=torch.device("cpu"), results={})
    banks = [m for m in model.modules() if isinstance(m, MultiSubjectLoraLinear)]
    assert banks, "no MultiSubjectLoraLinear found after run"
    assert len(banks) == N_LORA_LAYERS, \
        f"expected {N_LORA_LAYERS} wrapped layers, got {len(banks)}"
    for m in banks:
        assert m.lora_A.shape[0] == NUM_SUBJECTS, \
            f"lora_A bank has {m.lora_A.shape[0]} subjects, expected {NUM_SUBJECTS}"
        assert m.lora_B.shape[0] == NUM_SUBJECTS


def test_backbone_frozen_head_and_lora_move(make_model, model_name):
    _patch_loaders()
    args = _default_args(model=model_name)
    model = make_model()
    head_before = {n: p.detach().clone()
                   for n, p in model.final_layer.named_parameters()}
    stages.run_joint_multilora(model, pos_bank=None, args=args,
                               device=torch.device("cpu"), results={})

    # Backbone (base.* of every wrapped Linear) stays requires_grad=False.
    for m in model.modules():
        if isinstance(m, MultiSubjectLoraLinear):
            for p in m.base.parameters():
                assert p.requires_grad is False, "base param became trainable"

    moved_head = any(
        (head_before[n] - p.detach()).abs().max().item() > 0
        for n, p in model.final_layer.named_parameters()
    )
    assert moved_head, "shared head (final_layer) did not move"

    any_lora_moved = any(
        m.lora_B.detach().abs().sum().item() > 0
        for m in model.modules() if isinstance(m, MultiSubjectLoraLinear)
    )
    assert any_lora_moved, "no per-subject lora_B moved away from zero"


def test_only_present_subjects_receive_lora_updates(make_model, model_name):
    """Drop one subject from the pooled loaders; its adapter slot must stay
    at the zero init."""
    HIDDEN = 2

    def _fake(dataset, pos_bank, batch_size, seed=None, num_subjects=None,
              model_name="reve"):
        pooled, subj, ch_names = _loaders(batch_size=batch_size, seed=seed or 0)

        def _filter(loader):
            xs, ys, ss = [], [], []
            for b in loader:
                m = b["subject_id"] != HIDDEN
                if m.any():
                    xs.append(b["sample"][m])
                    ys.append(b["label"][m])
                    ss.append(b["subject_id"][m])
            X = torch.cat(xs); Y = torch.cat(ys); S = torch.cat(ss)
            return torch.utils.data.DataLoader(
                _Set(X, Y, S), batch_size=batch_size, shuffle=False,
                collate_fn=_collate
            )
        pooled = tuple(_filter(l) for l in pooled)
        return pooled, subj, ch_names

    stages.load_loaders_per_subject = _fake
    args = _default_args(model=model_name)
    model = make_model()
    stages.run_joint_multilora(model, pos_bank=None, args=args,
                               device=torch.device("cpu"), results={})

    for m in model.modules():
        if isinstance(m, MultiSubjectLoraLinear):
            assert m.lora_B[HIDDEN].abs().sum().item() == 0, \
                "hidden subject's lora_B drifted from zero"
            present_moved = any(
                m.lora_B[s].abs().sum().item() > 0
                for s in range(NUM_SUBJECTS) if s != HIDDEN
            )
            assert present_moved, "no present subject's lora_B moved"


def test_global_mode_trains_global_adapter(make_model, model_name):
    _patch_loaders()
    args = _default_args(model=model_name)
    model = make_model()
    stages.run_joint_multilora_global(model, pos_bank=None, args=args,
                                      device=torch.device("cpu"), results={})

    has_global = False
    for m in model.modules():
        if isinstance(m, MultiSubjectLoraLinear):
            assert hasattr(m, "global_A") and hasattr(m, "global_B"), \
                "global adapter missing in joint_multilora_global"
            assert m.global_rank == args.gl_rank
            if m.global_B.detach().abs().sum().item() > 0:
                has_global = True
    assert has_global, "no global_B moved away from zero in _global mode"


def test_non_global_mode_has_no_global_adapter(make_model, model_name):
    _patch_loaders()
    args = _default_args(model=model_name)
    model = make_model()
    stages.run_joint_multilora(model, pos_bank=None, args=args,
                               device=torch.device("cpu"), results={})
    for m in model.modules():
        if isinstance(m, MultiSubjectLoraLinear):
            assert m.global_rank == 0
            assert not hasattr(m, "global_A")
            assert not hasattr(m, "global_B")


def test_results_block_structure(make_model, model_name):
    _patch_loaders()
    args = _default_args(model=model_name)
    results = {}
    stages.run_joint_multilora(make_model(), None, args, torch.device("cpu"),
                               results=results)
    block = results["stages"]["multilora"]
    for key in ("history", "test", "per_subject", "num_subjects", "n_lora_layers"):
        assert key in block, f"missing key {key} in non-global results"
    assert "global_rank" not in block
    assert block["num_subjects"] == NUM_SUBJECTS
    assert block["n_lora_layers"] == N_LORA_LAYERS
    assert set(block["per_subject"].keys()) == {str(i + 1) for i in range(NUM_SUBJECTS)}
    for k in ("acc", "balanced_acc", "f1"):
        assert k in block["test"]
    assert len(block["history"]) == args.epochs

    results = {}
    stages.run_joint_multilora_global(make_model(), None, args, torch.device("cpu"),
                                      results=results)
    block = results["stages"]["multilora_global"]
    assert block["global_rank"] == args.gl_rank


def test_subject_id_context_cleared_after_run(make_model, model_name):
    _patch_loaders()
    multilora.set_subject_ids(torch.tensor([42]))   # poison
    stages.run_joint_multilora(make_model(), None, _default_args(model=model_name),
                               torch.device("cpu"), results={})
    assert multilora._CTX.subject_ids is None, \
        "_SubjectCtx not cleared at end of training run"


def test_best_state_round_trip_matches_recorded_best(make_model, model_name):
    """Spy on val-eval; after the run, recompute val on the final model — it
    must equal the *best* recorded val (i.e. `_load_trainable_state` restored
    the best checkpoint, not the last one)."""
    _patch_loaders()
    args = _default_args(model=model_name, epochs=4)
    model = make_model()

    from engine import eval_model_multilora as real_eval
    seen = []

    def _spy(m, loader, device, n_classes=None):
        out = real_eval(m, loader, device, n_classes=n_classes)
        seen.append(out["balanced_acc"])
        return out

    stages.eval_model_multilora = _spy
    try:
        stages.run_joint_multilora(model, None, args, torch.device("cpu"),
                                   results={})
    finally:
        stages.eval_model_multilora = real_eval

    assert seen, "eval_model_multilora was never called"
    best_recorded = max(seen)

    pooled, _, _ = _loaders(batch_size=args.batch_size, seed=args.seed)
    final_val = real_eval(model, pooled[1], torch.device("cpu"))["balanced_acc"]
    assert abs(final_val - best_recorded) < 1e-6, \
        f"final val ({final_val:.6f}) != best recorded ({best_recorded:.6f})"


def test_training_drives_loss_down_on_overfittable_task(make_model, model_name):
    """Sanity check that the optimization loop is wired up: train_acc and
    val_balanced_acc should both end above chance. Threshold kept loose because
    TinyReve / TinyLabram have different residual structures and converge at
    different rates on this toy task."""
    _patch_loaders()
    args = _default_args(model=model_name, epochs=20, lr=5e-3, patience=0)
    results = {}
    stages.run_joint_multilora(make_model(), None, args, torch.device("cpu"),
                               results=results)
    history = results["stages"]["multilora"]["history"]
    final_train = history[-1]["train_acc"]
    best_val = max(h["val_balanced_acc"] for h in history)
    chance = 1.0 / N_CLASSES
    assert final_train > chance + 0.05 and best_val > chance + 0.05, \
        f"training did not improve above chance: {history}"


# ----- Runner --------------------------------------------------------------- #

TESTS = [
    test_lora_bank_size_matches_num_subjects,
    test_backbone_frozen_head_and_lora_move,
    test_only_present_subjects_receive_lora_updates,
    test_global_mode_trains_global_adapter,
    test_non_global_mode_has_no_global_adapter,
    test_results_block_structure,
    test_subject_id_context_cleared_after_run,
    test_best_state_round_trip_matches_recorded_best,
    test_training_drives_loss_down_on_overfittable_task,
]


def main():
    failed = 0
    total = 0
    for model_name, make_model in BACKBONES:
        for t in TESTS:
            multilora.set_subject_ids(None)
            total += 1
            label = f"{t.__name__}[{model_name}]"
            try:
                t(make_model, model_name)
                print(f"PASS  {label}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {label}: {e}")
            except Exception as e:
                failed += 1
                print(f"ERROR {label}: {type(e).__name__}: {e}")
    print(f"\n{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
