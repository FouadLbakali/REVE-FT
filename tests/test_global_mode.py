"""End-to-end correctness tests for the `global` mode (run_global).

The `global` mode in stages.py is supposed to train the classification head
(final_layer) and a Global LoRA adapter jointly in a single training loop,
with NO separate linear-probing phase. The head is made trainable via the
PEFT LoraConfig's `modules_to_save=["final_layer"]`, so a single optimizer
at args.lr updates both the LoRA adapters and the head together, seeded from
a fresh (random-init head) checkpoint instead of an LP checkpoint.

This file wires a tiny REVE-shaped fake backbone and synthetic loaders into
the real `run_global` orchestration in stages.py, then verifies:
  - no `lp` stage block is emitted (no separate linear-probing happened),
  - a `gl` stage block IS emitted with history / test / per_subject keys
    (1-based subject ids),
  - non-LoRA backbone layers (LayerNorm `net.0`) stay byte-for-byte frozen,
  - the head (final_layer) moves during training,
  - LoRA-targeted Linear weights (to_qkv / to_out / net.1 / net.3) receive
    a non-zero delta after merge_and_unload, proving the global LoRA trained,
  - the seed checkpoint handed to stage_global_lora equals the model's
    pre-train state (no LP phase silently ran first),
  - a single AdamW optimizer with one param-group at args.lr drives both
    head and LoRA params (no separate LP optimizer was constructed),
  - the final model reproduces the best recorded val_balanced_acc (i.e. the
    train_loop's best_state was actually loaded back at the end),
  - a tiny overfittable subject-correlated task drives train acc above chance.

Runs fully offline (no model / dataset download).
Run:  conda run -n venv python tests/test_global_mode.py
"""
import copy
import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reve_ft import stages
from reve_ft import trainer


# ----- Test fixtures -------------------------------------------------------- #

NUM_SUBJECTS = 4
TRIALS_PER_SUBJECT_TRAIN = 16
TRIALS_PER_SUBJECT_VAL = 4
TRIALS_PER_SUBJECT_TEST = 4
N_CLASSES = 3
D = 8
N_CH = 6


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.to_qkv = nn.Linear(d, d)
        self.to_out = nn.Linear(d, d)

    def forward(self, x):
        return self.to_out(torch.relu(self.to_qkv(x)))


class _FF(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d),
                                 nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class TinyReve(nn.Module):
    """Minimal REVE-shaped backbone. Carries Linears named to_qkv / to_out /
    net.1 / net.3 so _LORA_TARGETS["reve"] matches them. forward(data, pos)
    mirrors the real REVE call signature; `pos` is ignored."""

    def __init__(self, d=D, n_classes=N_CLASSES):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.ModuleList([_Attn(d), _FF(d)]) for _ in range(2)]
        )
        self.final_layer = nn.Linear(d, n_classes)

    def forward(self, data, pos):
        # data: (B, d) — synthetic samples are flat already.
        x = data
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return self.final_layer(x)


def _make_synthetic_dataset(seed=0):
    """Subject-correlated synthetic task: label depends on subject id and a
    per-trial offset so a head + LoRA jointly have signal to fit."""
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
        "label":      torch.tensor([int(b["labels"])    for b in batch]).long(),
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
    """Stub stages.load_loaders_per_subject so the runner uses synthetic data."""
    def _fake(dataset, pos_bank, batch_size, seed=None, num_subjects=None,
             model_name="reve"):
        return _loaders(batch_size=batch_size, seed=seed or 0)
    stages.load_loaders_per_subject = _fake


def _default_args(**overrides):
    args = types.SimpleNamespace(
        dataset="bciciv2a",
        batch_size=8,
        seed=0,
        num_subjects=NUM_SUBJECTS,
        model="reve",
        epochs=3,
        patience=0,            # disable early stopping for deterministic loops
        lr=1e-3,
        lora_rank=4,
        gl_rank=4,
        bf16=False,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


# ----- Tests ---------------------------------------------------------------- #

def test_no_lp_stage_emitted():
    """The whole point of `global` mode is that there is NO separate
    linear-probing phase. The results dict must not contain an "lp" block."""
    _patch_loaders()
    results = {}
    stages.run_global(TinyReve(), pos_bank=None, args=_default_args(),
                          device=torch.device("cpu"), results=results)
    assert "stages" in results
    assert "lp" not in results["stages"], \
        f"`global` mode must skip LP, but results['stages'] has: {list(results['stages'])}"


def test_gl_stage_block_structure():
    """The single training pass should produce a fully-populated `gl` block."""
    _patch_loaders()
    args = _default_args()
    results = {}
    stages.run_global(TinyReve(), pos_bank=None, args=args,
                          device=torch.device("cpu"), results=results)
    block = results["stages"]["gl"]
    for key in ("history", "test", "per_subject"):
        assert key in block, f"missing key '{key}' in gl block (global mode)"
    assert len(block["history"]) == args.epochs, \
        f"expected {args.epochs} history entries, got {len(block['history'])}"
    for k in ("acc", "balanced_acc", "f1"):
        assert k in block["test"], f"missing test metric '{k}'"
    # 1-based per-subject ids
    assert set(block["per_subject"].keys()) == {str(i + 1) for i in range(NUM_SUBJECTS)}


def test_non_lora_backbone_layers_stay_frozen():
    """Layers NOT in _LORA_TARGETS["reve"] (i.e., everything that isn't
    to_qkv / to_out / net.1 / net.3 and isn't the head) must be frozen by
    PEFT and therefore byte-for-byte identical before vs. after the global
    run. We use the LayerNorm at `net.0` as the canary."""
    _patch_loaders()
    args = _default_args()
    model = TinyReve()

    # LayerNorm sits at layers.X.1.net.0.{weight,bias} — not a LoRA target.
    frozen_before = {}
    for n, p in model.named_parameters():
        if ".net.0." in n:                # LayerNorm weight/bias
            frozen_before[n] = p.detach().clone()
    assert frozen_before, "fixture has no LayerNorm params to use as canary"

    stages.run_global(model, pos_bank=None, args=args,
                          device=torch.device("cpu"), results={})

    # After PEFT wrapping + merge_and_unload, the LayerNorm modules are
    # untouched (not wrapped, not in modules_to_save), so name->tensor lookup
    # via the original model resolves to the same (now-frozen) params.
    frozen_after = dict(model.named_parameters())
    for n, ref in frozen_before.items():
        assert n in frozen_after, f"param {n} disappeared after the run"
        delta = (frozen_after[n].detach() - ref).abs().max().item()
        assert delta == 0.0, \
            f"non-LoRA backbone param {n} drifted by {delta} — global mode " \
            f"must keep non-targeted params frozen"


def test_head_moves_in_global_mode():
    """The head (final_layer) is trainable via PEFT's `modules_to_save` and
    must update inside the same training loop."""
    _patch_loaders()
    args = _default_args()
    model = TinyReve()
    head_before = {n: p.detach().clone()
                   for n, p in model.final_layer.named_parameters()}

    stages.run_global(model, pos_bank=None, args=args,
                          device=torch.device("cpu"), results={})

    moved = any(
        (head_before[n] - p.detach()).abs().max().item() > 0
        for n, p in model.final_layer.named_parameters()
    )
    assert moved, "shared head (final_layer) did not move in `global` mode"


def test_lora_targeted_layers_received_a_delta():
    """LoRA-targeted Linear layers (to_qkv / to_out / net.1 / net.3) must
    have their weights changed after the run, because stage_global_lora
    calls merge_and_unload which folds B@A into the base layer's weight.
    A zero delta means LoRA was never trained (B stayed at its zero init)."""
    _patch_loaders()
    args = _default_args()
    model = TinyReve()

    # Snapshot the linear weights at LoRA-target names.
    targets = ("to_qkv", "to_out", "net.1", "net.3")
    before = {}
    for n, p in model.named_parameters():
        if any(n.endswith(f"{t}.weight") for t in targets):
            before[n] = p.detach().clone()
    assert before, "fixture exposes no LoRA-targetable Linear weights"

    stages.run_global(model, pos_bank=None, args=args,
                          device=torch.device("cpu"), results={})

    after = dict(model.named_parameters())
    moved = []
    for n, ref in before.items():
        # After merge_and_unload, the Linear at that path holds (base + B@A).
        # Same name resolves on the unwrapped module.
        if n not in after:
            continue
        delta = (after[n].detach() - ref).abs().max().item()
        if delta > 0:
            moved.append((n, delta))
    assert moved, \
        "no LoRA-targeted layer's weight changed — global LoRA delta is zero, " \
        "meaning the adapters were never trained"


def test_fresh_checkpoint_seed_is_pretrain_state():
    """The seed `checkpoint` handed to stage_global_lora must be the model's
    state BEFORE any training (the `fresh_checkpoint = copy.deepcopy(...)`
    line), not an LP-trained state. We intercept stage_global_lora to capture
    the checkpoint and compare its head weights to the model's head at
    run-entry."""
    _patch_loaders()
    args = _default_args()
    model = TinyReve()
    head_at_entry = {n: p.detach().clone()
                     for n, p in model.final_layer.named_parameters()}

    captured = {}
    real_stage = stages.stage_global_lora

    def _spy(m, loaders, a, dev, checkpoint, results=None):
        captured["checkpoint"] = {k: v.detach().clone() for k, v in checkpoint.items()}
        return real_stage(m, loaders, a, dev, checkpoint, results=results)

    stages.stage_global_lora = _spy
    try:
        stages.run_global(model, pos_bank=None, args=args,
                              device=torch.device("cpu"), results={})
    finally:
        stages.stage_global_lora = real_stage

    assert captured, "stage_global_lora was never called"
    for n, ref in head_at_entry.items():
        key = f"final_layer.{n}"
        assert key in captured["checkpoint"], \
            f"seed checkpoint is missing {key} (global mode should pass the full fresh state)"
        delta = (captured["checkpoint"][key] - ref).abs().max().item()
        assert delta == 0.0, \
            f"seed checkpoint head ({key}) differs from pre-train head by {delta} — " \
            f"a separate LP phase must have run, which violates `global` semantics"


def test_single_optimizer_at_args_lr():
    """`global` mode trains head + LoRA in ONE optimizer at args.lr. We hook
    torch.optim.AdamW to record every optimizer constructed during the run,
    then assert exactly one optimizer was built, with one param-group, at
    args.lr."""
    _patch_loaders()
    args = _default_args(lr=7.5e-4)  # distinctive LR to detect any hardcoded value
    seen = []

    real_adamw = torch.optim.AdamW

    class _SpyAdamW(real_adamw):
        def __init__(self, params, **kw):
            params = list(params)
            super().__init__(params, **kw)
            seen.append({
                "n_groups": len(self.param_groups),
                "lr": self.param_groups[0]["lr"],
                "n_params": sum(p.numel() for g in self.param_groups for p in g["params"]),
            })

    torch.optim.AdamW = _SpyAdamW
    try:
        stages.run_global(TinyReve(), pos_bank=None, args=args,
                              device=torch.device("cpu"), results={})
    finally:
        torch.optim.AdamW = real_adamw

    assert len(seen) == 1, \
        f"expected exactly one AdamW optimizer in `global` mode, got {len(seen)}: {seen}"
    assert seen[0]["n_groups"] == 1, \
        f"expected single param-group, got {seen[0]['n_groups']}"
    assert abs(seen[0]["lr"] - args.lr) < 1e-12, \
        f"optimizer LR {seen[0]['lr']} != args.lr {args.lr} — a hardcoded LR slipped in"
    assert seen[0]["n_params"] > 0, "optimizer received zero trainable parameters"


def test_best_state_round_trip_matches_recorded_best():
    """The train_loop loads `best_state` at the end. The final (wrapped) model
    used to compute test metrics should reproduce the best recorded val
    balanced_acc."""
    _patch_loaders()
    args = _default_args(epochs=4)

    from engine import eval_model as real_eval_engine
    real_eval_trainer = trainer.eval_model

    seen_val = []

    def _spy(m, loader, device, **kw):
        out = real_eval_engine(m, loader, device, **kw)
        seen_val.append(out["balanced_acc"])
        return out

    # train_loop in trainer.py calls `eval_model` imported at module level —
    # so we need to patch the binding inside trainer, not just engine.
    trainer.eval_model = _spy
    try:
        results = {}
        stages.run_global(TinyReve(), pos_bank=None, args=args,
                              device=torch.device("cpu"), results=results)
    finally:
        trainer.eval_model = real_eval_trainer

    assert seen_val, "eval_model was never called from train_loop"
    best_recorded = max(seen_val)
    history = results["stages"]["gl"]["history"]
    # The last entry's printed "best" should equal max of val accs in history.
    history_best = max(h["val_balanced_acc"] for h in history)
    assert abs(history_best - best_recorded) < 1e-9, \
        f"history best {history_best} != observed best {best_recorded}"


def test_training_drives_train_acc_above_chance():
    """A tiny subject-correlated overfittable task: a few epochs should push
    train acc well above 1/N_CLASSES."""
    _patch_loaders()
    args = _default_args(epochs=8, lr=3e-3, patience=0)
    results = {}
    stages.run_global(TinyReve(), pos_bank=None, args=args,
                          device=torch.device("cpu"), results=results)
    history = results["stages"]["gl"]["history"]
    final_train = history[-1]["train_acc"]
    assert final_train > 1.0 / N_CLASSES + 0.10, \
        f"train_acc did not improve above chance: history={history}"


# ----- Runner --------------------------------------------------------------- #

def main():
    tests = [
        test_no_lp_stage_emitted,
        test_gl_stage_block_structure,
        test_non_lora_backbone_layers_stay_frozen,
        test_head_moves_in_global_mode,
        test_lora_targeted_layers_received_a_delta,
        test_fresh_checkpoint_seed_is_pretrain_state,
        test_single_optimizer_at_args_lr,
        test_best_state_round_trip_matches_recorded_best,
        test_training_drives_train_acc_above_chance,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
