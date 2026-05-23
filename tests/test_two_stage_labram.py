"""End-to-end correctness tests for the `subject_specific` mode (run_two_stage),
parametrized over the reve and labram backbones.

`subject_specific` runs in two stages: a pooled linear-probing phase
(stage_linear_probing) followed by a per-subject loop (stage_per_subject_lora)
that, for each subject, reloads the LP checkpoint, wraps the backbone with
`get_peft_model(make_lora_config(...))`, trains, and `merge_and_unload`s back to
a clean backbone before the next subject.

That per-subject `get_peft_model` / `merge_and_unload` loop had never been
exercised on the LaBraM backbone (the mode was gated to reve only in main.py).
These tests wire a tiny LaBraM-shaped fake backbone (qkv / proj / mlp.0 / mlp.2)
— and a REVE-shaped one for parity — into the real `run_two_stage` orchestration
and verify:
  - the run completes end-to-end and emits the LP stage block, a per-subject
    `subjects` block (1-based ids) and an `aggregate_subjects` mean/std block,
  - `get_peft_model` is called exactly once per subject and the wrapped model
    carries trainable LoRA params (i.e. make_lora_config's labram target
    suffixes qkv/proj/mlp.0/mlp.2 actually matched),
  - a tiny overfittable per-subject task drives train accuracy above chance
    (the optimizer loop really back-propagates through the labram backbone).

Runs fully offline (no model / dataset download).
Run:  uv run python tests/test_two_stage_labram.py
"""
import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import stages

# Reuse the fixtures (fake backbones + synthetic loaders) from the multilora
# end-to-end tests; they expose exactly the forward(data, pos) / final_layer /
# subject-mixed-loader contract run_two_stage needs.
from test_joint_multilora_modes import (
    BACKBONES,
    NUM_SUBJECTS,
    N_CLASSES,
    _loaders,
    _patch_loaders,
    _default_args,
)


# ----- Tests (parametrized over reve + labram) ------------------------------ #

def test_run_two_stage_completes_and_emits_results(make_model, model_name):
    _patch_loaders()
    args = _default_args(model=model_name)
    results = {"stages": {}}
    stages.run_two_stage(make_model(), pos_bank=None, args=args,
                         device=torch.device("cpu"), results=results)

    lp = results["stages"]["lp"]
    for key in ("history", "test", "per_subject"):
        assert key in lp, f"missing key '{key}' in LP stage block"
    assert lp.get("skipped") is not True, "LP stage was unexpectedly skipped"

    subjects = results["subjects"]
    assert set(subjects.keys()) == {str(i + 1) for i in range(NUM_SUBJECTS)}, \
        f"per-subject block keys {set(subjects.keys())} != 1..{NUM_SUBJECTS}"
    for sid, block in subjects.items():
        for key in ("history", "test", "n_trials"):
            assert key in block, f"subject {sid} missing '{key}'"

    agg = results["aggregate_subjects"]
    for k in ("acc", "balanced_acc", "cohen_kappa", "f1", "auroc", "auc_pr"):
        assert "mean" in agg[k] and "std" in agg[k], \
            f"aggregate_subjects['{k}'] missing mean/std"


def test_get_peft_model_called_once_per_subject_with_lora(make_model, model_name):
    """The per-subject loop must wrap the labram backbone once per subject, and
    each wrapped model must carry trainable LoRA params (proving make_lora_config's
    target suffixes matched the backbone's Linears)."""
    _patch_loaders()
    args = _default_args(model=model_name)

    real_get_peft = stages.get_peft_model
    calls = []

    def _spy(model, config):
        peft_model = real_get_peft(model, config)
        lora_trainable = [
            n for n, p in peft_model.named_parameters()
            if p.requires_grad and "lora_" in n
        ]
        calls.append(lora_trainable)
        return peft_model

    stages.get_peft_model = _spy
    try:
        stages.run_two_stage(make_model(), pos_bank=None, args=args,
                             device=torch.device("cpu"), results={"stages": {}})
    finally:
        stages.get_peft_model = real_get_peft

    assert len(calls) == NUM_SUBJECTS, \
        f"get_peft_model called {len(calls)} times, expected {NUM_SUBJECTS}"
    for i, lora_trainable in enumerate(calls):
        assert lora_trainable, \
            f"subject {i + 1}: no trainable lora_ params — target suffixes " \
            f"did not match the {model_name} backbone"


def test_per_subject_training_above_chance(make_model, model_name):
    """Sanity check that the per-subject optimizer loop back-propagates through
    the backbone: each subject's best train accuracy should beat chance on the
    overfittable subject-correlated task. Seeded so the fake backbone's init
    (global-RNG) is deterministic; we look at the best epoch rather than the
    last because OneCycleLR can wobble the final step on this tiny task."""
    torch.manual_seed(0)
    _patch_loaders()
    args = _default_args(model=model_name, epochs=25, lr=5e-3, patience=0)
    results = {"stages": {}}
    stages.run_two_stage(make_model(), pos_bank=None, args=args,
                         device=torch.device("cpu"), results=results)

    chance = 1.0 / N_CLASSES
    for sid, block in results["subjects"].items():
        best_train = max(h["train_acc"] for h in block["history"])
        assert best_train > chance + 0.05, \
            f"subject {sid}: best train acc {best_train:.3f} not above chance " \
            f"({chance:.3f}) — per-subject loop did not learn"


# ----- Runner --------------------------------------------------------------- #

TESTS = [
    test_run_two_stage_completes_and_emits_results,
    test_get_peft_model_called_once_per_subject_with_lora,
    test_per_subject_training_above_chance,
]


def main():
    failed = 0
    total = 0
    for model_name, make_model in BACKBONES:
        for t in TESTS:
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
