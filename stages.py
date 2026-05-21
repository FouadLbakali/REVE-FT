import copy
import sys
import types

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model

sys.path.insert(0, "reve-repro-main/src")
from models.lora import get_lora_config, CustomGetLora

from data import load_loaders, load_loaders_per_subject
from engine import (
    eval_model,
    eval_model_per_subject,
    eval_model_multilora,
    eval_model_multilora_per_subject,
    extract_features,
    train_head_one_epoch,
    train_one_epoch_multilora,
    eval_head,
    eval_head_per_subject,
)
from labram_zoo import LabramSpec
from labram_zoo import LORA_TARGET_MODULES as LABRAM_LORA_TARGET_MODULES
from luna_zoo import LunaSpec
from luna_zoo import LUNA_LORA_TARGET_MODULES
from multilora import inject_multi_subject_lora
from trainer import train_loop, print_metrics, make_scheduler, EarlyStopper


def _floatify(d):
    return {k: float(v) for k, v in d.items()}


def _aggregate_subjects(subject_results):
    keys = ("acc", "balanced_acc", "cohen_kappa", "f1", "auroc", "auc_pr")
    out = {}
    for k in keys:
        values = [subject_results[s][k] for s in sorted(subject_results)]
        out[k] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return out


def _format_per_subject(per_subject):
    """Turn {subject_id (0-based): metrics} into a JSON-ready block keyed by the
    1-based subject id."""
    keys = ("acc", "balanced_acc", "cohen_kappa", "f1", "auroc", "auc_pr")
    block = {}
    for s in sorted(per_subject):
        m = per_subject[s]
        block[str(s + 1)] = {
            **{k: (float(m[k]) if m[k] is not None else None) for k in keys},
            "n_test": int(m["n_test"]),
        }
    return block


# Per-backbone LoRA targets: attention (q/k/v + output proj) + the two FFN
# linears of every transformer block. REVE: to_qkv/to_out + net.1/net.3;
# LaBraM: qkv/proj + mlp.0/mlp.2; LUNA: qkv_proj/proj + fc1/fc2 (see the
# *_zoo.py modules for details).
_LORA_TARGETS = {
    "reve": ["to_qkv", "to_out", "net.1", "net.3"],
    "labram": LABRAM_LORA_TARGET_MODULES,
    "luna": LUNA_LORA_TARGET_MODULES,
}


def make_lora_config(rank, model_name="reve"):
    return LoraConfig(
        r=rank,
        lora_alpha=32,
        target_modules=_LORA_TARGETS[model_name],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["final_layer"],
    )


def _print_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


# --------------------------------------------------------------------------- #
# Stages used by the multi-stage runs                                          #
# --------------------------------------------------------------------------- #

def stage_linear_probing(model, pooled_loaders, args, device, results=None):
    pooled_train, pooled_val, pooled_test = pooled_loaders

    if args.load_final_layer or getattr(args, "load_global_lora", None):
        reason = "final layer loaded" if args.load_final_layer else "global LoRA checkpoint will be loaded"
        print("=" * 60)
        print(f"  Stage 1 — SKIPPED ({reason})")
        print("=" * 60)
        model.to(device)
        if results is not None:
            results.setdefault("stages", {})["lp"] = {"skipped": True, "reason": reason}
        return copy.deepcopy(model.state_dict())

    print("=" * 60)
    print("  Stage 1 — Linear Probing (cached features)")
    print("=" * 60)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.final_layer.parameters():
        param.requires_grad = True

    _print_trainable(model)
    model.to(device)

    print("Extracting cached features (one-shot backbone forward)...")
    train_features, train_labels = extract_features(model, pooled_train, device)
    val_features, val_labels = extract_features(model, pooled_val, device)
    test_features, test_labels, test_subjects = extract_features(
        model, pooled_test, device, return_subjects=True
    )
    print(f"  Train: {tuple(train_features.shape)} | Val: {tuple(val_features.shape)} | Test: {tuple(test_features.shape)}")

    head = model.final_layer
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    steps_per_epoch = (train_features.size(0) + args.batch_size - 1) // args.batch_size
    scheduler = make_scheduler(optimizer, args.epochs, steps_per_epoch)

    stopper = EarlyStopper(args.patience)
    best_val = -float("inf")
    best_head_state = copy.deepcopy(head.state_dict())
    history = []
    for epoch in range(args.epochs):
        print(f"\n[LP] Epoch {epoch + 1}/{args.epochs}")
        _, train_acc = train_head_one_epoch(
            head, optimizer, train_features, train_labels, args.batch_size, device,
            scheduler=scheduler
        )
        b_acc = eval_head(head, val_features, val_labels, device)["balanced_acc"]
        improved = b_acc > best_val
        if improved:
            best_val = b_acc
            best_head_state = copy.deepcopy(head.state_dict())
        print(f"Train acc: {train_acc:.4f} | Val balanced_acc: {b_acc:.4f} (best: {best_val:.4f})")
        history.append({"epoch": epoch + 1,
                        "train_acc": float(train_acc),
                        "val_balanced_acc": float(b_acc)})
        if stopper.step(improved):
            print(f"[LP] Early stopping at epoch {epoch + 1} "
                  f"(no improvement for {args.patience} epochs)")
            break

    head.load_state_dict(best_head_state)
    print("\n--- Stage 1 (LP) test results ---")
    test_metrics, per_subject = eval_head_per_subject(
        head, test_features, test_labels, test_subjects, device
    )
    print_metrics(test_metrics)

    if args.save_final_layer:
        torch.save(best_head_state, args.save_final_layer)
        print(f"Final layer saved to {args.save_final_layer}")

    if results is not None:
        ps_block = _format_per_subject(per_subject)
        results.setdefault("stages", {})["lp"] = {
            "history": history,
            "test": _floatify(test_metrics),
            "per_subject": ps_block,
        }
    return copy.deepcopy(model.state_dict())


def stage_global_lora(model, pooled_loaders, args, device, checkpoint, results=None):
    pooled_train, pooled_val, pooled_test = pooled_loaders

    if args.load_global_lora:
        print("=" * 60)
        print("  Global LoRA — SKIPPED (adapters loaded from checkpoint)")
        print("=" * 60)
        lora_model = PeftModel.from_pretrained(model, args.load_global_lora, is_trainable=False)
        lora_model.to(device)
        print(f"Global LoRA adapters loaded from {args.load_global_lora}")
        merged_model = lora_model.merge_and_unload()
        if results is not None:
            results.setdefault("stages", {})["gl"] = {"skipped": True,
                                                      "loaded_from": args.load_global_lora}
        return merged_model, copy.deepcopy(merged_model.state_dict())

    print(f"\n{'=' * 60}")
    print("  Global LoRA (all subjects)")
    print(f"{'=' * 60}")

    model.load_state_dict(checkpoint)
    lora_model = get_peft_model(model, make_lora_config(args.gl_rank, args.model))
    lora_model.to(device)
    _print_trainable(lora_model)

    optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad], lr=args.lr
    )
    best_state, history = train_loop(
        lora_model, pooled_train, pooled_val, optimizer, args.epochs, device,
        label="GL", patience=args.patience,
    )

    lora_model.load_state_dict(best_state)
    print("\n--- Global LoRA test results ---")
    test_metrics, per_subject = eval_model_per_subject(lora_model, pooled_test, device)
    print_metrics(test_metrics)

    if args.save_global_lora:
        lora_model.save_pretrained(args.save_global_lora)
        print(f"Global LoRA adapters saved to {args.save_global_lora}")

    if results is not None:
        ps_block = _format_per_subject(per_subject)
        results.setdefault("stages", {})["gl"] = {
            "history": history,
            "test": _floatify(test_metrics),
            "per_subject": ps_block,
        }

    merged_model = lora_model.merge_and_unload()
    return merged_model, copy.deepcopy(merged_model.state_dict())


def stage_per_subject_lora(model, subject_loaders, args, device, checkpoint, results=None):
    subject_results = {}
    subjects_block = {} if results is not None else None

    for subj, (subj_train, subj_val, subj_test) in sorted(subject_loaders.items()):
        print(f"\n{'=' * 60}")
        print(f"  Per-subject LoRA — Subject {subj}")
        print(f"{'=' * 60}")
        print(f"  Trials — train: {len(subj_train.dataset)}, val: {len(subj_val.dataset)}, test: {len(subj_test.dataset)}")

        model.load_state_dict(checkpoint)
        lora_model = get_peft_model(model, make_lora_config(args.lora_rank, args.model))
        lora_model.to(device)

        optimizer = torch.optim.AdamW(
            [p for p in lora_model.parameters() if p.requires_grad], lr=args.lr
        )
        best_state, history = train_loop(
            lora_model, subj_train, subj_val, optimizer, args.epochs, device,
            label=f"FT-S{subj}", patience=args.patience,
        )

        lora_model.load_state_dict(best_state)
        test_metrics = eval_model(lora_model, subj_test, device)
        subject_results[subj] = test_metrics

        print(f"\n  --- Subject {subj} test results ---")
        print_metrics(test_metrics)

        if subjects_block is not None:
            subjects_block[str(subj)] = {
                "history": history,
                "test": _floatify(test_metrics),
                "n_trials": {"train": len(subj_train.dataset),
                             "val": len(subj_val.dataset),
                             "test": len(subj_test.dataset)},
            }

        model = lora_model.merge_and_unload()

    if results is not None:
        results["subjects"] = subjects_block
        results["aggregate_subjects"] = _aggregate_subjects(subject_results)

    return subject_results


def summarize_subject_results(subject_results, title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for k in ("acc", "balanced_acc", "cohen_kappa", "f1", "auroc", "auc_pr"):
        values = [subject_results[s][k] for s in sorted(subject_results)]
        print(f"  {k}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


# --------------------------------------------------------------------------- #
# Top-level run modes                                                          #
# --------------------------------------------------------------------------- #

def _materialize_lazy(model, train_loader, ch_names):
    """LaBraM and LUNA are built lazily: their montage (chs_info / channel
    positions) and trial length are only known once the loaders exist. REVE is
    already an nn.Module -> passthrough."""
    if isinstance(model, (LabramSpec, LunaSpec)):
        n_times = train_loader.dataset[0]["data"].shape[-1]
        print(f"Building {type(model).__name__} ({len(ch_names)} ch, n_times={n_times})")
        model = model.build(ch_names, n_times)
    return model


def run_linear_probing_cached(model, pos_bank, args, device, results=None):
    """Linear probing with feature caching: the backbone forward is run once per
    split, then only the head is trained."""
    (train_loader, val_loader, test_loader), ch_names = load_loaders(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects,
        model_name=args.model,
    )
    model = _materialize_lazy(model, train_loader, ch_names)
    model.to(device)

    print("Extracting cached features (one-shot backbone forward)...")
    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)
    test_features, test_labels, test_subjects = extract_features(
        model, test_loader, device, return_subjects=True
    )
    print(f"  Train: {tuple(train_features.shape)} | Val: {tuple(val_features.shape)} | Test: {tuple(test_features.shape)}")

    head = model.final_layer
    print(f"Total paramètres : {sum(p.numel() for p in head.parameters()):,}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.1)
    steps_per_epoch = (train_features.size(0) + args.batch_size - 1) // args.batch_size
    scheduler = make_scheduler(optimizer, args.epochs, steps_per_epoch)

    stopper = EarlyStopper(args.patience)
    best_val_acc = -float("inf")
    best_head_state = copy.deepcopy(head.state_dict())
    history = []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        _, train_acc = train_head_one_epoch(
            head, optimizer, train_features, train_labels, args.batch_size, device,
            scheduler=scheduler
        )
        b_acc = eval_head(head, val_features, val_labels, device)["balanced_acc"]
        improved = b_acc > best_val_acc
        if improved:
            best_val_acc = b_acc
            best_head_state = copy.deepcopy(head.state_dict())
        print(f"Train acc: {train_acc:.4f} | Validation balanced accuracy: {b_acc:.4f}, best: {best_val_acc:.4f}")
        history.append({"epoch": epoch + 1,
                        "train_acc": float(train_acc),
                        "val_balanced_acc": float(b_acc)})
        if stopper.step(improved):
            print(f"Early stopping at epoch {epoch + 1} "
                  f"(no improvement for {args.patience} epochs)")
            break

    head.load_state_dict(best_head_state)

    if args.save_final_layer:
        torch.save(best_head_state, args.save_final_layer)
        print(f"Final layer saved to {args.save_final_layer}")

    test_metrics, per_subject = eval_head_per_subject(
        head, test_features, test_labels, test_subjects, device
    )
    print_metrics(test_metrics)

    if results is not None:
        ps_block = _format_per_subject(per_subject)
        results.setdefault("stages", {})["lp"] = {
            "history": history,
            "test": _floatify(test_metrics),
            "per_subject": ps_block,
        }


def run_single_stage(model, pos_bank, args, device):
    (train_loader, val_loader, test_loader), _ = load_loaders(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects
    )

    if args.mode == "full":
        params = list(model.parameters())
    elif args.mode == "lora":
        lora_config = get_lora_config(
            types.SimpleNamespace(encoder=model), rank=8,
            apply_to=("patch", "mlp4d", "attention", "ffw"),
        )
        model = CustomGetLora(lora_config).get_model(model)
        model.final_layer.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        raise ValueError(f"Unsupported single-stage mode: {args.mode}")

    print(f"Total paramètres : {sum(p.numel() for p in params):,}")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.1)
    model.to(device)

    best_state, _ = train_loop(
        model, train_loader, val_loader, optimizer, args.epochs, device,
        save_module=model.final_layer,
    )
    model.final_layer.load_state_dict(best_state)

    print_metrics(eval_model(model, test_loader, device))


def run_two_stage(model, pos_bank, args, device, results=None):
    pooled_loaders, subject_loaders, _ = load_loaders_per_subject(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects,
        model_name=args.model,
    )
    lp_checkpoint = stage_linear_probing(model, pooled_loaders, args, device, results=results)
    subject_results = stage_per_subject_lora(model, subject_loaders, args, device, lp_checkpoint, results=results)
    summarize_subject_results(subject_results, "Two-Stage Fine-Tuning — Summary")


def run_global_lora(model, pos_bank, args, device, results=None):
    pooled_loaders, _, _ = load_loaders_per_subject(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects,
        model_name=args.model,
    )
    lp_checkpoint = stage_linear_probing(model, pooled_loaders, args, device, results=results)
    stage_global_lora(model, pooled_loaders, args, device, lp_checkpoint, results=results)


def run_joint_lora(model, pos_bank, args, device, results=None):
    """Train the head (final_layer) and Global LoRA jointly in a single loop,
    with no separate linear-probing phase. The head is trainable via the LoRA
    config's modules_to_save, so a single optimizer at args.lr updates both
    the LoRA adapters and the head together."""
    pooled_loaders, _, ch_names = load_loaders_per_subject(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects,
        model_name=args.model,
    )
    model = _materialize_lazy(model, pooled_loaders[0], ch_names)
    print("=" * 60)
    print("  Joint training — head + Global LoRA (no separate LP phase)")
    print("=" * 60)
    fresh_checkpoint = copy.deepcopy(model.state_dict())
    stage_global_lora(model, pooled_loaders, args, device, fresh_checkpoint, results=results)


def run_three_stage(model, pos_bank, args, device, results=None):
    pooled_loaders, subject_loaders, _ = load_loaders_per_subject(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects,
        model_name=args.model,
    )
    lp_checkpoint = stage_linear_probing(model, pooled_loaders, args, device, results=results)
    model, global_checkpoint = stage_global_lora(model, pooled_loaders, args, device, lp_checkpoint, results=results)
    subject_results = stage_per_subject_lora(model, subject_loaders, args, device, global_checkpoint, results=results)
    summarize_subject_results(subject_results, "Three-Stage Fine-Tuning — Summary")


def run_joint_stacked(model, pos_bank, args, device, results=None):
    """Head + Global LoRA + per-subject LoRA, with no separate linear-probing
    phase. The head is trained jointly with the Global LoRA in a single loop
    (head trainable via the LoRA config's modules_to_save), then per-subject
    LoRA is run on top of the merged backbone."""
    pooled_loaders, subject_loaders, _ = load_loaders_per_subject(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects,
        model_name=args.model,
    )
    print("=" * 60)
    print("  Joint training — head + Global LoRA (no separate LP phase)")
    print("=" * 60)
    fresh_checkpoint = copy.deepcopy(model.state_dict())
    model, global_checkpoint = stage_global_lora(model, pooled_loaders, args, device, fresh_checkpoint, results=results)
    subject_results = stage_per_subject_lora(model, subject_loaders, args, device, global_checkpoint, results=results)
    summarize_subject_results(subject_results, "Joint Stacked Fine-Tuning — Summary")


def run_joint_subject_specific(model, pos_bank, args, device, results=None):
    """Per-subject LoRA with no separate linear-probing phase. For each subject
    the head is trained jointly with the per-subject LoRA in a single loop (head
    trainable via the LoRA config's modules_to_save), seeded from a fresh
    checkpoint instead of an LP checkpoint."""
    _, subject_loaders, _ = load_loaders_per_subject(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects,
        model_name=args.model,
    )
    print("=" * 60)
    print("  Joint training — head + per-subject LoRA (no separate LP phase)")
    print("=" * 60)
    fresh_checkpoint = copy.deepcopy(model.state_dict())
    subject_results = stage_per_subject_lora(model, subject_loaders, args, device, fresh_checkpoint, results=results)
    summarize_subject_results(subject_results, "Joint Subject-Specific Fine-Tuning — Summary")


def _trainable_state(model):
    return {n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def _load_trainable_state(model, state):
    own = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in state.items():
            own[n].copy_(t.to(own[n].device))


def _run_joint_multilora(model, pos_bank, args, device, global_rank,
                         results_key, banner, results=None):
    """Shared head + one LoRA adapter per subject, trained jointly. Batches mix
    subjects; each sample is routed through its subject's adapter, so a single
    forward + single backward updates the shared head and the adapters of the
    subjects present in the batch (no per-subject loop). With global_rank>0 a
    shared global LoRA adapter is trained jointly on top, in the same pass."""
    pooled_loaders, subject_loaders, ch_names = load_loaders_per_subject(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects,
        model_name=args.model,
    )
    pooled_train, pooled_val, pooled_test = pooled_loaders
    model = _materialize_lazy(model, pooled_train, ch_names)
    num_subjects = max(subject_loaders) + 1

    print("=" * 60)
    print(f"  {banner} ({num_subjects} subjects, single forward/backward)")
    print("=" * 60)

    model, n_targets = inject_multi_subject_lora(
        model, num_subjects, args.lora_rank, alpha=32, dropout=0.05,
        global_rank=global_rank, global_alpha=32,
        target_suffixes=_LORA_TARGETS[args.model],
    )
    model.to(device)
    g = f", global rank={global_rank}" if global_rank else ""
    print(f"  Multi-LoRA on {n_targets} linear layers "
          f"(per-subject rank={args.lora_rank}{g}, subjects={num_subjects})")
    _print_trainable(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = make_scheduler(optimizer, args.epochs, len(pooled_train),
                               max_lr=args.lr)
    stopper = EarlyStopper(args.patience)

    best_val = -float("inf")
    best_state = _trainable_state(model)
    history = []
    for epoch in range(args.epochs):
        print(f"\n[ML] Epoch {epoch + 1}/{args.epochs}")
        _, train_acc = train_one_epoch_multilora(
            model, optimizer, pooled_train, device, scheduler=scheduler
        )
        b_acc = eval_model_multilora(model, pooled_val, device)["balanced_acc"]
        improved = b_acc > best_val
        if improved:
            best_val = b_acc
            best_state = _trainable_state(model)
        print(f"Train acc: {train_acc:.4f} | Val balanced_acc: {b_acc:.4f} (best: {best_val:.4f})")
        history.append({"epoch": epoch + 1,
                        "train_acc": float(train_acc),
                        "val_balanced_acc": float(b_acc)})
        if stopper.step(improved):
            print(f"[ML] Early stopping at epoch {epoch + 1} "
                  f"(no improvement for {args.patience} epochs)")
            break

    _load_trainable_state(model, best_state)
    print(f"\n--- {banner} test results ---")
    test_metrics, per_subject = eval_model_multilora_per_subject(
        model, pooled_test, device
    )
    print_metrics(test_metrics)

    if results is not None:
        block = {
            "history": history,
            "test": _floatify(test_metrics),
            "per_subject": _format_per_subject(per_subject),
            "num_subjects": int(num_subjects),
            "n_lora_layers": int(n_targets),
        }
        if global_rank:
            block["global_rank"] = int(global_rank)
        results.setdefault("stages", {})[results_key] = block


def run_joint_multilora(model, pos_bank, args, device, results=None):
    _run_joint_multilora(
        model, pos_bank, args, device, global_rank=0,
        results_key="multilora",
        banner="Joint multi-LoRA — shared head + per-subject LoRA",
        results=results,
    )


def run_joint_multilora_global(model, pos_bank, args, device, results=None):
    """Global LoRA and per-subject multi-LoRA trained at the same time: each
    targeted layer applies a shared global adapter plus its per-sample routed
    subject adapter, jointly with the shared head, in one forward/backward."""
    _run_joint_multilora(
        model, pos_bank, args, device, global_rank=args.gl_rank,
        results_key="multilora_global",
        banner="Joint multi-LoRA + Global LoRA — shared head",
        results=results,
    )
