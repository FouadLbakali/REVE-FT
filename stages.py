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
    extract_features,
    train_head_one_epoch,
    eval_head,
)
from trainer import train_loop, print_metrics, make_scheduler, _step_scheduler


def make_lora_config(rank):
    return LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        target_modules=["to_qkv", "to_out"],
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

def stage_linear_probing(model, pooled_loaders, args, device):
    pooled_train, pooled_val, pooled_test = pooled_loaders

    if args.load_final_layer or getattr(args, "load_global_lora", None):
        reason = "final layer loaded" if args.load_final_layer else "global LoRA checkpoint will be loaded"
        print("=" * 60)
        print(f"  Stage 1 — SKIPPED ({reason})")
        print("=" * 60)
        model.to(device)
        return copy.deepcopy(model.state_dict())

    print("=" * 60)
    print("  Stage 1 — Linear Probing (all subjects)")
    print("=" * 60)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.final_layer.parameters():
        param.requires_grad = True

    _print_trainable(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.final_layer.parameters(), lr=args.lr)
    best_head_state = train_loop(
        model, pooled_train, pooled_val, optimizer, args.epochs, device,
        l1=args.l1, label="LP", save_module=model.final_layer,
        scheduler_name=args.scheduler,
    )

    model.final_layer.load_state_dict(best_head_state)
    print("\n--- Stage 1 (LP) test results ---")
    print_metrics(eval_model(model, pooled_test, device))

    if args.save_final_layer:
        torch.save(best_head_state, args.save_final_layer)
        print(f"Final layer saved to {args.save_final_layer}")

    return copy.deepcopy(model.state_dict())


def stage_global_lora(model, pooled_loaders, args, device, checkpoint):
    pooled_train, pooled_val, pooled_test = pooled_loaders

    if args.load_global_lora:
        print("=" * 60)
        print("  Global LoRA — SKIPPED (adapters loaded from checkpoint)")
        print("=" * 60)
        lora_model = PeftModel.from_pretrained(model, args.load_global_lora, is_trainable=False)
        lora_model.to(device)
        print(f"Global LoRA adapters loaded from {args.load_global_lora}")
        merged_model = lora_model.merge_and_unload()
        return merged_model, copy.deepcopy(merged_model.state_dict())

    print(f"\n{'=' * 60}")
    print("  Global LoRA (all subjects)")
    print(f"{'=' * 60}")

    model.load_state_dict(checkpoint)
    lora_model = get_peft_model(model, make_lora_config(args.gl_rank))
    lora_model.to(device)
    _print_trainable(lora_model)

    optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad], lr=args.gl_lr
    )
    best_state = train_loop(
        lora_model, pooled_train, pooled_val, optimizer, args.gl_epochs, device,
        l1=args.l1, label="GL", scheduler_name=args.scheduler,
    )

    lora_model.load_state_dict(best_state)
    print("\n--- Global LoRA test results ---")
    print_metrics(eval_model(lora_model, pooled_test, device))

    if args.save_global_lora:
        lora_model.save_pretrained(args.save_global_lora)
        print(f"Global LoRA adapters saved to {args.save_global_lora}")

    merged_model = lora_model.merge_and_unload()
    return merged_model, copy.deepcopy(merged_model.state_dict())


def stage_per_subject_lora(model, subject_loaders, args, device, checkpoint):
    subject_results = {}

    for subj, (subj_train, subj_val, subj_test) in sorted(subject_loaders.items()):
        print(f"\n{'=' * 60}")
        print(f"  Per-subject LoRA — Subject {subj}")
        print(f"{'=' * 60}")
        print(f"  Trials — train: {len(subj_train.dataset)}, val: {len(subj_val.dataset)}, test: {len(subj_test.dataset)}")

        model.load_state_dict(checkpoint)
        lora_model = get_peft_model(model, make_lora_config(args.lora_rank))
        lora_model.to(device)

        optimizer = torch.optim.AdamW(
            [p for p in lora_model.parameters() if p.requires_grad], lr=args.ft_lr
        )
        best_state = train_loop(
            lora_model, subj_train, subj_val, optimizer, args.ft_epochs, device,
            l1=args.l1, label=f"FT-S{subj}", scheduler_name=args.scheduler,
        )

        lora_model.load_state_dict(best_state)
        test_metrics = eval_model(lora_model, subj_test, device)
        subject_results[subj] = test_metrics

        print(f"\n  --- Subject {subj} test results ---")
        print_metrics(test_metrics)

        model = lora_model.merge_and_unload()

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

def run_linear_probing_cached(model, pos_bank, args, device):
    """Linear probing with feature caching: the backbone forward is run once per
    split, then only the head is trained."""
    train_loader, val_loader, test_loader = load_loaders(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects
    )
    model.to(device)

    print("Extracting cached features (one-shot backbone forward)...")
    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    print(f"  Train: {tuple(train_features.shape)} | Val: {tuple(val_features.shape)} | Test: {tuple(test_features.shape)}")

    head = model.final_layer
    print(f"Total paramètres : {sum(p.numel() for p in head.parameters()):,}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler, needs_metric = make_scheduler(args.scheduler, optimizer, args.epochs)

    best_val_acc = 0.0
    best_head_state = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        _, train_acc = train_head_one_epoch(
            head, optimizer, train_features, train_labels, args.batch_size, device, l1_lambda=args.l1
        )
        b_acc = eval_head(head, val_features, val_labels, device)["balanced_acc"]
        if b_acc > best_val_acc:
            best_val_acc = b_acc
            best_head_state = copy.deepcopy(head.state_dict())
        print(f"Train acc: {train_acc:.4f} | Validation balanced accuracy: {b_acc:.4f}, best: {best_val_acc:.4f}")
        _step_scheduler(scheduler, needs_metric, b_acc)

    head.load_state_dict(best_head_state)

    if args.save_final_layer:
        torch.save(best_head_state, args.save_final_layer)
        print(f"Final layer saved to {args.save_final_layer}")

    print_metrics(eval_head(head, test_features, test_labels, device))


def run_single_stage(model, pos_bank, args, device):
    train_loader, val_loader, test_loader = load_loaders(
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

    best_state = train_loop(
        model, train_loader, val_loader, optimizer, args.epochs, device,
        l1=args.l1, save_module=model.final_layer,
    )
    model.final_layer.load_state_dict(best_state)

    print_metrics(eval_model(model, test_loader, device))


def run_two_stage(model, pos_bank, args, device):
    pooled_loaders, subject_loaders = load_loaders_per_subject(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects
    )
    lp_checkpoint = stage_linear_probing(model, pooled_loaders, args, device)
    subject_results = stage_per_subject_lora(model, subject_loaders, args, device, lp_checkpoint)
    summarize_subject_results(subject_results, "Two-Stage Fine-Tuning — Summary")


def run_global_lora(model, pos_bank, args, device):
    pooled_loaders, _ = load_loaders_per_subject(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects
    )
    lp_checkpoint = stage_linear_probing(model, pooled_loaders, args, device)
    stage_global_lora(model, pooled_loaders, args, device, lp_checkpoint)


def run_three_stage(model, pos_bank, args, device):
    pooled_loaders, subject_loaders = load_loaders_per_subject(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects
    )
    lp_checkpoint = stage_linear_probing(model, pooled_loaders, args, device)
    model, global_checkpoint = stage_global_lora(model, pooled_loaders, args, device, lp_checkpoint)
    subject_results = stage_per_subject_lora(model, subject_loaders, args, device, global_checkpoint)
    summarize_subject_results(subject_results, "Three-Stage Fine-Tuning — Summary")
