import copy
import os

import torch
from peft import LoraConfig, get_peft_model

from data import load_loaders, load_loaders_per_subject, load_lora_targets
from engine import (
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
from luna_zoo import LunaSpec
from multilora import inject_multi_subject_lora, merge_subject_lora
from trainer import train_loop, print_metrics, make_scheduler, EarlyStopper


def _floatify(d):
    return {k: float(v) for k, v in d.items()}


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


def make_lora_config(rank, model_name="reve"):
    return LoraConfig(
        r=rank,
        lora_alpha=32,
        target_modules=load_lora_targets(model_name),
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["final_layer"],
    )


def _print_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def save_all_global(merged_model, save_dir, model_name):
    """Persist a `global`-mode run: the full merged model (backbone + Global LoRA
    merged + head) as <save_dir>/<model_name>.pt."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save(merged_model.state_dict(), path)
    print(f"Merged model saved to {path}")


def save_all_multilora(model, subject_ids, save_dir, model_name):
    """Persist a multi-LoRA run: one self-contained merged model per subject
    (that subject's LoRA — and the global adapter, if any — folded into the
    backbone, with the shared head) as <save_dir>/<model_name>_subject_<id>.pt.

    `subject_ids` is the iterable of `subject_loaders` keys, i.e. the 1-based raw
    subject ids (matching the per_subject ids in the results JSON). The matching
    multi-LoRA adapter index is the 0-based `subject_id` used at routing time
    (`raw - 1`), so adapter index = key - 1 and the file uses the 1-based key."""
    os.makedirs(save_dir, exist_ok=True)
    model = model.cpu()
    subject_ids = sorted(subject_ids)
    for raw_id in subject_ids:
        merged = merge_subject_lora(model, raw_id - 1)
        path = os.path.join(save_dir, f"{model_name}_subject_{raw_id:02d}.pt")
        torch.save(merged.state_dict(), path)
        del merged
    print(f"Saved {len(subject_ids)} per-subject merged models under {save_dir}")


# --------------------------------------------------------------------------- #
# Stages used by the multi-stage runs                                          #
# --------------------------------------------------------------------------- #

def stage_global_lora(model, pooled_loaders, args, device, checkpoint, results=None):
    pooled_train, pooled_val, pooled_test = pooled_loaders

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

    if results is not None:
        ps_block = _format_per_subject(per_subject)
        results.setdefault("stages", {})["gl"] = {
            "history": history,
            "test": _floatify(test_metrics),
            "per_subject": ps_block,
        }

    merged_model = lora_model.merge_and_unload()
    return merged_model, copy.deepcopy(merged_model.state_dict())


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


def run_global(model, pos_bank, args, device, results=None):
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
    print("  Global mode — head + Global LoRA (no separate LP phase)")
    print("=" * 60)
    fresh_checkpoint = copy.deepcopy(model.state_dict())
    merged_model, _ = stage_global_lora(
        model, pooled_loaders, args, device, fresh_checkpoint, results=results
    )

    if getattr(args, "save_all", None):
        save_all_global(merged_model, args.save_all, args.model)


def _trainable_state(model):
    return {n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def _load_trainable_state(model, state):
    own = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in state.items():
            own[n].copy_(t.to(own[n].device))


def _run_multilora(model, pos_bank, args, device, global_rank,
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
        target_suffixes=load_lora_targets(args.model),
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

    if getattr(args, "save_all", None):
        save_all_multilora(model, subject_loaders.keys(), args.save_all, args.model)


def run_subject_specific(model, pos_bank, args, device, results=None):
    _run_multilora(
        model, pos_bank, args, device, global_rank=0,
        results_key="multilora",
        banner="Subject-specific — shared head + per-subject LoRA",
        results=results,
    )


def run_stacked(model, pos_bank, args, device, results=None):
    """Global LoRA and per-subject multi-LoRA trained at the same time: each
    targeted layer applies a shared global adapter plus its per-sample routed
    subject adapter, jointly with the shared head, in one forward/backward."""
    _run_multilora(
        model, pos_bank, args, device, global_rank=args.gl_rank,
        results_key="multilora_global",
        banner="Stacked — per-subject LoRA + Global LoRA, shared head",
        results=results,
    )
