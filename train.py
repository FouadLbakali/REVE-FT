import argparse
import copy
from transformers import AutoModel
import torch
from transformers import set_seed
from data import load_bciciv2a, load_bciciv2a_per_subject
from engine import train_one_epoch, eval_model
import sys, types
sys.path.insert(0, "reve-repro-main/src")
from models.lora import get_lora_config, CustomGetLora
from peft import LoraConfig, get_peft_model
import types

def main():
    parser = argparse.ArgumentParser(description='REVE Training')
    parser.add_argument('--mode', default="linear", choices=["linear", "full", "lora", "two_stage"])
    parser.add_argument('--epochs', default=15, type=int, help='number of epoch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--seed', default=None, type=int, help='seed')
    parser.add_argument('--save-final-layer', default=None, type=str, help='path to save the best final layer (linear mode)')
    parser.add_argument('--load-final-layer', default=None, type=str, help='path to load a pretrained final layer')
    # Two-stage specific args
    parser.add_argument('--ft-epochs', default=15, type=int, help='stage 2 fine-tuning epochs')
    parser.add_argument('--ft-lr', default=1e-4, type=float, help='stage 2 learning rate')
    parser.add_argument('--lora-rank', default=8, type=int, help='LoRA rank for stage 2')
    args = parser.parse_args()

    # Loading REVE
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, dtype="auto")
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, dtype="auto")

    # Add Classification Head
    dim = 22 * 5 * 512
    model.final_layer = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.RMSNorm(dim),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(dim, 4),
    )

    if args.load_final_layer:
        state = torch.load(args.load_final_layer, weights_only=True)
        model.final_layer.load_state_dict(state)
        print(f"Final layer loaded from {args.load_final_layer}")

    if args.seed:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "two_stage":
        run_two_stage(model, pos_bank, args, device)
    else:
        run_single_stage(model, pos_bank, args, device)


def run_single_stage(model, pos_bank, args, device):
    # Load Dataset
    train_loader, val_loader, test_loader = load_bciciv2a(pos_bank, args.batch_size, seed=args.seed)

    mode = args.mode
    if mode == "linear":
        params = list(model.final_layer.parameters())
    elif mode == "full":
        params = list(model.parameters())
    elif mode == "lora":
        lora_config = get_lora_config(types.SimpleNamespace(encoder=model), rank=8, apply_to=("patch", "mlp4d", "attention", "ffw"))
        model = CustomGetLora(lora_config).get_model(model)
        model.final_layer.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]

    total = sum(p.numel() for p in params)
    print(f"Total paramètres : {total:,}")

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    model.to(device)

    best_val_acc = 0
    best_final_layer = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        b_acc = eval_model(model, val_loader, device)["balanced_acc"]
        if b_acc > best_val_acc:
            best_val_acc = b_acc
            best_final_layer = model.final_layer.state_dict()
        print(f"Train acc: {train_acc:.4f} | Validation balanced accuracy: {b_acc:.4f}, best: {best_val_acc:.4f}")
        scheduler.step(b_acc)

    model.final_layer.load_state_dict(best_final_layer)

    if args.mode == "linear" and args.save_final_layer:
        torch.save(best_final_layer, args.save_final_layer)
        print(f"Final layer saved to {args.save_final_layer}")

    results = eval_model(model, test_loader, device)

    # Results
    print("acc", results["acc"])
    print("balanced_acc", results["balanced_acc"])
    print("cohen_kappa", results["cohen_kappa"])
    print("f1", results["f1"])
    print("auroc", results["auroc"])
    print("auc_pr", results["auc_pr"])


def run_two_stage(model, pos_bank, args, device):
    pooled_loaders, subject_loaders = load_bciciv2a_per_subject(pos_bank, args.batch_size, seed=args.seed)
    pooled_train, pooled_val, pooled_test = pooled_loaders

    if args.load_final_layer:
        # ===================== Skip Stage 1: final layer already loaded =====================
        print("=" * 60)
        print("  Stage 1 — SKIPPED (final layer loaded from checkpoint)")
        print("=" * 60)
        model.to(device)
        lp_checkpoint = copy.deepcopy(model.state_dict())
    else:
        # ===================== Stage 1: Linear Probing =====================
        print("=" * 60)
        print("  Stage 1 — Linear Probing (all subjects)")
        print("=" * 60)

        for param in model.parameters():
            param.requires_grad = False
        for param in model.final_layer.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        model.to(device)

        optimizer_lp = torch.optim.AdamW(model.final_layer.parameters(), lr=args.lr)
        scheduler_lp = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lp, mode="max", factor=0.5, patience=3)

        best_val_acc = 0.0
        best_head_state = None

        for epoch in range(args.epochs):
            print(f"\n[LP] Epoch {epoch + 1}/{args.epochs}")
            train_loss, train_acc = train_one_epoch(model, optimizer_lp, pooled_train, device)
            b_acc = eval_model(model, pooled_val, device)["balanced_acc"]
            if b_acc > best_val_acc:
                best_val_acc = b_acc
                best_head_state = copy.deepcopy(model.final_layer.state_dict())
            print(f"Train acc: {train_acc:.4f} | Val balanced_acc: {b_acc:.4f} (best: {best_val_acc:.4f})")
            scheduler_lp.step(b_acc)

        model.final_layer.load_state_dict(best_head_state)
        lp_test = eval_model(model, pooled_test, device)
        print(f"\n--- Stage 1 (LP) test results ---")
        for k, v in lp_test.items():
            print(f"  {k}: {v:.4f}")

        # Save Stage 1 checkpoint
        lp_checkpoint = copy.deepcopy(model.state_dict())

    # ===================== Stage 2: Per-subject LoRA =====================
    subject_results = {}

    for subj, (subj_train, subj_val, subj_test) in sorted(subject_loaders.items()):
        print(f"\n{'=' * 60}")
        print(f"  Stage 2 — Subject {subj}")
        print(f"{'=' * 60}")
        print(f"  Trials — train: {len(subj_train.dataset)}, val: {len(subj_val.dataset)}, test: {len(subj_test.dataset)}")

        # Restore from LP checkpoint
        model.load_state_dict(lp_checkpoint)

        # Apply LoRA via peft
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            target_modules=["to_qkv", "to_out"],
            lora_dropout=0.05,
            bias="none",
            modules_to_save=["final_layer"],
        )
        lora_model = get_peft_model(model, lora_config)
        lora_model.to(device)

        trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
        optimizer_ft = torch.optim.AdamW(trainable_params, lr=args.ft_lr)
        scheduler_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode="max", factor=0.5, patience=3)

        best_val = 0.0
        best_state = None

        for epoch in range(args.ft_epochs):
            print(f"\n  [FT-S{subj}] Epoch {epoch + 1}/{args.ft_epochs}")
            train_loss, train_acc = train_one_epoch(lora_model, optimizer_ft, subj_train, device)
            m = eval_model(lora_model, subj_val, device)
            b_acc = m["balanced_acc"]
            if b_acc > best_val:
                best_val = b_acc
                best_state = copy.deepcopy(lora_model.state_dict())
            print(f"    Train acc: {train_acc:.4f} | Val balanced_acc: {b_acc:.4f} (best: {best_val:.4f})")
            scheduler_ft.step(b_acc)

        # Evaluate best model on subject's test set
        lora_model.load_state_dict(best_state)
        test_metrics = eval_model(lora_model, subj_test, device)
        subject_results[subj] = test_metrics

        print(f"\n  --- Subject {subj} test results ---")
        for k, v in test_metrics.items():
            print(f"    {k}: {v:.4f}")

        # Unwrap LoRA for next iteration
        model = lora_model.merge_and_unload()

    # ===================== Summary =====================
    print(f"\n{'=' * 60}")
    print("  Two-Stage Fine-Tuning — Summary")
    print(f"{'=' * 60}")
    metric_keys = ["acc", "balanced_acc", "cohen_kappa", "f1", "auroc", "auc_pr"]
    for k in metric_keys:
        values = [subject_results[s][k] for s in sorted(subject_results)]
        import numpy as np
        mean_v = np.mean(values)
        std_v = np.std(values)
        print(f"  {k}: {mean_v:.4f} +/- {std_v:.4f}")


if __name__ == "__main__":
    main()
