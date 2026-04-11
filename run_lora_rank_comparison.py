"""
Compare two-stage training with different LoRA ranks (1, 2, 4, 8, 16).
Plots the training loss evolution for each rank.
"""

import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, set_seed
from peft import LoraConfig, get_peft_model
import sys, types

sys.path.insert(0, "reve-repro-main/src")

from data import load_bciciv2a_per_subject
from engine import train_one_epoch, eval_model

SEED = 42
EPOCHS_FT = 15
LR_FT = 1e-4
BATCH_SIZE = 64
LORA_RANKS = [1, 2, 4, 8, 16]
FINAL_LAYER_PATH = "best_head.pt"


def build_model():
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, dtype="auto")
    dim = 22 * 5 * 512
    model.final_layer = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.RMSNorm(dim),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(dim, 4),
    )
    return model


def run_two_stage(model, pos_bank, device, lora_rank):
    """Load pretrained final layer, then run per-subject LoRA fine-tuning."""
    pooled_loaders, subject_loaders = load_bciciv2a_per_subject(pos_bank, BATCH_SIZE, seed=SEED)

    # ---------- Load pretrained final layer (replaces Stage 1) ----------
    state = torch.load(FINAL_LAYER_PATH, weights_only=True)
    model.final_layer.load_state_dict(state)
    print(f"  Final layer loaded from {FINAL_LAYER_PATH}")
    model.to(device)
    lp_checkpoint = copy.deepcopy(model.state_dict())

    # ---------- Stage 2: Per-subject LoRA ----------
    stage2_losses_per_epoch = [[] for _ in range(EPOCHS_FT)]  # [epoch] -> list of per-subject losses

    for subj, (subj_train, subj_val, subj_test) in sorted(subject_loaders.items()):
        print(f"  [FT] Subject {subj}, rank={lora_rank}")
        model.load_state_dict(lp_checkpoint)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            target_modules=["to_qkv", "to_out"],
            lora_dropout=0.05,
            bias="none",
            modules_to_save=["final_layer"],
        )
        lora_model = get_peft_model(model, lora_config)
        lora_model.to(device)

        trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
        optimizer_ft = torch.optim.AdamW(trainable_params, lr=LR_FT)
        scheduler_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode="max", factor=0.5, patience=3)

        for epoch in range(EPOCHS_FT):
            train_loss, train_acc = train_one_epoch(lora_model, optimizer_ft, subj_train, device)
            m = eval_model(lora_model, subj_val, device)
            stage2_losses_per_epoch[epoch].append(train_loss)
            scheduler_ft.step(m["balanced_acc"])
            print(f"    epoch {epoch+1:2d}/{EPOCHS_FT} — loss={train_loss:.4f} — train_acc={train_acc:.4f} — val_acc={m['acc']:.4f} — val_bal_acc={m['balanced_acc']:.4f}")

        model = lora_model.merge_and_unload()

    # Average stage 2 loss across subjects per epoch
    stage2_losses = [np.mean(epoch_losses) for epoch_losses in stage2_losses_per_epoch]

    return stage2_losses


def main():
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, dtype="auto")

    all_losses = {}  # rank -> list of stage2 losses

    for rank in LORA_RANKS:
        print(f"\n{'='*60}")
        print(f"  LoRA rank = {rank}")
        print(f"{'='*60}")

        model = build_model()
        stage2_losses = run_two_stage(model, pos_bank, device, rank)

        all_losses[rank] = stage2_losses
        print(f"  Done rank={rank} — final loss: {all_losses[rank][-1]:.4f}")

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(8, 5))

    for rank, losses in all_losses.items():
        ax.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3, label=f"rank={rank}")

    ax.set_xlabel("Fine-tuning Epoch")
    ax.set_ylabel("Training Loss (avg across subjects)")
    ax.set_title("LoRA Fine-Tuning Loss by Rank")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lora_rank_comparison.png", dpi=150)
    print(f"\nPlot saved to lora_rank_comparison.png")
    plt.show()


if __name__ == "__main__":
    import os, datetime
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/lora_rank_comparison_{timestamp}.log"
    log_file = open(log_path, "w")
    sys.stdout = type("Tee", (), {
        "write": lambda self, msg: (log_file.write(msg), log_file.flush(), sys.__stdout__.write(msg)),
        "flush": lambda self: (log_file.flush(), sys.__stdout__.flush()),
    })()
    sys.stderr = type("Tee", (), {
        "write": lambda self, msg: (log_file.write(msg), log_file.flush(), sys.__stderr__.write(msg)),
        "flush": lambda self: (log_file.flush(), sys.__stderr__.flush()),
    })()
    print(f"Logging to {log_path}")
    main()
    log_file.close()
