"""
LR Range Test (Smith 2018) — finds the optimal learning rate for a given mode.

Algorithm:
  1. Start with a very low LR (init_lr)
  2. Increase LR exponentially over `num_iter` batches up to final_lr
  3. Record the smoothed loss at each step
  4. Plot loss vs LR (log scale)
  5. Suggest the LR at the steepest loss descent

Usage:
  python lr_finder.py --mode subject_prompt --num-prompt 4
  python lr_finder.py --mode query_token --num-prompt 2
  python lr_finder.py --mode linear
"""

import argparse
import copy
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoModel, set_seed

from data import load_bciciv2a

sys.path.insert(0, "reve-repro-main/src")
from models.lora import get_lora_config, CustomGetLora

# Import model classes from train_v2
from train_v2 import ReveSubjectPrompt, ReveQueryToken


# ──────────────────────────────────────────────────────────────────────────────
# LR Finder
# ──────────────────────────────────────────────────────────────────────────────

class LRFinder:
    """
    Runs the LR range test over the training loader.
    Records (lr, smoothed_loss) at each batch step.
    """

    def __init__(self, model, optimizer, criterion, device, use_subject_id=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_subject_id = use_subject_id

        self._best_model_state = copy.deepcopy(model.state_dict())
        self._best_optim_state = copy.deepcopy(optimizer.state_dict())

    def range_test(self, loader, init_lr=1e-7, final_lr=10.0,
                   num_iter=None, smooth_f=0.05, diverge_th=5.0):
        """
        Args:
            loader:      training DataLoader
            init_lr:     starting learning rate
            final_lr:    ending learning rate
            num_iter:    number of batches to run (default: full epoch)
            smooth_f:    exponential smoothing factor for loss
            diverge_th:  stop if loss > diverge_th * best_loss
        Returns:
            lrs, losses  (lists)
        """
        if num_iter is None:
            num_iter = len(loader)

        # Set initial LR
        for pg in self.optimizer.param_groups:
            pg["lr"] = init_lr

        lr_mult = (final_lr / init_lr) ** (1.0 / (num_iter - 1))

        lrs, losses = [], []
        smoothed_loss = 0.0
        best_loss = float("inf")

        self.model.train()
        data_iter = iter(loader)

        for step in range(num_iter):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            data   = batch["sample"].to(self.device, non_blocking=True)
            target = batch["label"].to(self.device, non_blocking=True)
            pos    = batch["pos"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with torch.amp.autocast(
                dtype=torch.float16,
                device_type="cuda" if torch.cuda.is_available() else "cpu",
            ):
                if self.use_subject_id:
                    sid = batch["subject_id"].to(self.device, non_blocking=True)
                    output = self.model(data, pos, sid)
                else:
                    output = self.model(data, pos)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            raw_loss = loss.item()
            # Exponential smoothing
            smoothed_loss = smooth_f * raw_loss + (1 - smooth_f) * smoothed_loss if step > 0 else raw_loss
            # Bias correction
            debiased = smoothed_loss / (1 - (1 - smooth_f) ** (step + 1))

            current_lr = self.optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            losses.append(debiased)

            if debiased < best_loss:
                best_loss = debiased

            # Divergence check
            if step > 0 and debiased > diverge_th * best_loss:
                print(f"  Loss diverged at step {step} (lr={current_lr:.2e}), stopping early.")
                break

            # Update LR for next step
            for pg in self.optimizer.param_groups:
                pg["lr"] *= lr_mult

        # Restore original model/optimizer state
        self.model.load_state_dict(self._best_model_state)
        self.optimizer.load_state_dict(self._best_optim_state)

        return lrs, losses

    @staticmethod
    def suggest_lr(lrs, losses):
        """Return the LR at the steepest loss gradient (most negative slope)."""
        losses = np.array(losses)
        lrs = np.array(lrs)
        gradients = np.gradient(losses, np.log10(lrs))
        min_grad_idx = np.argmin(gradients)
        return lrs[min_grad_idx]

    @staticmethod
    def plot(lrs, losses, suggested_lr, save_path, title="LR Range Test"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lrs, losses, linewidth=1.5, label="Smoothed loss")
        ax.axvline(suggested_lr, color="red", linestyle="--",
                   label=f"Suggested LR = {suggested_lr:.2e}")
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"LR range test plot saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Build model (mirrors train_v2.py logic)
# ──────────────────────────────────────────────────────────────────────────────

def build_model(mode, reve_model, num_subject, num_prompt):
    if mode == "linear":
        model = reve_model
        dim = 22 * 5 * 512
        model.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.RMSNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, 4),
        )
        params = list(model.final_layer.parameters())
        use_subject_id = False

    elif mode == "full":
        model = reve_model
        dim = 22 * 5 * 512
        model.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.RMSNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, 4),
        )
        params = list(model.parameters())
        use_subject_id = False

    elif mode == "lora":
        model = reve_model
        dim = 22 * 5 * 512
        model.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.RMSNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, 4),
        )
        lora_config = get_lora_config(
            types.SimpleNamespace(encoder=model), rank=8,
            apply_to=("patch", "mlp4d", "attention", "ffw"),
        )
        model = CustomGetLora(lora_config).get_model(model)
        model.final_layer.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
        use_subject_id = False

    elif mode == "query_token":
        model = ReveQueryToken(reve_model, num_prompt=num_prompt, num_classes=4)
        params = [p for p in model.parameters() if p.requires_grad]
        use_subject_id = False

    elif mode == "subject_prompt":
        model = ReveSubjectPrompt(reve_model, num_subject=num_subject,
                                  num_prompt=num_prompt, num_classes=4)
        params = [p for p in model.parameters() if p.requires_grad]
        use_subject_id = True

    print(f"Trainable params: {sum(p.numel() for p in params):,}")
    return model, params, use_subject_id


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LR Range Test for REVE")
    parser.add_argument("--mode", default="linear",
                        choices=["linear", "full", "lora", "query_token", "subject_prompt"])
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num-prompt", default=4, type=int)
    parser.add_argument("--num-subject", default=9, type=int)
    parser.add_argument("--init-lr", default=1e-7, type=float, help="starting LR")
    parser.add_argument("--final-lr", default=10.0, type=float, help="ending LR")
    parser.add_argument("--num-iter", default=None, type=int,
                        help="number of batches (overrides --epochs if set)")
    parser.add_argument("--epochs", default=1, type=int,
                        help="number of epochs to sweep over (sets num_iter = epochs * len(loader))")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reve_model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, dtype="auto")
    pos_bank   = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, dtype="auto")

    train_loader, _, _ = load_bciciv2a(pos_bank, args.batch_size, seed=args.seed)

    model, params, use_subject_id = build_model(
        args.mode, reve_model, args.num_subject, args.num_prompt
    )
    model.to(device)

    optimizer = torch.optim.AdamW(params, lr=args.init_lr)
    criterion = torch.nn.CrossEntropyLoss()

    finder = LRFinder(model, optimizer, criterion, device, use_subject_id=use_subject_id)

    num_iter = args.num_iter if args.num_iter is not None else args.epochs * len(train_loader)
    print(f"\nRunning LR range test: {args.init_lr:.0e} → {args.final_lr:.0e}  ({num_iter} batches / {args.epochs} epochs)")
    lrs, losses = finder.range_test(
        train_loader,
        init_lr=args.init_lr,
        final_lr=args.final_lr,
        num_iter=num_iter,
    )

    suggested_lr = LRFinder.suggest_lr(lrs, losses)
    print(f"\nSuggested LR: {suggested_lr:.2e}")

    curve_name = f"lr_finder_{args.mode}"
    if args.mode in ("query_token", "subject_prompt"):
        curve_name += f"_np{args.num_prompt}"
    title = f"LR Range Test — {args.mode}"
    if args.mode in ("query_token", "subject_prompt"):
        title += f"  (num_prompt={args.num_prompt})"

    LRFinder.plot(lrs, losses, suggested_lr, f"{curve_name}.png", title=title)


if __name__ == "__main__":
    main()
