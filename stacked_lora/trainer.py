import copy
import math
import torch
from .engine import train_one_epoch, eval_model

def make_scheduler(optimizer, epochs, steps_per_epoch, max_lr=1.0e-4):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,                     # 1.0e-4 par défaut dans le config.yaml
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,   # Nombre de batchs dans votre DataLoader
        pct_start=0.1,                     # Correspond à vos 10% de warmup
        anneal_strategy='cos'              # Cosine-annealing
    )

class EarlyStopper:
    """Tracks the best val metric and signals when no improvement has occurred
    for `patience` consecutive epochs. patience <= 0 disables early stopping."""

    def __init__(self, patience):
        self.patience = patience
        self.epochs_without_improve = 0

    def step(self, improved):
        if self.patience <= 0:
            return False
        self.epochs_without_improve = 0 if improved else self.epochs_without_improve + 1
        return self.epochs_without_improve >= self.patience


def train_loop(model, train_loader, val_loader, optimizer, epochs, device,
               label="", save_module=None, patience=0):
    """Standard train/val loop. Returns (best_state, history) where history is a
    list of {epoch, train_acc, val_balanced_acc} dicts. save_module defaults to
    the full model. patience <= 0 disables early stopping."""
    save_module = save_module if save_module is not None else model
    max_lr = optimizer.param_groups[0]["lr"]
    scheduler = make_scheduler(optimizer, epochs, len(train_loader), max_lr=max_lr)
    stopper = EarlyStopper(patience)

    best_val = -float("inf")
    best_state = copy.deepcopy(save_module.state_dict())
    history = []
    prefix = f"[{label}] " if label else ""

    for epoch in range(epochs):
        print(f"\n{prefix}Epoch {epoch + 1}/{epochs}")
        _, train_acc = train_one_epoch(model, optimizer, train_loader, device,
                                       scheduler=scheduler)
        b_acc = eval_model(model, val_loader, device)["balanced_acc"]
        improved = b_acc > best_val
        if improved:
            best_val = b_acc
            best_state = copy.deepcopy(save_module.state_dict())
        print(f"Train acc: {train_acc:.4f} | Val balanced_acc: {b_acc:.4f} (best: {best_val:.4f})")
        history.append({"epoch": epoch + 1,
                        "train_acc": float(train_acc),
                        "val_balanced_acc": float(b_acc)})
        if stopper.step(improved):
            print(f"{prefix}Early stopping at epoch {epoch + 1} "
                  f"(no improvement for {patience} epochs)")
            break

    return best_state, history


def print_metrics(results):
    for k in ("acc", "balanced_acc", "cohen_kappa", "f1", "auroc", "auc_pr"):
        print(f"  {k}: {results[k]:.4f}")
