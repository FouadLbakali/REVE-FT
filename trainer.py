import copy
import torch
from engine import train_one_epoch, eval_model


def make_scheduler(name, optimizer, epochs):
    """Returns (scheduler, needs_metric). needs_metric=True means call .step(val_metric)."""
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3), True
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), False
    if name == "constant":
        return None, False
    raise ValueError(f"Unknown scheduler: {name}")


def _step_scheduler(scheduler, needs_metric, metric):
    if scheduler is None:
        return
    if needs_metric:
        scheduler.step(metric)
    else:
        scheduler.step()


def train_loop(model, train_loader, val_loader, optimizer, epochs, device,
               label="", save_module=None, scheduler_name="plateau"):
    """Standard train/val loop. Returns (best_state, history) where history is a
    list of {epoch, train_acc, val_balanced_acc} dicts. save_module defaults to
    the full model."""
    save_module = save_module if save_module is not None else model
    scheduler, needs_metric = make_scheduler(scheduler_name, optimizer, epochs)

    best_val = -float("inf")
    best_state = copy.deepcopy(save_module.state_dict())
    history = []
    prefix = f"[{label}] " if label else ""

    for epoch in range(epochs):
        print(f"\n{prefix}Epoch {epoch + 1}/{epochs}")
        _, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        b_acc = eval_model(model, val_loader, device)["balanced_acc"]
        if b_acc > best_val:
            best_val = b_acc
            best_state = copy.deepcopy(save_module.state_dict())
        print(f"Train acc: {train_acc:.4f} | Val balanced_acc: {b_acc:.4f} (best: {best_val:.4f})")
        history.append({"epoch": epoch + 1,
                        "train_acc": float(train_acc),
                        "val_balanced_acc": float(b_acc)})
        _step_scheduler(scheduler, needs_metric, b_acc)

    return best_state, history


def print_metrics(results):
    for k in ("acc", "balanced_acc", "cohen_kappa", "f1", "auroc", "auc_pr"):
        print(f"  {k}: {results[k]:.4f}")
