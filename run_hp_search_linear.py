import itertools
import copy
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, set_seed
from data import load_bciciv2a
from engine import train_one_epoch, eval_model

DROPOUTS = [0.1, 0.5, 0.7]
WEIGHT_DECAYS = [0.01, 0.1, 0.2]

EPOCHS = 15
LR = 0.001
BATCH_SIZE = 64
SEED = 42

def build_head(dim, dropout):
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.RMSNorm(dim),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(dim, 4),
    )


def run_one(model_base, head_state_template, pos_bank, dropout, weight_decay, device):
    """Train linear mode with given dropout and weight_decay, return best val acc and test metrics."""
    model = copy.deepcopy(model_base)
    dim = 22 * 5 * 512
    model.final_layer = build_head(dim, dropout)
    model.to(device)

    train_loader, val_loader, test_loader = load_bciciv2a(pos_bank, BATCH_SIZE, seed=SEED)

    params = list(model.final_layer.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0.0
    best_final_layer = None

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        b_acc = eval_model(model, val_loader, device)["balanced_acc"]
        if b_acc > best_val_acc:
            best_val_acc = b_acc
            best_final_layer = copy.deepcopy(model.final_layer.state_dict())
        print(f"  Epoch {epoch+1}/{EPOCHS} | train_acc: {train_acc:.4f} | val_bacc: {b_acc:.4f} (best: {best_val_acc:.4f})")
        scheduler.step(b_acc)

    model.final_layer.load_state_dict(best_final_layer)
    test_results = eval_model(model, test_loader, device)
    return best_val_acc, test_results, best_final_layer


def main():
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading REVE model...")
    model_base = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, dtype="auto")
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, dtype="auto")

    # Freeze backbone
    for param in model_base.parameters():
        param.requires_grad = False

    combos = list(itertools.product(DROPOUTS, WEIGHT_DECAYS))
    results = []

    for i, (dropout, wd) in enumerate(combos):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(combos)}] dropout={dropout}, weight_decay={wd}")
        print(f"{'='*60}")
        set_seed(SEED)
        best_val, test_metrics, head_state = run_one(model_base, None, pos_bank, dropout, wd, device)
        results.append({
            "dropout": dropout,
            "weight_decay": wd,
            "best_val_bacc": best_val,
            "head_state": head_state,
            **test_metrics,
        })

    # Summary
    print(f"\n{'='*60}")
    print("  Hyperparameter Search Results")
    print(f"{'='*60}")
    print(f"{'dropout':>10} {'wd':>10} {'val_bacc':>10} {'test_bacc':>10} {'test_acc':>10} {'kappa':>10} {'f1':>10}")
    print("-" * 70)

    best = None
    for r in results:
        print(f"{r['dropout']:>10.2f} {r['weight_decay']:>10.3f} {r['best_val_bacc']:>10.4f} {r['balanced_acc']:>10.4f} {r['acc']:>10.4f} {r['cohen_kappa']:>10.4f} {r['f1']:>10.4f}")
        if best is None or r["best_val_bacc"] > best["best_val_bacc"]:
            best = r

    print(f"\nBest config: dropout={best['dropout']}, weight_decay={best['weight_decay']}")
    print(f"  Val balanced_acc:  {best['best_val_bacc']:.4f}")
    print(f"  Test balanced_acc: {best['balanced_acc']:.4f}")
    print(f"  Test acc:          {best['acc']:.4f}")
    print(f"  Cohen kappa:       {best['cohen_kappa']:.4f}")
    print(f"  F1:                {best['f1']:.4f}")
    print(f"  AUROC:             {best['auroc']:.4f}")
    print(f"  AUC-PR:            {best['auc_pr']:.4f}")

    # Save best head
    torch.save(best["head_state"], "best_head.pt")
    print(f"\nBest final layer saved to best_head.pt")

    # Comparison plot
    labels = [f"d={r['dropout']}\nwd={r['weight_decay']}" for r in results]
    val_baccs = [r["best_val_bacc"] for r in results]
    test_baccs = [r["balanced_acc"] for r in results]

    x = np.arange(len(results))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_val = ax.bar(x - width / 2, val_baccs, width, label="Val balanced acc")
    bars_test = ax.bar(x + width / 2, test_baccs, width, label="Test balanced acc")

    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Hyperparameter Search — Linear Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1)

    # Highlight best config
    best_idx = val_baccs.index(max(val_baccs))
    bars_val[best_idx].set_edgecolor("red")
    bars_val[best_idx].set_linewidth(2)
    bars_test[best_idx].set_edgecolor("red")
    bars_test[best_idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig("hp_search_comparison.png", dpi=150)
    print("Comparison plot saved to hp_search_comparison.png")


if __name__ == "__main__":
    import os, datetime
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/hp_search_linear_{timestamp}.log"
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
