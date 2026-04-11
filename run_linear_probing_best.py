"""Linear Probing — Recherche exhaustive de la meilleure configuration.

Applique les meilleures pratiques:
  - Backbone strictement gelé (requires_grad=False)
  - Features extraites UNE SEULE FOIS et mises en cache (grid search rapide)
  - Standardisation des features (StandardScaler fit sur train uniquement)
  - Tête purement linéaire (pas de couches cachées — définition stricte du LP)
  - Optimiseurs testés: AdamW, SGD+momentum, L-BFGS
  - Learning rates agressifs (1e-3 -> 1e-1) — pas de risque puisque backbone gelé
  - Weight decay large grille logarithmique (1e-5 -> 1e-1)
  - Cosine Annealing LR schedule
  - Weighted Cross-Entropy pour gérer le déséquilibre des classes
"""

import copy
import datetime
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModel, set_seed

from data import load_bciciv2a

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 50  # Augmenté: la tête est minuscule, chaque epoch est très rapide
N_CLASSES = 4
FEATURE_DIM = 22 * 5 * 512  # dimensions des features du backbone REVE

# Grille d'hyperparamètres — suit les recommandations du cours
OPTIMIZERS = ["adamw", "sgd", "lbfgs"]
LEARNING_RATES = {
    "adamw": [1e-3, 1e-2, 1e-1],      # agressif puisque backbone gelé
    "sgd":   [1e-2, 1e-1, 3e-1],      # SGD préfère des LR plus grands
    "lbfgs": [1.0],                    # L-BFGS gère son propre step size
}
WEIGHT_DECAYS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # large grille log


# ---------------------------------------------------------------------------
# Extraction (une seule fois) des features du backbone gelé
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model_base, loader, device):
    """Exécute le backbone sur un loader et retourne (features, labels).

    Le `final_layer` du modèle est temporairement remplacé par Identity pour
    récupérer le tenseur avant la tête de classification, puis aplati.
    """
    model = copy.deepcopy(model_base)
    model.final_layer = nn.Identity()
    model.to(device).eval()

    feats, labels = [], []
    for batch in tqdm(loader, desc="Extracting features"):
        x = batch["sample"].to(device, non_blocking=True)
        pos = batch["pos"].to(device, non_blocking=True)
        y = batch["label"]
        with torch.amp.autocast(dtype=torch.float16, device_type=device.type):
            out = model(x, pos)
        feats.append(out.float().flatten(start_dim=1).cpu())
        labels.append(y)

    del model
    torch.cuda.empty_cache()
    return torch.cat(feats), torch.cat(labels)


# ---------------------------------------------------------------------------
# Entraînement d'une tête linéaire sur features cachées
# ---------------------------------------------------------------------------
def build_linear_head():
    """Tête STRICTEMENT linéaire — pas de couche cachée (définition du LP)."""
    return nn.Linear(FEATURE_DIM, N_CLASSES)


def make_optimizer(name, params, lr, weight_decay):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if name == "lbfgs":
        # L-BFGS ne supporte pas directement weight_decay — on l'ajoute manuellement dans la loss
        return torch.optim.LBFGS(params, lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    raise ValueError(name)


def train_head(
    train_feats, train_y, val_feats, val_y,
    optimizer_name, lr, weight_decay, class_weights, device,
):
    """Entraîne une tête linéaire et retourne (best_val_bacc, best_state)."""
    head = build_linear_head().to(device)
    optimizer = make_optimizer(optimizer_name, head.parameters(), lr, weight_decay)

    # Cosine annealing — recommandé par le cours
    if optimizer_name != "lbfgs":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    else:
        scheduler = None

    # Weighted CE pour gérer le déséquilibre des classes
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_ds = TensorDataset(train_feats, train_y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_feats_d = val_feats.to(device)
    val_y_d = val_y.to(device)

    best_val, best_state = 0.0, None

    if optimizer_name == "lbfgs":
        # L-BFGS: full-batch, très peu d'itérations extérieures
        X = train_feats.to(device)
        y = train_y.to(device)
        for epoch in range(10):
            def closure():
                optimizer.zero_grad()
                logits = head(X)
                loss = criterion(logits, y)
                # L2 manuel (L-BFGS ne supporte pas weight_decay natif)
                if weight_decay > 0:
                    l2 = sum(p.pow(2).sum() for p in head.parameters())
                    loss = loss + weight_decay * l2
                loss.backward()
                return loss
            optimizer.step(closure)

            with torch.no_grad():
                preds = head(val_feats_d).argmax(1).cpu().numpy()
            bacc = balanced_accuracy_score(val_y.numpy(), preds)
            if bacc > best_val:
                best_val = bacc
                best_state = copy.deepcopy(head.state_dict())
    else:
        for epoch in range(EPOCHS):
            head.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(head(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            head.eval()
            with torch.no_grad():
                preds = head(val_feats_d).argmax(1).cpu().numpy()
            bacc = balanced_accuracy_score(val_y.numpy(), preds)
            if bacc > best_val:
                best_val = bacc
                best_state = copy.deepcopy(head.state_dict())

    return best_val, best_state


# ---------------------------------------------------------------------------
# Évaluation finale sur le test set
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_head(head_state, test_feats, test_y, device):
    head = build_linear_head().to(device)
    head.load_state_dict(head_state)
    head.eval()
    logits = head(test_feats.to(device))
    probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    gt = test_y.numpy()

    return {
        "acc": float((preds == gt).mean()),
        "balanced_acc": balanced_accuracy_score(gt, preds),
        "cohen_kappa": cohen_kappa_score(gt, preds),
        "f1": f1_score(gt, preds, average="weighted"),
        "auroc": roc_auc_score(gt, probs, multi_class="ovr"),
        "auc_pr": average_precision_score(label_binarize(gt, classes=range(N_CLASSES)), probs, average="macro"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading REVE backbone...")
    model_base = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, dtype="auto")
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, dtype="auto")

    # Gel strict du backbone
    for p in model_base.parameters():
        p.requires_grad = False

    train_loader, val_loader, test_loader = load_bciciv2a(pos_bank, BATCH_SIZE, seed=SEED)

    # --- Extraction features (une seule fois) ---
    print("\nExtracting features from frozen backbone (one-shot)...")
    train_feats, train_y = extract_features(model_base, train_loader, device)
    val_feats, val_y = extract_features(model_base, val_loader, device)
    test_feats, test_y = extract_features(model_base, test_loader, device)
    print(f"  train: {tuple(train_feats.shape)} | val: {tuple(val_feats.shape)} | test: {tuple(test_feats.shape)}")

    # Libérer le backbone — plus besoin, tout le HP search utilise les features cachées
    del model_base
    torch.cuda.empty_cache()

    # --- Standardisation (fit sur train uniquement) ---
    mean = train_feats.mean(dim=0, keepdim=True)
    std = train_feats.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_feats = (train_feats - mean) / std
    val_feats = (val_feats - mean) / std
    test_feats = (test_feats - mean) / std
    print("  Features standardisées (mean=0, std=1) avec stats de train")

    # --- Class weights (weighted CE pour déséquilibre) ---
    counts = torch.bincount(train_y, minlength=N_CLASSES).float()
    class_weights = counts.sum() / (N_CLASSES * counts)
    print(f"  Class counts: {counts.tolist()} -> weights: {class_weights.tolist()}")

    # --- Grid search ---
    configs = []
    for opt in OPTIMIZERS:
        for lr in LEARNING_RATES[opt]:
            for wd in WEIGHT_DECAYS:
                configs.append({"optimizer": opt, "lr": lr, "weight_decay": wd})

    print(f"\n{'='*70}")
    print(f"  Grid search: {len(configs)} configurations")
    print(f"{'='*70}")

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1:3d}/{len(configs)}] opt={cfg['optimizer']:6s} lr={cfg['lr']:<8g} wd={cfg['weight_decay']:<8g}", end=" ")
        set_seed(SEED)
        best_val, best_state = train_head(
            train_feats, train_y, val_feats, val_y,
            cfg["optimizer"], cfg["lr"], cfg["weight_decay"], class_weights, device,
        )
        test_metrics = evaluate_head(best_state, test_feats, test_y, device)
        print(f"-> val_bacc={best_val:.4f} | test_bacc={test_metrics['balanced_acc']:.4f}")
        results.append({**cfg, "best_val_bacc": best_val, "head_state": best_state, **test_metrics})

    # --- Sélection meilleure config (sur val) ---
    best = max(results, key=lambda r: r["best_val_bacc"])

    # --- Résumé ---
    print(f"\n{'='*70}")
    print("  Résultats triés par val_balanced_acc")
    print(f"{'='*70}")
    print(f"{'opt':>6} {'lr':>10} {'wd':>10} {'val_bacc':>10} {'test_bacc':>10} {'kappa':>8} {'f1':>8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: -x["best_val_bacc"]):
        print(f"{r['optimizer']:>6} {r['lr']:>10g} {r['weight_decay']:>10g} "
              f"{r['best_val_bacc']:>10.4f} {r['balanced_acc']:>10.4f} "
              f"{r['cohen_kappa']:>8.4f} {r['f1']:>8.4f}")

    print(f"\n{'='*70}")
    print("  Meilleure configuration (sélectionnée sur val_bacc)")
    print(f"{'='*70}")
    print(f"  Optimizer:     {best['optimizer']}")
    print(f"  Learning rate: {best['lr']}")
    print(f"  Weight decay:  {best['weight_decay']}")
    print(f"  Val  bacc: {best['best_val_bacc']:.4f}")
    print(f"  Test bacc: {best['balanced_acc']:.4f}")
    print(f"  Test acc:  {best['acc']:.4f}")
    print(f"  Kappa:     {best['cohen_kappa']:.4f}")
    print(f"  F1:        {best['f1']:.4f}")
    print(f"  AUROC:     {best['auroc']:.4f}")
    print(f"  AUC-PR:    {best['auc_pr']:.4f}")

    # Sauvegarder la meilleure tête + stats de standardisation
    torch.save({
        "head_state": best["head_state"],
        "mean": mean,
        "std": std,
        "config": {k: best[k] for k in ("optimizer", "lr", "weight_decay")},
    }, "best_linear_head.pt")
    print("\nBest head (+ normalisation stats) sauvegardé dans best_linear_head.pt")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [f"{r['optimizer']}\nlr={r['lr']:g}\nwd={r['weight_decay']:g}" for r in results]
    val_baccs = [r["best_val_bacc"] for r in results]
    test_baccs = [r["balanced_acc"] for r in results]
    x = np.arange(len(results))
    w = 0.4
    ax.bar(x - w/2, val_baccs, w, label="Val bacc")
    ax.bar(x + w/2, test_baccs, w, label="Test bacc")
    best_idx = int(np.argmax(val_baccs))
    ax.axvline(best_idx, color="red", linestyle="--", alpha=0.5, label="Best (val)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Linear Probing — Grid Search")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("linear_probing_grid_search.png", dpi=150)
    print("Plot sauvegardé dans linear_probing_grid_search.png")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/linear_probing_best_{timestamp}.log"
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
