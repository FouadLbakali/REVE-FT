"""LoRA-rank ablation for REVE on BCIC IV-2a, in the style of the LoRA-rank
figure (one colored line per rank, log-scale training step on x).

Two panels are produced:
  (a) global mode          -- sweep --gl-rank   over {1,2,4,8,16,32,64,128,256}
  (b) subject-specific mode -- sweep --lora-rank over {1,2,4,8,16,32,64}

Each line is the per-step training loss of a single 3-epoch run at a fixed
learning rate (no learning-rate sweep / no full fine-tuning). Lines are
EMA-smoothed for readability; the raw per-step losses are kept in the JSON.

Usage:
    uv run python scripts/ablation_lora_rank.py            # train + plot
    uv run python scripts/ablation_lora_rank.py --bf16     # bf16 autocast
    uv run python scripts/ablation_lora_rank.py --plot-only # re-plot from JSON
"""

import argparse
import copy
import gc
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from peft import get_peft_model
from transformers import set_seed

from reve_ft.data import load_loaders_per_subject, load_lora_targets
from reve_ft.engine import _autocast, set_bf16
from reve_ft.main import build_model
from reve_ft.multilora import inject_multi_subject_lora, set_subject_ids
from reve_ft.stages import make_lora_config
from reve_ft.trainer import make_scheduler

DATASET = "bciciv2a"
MODEL = "reve"
GLOBAL_RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SUBJECT_RANKS = [1, 2, 4, 8, 16, 32, 64]
RESULTS_OUT = "results/ablation/lora_rank_bciciv2a.json"
FIG_OUT = "figures/ablation_lora_rank_bciciv2a"
EMA_ALPHA = 0.1  # smoothing for the plotted curves (raw losses are kept in JSON)


def parse_args():
    p = argparse.ArgumentParser(description="LoRA-rank ablation (REVE, bciciv2a)")
    p.add_argument("--epochs", default=3, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--results-out", default=RESULTS_OUT)
    p.add_argument("--fig-out", default=FIG_OUT)
    p.add_argument("--plot-only", action="store_true",
                   help="skip training; plot from --results-out")
    return p.parse_args()


def train_record(model, loader, optimizer, scheduler, epochs, device, multilora):
    """Train `model` for `epochs` and return the per-step training loss.

    Mirrors engine.train_one_epoch / train_one_epoch_multilora but records the
    loss of every optimization step instead of only the epoch average."""
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    losses = []
    for epoch in range(epochs):
        for batch in loader:
            data = batch["sample"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            pos = batch["pos"].to(device, non_blocking=True)
            if multilora:
                set_subject_ids(batch["subject_id"].to(device, non_blocking=True))
            optimizer.zero_grad()
            with _autocast():
                output = model(data, pos)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
    if multilora:
        set_subject_ids(None)
    return losses


def _make_optimizer_scheduler(model, lr, epochs, steps_per_epoch):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = make_scheduler(optimizer, epochs, steps_per_epoch, max_lr=lr)
    return optimizer, scheduler


def run_global_rank(pristine, train_loader, rank, args, device):
    model = get_peft_model(copy.deepcopy(pristine), make_lora_config(rank, MODEL))
    model.to(device)
    opt, sched = _make_optimizer_scheduler(model, args.lr, args.epochs, len(train_loader))
    return train_record(model, train_loader, opt, sched, args.epochs, device, multilora=False)


def run_subject_rank(pristine, train_loader, num_subjects, rank, args, device):
    model = copy.deepcopy(pristine)
    model, _ = inject_multi_subject_lora(
        model, num_subjects, rank, alpha=32, dropout=0.05, global_rank=0,
        target_suffixes=load_lora_targets(MODEL),
    )
    model.to(device)
    opt, sched = _make_optimizer_scheduler(model, args.lr, args.epochs, len(train_loader))
    return train_record(model, train_loader, opt, sched, args.epochs, device, multilora=True)


def sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bf16:
        set_bf16(True)
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pristine, pos_bank = build_model(DATASET, MODEL)
    pooled_loaders, subject_loaders, _ = load_loaders_per_subject(
        DATASET, pos_bank, args.batch_size, args.seed, model_name=MODEL,
    )
    train_loader = pooled_loaders[0]
    num_subjects = max(subject_loaders) + 1

    results = {
        "config": {"model": MODEL, "dataset": DATASET, "epochs": args.epochs,
                   "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed},
        "global": {},
        "subject-specific": {},
    }

    for i, rank in enumerate(GLOBAL_RANKS, 1):
        print(f"\n[global {i}/{len(GLOBAL_RANKS)}] gl-rank={rank}", flush=True)
        results["global"][str(rank)] = run_global_rank(
            pristine, train_loader, rank, args, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for i, rank in enumerate(SUBJECT_RANKS, 1):
        print(f"\n[subject-specific {i}/{len(SUBJECT_RANKS)}] lora-rank={rank}", flush=True)
        results["subject-specific"][str(rank)] = run_subject_rank(
            pristine, train_loader, num_subjects, rank, args, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.results_out) or ".", exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump(results, f)
    print(f"\nResults dumped to {args.results_out}")
    return results


def _ema(x, alpha=EMA_ALPHA):
    out, m = [], x[0]
    for v in x:
        m = alpha * v + (1 - alpha) * m
        out.append(m)
    return out


def _plot_panel(ax, curves, title):
    """curves: {rank: [per-step loss]}, drawn one viridis line per rank."""
    ranks = sorted(curves, key=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(ranks)))
    for rank, color in zip(ranks, colors):
        y = _ema(curves[rank])
        ax.plot(np.arange(1, len(y) + 1), y, color=color, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.set_title(title, loc="left", color="#4D2600")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.85", linewidth=0.5, linestyle=":")
    handles = [Line2D([0], [0], color=c, lw=1.4, label=r)
               for r, c in zip(ranks, colors)]
    ax.legend(handles=handles, title="Rank", loc="upper right",
              frameon=False, fontsize=7, ncol=2, columnspacing=1.0,
              handlelength=1.2, labelspacing=0.3)


def plot(results, fig_out):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
        "axes.linewidth": 0.6,
    })
    fig, (ax_g, ax_s) = plt.subplots(1, 2, figsize=(7.16, 3.2))
    _plot_panel(ax_g, results["global"],
                "(a) Global LoRA — REVE / BCIC IV-2a")
    _plot_panel(ax_s, results["subject-specific"],
                "(b) Subject-specific LoRA — REVE / BCIC IV-2a")
    fig.tight_layout()

    d = os.path.dirname(fig_out)
    if d:
        os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_out}.{ext}", dpi=300, bbox_inches="tight")
        print(f"[plot] saved to {fig_out}.{ext}")


def main():
    args = parse_args()
    if args.plot_only:
        with open(args.results_out) as f:
            results = json.load(f)
    else:
        results = sweep(args)
    plot(results, args.fig_out)


if __name__ == "__main__":
    main()
