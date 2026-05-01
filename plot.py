"""Plotting helpers for sweep results.

`plot_sweep` shows VAL only (test is intentionally hidden during selection).
`plot_summary` shows test for the winning configs ONLY at the very end.
"""
import os

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_sweep(rows, mode, param, out_path):
    """Heatmap of val_mean (value × scheduler) + grouped bar with SEM error bars.
    Test is intentionally NOT plotted to avoid biasing selection."""
    _ensure_dir(out_path)
    values = sorted({r["value"] for r in rows})
    scheds = sorted({r["scheduler"] for r in rows})

    val_grid = np.full((len(values), len(scheds)), np.nan)
    sem_grid = np.full((len(values), len(scheds)), np.nan)
    for r in rows:
        if r.get("n_ok", 1) == 0:
            continue
        i = values.index(r["value"]); j = scheds.index(r["scheduler"])
        val_grid[i, j] = r.get("val_mean", r.get("val", np.nan))
        sem_grid[i, j] = r.get("val_sem", 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    im = ax.imshow(val_grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(scheds)), scheds)
    ax.set_yticks(range(len(values)), [f"{v:.0e}" for v in values])
    ax.set_xlabel("scheduler"); ax.set_ylabel(param)
    ax.set_title("val balanced_acc (mean over seeds)")
    vmin, vmax = np.nanmin(val_grid), np.nanmax(val_grid)
    mid = (vmin + vmax) / 2
    for i in range(len(values)):
        for j in range(len(scheds)):
            v, s = val_grid[i, j], sem_grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}\n±{s:.3f}", ha="center", va="center",
                        color="white" if v < mid else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    width = 0.8 / max(len(scheds), 1)
    x = np.arange(len(values))
    for j, s in enumerate(scheds):
        ys = [val_grid[i, j] for i in range(len(values))]
        es = [sem_grid[i, j] for i in range(len(values))]
        ax.bar(x + j * width, ys, width, yerr=es, capsize=4, label=s)
    ax.set_xticks(x + width * (len(scheds) - 1) / 2, [f"{v:.0e}" for v in values])
    ax.set_xlabel(param); ax.set_ylabel("val balanced_acc")
    ax.set_title("Val ± SEM by scheduler")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Sweep — mode={mode}  param={param}  (val only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] {out_path}")


def plot_summary(summary, out_path):
    """Final test report. `summary[mode]` must include test_mean and test_sem."""
    _ensure_dir(out_path)
    modes = [m for m, b in summary.items() if b is not None]
    test_m = [summary[m].get("test_mean", summary[m].get("test", np.nan)) for m in modes]
    test_s = [summary[m].get("test_sem", 0.0) for m in modes]
    val_m = [summary[m].get("val_mean", summary[m].get("val", np.nan)) for m in modes]
    val_s = [summary[m].get("val_sem", 0.0) for m in modes]
    configs = [f"{summary[m]['value']:.0e} / {summary[m]['scheduler']}" for m in modes]

    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(modes)), 5))
    x = np.arange(len(modes)); width = 0.38
    bv = ax.bar(x - width / 2, val_m, width, yerr=val_s, capsize=4, label="val", color="#4C78A8")
    bt = ax.bar(x + width / 2, test_m, width, yerr=test_s, capsize=4, label="test (final)", color="#F58518")
    for b, v, s in zip(bv, val_m, val_s):
        ax.text(b.get_x() + b.get_width() / 2, v + s, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for b, v, s in zip(bt, test_m, test_s):
        ax.text(b.get_x() + b.get_width() / 2, v + s, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for xi, cfg in zip(x, configs):
        ax.text(xi, -0.06, cfg, ha="center", va="top", fontsize=8,
                color="gray", transform=ax.get_xaxis_transform())
    ax.set_xticks(x, modes)
    ax.set_ylabel("balanced_acc")
    ax.set_title("Best per mode — val (used for selection) vs test (held-out, single final eval)")
    top = max((max(val_m + test_m) if val_m else 1.0), 0.1)
    ax.set_ylim(0, top * 1.20)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] {out_path}")
