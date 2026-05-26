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
    """Bar plot of val_mean ± SEM over the swept `param` values.
    Test is intentionally NOT plotted to avoid biasing selection."""
    _ensure_dir(out_path)
    rows = sorted((r for r in rows if r.get("n_ok", 1) > 0),
                  key=lambda r: r["value"])
    values = [r["value"] for r in rows]
    val_m = [r.get("val_mean", r.get("val", np.nan)) for r in rows]
    val_s = [r.get("val_sem", 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(values)), 4.8))
    x = np.arange(len(values))
    bars = ax.bar(x, val_m, 0.6, yerr=val_s, capsize=4, color="#4C78A8")
    for b, v, s in zip(bars, val_m, val_s):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + s,
                    f"{v:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"{v:.0e}" for v in values])
    ax.set_xlabel(param); ax.set_ylabel("val balanced_acc")
    ax.set_title(f"Sweep — mode={mode}  param={param}  (val ± SEM, val only)")
    ax.grid(axis="y", alpha=0.3)

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
    configs = [f"{summary[m]['value']:.0e}" for m in modes]

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
