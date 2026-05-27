"""Compare per-subject accuracy of the 4 modes (linear probing, global,
subject specific, stacked) on bciciv2a, model REVE.

Aggregates over seeds 42/67/1331 (mean +/- SEM) and draws a publication-quality
grouped bar chart of per-subject test accuracy plus a side panel with each
mode's overall accuracy. Exports vector PDF + 300 dpi PNG.
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

RESULTS_DIR = "results/new_reve"
SEEDS = [42, 67, 1331]
OUT_BASE = "figures/bciciv2a_subjects_bars"
CHANCE = 0.25  # bciciv2a: 4-class motor imagery

# label -> (file tag, stage key, color)
# warm orange -> brick-red palette (light = lower mode, dark = stronger mode)
MODES = {
    "linear probing": ("lp", "lp", "#FAC284"),
    "global": ("global", "gl", "#FDA64D"),
    "subject specific": ("multi", "multilora", "#E94E1B"),
    "stacked": ("stacked", "multilora_global", "#C44601"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


def _sem(a, axis=None):
    """Standard error of the mean across seeds (sample std, ddof=1)."""
    return np.std(a, axis=axis, ddof=1) / np.sqrt(a.shape[0])


def load_mode(file_tag, stage_key):
    """Aggregate one mode over SEEDS.

    Returns (subjects, ps_mean, ps_sem, overall_mean, overall_sem).
    """
    subjects, per_subject_runs, overall_runs = None, [], []
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, f"bciciv2a_{file_tag}_s{seed}.json")
        with open(path) as f:
            stage = json.load(f)["stages"][stage_key]
        if subjects is None:
            subjects = sorted(stage["per_subject"].keys(), key=int)
        per_subject_runs.append([stage["per_subject"][s]["acc"] for s in subjects])
        overall_runs.append(stage["test"]["acc"])
    ps = np.array(per_subject_runs)      # (n_seeds, n_subjects)
    ov = np.array(overall_runs)          # (n_seeds,)
    return subjects, ps.mean(0), _sem(ps, axis=0), ov.mean(), _sem(ov)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.8", linewidth=0.6, linestyle=":")


def main():
    results = {label: load_mode(tag, key) for label, (tag, key, _) in MODES.items()}
    subjects = next(iter(results.values()))[0]
    x = np.arange(len(subjects))
    width = 0.8 / len(MODES)
    err_kw = dict(elinewidth=0.6, capthick=0.6, ecolor="0.15")

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 3.0), sharey=True,
        gridspec_kw={"width_ratios": [3.6, 1], "wspace": 0.05})

    # --- left: per-subject grouped bars (mean +/- SEM) ---
    for i, (label, (_, _, color)) in enumerate(MODES.items()):
        _, mean, sem, _, _ = results[label]
        offset = (i - (len(MODES) - 1) / 2) * width
        ax.bar(x + offset, mean, width, color=color,
               edgecolor="white", linewidth=0.3,
               yerr=sem, capsize=1.2, error_kw=err_kw)
    ax.axhline(CHANCE, ls="--", lw=0.7, color="0.45", zorder=0)
    ax.set_xticks(x, [f"S{s}" for s in subjects])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Per-subject", loc="left", color="#4D2600")
    style_axis(ax)

    # --- right: overall accuracy per mode (mean +/- SEM) ---
    labels = list(MODES.keys())
    ov_mean = [results[l][3] for l in labels]
    ov_sem = [results[l][4] for l in labels]
    colors = [MODES[l][2] for l in labels]
    xb = np.arange(len(labels))
    bars = ax2.bar(xb, ov_mean, 0.62, color=colors, edgecolor="white",
                   linewidth=0.3, yerr=ov_sem, capsize=2, error_kw=err_kw)
    for b, m, e in zip(bars, ov_mean, ov_sem):
        ax2.text(b.get_x() + b.get_width() / 2, m + e + 0.008, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=6.5, color="0.3")
    ax2.axhline(CHANCE, ls="--", lw=0.7, color="0.45", zorder=0)
    ax2.set_xticks([])
    ax2.set_title("(b) Overall", loc="left", color="#4D2600")
    style_axis(ax2)

    # --- shared legend (bottom) ---
    handles = [Patch(facecolor=MODES[l][2], label=l) for l in labels]
    handles.append(Line2D([0], [0], ls="--", lw=0.7, color="0.45",
                          label="chance (4-class)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.4, columnspacing=1.4)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.9, bottom=0.16, wspace=0.05)
    d = os.path.dirname(OUT_BASE)
    if d:
        os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_BASE}.{ext}", dpi=300, bbox_inches="tight")
        print(f"[plot] saved to {OUT_BASE}.{ext}")


if __name__ == "__main__":
    main()
