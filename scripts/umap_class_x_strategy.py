"""UMAP of REVE embeddings on bciciv2a: 3 strategies x 4 motor-imagery classes.

Reproduces the `umap_class_x_strategy_subjectcolor` notebook inside the repo,
reusing the repo's preprocessing (reve.yaml: 0.5-99.5 Hz bandpass, per-channel
z-score on train stats, clamp 15) and the already-merged per-strategy
checkpoints under ckpts/ instead of the notebook's ad-hoc LoRA reconstruction.

For each strategy we extract the pre-head features (input to final_layer), L2
-normalize, reduce with PCA(500) then UMAP, and draw a 3-row (strategy) x 4
-column (class) grid of the test set coloured by subject, with global density
contours in the background. A silhouette table (class vs. subject) is printed.

  Global:           ckpts/global_reve/reve.pt           (one merged model)
  Subject-Specific: ckpts/subj_reve/reve_subject_<id>.pt (one per subject)
  Stacked:          ckpts/stacked_reve/reve_subject_<id>.pt
"""
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from stacked_lora.data import load_loaders_per_subject
from stacked_lora.engine import extract_features
from stacked_lora.main import build_model

SEED = 42
BATCH_SIZE = 64
EMBED_DIR = "figures/umap_embeddings"
OUT_BASE = "figures/umap_class_x_strategy_subjectcolor"

# 1-based subject ids to force-drop from every strategy. Per-subject
# checkpoints that fail to load (e.g. corrupt files) are skipped automatically
# on top of this, so leave empty to use the full test set.
EXCLUDE_SUBJECTS = set()

# strategy key -> (checkpoint dir, per-subject?, display label)
STRATEGIES = {
    "Global": ("ckpts/global_reve", False, "Global LoRA"),
    "Subject-Specific": ("ckpts/subj_reve", True, "Subject-Specific LoRA"),
    "Stacked": ("ckpts/stacked_reve", True, "Stacked LoRA"),
}

# Columns ordered as in the reference figure: Tongue, Left Hand, Right Hand, Feet.
# label_map (see data.py): left_hand=0, right_hand=1, feet=2, tongue=3.
COL_ORDER = [3, 0, 1, 2]
COL_LABELS = ["Tongue", "Left Hand", "Right Hand", "Feet"]


def _load_into(model, ckpt_path, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)


def extract_strategy(name, ckpt_dir, per_subject, model, pooled_test,
                     subject_loaders, device):
    """Return (Z, Y, S) for one strategy, caching to <EMBED_DIR>/*_<name>.npy.

    Global runs the single merged model over the pooled test loader. The
    per-subject strategies run each subject's merged checkpoint over that
    subject's test subloader and stitch the results back together.
    """
    z_path = os.path.join(EMBED_DIR, f"embeddings_{name}.npy")
    if os.path.exists(z_path):
        print(f"  -> cache hit: {name}")
        return (np.load(z_path),
                np.load(os.path.join(EMBED_DIR, f"labels_{name}.npy")),
                np.load(os.path.join(EMBED_DIR, f"subjects_{name}.npy")))

    if not per_subject:
        _load_into(model, os.path.join(ckpt_dir, "reve.pt"), device)
        Zt, Yt, St = extract_features(model, pooled_test, device, return_subjects=True)
        Z = Zt.reshape(Zt.shape[0], -1).numpy()
        Y, S = Yt.numpy(), St.numpy()
        keep = ~np.isin(S + 1, list(EXCLUDE_SUBJECTS))  # S is 0-based
        Z, Y, S = Z[keep], Y[keep], S[keep]
    else:
        zs, ys, ss = [], [], []
        for raw_id in sorted(subject_loaders):
            if raw_id in EXCLUDE_SUBJECTS:
                print(f"  subject {raw_id} excluded")
                continue
            ckpt = os.path.join(ckpt_dir, f"reve_subject_{raw_id:02d}.pt")
            print(f"  subject {raw_id}...")
            try:
                _load_into(model, ckpt, device)
            except (RuntimeError, FileNotFoundError) as e:
                print(f"  subject {raw_id} skipped (checkpoint unusable: {e})")
                continue
            _, _, test_loader = subject_loaders[raw_id]
            Zt, Yt, St = extract_features(model, test_loader, device, return_subjects=True)
            zs.append(Zt.reshape(Zt.shape[0], -1).numpy())
            ys.append(Yt.numpy())
            ss.append(St.numpy())
        Z, Y, S = np.concatenate(zs), np.concatenate(ys), np.concatenate(ss)

    os.makedirs(EMBED_DIR, exist_ok=True)
    np.save(z_path, Z)
    np.save(os.path.join(EMBED_DIR, f"labels_{name}.npy"), Y)
    np.save(os.path.join(EMBED_DIR, f"subjects_{name}.npy"), S)
    print(f"  saved {name}: {Z.shape}")
    return Z, Y, S


def reduce_umap(Z):
    """L2-normalize -> PCA(500) -> UMAP, as in the notebook."""
    Z_norm = normalize(Z, norm="l2")
    pca = PCA(n_components=min(500, *Z_norm.shape), random_state=SEED)
    Z_pca = pca.fit_transform(Z_norm)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    reducer = umap.UMAP(n_neighbors=100, min_dist=0.1, metric="cosine",
                        random_state=SEED)
    return reducer.fit_transform(Z_pca)


def add_density_contours(ax, U, alpha=0.25):
    """Ghost background: global KDE density contours over all classes."""
    try:
        kernel = gaussian_kde(U.T)
        x_min, x_max = U[:, 0].min() - 1, U[:, 0].max() + 1
        y_min, y_max = U[:, 1].min() - 1, U[:, 1].max() + 1
        xi, yi = np.mgrid[x_min:x_max:120j, y_min:y_max:120j]
        zi = np.reshape(kernel(np.vstack([xi.ravel(), yi.ravel()])).T, xi.shape)
        ax.contour(xi, yi, zi, levels=10, linewidths=0.5, colors="k", alpha=alpha)
    except Exception:
        pass


def plot_grid(umap_results, subjects):
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(subjects)))
    keys = list(STRATEGIES)
    n_rows, n_cols = len(keys), len(COL_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 4.0 * n_rows), dpi=150)

    for row, key in enumerate(keys):
        U, Y, S = umap_results[key]
        for col, class_idx in enumerate(COL_ORDER):
            ax = axes[row, col]
            add_density_contours(ax, U)
            class_mask = Y == class_idx
            for i, sid in enumerate(subjects):
                m = class_mask & (S == sid)
                if m.sum():
                    ax.scatter(U[m, 0], U[m, 1], color=colors[i], s=8,
                               alpha=0.6, linewidths=0)
            ax.text(0.98, 0.98, COL_LABELS[col], transform=ax.transAxes,
                    ha="right", va="top", fontsize=10, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"))
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(STRATEGIES[key][2], fontsize=11,
                                fontweight="bold", labelpad=8)

    patches = [mpatches.Patch(color=colors[i], label=f"Subject {sid + 1}")
               for i, sid in enumerate(subjects)]
    fig.legend(handles=patches, title="Subject", title_fontsize=10, fontsize=9,
               ncol=len(subjects), loc="lower center", bbox_to_anchor=(0.5, -0.04),
               frameon=True, edgecolor="gray")
    fig.suptitle("UMAP of REVE embeddings — BCI-IV-2a\n"
                 "Each subplot: one MI class · points coloured by subject",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_BASE), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_BASE}.{ext}", bbox_inches="tight", dpi=300)
        print(f"[plot] saved {OUT_BASE}.{ext}")


def print_silhouettes(umap_results):
    rows = []
    for key in STRATEGIES:
        U, Y, S = umap_results[key]
        sc = silhouette_score(U, Y, sample_size=2000, random_state=SEED)
        ss = silhouette_score(U, S, sample_size=2000, random_state=SEED)
        rows.append({"Strategy": key, "Silhouette class (up)": round(sc, 4),
                     "Silhouette subject (down)": round(ss, 4)})
        print(f"{key:18s} | class: {sc:.4f} | subject: {ss:.4f}")
    print()
    print(pd.DataFrame(rows).set_index("Strategy"))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, pos_bank = build_model("bciciv2a", "reve")
    pooled_loaders, subject_loaders, _ = load_loaders_per_subject(
        "bciciv2a", pos_bank, BATCH_SIZE, seed=SEED, model_name="reve")
    pooled_test = pooled_loaders[2]

    umap_results = {}
    for key, (ckpt_dir, per_subject, _) in STRATEGIES.items():
        print(f"\n{'=' * 60}\n  Strategy: {key}\n{'=' * 60}")
        Z, Y, S = extract_strategy(key, ckpt_dir, per_subject, model,
                                   pooled_test, subject_loaders, device)
        print(f"-- {key}: reducing --")
        U = reduce_umap(Z)
        umap_results[key] = (U, Y, S)

    subjects = sorted(np.unique(umap_results["Global"][2]))
    plot_grid(umap_results, subjects)
    print("\n--- Silhouette scores ---")
    print_silhouettes(umap_results)


if __name__ == "__main__":
    main()
