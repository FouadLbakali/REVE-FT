"""Plot all loss curves from curves/*.csv into a single comparison figure."""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

CURVEDIR = "curves"
OUTPUT = os.path.join(CURVEDIR, "comparison.png")

csv_files = sorted(glob.glob(os.path.join(CURVEDIR, "*.csv")))
if not csv_files:
    print("No CSV files found in curves/")
    exit(1)

# Color groups
COLORS = {
    "linear": "black",
    "query_token": "tab:blue",
    "subject_prompt": "tab:orange",
}
STYLES = {1: "-", 2: "--", 4: "-.", 8: ":"}

fig, ax = plt.subplots(figsize=(11, 5))

for path in csv_files:
    name = os.path.basename(path).replace(".csv", "").replace("loss_", "")
    df = pd.read_csv(path)

    # Determine color and linestyle from name
    if name.startswith("subject_prompt"):
        base, color = "subject_prompt", COLORS["subject_prompt"]
    elif name.startswith("query_token"):
        base, color = "query_token", COLORS["query_token"]
    else:
        base, color = "linear", COLORS["linear"]

    np_val = None
    if "_np" in name:
        np_val = int(name.split("_np")[-1])

    linestyle = STYLES.get(np_val, "-") if np_val is not None else "-"
    label = name.replace("_np", " np=")

    ax.plot(df["epoch"], df["loss"], linestyle=linestyle, color=color,
            linewidth=1.8, marker="o", markersize=3, label=label)

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Loss curves — all experiments")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(OUTPUT, dpi=150)
plt.close()
print(f"Comparison plot saved to {OUTPUT}")
