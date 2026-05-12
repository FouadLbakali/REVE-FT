"""LR sweep for `linear` mode on `zuo2025`, 1 epoch.

Usage:
    python run_lr_sweep_linear_zuo.py [--seed 42]
"""
import argparse
import json
import os
import sys
import time

# Redirect MNE / MOABB caches away from $HOME (which is not writable here).
_CACHE_ROOT = "/users/local/REVE-FT/.cache"
os.makedirs(_CACHE_ROOT, exist_ok=True)
os.makedirs(os.path.join(_CACHE_ROOT, "mne_data"), exist_ok=True)
os.environ["_MNE_FAKE_HOME_DIR"] = _CACHE_ROOT
os.environ["MNE_DATA"] = os.path.join(_CACHE_ROOT, "mne_data")
os.environ["MNE_DATASETS_BNCI_PATH"] = os.environ["MNE_DATA"]
os.environ["MOABB_RESULTS"] = os.path.join(_CACHE_ROOT, "moabb_results")
os.environ["XDG_CACHE_HOME"] = _CACHE_ROOT

from sweep import run_one

DATASET = "zuo2025"
MODE = "linear"
EPOCHS = 1
SCHEDULER = "cosine"
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def _fmt_hms(seconds):
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lrs", nargs="+", type=float, default=LRS)
    args = p.parse_args()

    log_dir = f"sweep_logs/lr_linear_{DATASET}_s{args.seed}"
    os.makedirs(log_dir, exist_ok=True)

    base = [sys.executable, "main.py",
            "--mode", MODE, "--dataset", DATASET,
            "--epochs", str(EPOCHS)]

    print(f"\n{'=' * 78}\n  LR sweep — mode={MODE}  dataset={DATASET}  "
          f"epochs={EPOCHS}  scheduler={SCHEDULER}  seed={args.seed}\n"
          f"  lrs={args.lrs}\n{'=' * 78}")

    rows = []
    t_start = time.time()
    for i, lr in enumerate(args.lrs, 1):
        t0 = time.time()
        val, test, rc = run_one(base, "--lr", lr, SCHEDULER, args.seed)
        dt = time.time() - t0
        rows.append({"lr": lr, "val": val, "test": test, "rc": rc, "seconds": dt})
        wall = time.time() - t_start
        avg = wall / i
        eta = avg * (len(args.lrs) - i)
        print(f"  [{i}/{len(args.lrs)}] lr={lr:.1e}  val={val:.4f}  test={test:.4f}  "
              f"rc={rc}  dt={_fmt_hms(dt)}  ETA={_fmt_hms(eta)}")

    rows.sort(key=lambda r: (r["rc"] != 0,
                             -(r["val"] if r["val"] == r["val"] else -1)))
    best = next((r for r in rows if r["rc"] == 0 and r["val"] == r["val"]), None)

    lines = ["=" * 78,
             f"  LR sweep — mode={MODE}  dataset={DATASET}  seed={args.seed}",
             "=" * 78,
             f"  {'lr':>10}  {'val':>8}  {'test':>8}  rc"]
    for r in rows:
        lines.append(f"  {r['lr']:10.1e}  {r['val']:8.4f}  {r['test']:8.4f}  {r['rc']}")
    if best is not None:
        lines.append("")
        lines.append(f"  >>> BEST lr={best['lr']:.1e}  "
                     f"val={best['val']:.4f}  test={best['test']:.4f}")
        lmin, lmax = min(args.lrs), max(args.lrs)
        if best["lr"] == lmin:
            lines.append(f"  ⚠  BEST lr is at LOWER edge of grid ({lmin:.1e}). "
                         f"Extend the grid downward.")
        elif best["lr"] == lmax:
            lines.append(f"  ⚠  BEST lr is at UPPER edge of grid ({lmax:.1e}). "
                         f"Extend the grid upward.")
    print("\n" + "\n".join(lines))

    with open(os.path.join(log_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(log_dir, "rows.json"), "w") as f:
        json.dump({"dataset": DATASET, "mode": MODE, "epochs": EPOCHS,
                   "scheduler": SCHEDULER, "seed": args.seed,
                   "lrs": args.lrs, "rows": rows, "best": best}, f, indent=2)
    print(f"\n  Summary -> {log_dir}/")


if __name__ == "__main__":
    main()
