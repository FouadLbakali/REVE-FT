"""Sequential LR + LoRA-rank sweep for `global` and `subject_specific`.

Pipeline (single seed, cosine scheduler everywhere, --epochs 1 outer):
  0. Train one shared LP -> ckpts/lp_<dataset>_s<seed>.pt
  For each mode in {global, subject_specific}:
    1. LR sweep at fixed rank = 8                 -> best lr*
    2. Rank sweep at lr = lr* over {1..128}       -> best rank*

ETA is printed after every child run, spanning the whole pipeline.

Usage:
    python run_rank_sweep.py [--dataset bciciv2a] [--seed 42]
"""
import argparse
import json
import os
import subprocess
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

RANKS = [1, 2, 4, 8, 16, 32, 64, 128]
FIXED_RANK_FOR_LR_SEARCH = 64
GL_LRS = [5e-5, 1e-4, 3e-4, 1e-3]
FT_LRS = [1e-4, 3e-4, 1e-3]
SCHEDULER = "cosine"


def _fmt_hms(seconds):
    if seconds < 0 or seconds != seconds:  # NaN guard
        return "--:--:--"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ETA:
    """Tracks per-run wall time and prints remaining-time estimate."""

    def __init__(self, total_runs, label):
        self.total = total_runs
        self.label = label
        self.done = 0
        self.elapsed = 0.0
        self.t0 = time.time()

    def tick(self, run_seconds):
        self.done += 1
        self.elapsed += run_seconds
        avg = self.elapsed / self.done
        remaining_runs = self.total - self.done
        eta = avg * remaining_runs
        wall = time.time() - self.t0
        print(f"  [{self.label}]  run {self.done}/{self.total}  "
              f"last={_fmt_hms(run_seconds)}  avg={_fmt_hms(avg)}  "
              f"elapsed={_fmt_hms(wall)}  ETA={_fmt_hms(eta)}")
        return eta


def train_lp(dataset, seed, num_subjects, ckpt_path, lp_epochs, lp_lr):
    print(f"\n{'=' * 78}\n  STEP 0 — Train shared LP "
          f"(epochs={lp_epochs}, lr={lp_lr}, scheduler={SCHEDULER}, seed={seed})"
          f"\n{'=' * 78}")
    t0 = time.time()
    cmd = [sys.executable, "main.py",
           "--mode", "linear", "--dataset", dataset,
           "--seed", str(seed), "--num-subjects", str(num_subjects),
           "--lr", str(lp_lr), "--scheduler", SCHEDULER,
           "--epochs", str(lp_epochs), "--save-final-layer", ckpt_path]
    print(">>> " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"LP training failed (rc={rc})")
    print(f"  LP done in {_fmt_hms(time.time() - t0)} -> {ckpt_path}")


def _build_base(mode, dataset, num_subjects, lp_ckpt,
                epochs_extra_flag, epochs_extra_value):
    return [sys.executable, "main.py",
            "--mode", mode, "--dataset", dataset,
            "--epochs", "1",
            "--num-subjects", str(num_subjects),
            epochs_extra_flag, str(epochs_extra_value),
            "--load-final-layer", lp_ckpt]


def lr_sweep(mode, dataset, seed, num_subjects, lp_ckpt, lrs, fixed_rank,
             epochs_extra_flag, epochs_extra_value, log_dir, eta):
    """Sweep lr at fixed LoRA rank. Returns (rows, best)."""
    lr_flag = "--gl-lr" if mode == "global" else "--ft-lr"
    rank_flag = "--gl-rank" if mode == "global" else "--lora-rank"

    os.makedirs(log_dir, exist_ok=True)
    base = _build_base(mode, dataset, num_subjects, lp_ckpt,
                       epochs_extra_flag, epochs_extra_value) + \
           [rank_flag, str(fixed_rank)]

    print(f"\n{'=' * 78}\n  LR sweep — mode={mode}  lrs={lrs}  "
          f"rank={fixed_rank}  scheduler={SCHEDULER}  seed={seed}\n{'=' * 78}")
    rows = []
    for lr in lrs:
        t0 = time.time()
        val, test, rc = run_one(base, lr_flag, lr, SCHEDULER, seed)
        dt = time.time() - t0
        rows.append({"lr": lr, "rank": fixed_rank, "val": val, "test": test,
                     "rc": rc, "seconds": dt})
        eta.tick(dt)

    rows.sort(key=lambda r: (r["rc"] != 0,
                             -(r["val"] if r["val"] == r["val"] else -1)))
    best = next((r for r in rows if r["rc"] == 0 and r["val"] == r["val"]), None)

    lines = ["=" * 78,
             f"  LR sweep — mode={mode}  rank={fixed_rank}  seed={seed}",
             "=" * 78,
             f"  {'lr':>10}  {'val':>8}  {'test':>8}  rc"]
    for r in rows:
        lines.append(f"  {r['lr']:10.1e}  {r['val']:8.4f}  {r['test']:8.4f}  {r['rc']}")
    boundary_warning = None
    if best is not None:
        lines.append("")
        lines.append(f"  >>> BEST lr={best['lr']:.1e}  "
                     f"val={best['val']:.4f}  test={best['test']:.4f}")
        lmin, lmax = min(lrs), max(lrs)
        if best["lr"] == lmin:
            boundary_warning = (f"  ⚠  BEST lr is at LOWER edge of grid ({lmin:.1e}). "
                                f"Extend the grid downward.")
        elif best["lr"] == lmax:
            boundary_warning = (f"  ⚠  BEST lr is at UPPER edge of grid ({lmax:.1e}). "
                                f"Extend the grid upward.")
        if boundary_warning:
            lines.append(boundary_warning)
            best["boundary"] = boundary_warning.strip()
    print("\n" + "\n".join(lines))

    with open(os.path.join(log_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(log_dir, "rows.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows, best


def rank_sweep(mode, dataset, seed, num_subjects, lp_ckpt, lr_value,
               epochs_extra_flag, epochs_extra_value, log_dir, eta):
    """Sweep LoRA rank at fixed lr. Returns (rows, best)."""
    rank_flag = "--gl-rank" if mode == "global" else "--lora-rank"
    lr_flag = "--gl-lr" if mode == "global" else "--ft-lr"

    os.makedirs(log_dir, exist_ok=True)
    base = _build_base(mode, dataset, num_subjects, lp_ckpt,
                       epochs_extra_flag, epochs_extra_value) + \
           [lr_flag, str(lr_value)]

    print(f"\n{'=' * 78}\n  Rank sweep — mode={mode}  ranks={RANKS}  "
          f"lr={lr_value:.1e}  scheduler={SCHEDULER}  seed={seed}\n{'=' * 78}")
    rows = []
    for rank in RANKS:
        t0 = time.time()
        val, test, rc = run_one(base, rank_flag, rank, SCHEDULER, seed)
        dt = time.time() - t0
        rows.append({"lr": lr_value, "rank": rank, "val": val, "test": test,
                     "rc": rc, "seconds": dt})
        eta.tick(dt)

    rows.sort(key=lambda r: (r["rc"] != 0,
                             -(r["val"] if r["val"] == r["val"] else -1)))
    best = next((r for r in rows if r["rc"] == 0 and r["val"] == r["val"]), None)

    lines = ["=" * 78,
             f"  Rank sweep — mode={mode}  lr={lr_value:.1e}  seed={seed}",
             "=" * 78,
             f"  {'rank':>4}  {'val':>8}  {'test':>8}  rc"]
    for r in rows:
        lines.append(f"  {r['rank']:>4}  {r['val']:8.4f}  {r['test']:8.4f}  {r['rc']}")
    if best is not None:
        lines.append("")
        lines.append(f"  >>> BEST rank={best['rank']}  "
                     f"val={best['val']:.4f}  test={best['test']:.4f}")
    print("\n" + "\n".join(lines))

    with open(os.path.join(log_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(log_dir, "rows.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bciciv2a")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-subjects", type=int, default=109)
    p.add_argument("--lp-epochs", type=int, default=15)
    p.add_argument("--lp-lr", type=float, default=3e-2)
    p.add_argument("--gl-epochs", type=int, default=1,
                   help="inner --gl-epochs passed to main.py")
    p.add_argument("--ft-epochs", type=int, default=1,
                   help="inner --ft-epochs passed to main.py")
    args = p.parse_args()

    os.makedirs("ckpts", exist_ok=True)
    lp_ckpt = f"ckpts/lp_{args.dataset}_s{args.seed}.pt"
    log_root = f"sweep_logs/lr_then_rank_{args.dataset}_s{args.seed}"
    os.makedirs(log_root, exist_ok=True)

    total_runs = len(GL_LRS) + len(RANKS) + len(FT_LRS) + len(RANKS)
    eta = ETA(total_runs=total_runs, label="sweep")

    overall_t0 = time.time()

    # 0) Shared LP
    train_lp(args.dataset, args.seed, args.num_subjects, lp_ckpt,
             args.lp_epochs, args.lp_lr)

    # 1) GLOBAL: lr sweep -> rank sweep at best lr
    gl_lr_rows, gl_lr_best = lr_sweep(
        mode="global", dataset=args.dataset, seed=args.seed,
        num_subjects=args.num_subjects, lp_ckpt=lp_ckpt,
        lrs=GL_LRS, fixed_rank=FIXED_RANK_FOR_LR_SEARCH,
        epochs_extra_flag="--gl-epochs", epochs_extra_value=args.gl_epochs,
        log_dir=f"{log_root}/global/lr", eta=eta,
    )
    if gl_lr_best is None:
        raise RuntimeError("global lr sweep produced no successful run")
    gl_rank_rows, gl_rank_best = rank_sweep(
        mode="global", dataset=args.dataset, seed=args.seed,
        num_subjects=args.num_subjects, lp_ckpt=lp_ckpt,
        lr_value=gl_lr_best["lr"],
        epochs_extra_flag="--gl-epochs", epochs_extra_value=args.gl_epochs,
        log_dir=f"{log_root}/global/rank", eta=eta,
    )

    # 2) SUBJECT_SPECIFIC: lr sweep -> rank sweep at best lr
    ss_lr_rows, ss_lr_best = lr_sweep(
        mode="subject_specific", dataset=args.dataset, seed=args.seed,
        num_subjects=args.num_subjects, lp_ckpt=lp_ckpt,
        lrs=FT_LRS, fixed_rank=FIXED_RANK_FOR_LR_SEARCH,
        epochs_extra_flag="--ft-epochs", epochs_extra_value=args.ft_epochs,
        log_dir=f"{log_root}/subject_specific/lr", eta=eta,
    )
    if ss_lr_best is None:
        raise RuntimeError("subject_specific lr sweep produced no successful run")
    ss_rank_rows, ss_rank_best = rank_sweep(
        mode="subject_specific", dataset=args.dataset, seed=args.seed,
        num_subjects=args.num_subjects, lp_ckpt=lp_ckpt,
        lr_value=ss_lr_best["lr"],
        epochs_extra_flag="--ft-epochs", epochs_extra_value=args.ft_epochs,
        log_dir=f"{log_root}/subject_specific/rank", eta=eta,
    )

    # Final report
    total = time.time() - overall_t0
    print("\n" + "=" * 78)
    print(f"  FINAL — dataset={args.dataset}  seed={args.seed}  "
          f"total wall time={_fmt_hms(total)}")
    print("=" * 78)
    print(f"  {'mode':<18}  {'best lr':>10}  {'best rank':>9}  {'val':>8}  {'test':>8}")
    for name, lr_best, rank_best in [
        ("global", gl_lr_best, gl_rank_best),
        ("subject_specific", ss_lr_best, ss_rank_best),
    ]:
        if rank_best is None:
            print(f"  {name:<18}  (no successful rank run)")
            continue
        print(f"  {name:<18}  {lr_best['lr']:10.1e}  {rank_best['rank']:>9}  "
              f"{rank_best['val']:8.4f}  {rank_best['test']:8.4f}")
        if lr_best.get("boundary"):
            print(f"    {lr_best['boundary']}")

    summary = {
        "dataset": args.dataset, "seed": args.seed, "scheduler": SCHEDULER,
        "lp_ckpt": lp_ckpt, "lp_epochs": args.lp_epochs, "lp_lr": args.lp_lr,
        "ranks": RANKS, "fixed_rank_for_lr_search": FIXED_RANK_FOR_LR_SEARCH,
        "global": {
            "lr_grid": GL_LRS, "lr_rows": gl_lr_rows, "lr_best": gl_lr_best,
            "rank_rows": gl_rank_rows, "rank_best": gl_rank_best,
        },
        "subject_specific": {
            "lr_grid": FT_LRS, "lr_rows": ss_lr_rows, "lr_best": ss_lr_best,
            "rank_rows": ss_rank_rows, "rank_best": ss_rank_best,
        },
        "wall_seconds": total,
    }
    with open(f"{log_root}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary -> {log_root}/summary.json")


if __name__ == "__main__":
    main()
