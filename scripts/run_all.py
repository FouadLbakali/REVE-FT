"""End-to-end auto sweep — multi-seed, test held out during selection.

Pipeline (each stage propagates its best to the next, per seed):
  1. linear   sweep --lr × scheduler   (selection on val_mean over seeds)
  2. for each seed: retrain LP with the winning config -> ckpts/lp_<ds>_s{seed}.pt
  3. global   sweep --gl-lr × scheduler  (each seed loads its own LP ckpt)
  4. for each seed: retrain GL with the winning config -> ckpts/gl_<ds>_s{seed}/
  5. stacked  sweep --ft-lr × scheduler  (each seed loads its own GL adapters)
  6. subject_specific sweep --ft-lr × scheduler (each seed loads its own LP)

`test` is hidden during the sweeps and reported only at the FINAL step, on the
winning config's logs (already trained across all seeds during the sweep).

Usage:  python run_all.py [--dataset bciciv2a] [--seeds 0 1 2] [--num-subjects 109]
"""
import argparse
import json
import os
import subprocess
import sys

# Redirect MNE / MOABB caches away from $HOME (which is not writable here).
_CACHE_ROOT = "/users/local/REVE-FT/.cache"
os.makedirs(_CACHE_ROOT, exist_ok=True)
os.makedirs(os.path.join(_CACHE_ROOT, "mne_data"), exist_ok=True)
os.environ["_MNE_FAKE_HOME_DIR"] = _CACHE_ROOT          # -> .mne config dir
os.environ["MNE_DATA"] = os.path.join(_CACHE_ROOT, "mne_data")
os.environ["MNE_DATASETS_BNCI_PATH"] = os.environ["MNE_DATA"]
os.environ["MOABB_RESULTS"] = os.path.join(_CACHE_ROOT, "moabb_results")
os.environ["XDG_CACHE_HOME"] = _CACHE_ROOT

from sweep import sweep


def _run(cmd, what):
    print(f"\n>>> {what}")
    print(f">>> {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"{what} failed (rc={rc})")


def retrain_lp_per_seed(dataset, seeds, lr, scheduler, epochs, num_subjects, ckpt_pattern):
    for seed in seeds:
        ckpt = ckpt_pattern.format(seed=seed)
        _run([sys.executable, "main.py",
              "--mode", "linear", "--dataset", dataset,
              "--seed", str(seed), "--num-subjects", str(num_subjects),
              "--lr", str(lr), "--scheduler", scheduler,
              "--epochs", str(epochs), "--save-final-layer", ckpt],
             f"Retrain LP seed={seed} -> {ckpt}")


def retrain_gl_per_seed(dataset, seeds, gl_lr, scheduler, gl_epochs,
                        num_subjects, lp_ckpt_pattern, gl_ckpt_pattern):
    for seed in seeds:
        _run([sys.executable, "main.py",
              "--mode", "global", "--dataset", dataset,
              "--seed", str(seed), "--num-subjects", str(num_subjects),
              "--gl-lr", str(gl_lr), "--scheduler", scheduler,
              "--gl-epochs", str(gl_epochs),
              "--load-final-layer", lp_ckpt_pattern.format(seed=seed),
              "--save-global-lora", gl_ckpt_pattern.format(seed=seed)],
             f"Retrain GL seed={seed}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bciciv2a")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--num-subjects", type=int, default=109)
    p.add_argument("--lp-epochs", type=int, default=50)
    p.add_argument("--gl-epochs", type=int, default=30)
    p.add_argument("--ft-epochs", type=int, default=20)
    args = p.parse_args()

    seeds = args.seeds
    seeds_tag = "_".join(str(s) for s in seeds)
    os.makedirs("ckpts", exist_ok=True)
    log_root = f"sweep_logs/{args.dataset}_seeds{seeds_tag}"
    lp_ckpt_pat = f"ckpts/lp_{args.dataset}_s{{seed}}.pt"
    gl_ckpt_pat = f"ckpts/gl_{args.dataset}_s{{seed}}"
    summary = {}

    # 1) Linear -----------------------------------------------------------
    _, best_lp = sweep(
        mode="linear", dataset=args.dataset, param="--lr",
        values=[1e-3, 3e-3, 1e-2, 3e-2],
        schedulers=["plateau", "cosine", "constant"],
        epochs=args.lp_epochs, seeds=seeds, num_subjects=args.num_subjects,
        log_dir=f"{log_root}/linear",
    )
    if best_lp is None:
        raise RuntimeError("Linear sweep produced no successful run")
    summary["linear"] = best_lp

    # 2) Retrain LP per seed ----------------------------------------------
    retrain_lp_per_seed(args.dataset, seeds, best_lp["value"], best_lp["scheduler"],
                        args.lp_epochs, args.num_subjects, lp_ckpt_pat)

    # 3) Global -----------------------------------------------------------
    _, best_gl = sweep(
        mode="global", dataset=args.dataset, param="--gl-lr",
        values=[5e-5, 1e-4, 3e-4, 1e-3],
        schedulers=["cosine", "plateau", "constant"],
        epochs=1, seeds=seeds, num_subjects=args.num_subjects,
        extra=["--gl-epochs", str(args.gl_epochs),
               "--load-final-layer", lp_ckpt_pat],
        log_dir=f"{log_root}/global",
    )
    if best_gl is None:
        raise RuntimeError("Global sweep produced no successful run")
    summary["global"] = best_gl

    # 4) Retrain GL per seed ----------------------------------------------
    retrain_gl_per_seed(args.dataset, seeds, best_gl["value"], best_gl["scheduler"],
                        args.gl_epochs, args.num_subjects, lp_ckpt_pat, gl_ckpt_pat)

    # 5) Stacked ----------------------------------------------------------
    _, best_st = sweep(
        mode="stacked", dataset=args.dataset, param="--ft-lr",
        values=[3e-5, 1e-4, 3e-4],
        schedulers=["cosine", "plateau"],
        epochs=1, seeds=seeds, num_subjects=args.num_subjects,
        extra=["--ft-epochs", str(args.ft_epochs),
               "--load-global-lora", gl_ckpt_pat],
        log_dir=f"{log_root}/stacked",
    )
    summary["stacked"] = best_st

    # 6) Subject-specific -------------------------------------------------
    _, best_ss = sweep(
        mode="subject_specific", dataset=args.dataset, param="--ft-lr",
        values=[1e-4, 3e-4, 1e-3],
        schedulers=["cosine", "plateau"],
        epochs=1, seeds=seeds, num_subjects=args.num_subjects,
        extra=["--ft-epochs", str(args.ft_epochs),
               "--load-final-layer", lp_ckpt_pat],
        log_dir=f"{log_root}/subject_specific",
    )
    summary["subject_specific"] = best_ss

    # Final report --------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  FINAL REPORT — dataset={args.dataset}  seeds={seeds}")
    print(f"  (test reported ONLY here, on the winning config of each mode)")
    print("=" * 78)
    print(f"  {'mode':<18}  {'val_mean ±sem':>16}  {'test_mean ±sem':>17}  config")
    for mode, b in summary.items():
        if b is None:
            print(f"  {mode:<18}  (no successful run)")
            continue
        cfg = f"{b['value']:.1e} / {b['scheduler']}"
        print(f"  {mode:<18}  {b['val_mean']:8.4f} ±{b['val_sem']:.4f}  "
              f"{b['test_mean']:8.4f} ±{b['test_sem']:.4f}  {cfg}")

    summary_path = f"{log_root}/summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved -> {summary_path}")

    try:
        from plot import plot_summary
        plot_summary(summary, f"{log_root}/summary.png")
    except Exception as e:
        print(f"  [plot] skipped ({e})")


if __name__ == "__main__":
    main()
