"""LoRA-rank sweep for `joint` and `joint_multilora` on REVE / bciciv2a.

Each (mode, rank) is a fully isolated `main.py` subprocess (fresh model +
deterministic seed per run), so the only thing that varies within a mode is the
LoRA rank. Both modes train from scratch (no separate LP phase), so there is
nothing to amortize across runs and the subprocess path is the robust one.

Rank knob per mode (the rest of the LoRA config is identical):
  - joint            -> --gl-rank   (stage_global_lora, results in stages.gl)
  - joint_multilora  -> --lora-rank (inject_multi_subject_lora, stages.multilora)

Selection is on val balanced_acc (max over the history); test balanced_acc is
reported alongside but never used to pick the rank.

Usage:
    python scripts/run_joint_rank_sweep.py [--dataset bciciv2a] [--seed 42]
        [--ranks 1 2 4 8 16 32 64 128] [--modes joint joint_multilora]
        [--epochs 25] [--patience 5] [--lora-lr 1e-4] [--batch-size 32]
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

RANKS = [1, 2, 4, 8, 16, 32, 64, 128]
MODES = ["joint", "joint_multilora"]

# (rank CLI flag, results-JSON stage key) per mode.
_MODE_SPEC = {
    "joint":           ("--gl-rank", "gl"),
    "joint_multilora": ("--lora-rank", "multilora"),
}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fmt_hms(seconds):
    if seconds is None or seconds < 0 or seconds != seconds:
        return "--:--:--"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ETA:
    def __init__(self, total_runs):
        self.total = total_runs
        self.done = 0
        self.elapsed = 0.0
        self.t0 = time.time()

    def tick(self, run_seconds, label):
        self.done += 1
        self.elapsed += run_seconds
        avg = self.elapsed / self.done
        eta = avg * (self.total - self.done)
        wall = time.time() - self.t0
        print(f"  [eta] {label}  run {self.done}/{self.total}  "
              f"last={_fmt_hms(run_seconds)}  avg={_fmt_hms(avg)}  "
              f"elapsed={_fmt_hms(wall)}  ETA={_fmt_hms(eta)}")


def _val_from_history(history):
    vals = [h["val_balanced_acc"] for h in history
            if h.get("val_balanced_acc") == h.get("val_balanced_acc")]
    return max(vals) if vals else float("nan")


def _summarize(results_path, stage_key):
    """(val, test) from a main.py results JSON; raises if the stage is missing."""
    with open(results_path) as f:
        data = json.load(f)
    block = data.get("stages", {}).get(stage_key)
    if not block or block.get("skipped"):
        raise RuntimeError(f"stage '{stage_key}' missing/skipped in {results_path}")
    return _val_from_history(block["history"]), float(block["test"]["balanced_acc"])


def _run_one(mode, rank, args, results_path):
    flag, _ = _MODE_SPEC[mode]
    cmd = [sys.executable, "main.py",
           "--mode", mode, "--model", "reve", "--dataset", args.dataset,
           "--seed", str(args.seed), "--num-subjects", str(args.num_subjects),
           "--epochs", str(args.epochs), "--patience", str(args.patience),
           "--lora-lr", str(args.lora_lr), "--batch-size", str(args.batch_size),
           flag, str(rank),
           "--results-out", results_path]
    print(f"\n>>> [{mode}] rank={rank}")
    print(">>> " + " ".join(cmd))
    return subprocess.run(cmd, cwd=_REPO_ROOT).returncode


def _format_rows(rows, mode, seed):
    lines = ["=" * 78,
             f"  Joint rank sweep — mode={mode}  dataset  seed={seed}",
             "=" * 78,
             f"  {'rank':>4}  {'val':>8}  {'test':>8}  rc"]
    for r in rows:
        lines.append(f"  {r['rank']:>4}  {r['val']:8.4f}  {r['test']:8.4f}  {r['rc']}")
    best = next((r for r in rows if r["rc"] == 0 and r["val"] == r["val"]), None)
    if best is not None:
        lines += ["",
                  f"  >>> BEST rank={best['rank']}  "
                  f"val={best['val']:.4f}  test={best['test']:.4f}"]
    return lines, best


def sweep_mode(mode, ranks, args, log_dir, eta):
    os.makedirs(log_dir, exist_ok=True)
    _, stage_key = _MODE_SPEC[mode]
    print(f"\n{'=' * 78}\n  Joint rank sweep — mode={mode}  ranks={ranks}  "
          f"seed={args.seed}\n{'=' * 78}")

    rows = []
    for rank in ranks:
        results_path = os.path.join(log_dir, f"rank_{rank}.json")
        t0 = time.time()
        rc = _run_one(mode, rank, args, results_path)
        try:
            if rc != 0:
                raise RuntimeError(f"main.py exited rc={rc}")
            val, test = _summarize(results_path, stage_key)
        except Exception as e:
            print(f"!!! [{mode}] rank={rank} failed: {e}")
            val, test, rc = float("nan"), float("nan"), (rc or 1)
        dt = time.time() - t0
        rows.append({"rank": rank, "val": val, "test": test,
                     "rc": rc, "seconds": dt})
        eta.tick(dt, f"{mode}/rank{rank}")

    rows.sort(key=lambda r: (r["rc"] != 0,
                             -(r["val"] if r["val"] == r["val"] else -1)))
    lines, best = _format_rows(rows, mode, args.seed)
    print("\n" + "\n".join(lines))
    with open(os.path.join(log_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(log_dir, "rows.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bciciv2a",
                   choices=["bciciv2a", "physionet", "zuo2025"])
    p.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    p.add_argument("--ranks", nargs="+", type=int, default=RANKS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-subjects", type=int, default=109,
                   help="ignored by bciciv2a (always loads all 9 subjects)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    log_root = os.path.join(_REPO_ROOT,
                            f"sweep_logs/joint_rank_{args.dataset}_s{args.seed}")
    os.makedirs(log_root, exist_ok=True)

    print("=" * 78)
    print(f"  Plan: {len(args.modes)} modes × {len(args.ranks)} ranks = "
          f"{len(args.modes) * len(args.ranks)} runs")
    print(f"  dataset={args.dataset}  seed={args.seed}  ranks={args.ranks}")
    print(f"  epochs={args.epochs}  patience={args.patience}  "
          f"lora_lr={args.lora_lr}  batch_size={args.batch_size}")
    print("=" * 78)

    overall_t0 = time.time()
    eta = ETA(total_runs=len(args.modes) * len(args.ranks))
    per_mode = {}
    for mode in args.modes:
        rows, best = sweep_mode(
            mode, args.ranks, args,
            log_dir=os.path.join(log_root, mode), eta=eta,
        )
        per_mode[mode] = {"rank_rows": rows, "rank_best": best}

    total = time.time() - overall_t0
    print("\n" + "=" * 78)
    print(f"  FINAL — dataset={args.dataset}  seed={args.seed}  "
          f"total wall time={_fmt_hms(total)}")
    print("=" * 78)
    print(f"  {'mode':<18}  {'best rank':>9}  {'val':>8}  {'test':>8}")
    for mode in args.modes:
        best = per_mode[mode]["rank_best"]
        if best is None:
            print(f"  {mode:<18}  (no successful rank run)")
        else:
            print(f"  {mode:<18}  {best['rank']:>9}  "
                  f"{best['val']:8.4f}  {best['test']:8.4f}")

    summary = {
        "dataset": args.dataset, "seed": args.seed,
        "modes": args.modes, "ranks": args.ranks,
        "config": {"epochs": args.epochs, "patience": args.patience,
                   "lora_lr": args.lora_lr, "batch_size": args.batch_size},
        "results": per_mode, "wall_seconds": total,
    }
    with open(os.path.join(log_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary -> {log_root}/summary.json")


if __name__ == "__main__":
    main()
