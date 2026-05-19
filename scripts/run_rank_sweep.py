"""Sequential LoRA-rank sweep for `global` and `subject_specific` (in-process).

The dataset and the linear-probing stage are shared across all rank runs:
  - dataset loaded once
  - backbone + pos_bank built once
  - LP trained once -> lp_checkpoint
  - for each mode in {global, subject_specific}:
      for each rank: only the LoRA stage runs, starting from lp_checkpoint

Usage:
    python run_rank_sweep.py [--dataset bciciv2a] [--seed 42]
"""
import argparse
import json
import math
import os
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

import torch
from transformers import set_seed

from data import load_loaders_per_subject
from main import build_model
from stages import (
    stage_global_lora,
    stage_linear_probing,
    stage_per_subject_lora,
)

RANKS = [1, 2, 4, 8, 16, 32, 64, 128]


def _fmt_hms(seconds):
    if seconds < 0 or seconds != seconds:
        return "--:--:--"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ETA:
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


def _make_args(**kw):
    """Build a Namespace with the same defaults main.py uses, overriding via kw."""
    a = argparse.Namespace(
        mode=None,
        dataset="bciciv2a",
        num_subjects=109,
        epochs=5,
        lr=2e-3,
        batch_size=32,
        seed=None,
        save_final_layer=None,
        load_final_layer=None,
        save_global_lora=None,
        load_global_lora=None,
        ft_epochs=1,
        lora_rank=8,
        gl_epochs=1,
        gl_rank=32,
        lora_lr=2e-4,
        results_out=None,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _val_from_history(history):
    vals = [h["val_balanced_acc"] for h in history if h.get("val_balanced_acc") == h.get("val_balanced_acc")]
    return max(vals) if vals else float("nan")


def _summarize_global(res):
    gl = res["stages"]["gl"]
    val = _val_from_history(gl["history"])
    test = float(gl["test"]["balanced_acc"])
    return val, test


def _summarize_subject_specific(res):
    subjects = res.get("subjects", {}) or {}
    per_subj_val = [_val_from_history(s["history"]) for s in subjects.values()]
    per_subj_val = [v for v in per_subj_val if not (isinstance(v, float) and math.isnan(v))]
    val = sum(per_subj_val) / len(per_subj_val) if per_subj_val else float("nan")
    test = float(res["aggregate_subjects"]["balanced_acc"]["mean"])
    return val, test


def _format_rows(rows, mode, seed):
    lines = ["=" * 78,
             f"  Rank sweep — mode={mode}  seed={seed}",
             "=" * 78,
             f"  {'rank':>4}  {'val':>8}  {'test':>8}  rc"]
    for r in rows:
        lines.append(f"  {r['rank']:>4}  {r['val']:8.4f}  {r['test']:8.4f}  {r['rc']}")
    best = next((r for r in rows if r["rc"] == 0 and r["val"] == r["val"]), None)
    if best is not None:
        lines.append("")
        lines.append(f"  >>> BEST rank={best['rank']}  "
                     f"val={best['val']:.4f}  test={best['test']:.4f}")
    return lines, best


def rank_sweep(mode, model, loaders, lp_checkpoint, args, device, log_dir, eta):
    """Run the rank sweep for one mode in-process. Returns (rows, best)."""
    pooled_loaders, subject_loaders = loaders
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'=' * 78}\n  Rank sweep — mode={mode}  ranks={RANKS}  "
          f"seed={args.seed}\n{'=' * 78}")

    rows = []
    for rank in RANKS:
        if mode == "global":
            run_args = _make_args(
                mode="global", dataset=args.dataset, seed=args.seed,
                num_subjects=args.num_subjects, batch_size=args.batch_size,
                gl_rank=rank, gl_epochs=args.gl_epochs, lora_lr=args.lora_lr,
            )
        else:
            run_args = _make_args(
                mode="subject_specific", dataset=args.dataset, seed=args.seed,
                num_subjects=args.num_subjects, batch_size=args.batch_size,
                lora_rank=rank, ft_epochs=args.ft_epochs, lora_lr=args.lora_lr,
            )

        res = {"stages": {}}
        t0 = time.time()
        rc = 0
        try:
            if mode == "global":
                stage_global_lora(model, pooled_loaders, run_args, device, lp_checkpoint, results=res)
                val, test = _summarize_global(res)
            else:
                stage_per_subject_lora(model, subject_loaders, run_args, device, lp_checkpoint, results=res)
                val, test = _summarize_subject_specific(res)
        except Exception as e:
            print(f"!!! rank={rank} failed: {e}")
            val, test, rc = float("nan"), float("nan"), 1
        dt = time.time() - t0

        rows.append({"rank": rank, "val": val, "test": test, "rc": rc, "seconds": dt})
        eta.tick(dt)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    p.add_argument("--dataset", default="bciciv2a")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-subjects", type=int, default=109)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lp-epochs", type=int, default=5, help="LP epochs (trained once, shared)")
    p.add_argument("--lp-lr", type=float, default=2e-3)
    p.add_argument("--gl-epochs", type=int, default=1)
    p.add_argument("--ft-epochs", type=int, default=1)
    p.add_argument("--lora-lr", type=float, default=2e-4)
    args = p.parse_args()

    log_root = f"sweep_logs/rank_{args.dataset}_s{args.seed}"
    os.makedirs(log_root, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    overall_t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 78}\n  Loading model + dataset (once)\n{'=' * 78}")
    t0 = time.time()
    model, pos_bank = build_model(args.dataset)
    pooled_loaders, subject_loaders = load_loaders_per_subject(
        args.dataset, pos_bank, args.batch_size, args.seed, args.num_subjects
    )
    print(f"  loaded in {_fmt_hms(time.time() - t0)}")

    print(f"\n{'=' * 78}\n  Shared linear probing (trained once)\n{'=' * 78}")
    lp_args = _make_args(
        mode="linear", dataset=args.dataset, seed=args.seed,
        num_subjects=args.num_subjects, batch_size=args.batch_size,
        epochs=args.lp_epochs, lr=args.lp_lr,
    )
    lp_results = {"stages": {}}
    t0 = time.time()
    lp_checkpoint = stage_linear_probing(model, pooled_loaders, lp_args, device, results=lp_results)
    lp_seconds = time.time() - t0
    print(f"  LP done in {_fmt_hms(lp_seconds)}")

    eta = ETA(total_runs=2 * len(RANKS), label="sweep")
    loaders = (pooled_loaders, subject_loaders)

    gl_rows, gl_best = rank_sweep(
        "global", model, loaders, lp_checkpoint, args, device,
        log_dir=f"{log_root}/global/rank", eta=eta,
    )
    ss_rows, ss_best = rank_sweep(
        "subject_specific", model, loaders, lp_checkpoint, args, device,
        log_dir=f"{log_root}/subject_specific/rank", eta=eta,
    )

    total = time.time() - overall_t0
    print("\n" + "=" * 78)
    print(f"  FINAL — dataset={args.dataset}  seed={args.seed}  "
          f"total wall time={_fmt_hms(total)}  (LP={_fmt_hms(lp_seconds)})")
    print("=" * 78)
    print(f"  {'mode':<18}  {'best rank':>9}  {'val':>8}  {'test':>8}")
    for name, best in [("global", gl_best), ("subject_specific", ss_best)]:
        if best is None:
            print(f"  {name:<18}  (no successful rank run)")
            continue
        print(f"  {name:<18}  {best['rank']:>9}  "
              f"{best['val']:8.4f}  {best['test']:8.4f}")

    summary = {
        "dataset": args.dataset, "seed": args.seed,
        "ranks": RANKS,
        "lp": {
            "epochs": args.lp_epochs, "lr": args.lp_lr,
            "seconds": lp_seconds,
            "test": lp_results.get("stages", {}).get("lp", {}).get("test"),
        },
        "global": {"rank_rows": gl_rows, "rank_best": gl_best},
        "subject_specific": {"rank_rows": ss_rows, "rank_best": ss_best},
        "wall_seconds": total,
    }
    with open(f"{log_root}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary -> {log_root}/summary.json")


if __name__ == "__main__":
    main()
