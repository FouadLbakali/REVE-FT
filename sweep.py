"""Grid sweep with multi-seed aggregation.

Selection is done on `val` aggregated over multiple seeds (`val_mean`). `test`
is parsed from logs but NOT printed in the per-config table — it must not bias
human selection. Test is reported only at the very end, on the winning config.
"""
import argparse
import math
import os
import re
import subprocess
import sys
import time
from itertools import product
from statistics import mean, stdev

VAL_RE = re.compile(r"best:\s*([0-9.]+)")
TEST_RE = re.compile(r"^\s*balanced_acc:\s*([0-9.]+)", re.MULTILINE)


def run_one(base_cmd, param, value, sched, seed):
    cmd = base_cmd + ["--seed", str(seed), param, str(value), "--scheduler", sched]
    print(f"\n>>> {' '.join(cmd)}")
    for attempt in range(3):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Retry on transient NFS EACCES when the interpreter can't open the script
        if proc.returncode == 2 and "Permission denied" in (proc.stdout or ""):
            print(f"!!! transient Permission denied (attempt {attempt+1}/3), retrying in 5s...")
            time.sleep(5)
            continue
        break
    out = proc.stdout
    if proc.returncode != 0:
        tail = "\n".join(out.splitlines()[-40:])
        print(f"!!! run failed (rc={proc.returncode}). Last 40 lines:\n{tail}")
    val_matches = [float(m) for m in VAL_RE.findall(out)]
    test_matches = [float(m) for m in TEST_RE.findall(out)]
    best_val = max(val_matches) if val_matches else float("nan")
    test_bacc = test_matches[-1] if test_matches else float("nan")
    return best_val, test_bacc, proc.returncode


def _agg(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return mean(xs), stdev(xs) / math.sqrt(len(xs))


def sweep(mode, dataset, param, values, schedulers, epochs, seeds,
          num_subjects=109, extra=None, log_dir="sweep_logs"):
    """Run the grid across `seeds` and return (sorted rows, best).

    Each row aggregates the seeds for one (value, scheduler):
      {value, scheduler, val_mean, val_sem, test_mean, test_sem,
       n_ok, n_total, logs}
    Selection is on val_mean. Boundary warning is emitted when the best
    `value` is at min/max of the grid.
    """
    extra = list(extra or [])
    os.makedirs(log_dir, exist_ok=True)
    if isinstance(seeds, int):
        seeds = [seeds]

    base_no_extra = [sys.executable, "main.py",
                     "--mode", mode, "--dataset", dataset,
                     "--epochs", str(epochs),
                     "--num-subjects", str(num_subjects)]

    param_tag = param.lstrip("-").replace("-", "_")
    rows = []
    for value, sched in product(values, schedulers):
        vals, tests, rcs = [], [], []
        for seed in seeds:
            extra_seed = [a.format(seed=seed) for a in extra]
            base = base_no_extra + extra_seed
            v, t, rc = run_one(base, param, value, sched, seed)
            vals.append(v); tests.append(t); rcs.append(rc)
        n_ok = sum(1 for rc in rcs if rc == 0)
        ok_vals = [v for v, rc in zip(vals, rcs) if rc == 0]
        ok_tests = [t for t, rc in zip(tests, rcs) if rc == 0]
        v_mean, v_sem = _agg(ok_vals)
        t_mean, t_sem = _agg(ok_tests)
        rows.append({
            "value": value, "scheduler": sched,
            "val_mean": v_mean, "val_sem": v_sem,
            "test_mean": t_mean, "test_sem": t_sem,
            "n_ok": n_ok, "n_total": len(seeds),
            # legacy keys for plot.py
            "val": v_mean, "test": t_mean, "rc": 0 if n_ok > 0 else 1,
        })

    rows.sort(key=lambda r: (r["n_ok"] == 0, -r["val_mean"]
                             if not math.isnan(r["val_mean"]) else float("inf")))

    lines = []
    lines.append("=" * 78)
    lines.append(f"  Sweep — mode={mode}  param={param}  seeds={seeds}  (test hidden)")
    lines.append("=" * 78)
    lines.append(f"  {'val_mean':>9} {'±sem':>7}  {param_tag:>10}  {'scheduler':<10}  ok")
    for r in rows:
        flag = "" if r["n_ok"] == len(seeds) else f"  [{r['n_ok']}/{r['n_total']}]"
        lines.append(f"  {r['val_mean']:9.4f} {r['val_sem']:7.4f}  "
                     f"{r['value']:10.1e}  {r['scheduler']:<10}  {r['n_ok']}/{r['n_total']}{flag}")

    best = next((r for r in rows if r["n_ok"] > 0 and not math.isnan(r["val_mean"])), None)
    if best is not None:
        lines.append("")
        lines.append(f"  >>> BEST: {param}={best['value']} scheduler={best['scheduler']}  "
                     f"val={best['val_mean']:.4f}±{best['val_sem']:.4f}  "
                     f"test={best['test_mean']:.4f}±{best['test_sem']:.4f}")
        vmin, vmax = min(values), max(values)
        if best["value"] == vmin:
            lines.append(f"  ⚠  BEST is at LOWER edge of grid ({vmin:.1e}). Extend the grid downward.")
        elif best["value"] == vmax:
            lines.append(f"  ⚠  BEST is at UPPER edge of grid ({vmax:.1e}). Extend the grid upward.")

    print("\n" + "\n".join(lines))
    try:
        with open(os.path.join(log_dir, "summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        print(f"  [summary] could not write summary.txt ({e})")

    try:
        from plot import plot_sweep
        plot_sweep(rows, mode, param, os.path.join(log_dir, "sweep.png"))
    except Exception as e:
        print(f"  [plot] skipped ({e})")

    return rows, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["linear", "global", "stacked", "subject_specific"])
    p.add_argument("--dataset", default="bciciv2a")
    p.add_argument("--param", default="--lr")
    p.add_argument("--values", nargs="+", type=float, default=[1e-3, 3e-3, 1e-2, 3e-2])
    p.add_argument("--schedulers", nargs="+", default=["plateau", "cosine", "constant"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--num-subjects", type=int, default=109)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    p.add_argument("--log-dir", default="sweep_logs")
    a = p.parse_args()
    sweep(a.mode, a.dataset, a.param, a.values, a.schedulers, a.epochs,
          a.seeds, a.num_subjects, a.extra, a.log_dir)


if __name__ == "__main__":
    main()
