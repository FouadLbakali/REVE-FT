"""Run `stacked` and `subject_specific` on bciciv2a, physionet, zuo2025.

Pipeline per dataset (single seed):
  1. stacked           -> trains LP + Global LoRA + per-subject in one run;
                          saves LP        -> ckpts/lp_<ds>_s<seed>.pt
                          saves Global LoRA -> ckpts/gl_<ds>_s<seed>/
  2. subject_specific  -> loads the LP from step 1, runs per-subject LoRA

Per-subject LoRA weights are NOT saved. ETA is printed after every step.

Usage: python run_modes.py [--seed 42] [--datasets bciciv2a physionet zuo2025]
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

LP_EPOCHS = 5
LP_LR = 2e-3
GL_EPOCHS = 10
GL_RANK = 32
FT_EPOCHS = 10
FT_RANK = 8
LORA_LR = 2e-4

STEP_ORDER = ["ST", "SS"]


def _fmt_hms(seconds):
    if seconds is None or seconds < 0 or seconds != seconds:
        return "--:--:--"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ETA:
    """Per-step-type averages -> remaining-time estimate over the full plan."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.done = 0
        self.per_type = {k: [] for k in STEP_ORDER}
        self.t0 = time.time()

    def _avg_for(self, tag):
        if self.per_type[tag]:
            return sum(self.per_type[tag]) / len(self.per_type[tag])
        flat = [x for v in self.per_type.values() for x in v]
        return (sum(flat) / len(flat)) if flat else None

    def tick(self, tag, run_seconds, label):
        self.per_type[tag].append(run_seconds)
        self.done += 1
        remaining_tags = self.plan[self.done:]
        avgs = [self._avg_for(t) for t in remaining_tags]
        eta = sum(a for a in avgs if a is not None) if all(a is not None for a in avgs) else None
        wall = time.time() - self.t0
        avg_tag = self._avg_for(tag)
        finish = (eta + wall) if eta is not None else None
        print(f"  [eta] {label}  step {self.done}/{len(self.plan)}  "
              f"last={_fmt_hms(run_seconds)}  avg[{tag}]={_fmt_hms(avg_tag)}  "
              f"elapsed={_fmt_hms(wall)}  ETA={_fmt_hms(eta)}  "
              f"finish~={_fmt_hms(finish)}")


def _run(cmd, what):
    print(f"\n>>> {what}")
    print(">>> " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"{what} failed (rc={rc})")


def step_stacked(dataset, seed, num_subjects, lp_ckpt, gl_ckpt, results_out):
    cmd = [sys.executable, "main.py",
           "--mode", "stacked", "--dataset", dataset,
           "--seed", str(seed), "--num-subjects", str(num_subjects),
           "--epochs", str(LP_EPOCHS), "--lr", str(LP_LR),
           "--gl-epochs", str(GL_EPOCHS), "--gl-rank", str(GL_RANK),
           "--ft-epochs", str(FT_EPOCHS), "--lora-rank", str(FT_RANK),
           "--lora-lr", str(LORA_LR),
           "--save-final-layer", lp_ckpt,
           "--save-global-lora", gl_ckpt,
           "--results-out", results_out]
    _run(cmd, f"[{dataset}] stacked (LP + GL + per-subject)")


def step_subject_specific(dataset, seed, num_subjects, lp_ckpt, results_out):
    cmd = [sys.executable, "main.py",
           "--mode", "subject_specific", "--dataset", dataset,
           "--seed", str(seed), "--num-subjects", str(num_subjects),
           "--ft-epochs", str(FT_EPOCHS), "--lora-rank", str(FT_RANK),
           "--lora-lr", str(LORA_LR),
           "--load-final-layer", lp_ckpt,
           "--results-out", results_out]
    _run(cmd, f"[{dataset}] subject_specific (LP loaded + per-subject)")


def main():
    global LP_EPOCHS, GL_EPOCHS, FT_EPOCHS
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["bciciv2a", "physionet", "zuo2025"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-subjects", type=int, default=109,
                   help="PhysioNet only (max 109); ignored by other datasets")
    p.add_argument("--lp-epochs", type=int, default=LP_EPOCHS)
    p.add_argument("--gl-epochs", type=int, default=GL_EPOCHS)
    p.add_argument("--ft-epochs", type=int, default=FT_EPOCHS)
    args = p.parse_args()

    LP_EPOCHS = args.lp_epochs
    GL_EPOCHS = args.gl_epochs
    FT_EPOCHS = args.ft_epochs

    os.makedirs("ckpts", exist_ok=True)
    seed_tag = f"s{args.seed}"
    results_root = f"results/{args.datasets}_{seed_tag}"
    os.makedirs(results_root, exist_ok=True)

    plan = []
    for _ in args.datasets:
        plan += STEP_ORDER
    eta = ETA(plan)

    print("=" * 78)
    print(f"  Plan: {len(args.datasets)} datasets × {len(STEP_ORDER)} modes = {len(plan)} runs")
    print(f"  datasets={args.datasets}  seed={args.seed}")
    print(f"  LP (in stacked): epochs={LP_EPOCHS}  lr={LP_LR}")
    print(f"  GL (in stacked): epochs={GL_EPOCHS}  rank={GL_RANK}")
    print(f"  FT             : epochs={FT_EPOCHS}  rank={FT_RANK}")
    print(f"  LoRA lr (GL+FT): {LORA_LR}")
    print("=" * 78)

    runs = []
    for dataset in args.datasets:
        lp_ckpt = f"ckpts/lp_{dataset}_{seed_tag}.pt"
        gl_ckpt = f"ckpts/gl_{dataset}_{seed_tag}"
        st_json = f"{results_root}/{dataset}_stacked.json"
        ss_json = f"{results_root}/{dataset}_subject_specific.json"

        t0 = time.time(); step_stacked(dataset, args.seed, args.num_subjects, lp_ckpt, gl_ckpt, st_json)
        eta.tick("ST", time.time() - t0, f"{dataset}/stacked")
        runs.append((dataset, "stacked", st_json))

        t0 = time.time(); step_subject_specific(dataset, args.seed, args.num_subjects, lp_ckpt, ss_json)
        eta.tick("SS", time.time() - t0, f"{dataset}/subject_specific")
        runs.append((dataset, "subject_specific", ss_json))

    total_wall = time.time() - eta.t0
    print("\n" + "=" * 78)
    print(f"  DONE — total wall time {_fmt_hms(total_wall)}")
    print("=" * 78)

    _write_summary(runs, results_root, total_wall, args)


def _write_summary(runs, results_root, total_wall, args):
    """Combined .json + human-readable .txt across all (dataset, mode) results."""
    combined = {"seed": args.seed,
                "datasets": args.datasets, "wall_seconds": total_wall, "runs": {}}
    txt_lines = ["=" * 78,
                 f"  RESULTS SUMMARY — seed={args.seed}",
                 f"  wall time: {_fmt_hms(total_wall)}",
                 "=" * 78]

    for dataset, mode, path in runs:
        key = f"{dataset}/{mode}"
        if not os.path.exists(path):
            txt_lines.append(f"\n[{key}] MISSING ({path})")
            continue
        with open(path) as f:
            data = json.load(f)
        combined["runs"][key] = data

        txt_lines.append(f"\n[{key}]  ({path})")
        stages = data.get("stages", {})
        for stage_name in ("lp", "gl"):
            s = stages.get(stage_name)
            if not s or s.get("skipped"):
                if s and s.get("skipped"):
                    txt_lines.append(f"  {stage_name}: SKIPPED ({s.get('reason') or s.get('loaded_from','')})")
                continue
            t = s.get("test", {})
            txt_lines.append(f"  {stage_name} test: " +
                             "  ".join(f"{k}={t[k]:.4f}" for k in
                                      ("acc","balanced_acc","cohen_kappa","f1","auroc","auc_pr") if k in t))
        agg = data.get("aggregate_subjects")
        if agg:
            txt_lines.append("  per-subject mean ± std:")
            for k in ("acc","balanced_acc","cohen_kappa","f1","auroc","auc_pr"):
                if k in agg:
                    txt_lines.append(f"    {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
        subs = data.get("subjects", {})
        if subs:
            txt_lines.append(f"  per-subject test balanced_acc (n={len(subs)}):")
            for sid in sorted(subs, key=lambda s: int(s) if s.isdigit() else s):
                bacc = subs[sid].get("test", {}).get("balanced_acc")
                acc = subs[sid].get("test", {}).get("acc")
                txt_lines.append(f"    s{sid}: balanced_acc={bacc:.4f}  acc={acc:.4f}")

    json_path = os.path.join(results_root, "summary.json")
    txt_path = os.path.join(results_root, "summary.txt")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"\n  Combined summary -> {json_path}")
    print(f"  Human-readable   -> {txt_path}")


if __name__ == "__main__":
    main()
