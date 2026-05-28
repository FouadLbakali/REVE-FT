import argparse
import json
import os

# # Redirect MNE / MOABB caches away from $HOME (which is not writable here).
# _CACHE_ROOT = "/users/local/REVE-FT/.cache"
# os.makedirs(_CACHE_ROOT, exist_ok=True)
# os.makedirs(os.path.join(_CACHE_ROOT, "mne_data"), exist_ok=True)
# os.environ.setdefault("_MNE_FAKE_HOME_DIR", _CACHE_ROOT)
# os.environ.setdefault("MNE_DATA", os.path.join(_CACHE_ROOT, "mne_data"))
# os.environ.setdefault("MNE_DATASETS_BNCI_PATH", os.environ["MNE_DATA"])
# os.environ.setdefault("MOABB_RESULTS", os.path.join(_CACHE_ROOT, "moabb_results"))
# os.environ.setdefault("XDG_CACHE_HOME", _CACHE_ROOT)

import torch
import yaml
from transformers import AutoModel, set_seed

from .labram_zoo import LabramSpec
from .luna_zoo import LunaSpec
from .engine import set_bf16
from .stages import (
    run_global,
    run_subject_specific,
    run_stacked,
    run_linear_probing_cached,
)

_DATASETS_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "datasets.yaml")
with open(_DATASETS_YAML) as f:
    _DATASETS = yaml.safe_load(f)

TIME_PATCHES = {name: cfg["time_patches"] for name, cfg in _DATASETS.items()}
NUM_CHANNELS = {name: cfg["num_channels"] for name, cfg in _DATASETS.items()}
NUM_CLASSES = {name: cfg["num_classes"] for name, cfg in _DATASETS.items()}


def parse_args():
    parser = argparse.ArgumentParser(description='REVE Training')
    parser.add_argument('--mode', default="linear",
                        choices=["linear", "global", "subject-specific",
                                 "stacked"])
    parser.add_argument('--model', default="reve", choices=["reve", "labram", "luna"],
                        help='backbone: reve (brain-bzh/reve-base), labram '
                             '(braindecode/labram-pretrained) or luna '
                             '(PulpBio/LUNA); ')
    parser.add_argument('--dataset', default="bciciv2a", choices=["bciciv2a", "physionet", "zuo2025"])
    parser.add_argument('--num-subjects', default=109, type=int, help='Number of subjects to load (PhysioNet max 109, bciciv2a max 9)')
    parser.add_argument('--epochs', default=25, type=int,
                        help='number of epochs (shared by every training stage: LP, GL, per-subject LoRA)')
    parser.add_argument('--patience', default=5, type=int,
                        help='early-stopping patience on val balanced_acc; <=0 disables early stopping')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate (shared by LP head and LoRA stages)')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--save-all', default=None, type=str,
                        help='directory to save the full merged model (backbone + LoRA '
                             'merged + head). For global: writes <model>.pt. For '
                             'subject-specific and stacked: writes one '
                             '<model>_subject_<id>.pt per subject (subject-specific LoRA '
                             'merged in). Only valid with --mode global, subject-specific '
                             'or stacked')
    parser.add_argument('--lora-rank', default=8, type=int, help='per-subject LoRA rank')
    parser.add_argument('--gl-rank', default=32, type=int, help='global LoRA rank')
    parser.add_argument('--results-out', default=None, type=str,
                        help='path to dump a JSON file with histories + test metrics + per-subject results')
    parser.add_argument('--bf16', action='store_true',
                        help='enable bfloat16 autocast (default: float32)')
    return parser.parse_args()

def build_model(dataset, model_name="reve"):
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, dtype="auto")

    if model_name == "labram":
        # The montage (chs_info) and trial length are only known once the data
        # loaders exist, so defer construction to the linear-probing runner.
        # pos_bank is still used by the loaders' collate (LaBraM ignores pos).
        return LabramSpec(NUM_CLASSES[dataset]), pos_bank

    if model_name == "luna":
        # Same deferred-build story as LaBraM: ch_names (-> 3-D positions) and
        # trial length are only known after the loaders are built.
        return LunaSpec(NUM_CLASSES[dataset]), pos_bank

    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, dtype="auto")

    dim = NUM_CHANNELS[dataset] * TIME_PATCHES[dataset] * 512
    model.final_layer = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.RMSNorm(dim),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(dim, NUM_CLASSES[dataset]),
    )

    return model, pos_bank


def main():
    args = parse_args()

    if args.bf16:
        set_bf16(True)

    if args.model == "labram" and args.mode not in (
        "linear", "global", "subject-specific", "stacked"
    ):
        raise SystemExit(f"--model {args.model} currently supports --mode linear, global, "
                          "subject-specific and stacked only")

    if args.save_all and args.mode not in (
        "global", "subject-specific", "stacked"
    ):
        raise SystemExit("--save-all is only valid with --mode global, subject-specific "
                          "or stacked")

    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    model, pos_bank = build_model(args.dataset, args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runners = {
        "linear": run_linear_probing_cached,
        "global": run_global,
        "subject-specific": run_subject_specific,
        "stacked": run_stacked,
    }

    results = {
        "dataset": args.dataset,
        "mode": args.mode,
        "seed": args.seed,
        "num_subjects": args.num_subjects,
        "epochs": args.epochs,
        "patience": args.patience,
        "hyperparams": {
            "lp": {"lr": args.lr},
            "gl": {"lr": args.lr, "rank": args.gl_rank},
            "ft": {"lr": args.lr, "rank": args.lora_rank},
        },
        "stages": {},
    }
    runners[args.mode](model, pos_bank, args, device, results=results)

    if args.results_out:
        os.makedirs(os.path.dirname(args.results_out) or ".", exist_ok=True)
        with open(args.results_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results dumped to {args.results_out}")


if __name__ == "__main__":
    main()
