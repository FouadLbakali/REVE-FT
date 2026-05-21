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
from transformers import AutoModel, set_seed

from labram_zoo import LabramSpec
from luna_zoo import LunaSpec
from engine import set_bf16
from stages import (
    run_global_lora,
    run_joint_lora,
    run_joint_multilora,
    run_joint_multilora_global,
    run_joint_stacked,
    run_joint_subject_specific,
    run_linear_probing_cached,
    run_three_stage,
    run_two_stage,
)

TIME_PATCHES = {"bciciv2a": 5, "physionet": 3, "zuo2025": 5}
NUM_CHANNELS = {"bciciv2a": 22, "physionet": 64, "zuo2025": 30}
NUM_CLASSES = {"bciciv2a": 4, "physionet": 4, "zuo2025": 2}


def parse_args():
    parser = argparse.ArgumentParser(description='REVE Training')
    parser.add_argument('--mode', default="linear",
                        choices=["linear", "global", "joint", "stacked", "subject_specific",
                                 "joint_stacked", "joint_subject_specific", "joint_multilora",
                                 "joint_multilora_global"])
    parser.add_argument('--model', default="reve", choices=["reve", "labram", "luna"],
                        help='backbone: reve (brain-bzh/reve-base), labram '
                             '(braindecode/labram-pretrained) or luna '
                             '(PulpBio/LUNA, --luna-variant picks base/large/huge); '
                             'labram and luna support --mode linear, joint, '
                             'joint_multilora and joint_multilora_global')
    parser.add_argument('--luna-variant', default="large",
                        choices=["base", "large", "huge"],
                        help='LUNA variant when --model luna')
    parser.add_argument('--pooling', default="flatten",
                        choices=["flatten", "mean", "labram"],
                        help="'labram' is the official LaBraM classifier head "
                             "(fc_norm + mean over patch tokens + linear) and "
                             "is only valid with --model labram")
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
    parser.add_argument('--save-final-layer', default=None, type=str, help='path to save the best final layer (linear mode)')
    parser.add_argument('--load-final-layer', default=None, type=str, help='path to load a pretrained final layer')
    parser.add_argument('--save-global-lora', default=None, type=str, help='directory to save Global LoRA adapters (PEFT save_pretrained)')
    parser.add_argument('--load-global-lora', default=None, type=str, help='directory to load Global LoRA adapters (skips LP and GL stages)')
    # Per-subject LoRA
    parser.add_argument('--lora-rank', default=8, type=int, help='per-subject LoRA rank')
    # Global LoRA (for three_stage and global_lora modes)
    parser.add_argument('--gl-rank', default=32, type=int, help='global LoRA rank')
    parser.add_argument('--results-out', default=None, type=str,
                        help='path to dump a JSON file with histories + test metrics + per-subject results')
    parser.add_argument('--bf16', action='store_true',
                        help='enable bfloat16 autocast (default: float32)')
    return parser.parse_args()

def build_model(dataset, load_final_layer=None, pooling="flatten", model_name="reve",
                luna_variant="large"):
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, dtype="auto")

    if model_name == "labram":
        # The montage (chs_info) and trial length are only known once the data
        # loaders exist, so defer construction to the linear-probing runner.
        # pos_bank is still used by the loaders' collate (LaBraM ignores pos).
        return LabramSpec(NUM_CLASSES[dataset], pooling=pooling), pos_bank

    if model_name == "luna":
        # Same deferred-build story as LaBraM: ch_names (-> 3-D positions) and
        # trial length are only known after the loaders are built.
        return LunaSpec(NUM_CLASSES[dataset], variant=luna_variant), pos_bank

    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, dtype="auto")

    if pooling == "mean":
        class MeanPool(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
            def forward(self, x):
                return x.mean(dim=self.dim)

        model.final_layer = torch.nn.Sequential(
            MeanPool(dim=(1, 2)),                     # (B, C, H, E) -> (B, E)
            torch.nn.Linear(512, NUM_CLASSES[dataset]),
        )
    elif pooling == "flatten":
        dim = NUM_CHANNELS[dataset] * TIME_PATCHES[dataset] * 512
        model.final_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.RMSNorm(dim),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(dim, NUM_CLASSES[dataset]),
        )
    else:
        raise ValueError(f"Unknown pooling {pooling!r}, expected 'flatten' or 'mean'")

    if load_final_layer:
        state = torch.load(load_final_layer, weights_only=True)
        model.final_layer.load_state_dict(state)
        print(f"Final layer loaded from {load_final_layer}")

    return model, pos_bank


def main():
    args = parse_args()

    if args.model == "labram" and args.mode not in (
        "linear", "joint", "joint_multilora", "joint_multilora_global"
    ):
        raise SystemExit(f"--model {args.model} currently supports --mode linear, joint, "
                          "joint_multilora and joint_multilora_global only")

    if args.pooling == "labram" and args.model != "labram":
        raise SystemExit("--pooling labram is only valid with --model labram")

    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    model, pos_bank = build_model(args.dataset, args.load_final_layer, args.pooling,
                                  args.model, luna_variant=args.luna_variant)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runners = {
        "linear": run_linear_probing_cached,
        "global": run_global_lora,
        "joint": run_joint_lora,
        "stacked": run_three_stage,
        "subject_specific": run_two_stage,
        "joint_stacked": run_joint_stacked,
        "joint_subject_specific": run_joint_subject_specific,
        "joint_multilora": run_joint_multilora,
        "joint_multilora_global": run_joint_multilora_global,
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
