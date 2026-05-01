import argparse

import torch
from transformers import AutoModel, set_seed

from stages import (
    run_global_lora,
    run_linear_probing_cached,
    run_three_stage,
    run_two_stage,
)

TIME_PATCHES = {"bciciv2a": 5, "physionet": 3, "zuo2025": 5}
NUM_CHANNELS = {"bciciv2a": 22, "physionet": 22, "zuo2025": 32}
NUM_CLASSES = {"bciciv2a": 4, "physionet": 4, "zuo2025": 2}


def parse_args():
    parser = argparse.ArgumentParser(description='REVE Training')
    parser.add_argument('--mode', default="linear",
                        choices=["linear", "global", "stacked", "subject_specific"])
    parser.add_argument('--dataset', default="bciciv2a", choices=["bciciv2a", "physionet", "zuo2025"])
    parser.add_argument('--num-subjects', default=109, type=int, help='PhysioNet: number of subjects to load (max 109)')
    parser.add_argument('--epochs', default=5, type=int, help='number of epoch')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--scheduler', default='plateau', choices=['plateau', 'cosine', 'constant'],
                        help='LR scheduler used by every training stage')
    parser.add_argument('--l1', default=1e-4, type=float, help='L1 regularization strength (0 disables)')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--seed', default=None, type=int, help='seed')
    parser.add_argument('--save-final-layer', default=None, type=str, help='path to save the best final layer (linear mode)')
    parser.add_argument('--load-final-layer', default=None, type=str, help='path to load a pretrained final layer')
    parser.add_argument('--save-global-lora', default=None, type=str, help='directory to save Global LoRA adapters (PEFT save_pretrained)')
    parser.add_argument('--load-global-lora', default=None, type=str, help='directory to load Global LoRA adapters (skips LP and GL stages)')
    # Per-subject LoRA
    parser.add_argument('--ft-epochs', default=5, type=int, help='per-subject LoRA epochs')
    parser.add_argument('--ft-lr', default=1e-4, type=float, help='per-subject LoRA learning rate')
    parser.add_argument('--lora-rank', default=8, type=int, help='per-subject LoRA rank')
    # Global LoRA (for three_stage and global_lora modes)
    parser.add_argument('--gl-epochs', default=5, type=int, help='global LoRA epochs')
    parser.add_argument('--gl-lr', default=1e-4, type=float, help='global LoRA learning rate')
    parser.add_argument('--gl-rank', default=8, type=int, help='global LoRA rank')
    return parser.parse_args()


def build_model(dataset, load_final_layer=None):
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, dtype="auto")
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, dtype="auto")

    dim = NUM_CHANNELS[dataset] * TIME_PATCHES[dataset] * 512
    model.final_layer = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.RMSNorm(dim),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(dim, NUM_CLASSES[dataset]),
    )

    if load_final_layer:
        state = torch.load(load_final_layer, weights_only=True)
        model.final_layer.load_state_dict(state)
        print(f"Final layer loaded from {load_final_layer}")

    return model, pos_bank


def main():
    args = parse_args()

    if args.seed:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    model, pos_bank = build_model(args.dataset, args.load_final_layer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runners = {
        "linear": run_linear_probing_cached,
        "global": run_global_lora,
        "stacked": run_three_stage,
        "subject_specific": run_two_stage,
    }
    runners[args.mode](model, pos_bank, args, device)


if __name__ == "__main__":
    main()
