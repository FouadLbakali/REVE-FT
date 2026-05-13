"""Per-subject test accuracy on BCI-IV-2a: global LoRA vs stacked vs subject_specific.

- Global LoRA: evaluated here from ckpts/gl_bciciv2a_s42 (final_layer is in the adapter).
- Stacked / subject_specific: already-computed per-subject accuracies are read from
  results/['bciciv2a', 'physionet', 'zuo2025']_s42/bciciv2a_{stacked,subject_specific}.json

The three modes are compared on the same per-subject test split (seed=42).
"""
import json
import os

# Redirect MNE / MOABB caches (mirrors main.py).
_CACHE_ROOT = "/users/local/REVE-FT/.cache"
os.makedirs(_CACHE_ROOT, exist_ok=True)
os.makedirs(os.path.join(_CACHE_ROOT, "mne_data"), exist_ok=True)
os.environ.setdefault("_MNE_FAKE_HOME_DIR", _CACHE_ROOT)
os.environ.setdefault("MNE_DATA", os.path.join(_CACHE_ROOT, "mne_data"))
os.environ.setdefault("MNE_DATASETS_BNCI_PATH", os.environ["MNE_DATA"])
os.environ.setdefault("MOABB_RESULTS", os.path.join(_CACHE_ROOT, "moabb_results"))
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModel, set_seed

from data import load_loaders_per_subject
from engine import eval_model

SEED = 42
DATASET = "bciciv2a"
GL_CKPT_DIR = "ckpts/gl_bciciv2a_s42"
RESULTS_DIR = "results/['bciciv2a', 'physionet', 'zuo2025']_s42"
STACKED_JSON = f"{RESULTS_DIR}/bciciv2a_stacked.json"
SUBJECT_SPECIFIC_JSON = f"{RESULTS_DIR}/bciciv2a_subject_specific.json"
CACHE_JSON = f"{RESULTS_DIR}/bciciv2a_global_lora_per_subject.json"
OUT_PNG = "results/bciciv2a_per_subject_accuracy.png"

NUM_CHANNELS = 22
TIME_PATCHES = 5
NUM_CLASSES = 4
FEAT_DIM = NUM_CHANNELS * TIME_PATCHES * 512


def build_model():
    model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True, dtype="auto")
    model.final_layer = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.RMSNorm(FEAT_DIM),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(FEAT_DIM, NUM_CLASSES),
    )
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True, dtype="auto")
    return model, pos_bank


def evaluate_global_lora_per_subject():
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pos_bank = build_model()
    _, subject_loaders = load_loaders_per_subject(DATASET, pos_bank, batch_size=32, seed=SEED)

    lora_model = PeftModel.from_pretrained(model, GL_CKPT_DIR, is_trainable=False)
    lora_model.to(device)
    print(f"Loaded Global LoRA adapter from {GL_CKPT_DIR}")

    per_subject = {}
    for subj, (_, _, test_loader) in sorted(subject_loaders.items()):
        metrics = eval_model(lora_model, test_loader, device)
        per_subject[str(subj)] = {k: float(v) for k, v in metrics.items()}
        print(f"  s{subj}: acc={metrics['acc']:.4f}  balanced_acc={metrics['balanced_acc']:.4f}")
    return per_subject


def get_global_lora_results():
    if os.path.exists(CACHE_JSON):
        print(f"Reading cached global LoRA per-subject results from {CACHE_JSON}")
        with open(CACHE_JSON) as f:
            return json.load(f)
    per_subject = evaluate_global_lora_per_subject()
    os.makedirs(os.path.dirname(CACHE_JSON), exist_ok=True)
    with open(CACHE_JSON, "w") as f:
        json.dump(per_subject, f, indent=2)
    print(f"Saved global LoRA per-subject results to {CACHE_JSON}")
    return per_subject


def read_subjects_acc(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return {sid: s["test"]["acc"] for sid, s in data["subjects"].items()}


def plot(gl_acc, stacked_acc, ss_acc, out_path):
    subjects = sorted(set(gl_acc) | set(stacked_acc) | set(ss_acc), key=int)
    x = np.arange(len(subjects))
    width = 0.27

    gl = [gl_acc[s] for s in subjects]
    st = [stacked_acc[s] for s in subjects]
    ss = [ss_acc[s] for s in subjects]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - width, gl, width, label=f"Global LoRA (mean={np.mean(gl):.3f})", color="#4C78A8")
    b2 = ax.bar(x,         st, width, label=f"Stacked (mean={np.mean(st):.3f})",      color="#F58518")
    b3 = ax.bar(x + width, ss, width, label=f"Subject-specific (mean={np.mean(ss):.3f})", color="#54A24B")

    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, [f"S{s}" for s in subjects])
    ax.set_xlabel("Subject")
    ax.set_ylabel("Test accuracy")
    ax.set_title("BCI-IV-2a — per-subject test accuracy (seed=42)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved plot to {out_path}")


def main():
    gl = {sid: m["acc"] for sid, m in get_global_lora_results().items()}
    stacked = read_subjects_acc(STACKED_JSON)
    ss = read_subjects_acc(SUBJECT_SPECIFIC_JSON)
    plot(gl, stacked, ss, OUT_PNG)


if __name__ == "__main__":
    main()
