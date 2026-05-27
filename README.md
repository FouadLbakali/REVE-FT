# REVE-FT

Subject-adaptive parameter-efficient fine-tuning of EEG foundation models (**REVE**, **LaBraM**, **LUNA**) using **Stacked LoRA** for motor-imagery decoding.

Code accompanying the paper *"Stacked LoRA for Subject-Adaptive EEG Foundation Models in Motor Imagery Decoding"* (Aymen Sarhane, Fouad Lbakali, Mouad Souissi, Jonathan Lys, and Giulia Lioi, 2026).

## Installation

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates a virtual environment, installs the dependencies (exact versions
pinned in `uv.lock`), and installs the `reve_ft` package in editable mode.
With plain pip instead: `pip install -e .`.

- Pretrained backbones are downloaded from the Hugging Face Hub on first use
  (`brain-bzh/reve-base`, `braindecode/labram-pretrained`, `PulpBio/LUNA`).
- EEG datasets are downloaded automatically via [MOABB](https://moabb.neurotechx.com/)
  into `mne_data/`.

## Usage

```bash
uv run reve-ft --model reve --mode linear --dataset bciciv2a --epochs 25 --seed 42
```

Key options (`uv run reve-ft --help` for the full list):

- `--model` — `reve` | `labram` | `luna`
- `--mode` — `linear` (linear probing) | `global` (single shared LoRA) |
  `subject-specific` (per-subject LoRA) | `stacked`
- `--dataset` — `bciciv2a` | `physionet` | `zuo2025`

## Project structure

```
reve_ft/              # library + CLI (installable package)
  main.py             # CLI entry point (the `reve-ft` command)
  data.py             # dataset loading & preprocessing (MOABB)
  engine.py           # train / eval primitives
  trainer.py          # training loop, scheduler, early stopping
  stages.py           # fine-tuning stages (linear / global / per-subject / stacked)
  multilora.py        # per-subject multi-LoRA routing
  labram_zoo.py       # LaBraM backbone wrapper
  luna_zoo.py         # LUNA backbone wrapper
  luna_module/        # LUNA model implementation
  configs/            # model preprocessing/LoRA configs + dataset shapes (package data)
scripts/              # experiment sweeps & plotting helpers
tests/                # offline correctness tests (no downloads required)
```

## Scripts

- `scripts/run_all.py` — the script used to obtain all the results from the paper.
- `scripts/plot_subjects_bciciv2a.py` — per-subject result figures for BCIC IV-2a.
- `scripts/umap_class_x_strategy.py`
