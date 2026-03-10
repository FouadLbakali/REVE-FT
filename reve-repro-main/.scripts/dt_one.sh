#!/bin/bash

OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu src/dt_hydra.py --config-name local_config.yaml --config-dir . pretrained_path=hf:brain-bzh/reve-base