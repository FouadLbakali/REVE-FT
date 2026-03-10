#!/bin/bash

# Example usage:
# ./train.sh

OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu src/train.py --config-name config_train.yaml data.subset=small \
    data.path=/Brain/private/y17eloua/preprocessed_foundation_data
