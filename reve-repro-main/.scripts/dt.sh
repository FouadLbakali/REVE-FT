#!/bin/bash

# Usage: ./dt.sh /path/to/data/root

if [ -z "$1" ]; then
    echo "Usage: $0 <data_root>"
    exit 1
fi

DATA_ROOT=$1
shift

TASKS=("physio" "hmc" "isruc" "faced" "mumtaz" "stress" "speech")

for task in "${TASKS[@]}"; do
    echo "Running task: $task"
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu src/dt_hydra.py \
        --config-name local_config.yaml \
        --config-dir . \
        task=$task \
        data_root=$DATA_ROOT \
        pretrained_path=hf:brain-bzh/reve-base \
        training_mode=lp \
        task.linear_probing.n_epochs=2 \
        "$@"
done