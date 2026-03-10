#!/bin/bash
#SBATCH --job-name=accelerate_multi-node
#SBATCH --output=logs/accelerate%j.out
#SBATCH --error=logs/accelerate%j.err 
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --cpus-per-task=32
#SBATCH -C a100

## load Pytorch module
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.3.0

## launch script on every node
set -x

srun idr_accelerate --mixed_precision fp16 \
train.py --wandb_offline True --load_subset 'nas' \
--data_path /lustre/fsn1/projects/rech/lnb/uby38mh/dataset/preprocessed_foundation_data \
--n_gpus 8 --num_workers 16 --batch_size 128 \
--peak_lr 7e-4 --heads 8 --depth 8 \
--decoder_depth 4 \
--bs_scheduler_warmup 0 --mlp_dim_ratio 2.66 \
--tag 'mlpdim' --comment 'mlp2.6' --save_model_path '/lustre/fsn1/projects/rech/lnb/uby38mh/models' 
