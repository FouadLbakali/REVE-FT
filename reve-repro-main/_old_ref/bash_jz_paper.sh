#!/bin/bash
#SBATCH --job-name=accelerate_multi-node
#SBATCH --output=logs/accelerate%j.out
#SBATCH --error=logs/accelerate%j.err
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --cpus-per-task=40
#SBATCH -C a100
#SBATCH -A kzl@a100

## load Pytorch module
module purge
module load arch/a100
module load pytorch-gpu/py3/2.5.0

## launch script on every node
set -x

srun idr_accelerate --mixed_precision fp16 \
train_noval.py --wandb_offline True --load_subset 'all' --log_wandb True \
--data_path /lustre/fsn1/projects/rech/lnb/uby38mh/dataset/preprocessed_foundation_data \
--n_gpus 8 --n_nodes 2 --num_workers 10 --batch_size 200 \
--depth 22 --heads 8 --decoder_depth 1 --decoder_heads 8 --embed_dim 512 --decoder_dim 512 \
--n_cycles 1 --peak_lr 2e-4 --start_lr 1e-7 \
--save_model_path '/lustre/fsn1/projects/rech/lnb/uby38mh/models' \
--load_model True --checkpoint_path /lustre/fsn1/projects/rech/lnb/uby38mh/warmup_models/warmup_base_22_8_1_8_512_l_model_.pth --use_flash False \
--token_avg True --tag 'all' --comment 'detective_conan'