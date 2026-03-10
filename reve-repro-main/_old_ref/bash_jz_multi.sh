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

## load Pytorch module
module purge
module load arch/a100
module load pytorch-gpu/py3/2.5.0

## launch script on every node
set -x

srun idr_accelerate --mixed_precision fp16 \
train_noval.py --wandb_offline True --load_subset 'all' \
--data_path /lustre/fsn1/projects/rech/lnb/uby38mh/dataset/preprocessed_foundation_data \
--n_gpus 8 --n_nodes 2 --num_workers 32 --batch_size 64 \
--depth 22 --heads 8 --decoder_depth 4 --decoder_heads 8 --embed_dim 512 --decoder_dim 512 \
--mlp_dim_ratio 2.66 --n_cycles 1 --peak_lr 4e-5 --adamw_beta1 0.9 --adamw_beta2 0.95 \
--tag 'all' --comment 'reload_beta_full_22_8_4_8_512' --save_model_path '/lustre/fsn1/projects/rech/lnb/uby38mh/models' \
--load_model True --checkpoint_path /lustre/fsn1/projects/rech/lnb/uby38mh/models/22_12_4_8runall_2nodes_model_6.pth