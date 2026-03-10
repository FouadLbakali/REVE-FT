import argparse
import random
import torch 
import numpy as np



def bl_str(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parse_args():
    parser = argparse.ArgumentParser(description='Pretraining SSL MAE EEG')
    args = parser.parse_args(args=[])
    parser.add_argument('--soup', type=bl_str, default=False, help='soup')
    parser.add_argument('--lora', type=bl_str, default=True, help='lora')
    parser.add_argument('--config_toxic', type=int, default=0, help='config toxic')
    parser.add_argument('--pt_epoch', type=int, default=0, help='pretraining epoch')
    parser.add_argument('--load_model',type=bl_str,default=True,help='load model')
    parser.add_argument('--checkpoint_path',type=str,default='',help='load model path')
    
    parser.add_argument('--token_avg', type=bl_str, default=True, help='')
    parser.add_argument('--reset_token_avg',type=bl_str, default=False, help='')
    parser.add_argument('--ft_path', type=str, default='/users/local/preprocessed_foundation_data/downstream_validation/', help='path to ft dataset') 
    parser.add_argument('--ft_dataset', type=str, default='PhysionetMI', help='ft dataset')
    parser.add_argument('--mixup_ft', type=bl_str, default=True, help='mixupft')
    parser.add_argument('--mixup_lp', type=bl_str, default=True, help='mixupft')
    parser.add_argument('--n_epochs_ft', type=int, default=200, help='number of training epochs')
    parser.add_argument('--n_epochs_lp', type=int, default=20, help='number of training epochs')
    parser.add_argument('--warmup_epochs_ft', type=int, default=5)
    parser.add_argument('--warmup_epochs_lp', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=1e-4,help='')#1e-4
    
    parser.add_argument('--last_pooling',type=bl_str,default=True,help='last pooling')
    parser.add_argument('--classic_pooling',type=bl_str,default=False,help='avg pooling')


    parser.add_argument('--lr_lp', type=float, default=5e-3)
    parser.add_argument('--lr_ft', type=float, default=1e-4)
    parser.add_argument('--patience_ft', type=int, default=3)
    parser.add_argument('--patience_lp', type=int, default=3)
    parser.add_argument('--ES_patience_ft', type=int, default=11,help='early stopping patience (default: 12)')
    parser.add_argument('--ES_patience_lp', type=int, default=10,help='early stopping patience (default: 12)')
    parser.add_argument('--dropout_ft', type=float, default=0.3,help='dropout')
    parser.add_argument('--runs_ft', type=int, default=10)
    
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--EA', type=bl_str, default=True, help='EA')
    parser.add_argument('--clip', type=int, default=100, help='clip value')

    parser.add_argument('--device', type=str, default='cuda:0', help='device to train on (default: "cuda:0")')
    parser.add_argument('--wandb_path', type=str, default='/lustre/fsn1/projects/rech/lnb/uby38mh', help='path to wandb directory')

    parser.add_argument('--patch_size', type=int,   default=200, help='patch size')
    parser.add_argument('--overlap_size', type=int, default=20, help='overlap size')
    # Spatial noise electrodes
    parser.add_argument('--noise_ratio', type=float,default=0.0025, help='noise ratio') #0.01 should be better
    # Model params
    parser.add_argument('--embed_dim', type=int,    default=512, help='embedding dimension')
    parser.add_argument('--depth', type=int,        default=4, help='depth')
    parser.add_argument('--heads', type=int,        default=4, help='heads')
    parser.add_argument('--mlp_dim_ratio', type=float,      default=2.66, help='mlp dimension')
    parser.add_argument('--dim_head', type=int,     default=64, help='dimension head')
    parser.add_argument('--geglu', type=bl_str,     default=True, help='use GEGLU')
    parser.add_argument('--use_flash', type=bl_str, default=False, help='use flash')    
    parser.add_argument('--masking_ratio', type=float, default=0.55, help='masking ratio')
    parser.add_argument('--block_masking', type=bl_str, default=True, help='block masking')
    parser.add_argument('--decoder_dim', type=int, default=512, help='decoder dimension')
    parser.add_argument('--decoder_depth', type=int, default=4, help='decoder depth')
    parser.add_argument('--decoder_heads', type=int, default=4, help='decoder heads')
    # # Training 
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience (default: 15)')
    parser.add_argument('--grad_clip', type=bl_str, default=True)#1e-4
    parser.add_argument('--grad_clip_norm', type=float, default=5)#1e-4
    parser.add_argument('--optimizer', type=str, default='stableadamw', choices=['sgd', 'adam', 'adamw', 'lars','stableadamw'], help='optimizer')
    ##### ADAMW args
    parser.add_argument('--init_megatron', type=bl_str, default=True, help='initialize with megatron')
    
    parser.add_argument('--tag', type=str, default='ft', help='tag to add to the run')
    parser.add_argument('--comment', type=str, default='', help='comment to add to the run')
    parser.add_argument('--wandbProjectName', type=str, default='foundation_EEG', help='wandb project folder')
    parser.add_argument('--entity', type=str, default='brain-imt', help='wandb entity')
    parser.add_argument('--wandb_offline', type=bl_str, default=False, help='wandb offline')
    parser.add_argument('--log_wandb', type=bl_str, default=True, help='wandb')
    args = parser.parse_args()

    return args
