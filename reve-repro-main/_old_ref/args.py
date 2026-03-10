import argparse


def bl_str(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


# fmt: off
def parse_args():
    parser = argparse.ArgumentParser(description='Pretraining SSL MAE EEG')
    args = parser.parse_args(args=[])
    parser.add_argument('--tiling', type=bl_str, default=False, help='only in train.py atm')
    parser.add_argument('--tiling_model_path', type=str, default='', help='only in train.py atm')
    parser.add_argument('--token_avg', type=bl_str, default=False, help='')
    parser.add_argument('--token_avg_lambda', type=float, default=0.1, help='')
    parser.add_argument('--remove_toxic', type=bl_str, default=False, help='remove toxic guys')
    parser.add_argument('--remove_toxic_number', type=str, default='', help='remove toxic config')
    parser.add_argument('--init_megatron', type=bl_str, default=True, help='initialize with megatron')
    parser.add_argument('--batch_size', type=int, default=300, help='batch_size')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to train on (default: "cuda:0")')
    parser.add_argument('--n_gpus', type=int, default=3, help='')
    parser.add_argument('--n_nodes', type=int, default=1, help='')
    parser.add_argument('--acc_steps', type=int, default=1, help='')
    parser.add_argument('--load_model',type=bl_str,default=False,help='load model')
    parser.add_argument('--checkpoint_path',type=str,default='',help='load model path')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--prefetch_factor', type=int, default=3, help='prefect factor for dataloader')
    parser.add_argument('--train_epochs', type=int, default=200, help='number of training epochs')
    # Data Loading
    parser.add_argument('--data_path', type=str, default='/users/local/EEG_subset/', help='path to dataset') #/lustre/fsn1/projects/rech/lnb/uby38mh/dataset/preprocessed_foundation_data
    parser.add_argument('--load_subset', type=str, default='all', choices=['nas','subset','all'], help='load a specific subset') #/lustre/fsn1/projects/rech/lnb/uby38mh/dataset/preprocessed_foundation_data
    parser.add_argument('--wandb_path', type=str, default='/lustre/fsn1/projects/rech/lnb/uby38mh', help='path to wandb directory')
    parser.add_argument('--dataloader', type=str, default='standard', choices= ['standard','multi'],help='dataloader')
    parser.add_argument('--clip', type=int, default=15, help='clip value')
    # Patching
    parser.add_argument('--patch_size', type=int,   default=200, help='patch size')
    parser.add_argument('--overlap_size', type=int, default=20, help='overlap size')
    # Spatial noise electrodes
    parser.add_argument('--noise_ratio', type=float,default=0.0025, help='noise ratio') #0.01 should be better
    # Encoder
    parser.add_argument('--embed_dim', type=int,    default=512, help='embedding dimension')
    parser.add_argument('--depth', type=int,        default=4, help='depth')
    parser.add_argument('--heads', type=int,        default=4, help='heads')
    parser.add_argument('--mlp_dim_ratio', type=float,      default=2.66, help='mlp dimension')
    parser.add_argument('--dim_head', type=int,     default=64, help='dimension head')
    parser.add_argument('--geglu', type=bl_str,     default=True, help='use GEGLU')
    parser.add_argument('--use_flash', type=bl_str, default=False, help='use flash')
    # Decoder
    parser.add_argument('--masking_ratio', type=float, default=0.55, help='masking ratio')
    parser.add_argument('--block_masking', type=bl_str, default=True, help='block masking')
    parser.add_argument('--decoder_dim', type=int, default=512, help='decoder dimension')
    parser.add_argument('--decoder_depth', type=int, default=4, help='decoder depth')
    parser.add_argument('--decoder_heads', type=int, default=4, help='decoder heads')
    # # Training
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience (default: 15)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-4), bypassed by trapezoid')
    parser.add_argument('--weight_decay', type=float, default=1e-6,help='bypassed by adamw_weight_decay')#1e-4
    parser.add_argument('--grad_clip', type=bl_str, default=True)#1e-4
    parser.add_argument('--grad_clip_norm', type=float, default=5)#1e-4
    parser.add_argument('--optimizer', type=str, default='stableadamw', choices=['sgd', 'adam', 'adamw', 'lars','stableadamw'], help='optimizer')
    ##### ADAMW args
    parser.add_argument('--adamw_beta1', type=float, default=0.92)
    parser.add_argument('--adamw_beta2', type=float, default=0.999)
    parser.add_argument('--adamw_epsilon', type=float, default=1e-9)
    parser.add_argument('--adamw_weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='trapezoid', choices=['linear', 'cosine','plateau','trapezoid'])
    # Plateau args
    parser.add_argument('--patience_plateau', type=int, default=5)
    # Trapez args
    parser.add_argument('--start_lr', type=float, default=1e-5)
    parser.add_argument('--peak_lr', type=float, default=7e-4) # 7e-4
    parser.add_argument('--end_lr', type=float, default=1e-6)
    parser.add_argument('--n_cycles', type=int, default=1) # à voir avec n_epochs 2 si small
    parser.add_argument('--warmup_steps', type=float, default=0.1)
    parser.add_argument('--cooldown_steps', type=float, default=0.1)
    parser.add_argument('--ft_path', type=str, default='/users/local/preprocessed_foundation_data/downstream_validation/', help='path to ft dataset')
    parser.add_argument('--ft_dataset', type=str, default='PhysionetMI', help='ft dataset')
    parser.add_argument('--mixup_ft', type=bl_str, default=True, help='mixupft')
    parser.add_argument('--n_epochs_ft', type=int, default=200, help='number of training epochs')
    parser.add_argument('--warmup_epochs_ft', type=int, default=5)
    parser.add_argument('--lr_ft', type=float, default=1e-4)
    parser.add_argument('--patience_ft', type=int, default=4)
    parser.add_argument('--ES_patience_ft', type=int, default=11,help='early stopping patience (default: 12)')
    parser.add_argument('--dropout_classif', type=float, default=0.3,help='dropout')
    parser.add_argument('--runs_ft', type=int, default=5)
    # # Saving model
    parser.add_argument('--save_model_path', type=str, default='checkpoints/', help='path to save the final model (default: "")')
    parser.add_argument('--save_all_epochs', type=bl_str, default=True, help='save all epochs')
    parser.add_argument('--tag', type=str, default='', help='tag to add to the run')
    parser.add_argument('--comment', type=str, default='', help='comment to add to the run')
    parser.add_argument('--wandbProjectName', type=str, default='foundation_EEG', help='wandb project folder')
    parser.add_argument('--entity', type=str, default='brain-imt', help='wandb entity')
    parser.add_argument('--wandb_offline', type=bl_str, default=False, help='wandb offline')
    parser.add_argument('--log_wandb', type=bl_str, default=True, help='wandb')
    parser.add_argument('--debug', type=bl_str, default=True, help='')
    args = parser.parse_args()
    # # deterministic mode for reproducibility
    # if args.seed >= 0:
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.use_deterministic_algorithms(True)

    return args
