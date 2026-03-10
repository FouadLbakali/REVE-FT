from dt.finetuning_token import train_test
from os.path import join as pjoin
import pandas as pd
from args_ft import parse_args
import wandb

args = parse_args()
args.ft_dataset = 'PhysionetMI'
args.checkpoint_path = '/Brain/private/y17eloua/yass_models/tiny_toxic2_48_18_512_model_15.pth'

args.runs_ft = 5
args.embed_dim = 512
args.decoder_dim = args.embed_dim
args.depth = 4
args.heads = 8
args.decoder_depth = 1
args.decoder_heads = 8
args.lora = False
args.soup = False

chkpt = args.checkpoint_path

if args.ft_dataset in ['TUEV', 'TUAB']:
    path = '/Brain/public/EEGNAS_DT/TUH'
elif args.ft_dataset in ['FACED']:
    path = '/Brain/public/EEGNAS_DT/Emotion2'
else:
    path = '/Brain/public/EEGNAS_DT/MOABB'
    
dataset_name = args.ft_dataset
dset_name_path = pjoin(path,dataset_name)
df_dataset = pd.read_csv(pjoin(dset_name_path,dataset_name+'_m.csv'))
df_dt = pd.read_csv('/Brain/private/y17eloua/downstream_tasks/DT_datasets.csv')
df_dt = df_dt[df_dt['dataset'] == dataset_name]

print(args)
print(df_dt.head())
print(chkpt)
scores = train_test(df_dataset,df_dt,path,dataset_name,chkpt,args)

if args.log_wandb:
    wandb.init(project='Foundation_DT', entity=args.entity, config=vars(args))     
    wandb.log({'ft':scores[0],'lp':scores[1],'metrics_ft':scores[2],'metrics_lp':scores[3],'soup':scores[4]})
    wandb.finish() 
