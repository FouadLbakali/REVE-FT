from dt.finetuning_token import train_test_ft_fixsplit
from os.path import join as pjoin
import pandas as pd
from args_ft import parse_args
import wandb

#'tiny_toxic1_48_18_512_model_20.pth'
#'tiny_toxic2_48_18_512_model_22.pth'
#'tiny_toxic3_48_18_512_model_35.pth'

config_toxic = 1
for i in range(21):
    args = parse_args()
    args.ft_dataset = 'BNCI2014_001'
    args.checkpoint_path = '/nasbrain/yass_models/tiny_toxic'+str(config_toxic)+'_48_18_512_model_'+str(i)+'.pth' #tiny_toxic1_48_18_512_model_12.pth' #'/nasbrain/yass_models/ablabase/decod4_model_.pth'
    args.pt_epoch = i
    args.config_toxic = config_toxic
    
    args.runs_ft = 6
    args.embed_dim = 512
    args.decoder_dim = args.embed_dim
    args.depth = 4
    args.heads = 8
    args.decoder_depth = 1
    args.decoder_heads = 8
    chkpt = args.checkpoint_path
    args.comment = 'search_best_tiny'


    path = '/users3/local/downstream_tasks/MOABB'
    dataset_name = args.ft_dataset
    dset_name_path = pjoin(path,dataset_name)
    df_dataset = pd.read_csv(pjoin(dset_name_path,dataset_name+'_m.csv'))
    df_dt = pd.read_csv('/users3/local/downstream_tasks/DT_datasets.csv')
    df_dt = df_dt[df_dt['dataset'] == dataset_name]


    print(args)
    print(df_dt.head())
    print(chkpt)
    scores = train_test_ft_fixsplit(df_dataset,df_dt,path,dataset_name,chkpt,args)

    if args.log_wandb:
        wandb.init(project='Foundation_DT', entity=args.entity, config=vars(args))     
        wandb.log({'ft':scores[0],'lp':scores[1],'metrics_ft':scores[2],'metrics_lp':scores[3]})
        wandb.finish() 


#logs les metrics + le binaire + mettre session en fonction du nom de la DT ! remettre TUHd edans en vrai je pense