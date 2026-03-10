import torch 
from torch import nn
import random
import numpy as np
import pandas as pd
from os.path import join as pjoin
from collections import OrderedDict

from models.mae_eeg import MAE
from models.transformer_eeg import TransformerEncoder
from torch.utils.data import Dataset,DataLoader
from utils.optim import StableAdamW
from dt.finetuning_core import train,test,EEGDTDataset,instanciate_models,freeze_model,unfreeze_model,instanciate_soup

from dt.lora import CustomGetLora,get_lora_config


import torch
import os
import pickle
from torch.utils.data import Dataset

class TUEV(Dataset):
    def __init__(self, file_names, data_path,pos):
        """
        Args:
            file_names (list of str): List of pickle file names.
            data_path (str): Path to the directory containing the pickle files.
        """
        self.file_names = file_names
        self.data_path = data_path
        self.pos = pos.float()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_path, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        X = torch.tensor(data['signal']) *1e4
        Y = torch.tensor(data['label']).squeeze(-1) -1

        return X.float(), Y.long(),self.pos.float()



def loaders(idx,df_dataset,df_dt,path,dataset_name,EA,clip,batch_size,session=True):    
    if dataset_name == 'TUEV':
        #path = '/Brain/private/y17eloua/downstream_tasks/TUH/TUEV2'
        path = '/users/local/TUEV2'
        path_train =pjoin(path,'processed_train')
        path_test = pjoin(path,'processed_eval')
        pos = torch.from_numpy(np.load('/Brain/private/y17eloua/downstream_tasks/TUH/TUEV2/pos_-_eeg_-_TUEV2.npy'))
        test_dataset = TUEV(os.listdir(path_test), path_test,pos)
        test_loader = DataLoader(test_dataset, batch_size=256, pin_memory=True, shuffle=False, num_workers=10, persistent_workers=True,drop_last=False)
        
        train_files = sorted(os.listdir(path_train))
        trainfull_dataset = TUEV(train_files, path_train,pos)
        trainfull_loader = DataLoader(trainfull_dataset, batch_size=256, pin_memory=True, shuffle=True, num_workers=10, persistent_workers=True,drop_last=True)

        train_sub = list(sorted(set([f.split("_")[0] for f in train_files]))) 
        seed = 2025#261296#261296#999999999
        print(seed)
        np.random.seed(seed)
        random.seed(seed)
        val_sub = np.random.choice(train_sub, size=int(
            len(train_sub) * 0.2), replace=False)
        train_sub = list(set(train_sub) - set(val_sub))
        val_files = [f for f in train_files if f.split("_")[0] in val_sub]
        train_files = [f for f in train_files if f.split("_")[0] in train_sub]


        train_dataset = TUEV(train_files, path_train,pos)
        train_loader = DataLoader(train_dataset, batch_size=256, pin_memory=True, shuffle=True, num_workers=10, persistent_workers=True,drop_last=True)

        val_dataset = TUEV(val_files, path_train,pos)
        val_loader = DataLoader(val_dataset, batch_size=256, pin_memory=True, shuffle=False, num_workers=10, persistent_workers=True,drop_last=True)

        print(len(train_dataset),len(val_dataset),len(test_dataset))
        print(len(train_loader),len(val_loader),len(test_loader))
        return train_loader, test_loader, val_loader,trainfull_loader

    elif dataset_name == 'TUAB':
        path = '/Brain/private/y17eloua/downstream_tasks/TUH/'
        batch_size = 256        
        df_train = df_dataset[df_dataset['split']=='train']
        df_test = df_dataset[df_dataset['split']=='eval']
        
        train_sub = df_dataset[df_dataset['split']=='train'].session.unique()

        seed = 26121996#26121996#2025#261296#261296#999999999
        print(seed)
        np.random.seed(seed)
        random.seed(seed)
        val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.2), replace=False)
        train_sub = list(set(train_sub) - set(val_sub))

        df_val = df_train[df_train['session'].isin(val_sub)].copy()
        df_train = df_train[df_train['session'].isin(train_sub)]
        
        #df_val = df_test.copy()
        # session_train = df_train.session.unique().tolist()
        # random.shuffle(session_train)
        # session_val = session_train[:len(session_train)//10]
        # session_train = session_train[len(session_train)//10:]
        # df_val = df_train[df_train['session'].isin(session_val)].copy()
        # df_train = df_train[df_train['session'].isin(session_train)]
        
        train_segs = [str(t)+'_-_'+str(se) for t,se in zip(df_train['trials'].values,df_train['session'].values)]
        val_segs = [str(t)+'_-_'+str(se) for t,se in zip(df_val['trials'].values,df_val['session'].values)]
        test_segs = [str(t)+'_-_'+str(se) for t,se in zip(df_test['trials'].values,df_test['session'].values)]
        
        print(len(train_segs),len(val_segs),len(test_segs))
        train_dataset = EEGDTDataset(train_segs,df_dt,df_dataset,path,dataset_name,clip=clip,EA=EA,session=session)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,shuffle = True, num_workers=10,persistent_workers=True)

        val_dataset = EEGDTDataset(val_segs, df_dt, df_dataset, path, dataset_name, clip=clip, EA=EA,session=session)
        val_loader = DataLoader(val_dataset, batch_size=256, pin_memory=True, shuffle=False, num_workers=10, persistent_workers=True,drop_last=False)
        
        test_dataset = EEGDTDataset(test_segs, df_dt, df_dataset, path, dataset_name, clip=clip, EA=EA,session=session)
        test_loader = DataLoader(test_dataset, batch_size=256, pin_memory=True, shuffle=False, num_workers=10, persistent_workers=True,drop_last=False)
    
    return train_loader, test_loader, val_loader,None




def train_test(df_dataset,df_dt,path,dataset_name,model_state_dict,args):
    runs = args.runs_ft
    n_split = 1
    scores,scores_lp = [],[]
    metrics_lp, metrics_ft = [],[]
    args.lr_lp = 5e-3
    warmup_epochs_lp = 1
    args.lr_ft = 1e-4
    warmup_epochs_ft = 2
    args.clip = 200
    print('lp',args.lr_lp,warmup_epochs_lp,'ft',args.lr_ft,warmup_epochs_ft)
    session = False
    if dataset_name == 'TUAB':
        args.patience_lp = 2
        args.patience_ft = 2
        args.ES_patience_lp = 6
        args.ES_patience_ft = 6
        args.mixup_ft = True
        args.mixup_lp = True
        
    elif dataset_name == 'TUEV':
        args.patience_lp = 3
        args.patience_ft = 3
        args.ES_patience_lp = 8
        args.ES_patience_ft = 8
        args.mixup_ft = True
        args.mixup_lp = True
        
    

    args.dropout_ft = 0.3

    list_state_dict = []
    binary = True if df_dt.n_classes.values[0] == 2 else False
    for idx_ in range(n_split):
        train_loader, test_loader,val_loader,_ = loaders(idx_,df_dataset,df_dt,path,dataset_name,args.EA,args.clip,args.batch_size,session)
        criterion = torch.nn.CrossEntropyLoss()
        for n_run in range(runs):
            best_val,best_test = 0,0
            device = args.device
            n_classes = df_dt.n_classes.values[0]
            model = instanciate_models(args,model_state_dict,n_classes)
            model.to(device)
            freeze_model(model)
            
            if args.optimizer=='stableadamw':
                optimizer = StableAdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_lp,weight_decay=1e-4)
            elif args.optimizer=='adamw':
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_lp,weight_decay=1e-4)
            elif args.optimizer=='adam':
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_lp,weight_decay=args.weight_decay)
            scaler = torch.amp.GradScaler(device)
            
            scheduler_lh = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience_lp)
            if warmup_epochs_lp > 0:
                total_steps = len(train_loader)* warmup_epochs_lp
                exponential_warmup_lambda = lambda step: min(1.0, (10**(step / total_steps) - 1) / 9) if step < total_steps else 1.0
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exponential_warmup_lambda)
        
            patience = 0
            for epoch in range(100):
                warmup = True if epoch < warmup_epochs_lp else False
                _ = train(epoch, model, criterion, optimizer, train_loader,mixup = args.mixup_lp,device=args.device,warmup_scheduler=warmup_scheduler,warmup=warmup,scaler=scaler)
                val_metrics = test(epoch, model, val_loader,device=device,binary=binary)
                val_acc,val_balanced_acc,val_cohen_kappa,val_f1,val_auroc,val_auc_pr = val_metrics
                if best_val < val_balanced_acc:
                    best_val = val_balanced_acc
                    test_metrics = test(epoch, model, test_loader,device=device,binary=binary)
                    test_acc,test_balanced_acc,test_cohen_kappa,test_f1,test_auroc,test_auc_pr = test_metrics
                    best_test = test_balanced_acc
                    patience = 0
                    if args.soup and not args.lora:
                        best_state_dict = model.state_dict()
                else:
                    patience += 1  
                print("\rVal (Split-run lp {:3d}-{:3d}): {:3d} best test: {:.3f},best val: {:.3f},val: {:.3f},  LR {:.8f}, ft patience{:3d}/{:3d} ".format(idx_,n_run,epoch, best_test,best_val,val_balanced_acc,optimizer.param_groups[0]['lr'],patience,args.ES_patience_lp+1), end = '')
                if epoch > warmup_epochs_lp:
                    scheduler_lh.step(val_acc)
                if patience > args.ES_patience_lp:
                    break
            test_linearprobe = best_test
            metrics_lp.append(test_metrics)
            model.kv_dropout.p = args.dropout_ft
            model.fc_dropout.p = args.dropout_ft
            unfreeze_model(model)
            print('')
            if args.optimizer=='stableadamw':
                optimizer = StableAdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_ft,weight_decay=args.weight_decay)
            elif args.optimizer=='adamw':
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_ft,weight_decay=args.weight_decay)
            elif args.optimizer=='adam':
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_ft,weight_decay=args.weight_decay)
            if warmup_epochs_ft > 0:
                total_steps = len(train_loader)* warmup_epochs_ft
                exponential_warmup_lambda = lambda step: min(1.0, (10**(step / total_steps) - 1) / 9) if step < total_steps else 1.0
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exponential_warmup_lambda)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.patience_ft)
            patience = 0
            scaler = torch.amp.GradScaler(device)
            
            if args.lora:
                config = get_lora_config(
                                        model=model,
                                        rank=4,
                                        patch=False,
                                        mlp4d=False, 
                                        attention=True,
                                        ffn=False)
                lora = CustomGetLora(config=config, train_all=True)
                lora_model = lora.get_model(model)
                params = lora.get_opt_params(lora_model)
                lora_model.print_trainable_parameters()
                optimizer = StableAdamW(params, lr=args.lr_ft,weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.patience_ft)
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exponential_warmup_lambda)

            for epoch in range(100):
                warmup = True if epoch < warmup_epochs_ft else False
                if warmup:
                    patience = 0 
                _ = train(epoch, model, criterion, optimizer, train_loader,mixup = args.mixup_ft,device=args.device,warmup_scheduler=warmup_scheduler,warmup=warmup,scaler=scaler)
                val_metrics = test(epoch, model, val_loader,device=device,binary=binary)
                val_acc,val_balanced_acc,val_cohen_kappa,val_f1,val_auroc,val_auc_pr = val_metrics
                if best_val < val_balanced_acc:
                    best_val = val_balanced_acc
                    test_metrics = test(epoch, model, test_loader,device=device,binary=binary)
                    test_acc,test_balanced_acc,test_cohen_kappa,test_f1,test_auroc,test_auc_pr = test_metrics
                    best_test = test_balanced_acc
                    patience = 0
                    if args.soup:
                        best_state_dict = model.state_dict()
                    
                    
                else:
                    patience += 1    
                print("\rVal (Split-run ft {:3d}-{:3d}): {:3d} best test: {:.3f},best val: {:.3f},val: {:.3f},  LR {:.8f}, ft patience{:3d}/{:3d} ".format(idx_,n_run,epoch, best_test,best_val,val_balanced_acc,optimizer.param_groups[0]['lr'],patience,args.ES_patience_ft+1), end = '')
                
                if patience > args.ES_patience_ft:
                    # print('...')
                    # optimizer = StableAdamW(params, lr=1e-5,weight_decay=args.weight_decay)
                    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1,2], gamma=0.1)
                    # for epoch in range(epoch_final):
                    #     warmup =  False
                    #     _ = train(epoch, model, criterion, optimizer, trainfull_loader,mixup = args.mixup_ft,device=args.device,warmup_scheduler=warmup_scheduler,warmup=warmup,scaler=scaler)
                    #     test_metrics = test(epoch, model, test_loader,device=device,binary=binary)
                    #     test_metrics = test(epoch, model, test_loader,device=device,binary=binary)
                    #     test_acc,test_balanced_acc,test_cohen_kappa,test_f1,test_auroc,test_auc_pr = test_metrics
                    #     best_test = test_balanced_acc
                    #     patience = 0
                    #     print("\rVal (Split-run ft {:3d}-{:3d}): {:3d} best test: {:.3f},best val: {:.3f},val: {:.3f},  LR {:.8f}, ft patience{:3d}/{:3d} ".format(idx_,n_run,epoch, best_test,best_val,val_balanced_acc,optimizer.param_groups[0]['lr'],patience,args.ES_patience_ft+1), end = '')
                    #     scheduler.step()
                    # print(' ')
                    break
                if epoch > warmup_epochs_ft:
                    scheduler.step(val_acc)
            scores.append(best_test)
            scores_lp.append(test_linearprobe)
            metrics_ft.append(test_metrics)
            if n_run == runs - 1:
                print(" average: {:.3f}".format(np.mean(scores[-runs:])))
            print('')
            if args.soup:
                list_state_dict.append(best_state_dict)
    
    if args.soup:
        model = instanciate_soup(args,list_state_dict,n_classes,model)
        model.to(device)
        test_metrics = test(epoch, model, test_loader,device=device,binary=binary)
        soup_scores = test_metrics
        test_acc,test_balanced_acc,test_cohen_kappa,test_f1,test_auroc,test_auc_pr = test_metrics
        print('')
        print('SOUP ACC',test_balanced_acc)
    else:
        soup_scores = None
    
    std = np.array(scores).std()
    std_lp = np.array(scores_lp).std()
    print("{:.3f} +- {:.3f}, lp: {:.3f} +- {:.3f}".format(np.mean(scores),std,np.mean(scores_lp),std_lp))
    return np.mean(scores),np.mean(scores_lp),metrics_ft,metrics_lp,soup_scores









# class FTViT(nn.Module):
#     def __init__(self, encoder,n_classes):
#         super().__init__()
#         self.encoder = encoder
#         self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.encoder.embed_dim))  ## accelerate
#         self.fc_dropout = nn.Dropout(0.1)
#         self.linear_head = torch.nn.Sequential(RMSNorm(self.encoder.embed_dim), ## accelerate
#                             self.fc_dropout,
#                             #torch.nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim//2),
#                             torch.nn.Linear(self.encoder.embed_dim, n_classes)
#                             )
#         self.kv_dropout = nn.Dropout(0.1)
#         #self.param_weight = nn.Parameter(torch.ones(4))
#         #self.norm = RMSNorm(self.encoder.embed_dim)
#         #self.ln = nn.LayerNorm(self.encoder.embed_dim)
        
#     def forward(self,x,pos):
#         x = torch.cat(self.encoder(x,pos,True),dim=1)
#         #x = self.encoder(x,pos,False)
#         #x = self.encoder(x,pos,True)[1:]
#         #x = [self.ln(xx) for xx in x]
#         #x = torch.stack(x, dim=1)
#         #x = self.norm(torch.sum(x * self.param_weight.view(1, -1, 1, 1), dim=1))
#         b = x.shape[0]
#         query_output = self.cls_query_token.expand(b, -1, -1) 
#         x = self.kv_dropout(x)
#         key_value_tokens = x 
#         attention_scores = torch.matmul(query_output, key_value_tokens.transpose(-1, -2)) / (self.encoder.embed_dim ** 0.5)
#         attention_weights = torch.softmax(attention_scores, dim=-1)
#         context = torch.matmul(attention_weights, key_value_tokens).squeeze(1)
#         return self.linear_head(context)

# class FTViT(nn.Module):
#     def __init__(self, encoder):
#         super().__init__()
#         self.encoder = encoder
#         self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.encoder.embed_dim))  ## accelerate
#         self.fc_dropout = nn.Dropout(0.1)
#         self.linear_head = torch.nn.Sequential(RMSNorm(self.encoder.embed_dim), ## accelerate
#                             self.fc_dropout,
#                             #torch.nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim//2),
#                             torch.nn.Linear(self.encoder.embed_dim, 2)
#                             )
#         self.kv_dropout = nn.Dropout(0.1)
#         #self.param_weight = nn.Parameter(torch.ones(4))
#         #self.norm = RMSNorm(self.encoder.embed_dim)
#         #self.ln = nn.LayerNorm(self.encoder.embed_dim)
        
#     def forward(self,x,pos):
#         x = self.encoder(x,pos,False)
#         context = x.mean(dim=1)
#         #x = self.encoder(x,pos,True)[1:]
#         #x = [self.ln(xx) for xx in x]
#         #x = torch.stack(x, dim=1)
#         #x = self.norm(torch.sum(x * self.param_weight.view(1, -1, 1, 1), dim=1))
#         #b = x.shape[0]
#         #query_output = self.cls_query_token.expand(b, -1, -1) 
#         ##x = self.kv_dropout(x)
#         #key_value_tokens = x 
#         #attention_scores = torch.matmul(query_output, key_value_tokens.transpose(-1, -2)) / (self.encoder.embed_dim ** 0.5)
#         #attention_weights = torch.softmax(attention_scores, dim=-1)
#         #context = torch.matmul(attention_weights, key_value_tokens).squeeze(1)
#         return self.linear_head(context)





# def train_test_ft_fixsplit(df_dataset,df_dt,path,dataset_name,model_state_dict,args):
#     runs = args.runs_ft
#     n_split = 1
#     scores,scores_lp = [],[]
#     metrics_lp, metrics_ft = [],[]
#     if args.ft_dataset in ['TUAB','TUEV']:
#         args.lr_lp = 5e-3
#         warmup_epochs_lp = 3
#         epoch_lp = 15
#         steps_lp = [3,5,10] #[6,10,17]
#         args.lr_ft = 1e-4
#         warmup_epochs_ft = 5
#         args.clip = 200
#         print('lp',args.lr_lp,warmup_epochs_lp,epoch_lp,'ft',args.lr_ft,warmup_epochs_ft)
#         session = False
#         args.mixup_ft = False
#         args.mixup_lp = False
#     else:
#         warmup_epochs_ft = args.warmup_epochs_ft
#         warmup_epochs_lp = args.warmup_epochs_lp
#         epoch_lp = args.n_epochs_lp
#         session = True
#         steps_lp = [(epoch_lp-warmup_epochs_lp)//(5/3),(epoch_lp-warmup_epochs_lp)//(5/4)]
        
#     binary = True if df_dt.n_classes.values[0] == 2 else False
#     for idx_ in range(n_split):
#         train_loader, test_loader,val_loader = loaders(idx_,df_dataset,df_dt,path,dataset_name,args.EA,args.clip,args.batch_size,session)
#         criterion = torch.nn.CrossEntropyLoss()
#         for n_run in range(runs):
#             best_val,best_test = 0,0
#             device = args.device
#             n_classes = df_dt.n_classes.values[0]
#             model = instanciate_models(args,model_state_dict,n_classes)
#             model.to(device)
#             freeze_model(model)
            
#             if args.optimizer=='stableadamw':
#                 optimizer = StableAdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_lp,weight_decay=1e-4)
#             elif args.optimizer=='adamw':
#                 optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_lp,weight_decay=1e-4)
#             elif args.optimizer=='adam':
#                 optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_lp,weight_decay=args.weight_decay)
#             scaler = torch.amp.GradScaler(device)
            
#             scheduler_lh = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
#             if warmup_epochs_lp > 0:
#                 total_steps = len(train_loader)* warmup_epochs_lp
#                 exponential_warmup_lambda = lambda step: min(1.0, (10**(step / total_steps) - 1) / 9) if step < total_steps else 1.0
#                 warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exponential_warmup_lambda)
#             patience = 0
#             for epoch in range(epoch_lp):
#                 warmup = True if epoch < warmup_epochs_lp else False
#                 _ = train(epoch, model, criterion, optimizer, train_loader,mixup = args.mixup_lp,device=args.device,warmup_scheduler=warmup_scheduler,warmup=warmup,scaler=scaler)
#                 val_metrics = test(epoch, model, val_loader,device=device,binary=binary)
#                 val_acc,val_balanced_acc,val_cohen_kappa,val_f1,val_auroc,val_auc_pr = val_metrics
#                 if best_val < val_balanced_acc:
#                     best_val = val_balanced_acc
#                     test_metrics = test(epoch, model, test_loader,device=device,binary=binary)
#                     test_acc,test_balanced_acc,test_cohen_kappa,test_f1,test_auroc,test_auc_pr = test_metrics
#                     best_test = test_balanced_acc
#                     patience = 0
#                 else:
#                     patience += 1  
#                 print("\rVal (Split-run lp {:3d}-{:3d}): {:3d} best test: {:.3f},best val: {:.3f},val: {:.3f},  LR {:.8f}, ft patience{:3d}/{:3d} ".format(idx_,n_run,epoch, best_test,best_val,val_balanced_acc,optimizer.param_groups[0]['lr'],0,args.ES_patience_ft+1), end = '')
#                 if epoch > warmup_epochs_lp:
#                     scheduler_lh.step(val_acc)
#                 if patience > 6:
#                     break
#             test_linearprobe = best_test
#             metrics_lp.append(test_metrics)
            
#             model.kv_dropout.p = args.dropout_ft
#             model.fc_dropout.p = args.dropout_ft
#             unfreeze_model(model)
#             print('')
#             if args.optimizer=='stableadamw':
#                 optimizer = StableAdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_ft,weight_decay=args.weight_decay)
#             elif args.optimizer=='adamw':
#                 optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_ft,weight_decay=args.weight_decay)
#             elif args.optimizer=='adam':
#                 optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_ft,weight_decay=args.weight_decay)
#             if warmup_epochs_ft > 0:
#                 total_steps = len(train_loader)* warmup_epochs_ft
#                 exponential_warmup_lambda = lambda step: min(1.0, (10**(step / total_steps) - 1) / 9) if step < total_steps else 1.0
#                 warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exponential_warmup_lambda)
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.patience_ft)
#             patience = 0
#             scaler = torch.amp.GradScaler(device)
#             for epoch in range(args.n_epochs_ft):
#                 warmup = True if epoch < warmup_epochs_ft else False
#                 if warmup:
#                     patience = 0 
#                 _ = train(epoch, model, criterion, optimizer, train_loader,mixup = args.mixup_ft,device=args.device,warmup_scheduler=warmup_scheduler,warmup=warmup,scaler=scaler)
#                 val_metrics = test(epoch, model, val_loader,device=device,binary=binary)
#                 val_acc,val_balanced_acc,val_cohen_kappa,val_f1,val_auroc,val_auc_pr = val_metrics
#                 if best_val < val_balanced_acc:
#                     best_val = val_balanced_acc
#                     test_metrics = test(epoch, model, test_loader,device=device,binary=binary)
#                     test_acc,test_balanced_acc,test_cohen_kappa,test_f1,test_auroc,test_auc_pr = test_metrics
#                     best_test = test_balanced_acc
#                     patience = 0
#                 else:
#                     patience += 1    
#                 print("\rVal (Split-run ft {:3d}-{:3d}): {:3d} best test: {:.3f},best val: {:.3f},val: {:.3f},  LR {:.8f}, ft patience{:3d}/{:3d} ".format(idx_,n_run,epoch, best_test,best_val,val_balanced_acc,optimizer.param_groups[0]['lr'],patience,args.ES_patience_ft+1), end = '')

#                 if patience > args.ES_patience_ft:
#                     break
#                 if epoch > warmup_epochs_ft:
#                     scheduler.step(val_acc)
#             scores.append(best_test)
#             scores_lp.append(test_linearprobe)
#             metrics_ft.append(test_metrics)
#             if n_run == runs - 1:
#                 print(" average: {:.3f}".format(np.mean(scores[-runs:])))
#             print('')
#     std = np.array(scores).std()
#     std_lp = np.array(scores_lp).std()
#     print("{:.3f} +- {:.3f}, lp: {:.3f} +- {:.3f}".format(np.mean(scores),std,np.mean(scores_lp),std_lp))
#     return np.mean(scores),np.mean(scores_lp),metrics_ft,metrics_lp
