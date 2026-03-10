import torch 
from torch import nn
import random
import numpy as np
import pandas as pd
from os.path import join as pjoin
from collections import OrderedDict
from sklearn.metrics import balanced_accuracy_score,cohen_kappa_score,f1_score,average_precision_score,roc_auc_score

from models.mae_eeg import MAE
from models.transformer_eeg import TransformerEncoder
from torch.utils.data import Dataset,DataLoader
from initialize import ConfigInit,init_ft
from utils.optim import StableAdamW
from models.flash_vit_utils import RMSNorm

#from dt.lora import CustomGetLora,get_lora_config

class FTViT(nn.Module):
    def __init__(self, encoder,n_classes,cls_token,last_pooling=False,classic_pooling=False):
        super().__init__()
        self.encoder = encoder
        if cls_token is None:
            self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.encoder.embed_dim))
        else:
            self.cls_query_token = cls_token
        self.fc_dropout = nn.Dropout(0.05)
        self.linear_head = torch.nn.Sequential(RMSNorm(self.encoder.embed_dim),
                            self.fc_dropout,
                            torch.nn.Linear(self.encoder.embed_dim, n_classes)
                            )
        self.kv_dropout = nn.Identity()#nn.Dropout(0.05)
        self.last_pooling = last_pooling
        self.classic_pooling = classic_pooling
        
    def forward(self,x,pos):
        if self.classic_pooling:
            x = self.encoder(x,pos,False)
            x = x.mean(dim=1)
            return self.linear_head(x)
        elif self.last_pooling:
            x = self.encoder(x,pos,False)
        else:        
            x = torch.cat(self.encoder(x,pos,True),dim=1)
        b = x.shape[0]
        query_output = self.cls_query_token.expand(b, -1, -1) 
        x = self.kv_dropout(x)
        key_value_tokens = x 
        attention_scores = torch.matmul(query_output, key_value_tokens.transpose(-1, -2)) / (self.encoder.embed_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, key_value_tokens).squeeze(1)
        return self.linear_head(context)
    
    

class EEGDTDataset(Dataset):
    def __init__(self,segs,df_dt,df_dataset,path,dataset_name,clip=15,EA=True,session=False):
        self.path = path
        self.df_dataset = df_dataset
        self.df_dt = df_dt
        self.segs = segs
        self.clip = clip
        X_n = df_dt['N_trials'].values[0]
        X_t = df_dt['duration'].values[0]
        X_c = df_dt['n_chans'].values[0]
        eeg_files = np.memmap(pjoin(path,dataset_name,'X_-_eeg_-_'+dataset_name+'.npy'), mode='r', shape=(X_n, X_c, X_t), dtype='float32')
        eeg_files[:]
        self.eeg_files = eeg_files         
        y_files = np.memmap(pjoin(path,dataset_name,'Y_-_eeg_-_'+dataset_name+'.npy'), mode='r', shape=(X_n), dtype='int64')
        y_files[:]
        self.y_files = y_files
        self.stats_zscore = np.load(pjoin(path,dataset_name,'stats_-_zscore_-_'+dataset_name+'.npy'))
        self.stats_EA = np.load(pjoin(path,dataset_name,'stats_-_EA_-_'+dataset_name+'.npy'))
        self.pos = np.load(pjoin(path,dataset_name,'pos_-_eeg_-_'+dataset_name+'.npy'))
        self.EA = EA
        self.session = session
        self.tuh = True if df_dt['source'].values[0] == 'TUH' else False
        #self.tuh = False
        self.faced = True if df_dt['dataset'].values[0] == 'FACED' else False
        #self.faced = False
        self.phys = True if 'Physionet' in df_dt['dataset'].values[0]  else False
        #self.phys = True
        self.bnci = False
    def __getitem__(self, index):
        if self.session:
            n_trial, n_sub, n_sess = self.segs[index].split('_-_')
            n_trial, n_sub, n_sess = int(n_trial), int(n_sub), int(n_sess)
        else:
            n_trial, n_sess = self.segs[index].split('_-_')
            n_trial, n_sess = int(n_trial), int(n_sess)
            
        eeg = self.eeg_files[n_trial].copy()
        target = self.y_files[n_trial].copy()
        positions = self.pos
        eeg = torch.from_numpy(eeg)
        n_chans = eeg.shape[0]
        if self.bnci:
            eeg  = eeg*1e4
            return eeg.float(),target, torch.from_numpy(positions).float()
            
        if self.faced:
            eeg = eeg*0.1
            return eeg.float(),target, torch.from_numpy(positions).float()
        if self.tuh:
            eeg  = eeg*1e4
            return eeg.float(),target, torch.from_numpy(positions).float()
        if self.phys:
            eeg  = eeg*1e4
            return eeg.float(),target, torch.from_numpy(positions).float()
        else:
            if not self.EA:
                if self.session:
                    stats = torch.from_numpy(self.stats_zscore[n_sub][n_sess]).unsqueeze(-1)
                else:
                    stats = torch.from_numpy(self.stats_zscore[n_sess]).unsqueeze(-1)
                eeg = (eeg - stats[0]) / (stats[1])
            else:
                if self.session:
                    stats = torch.from_numpy(self.stats_EA[n_sub][n_sess])
                else:
                    stats = torch.from_numpy(self.stats_EA[n_sess])
                sqrt_R = stats[:n_chans]
                mean = stats[n_chans:][0].unsqueeze(-1)
                std = stats[n_chans:][1].unsqueeze(-1)
                eeg = torch.einsum("fe,bet->bft",sqrt_R, eeg.unsqueeze(0)).squeeze(0)
                eeg = (eeg - mean) / std
            #return eeg.float().clip(-self.clip,self.clip),target, torch.from_numpy(positions).float()
            return eeg.float(),target, torch.from_numpy(positions).float()
    
    def __len__(self,):
        return len(self.segs)  
    
    
# def train(epoch, model, criterion, optimizer, train_loader, mixup = False,device='cuda',warmup_scheduler = None,warmup = False,scaler=None):
#     losses, scores = [], []
#     model.train()      
#     device_type = 'cuda' if 'cuda' in device else 'cpu'
#     len_dl = len(train_loader)
#     for batch_idx, batch_data in enumerate(train_loader):
        
#         with torch.autocast(device_type=device_type,enabled=True,dtype= torch.float16):
#             data,target,pos = batch_data
#             data, target,pos = data.to(device,non_blocking=True), target.to(device,non_blocking=True),pos.to(device,non_blocking=True)
#             optimizer.zero_grad()
#             if mixup:
#                 mm = random.random()
#                 perm = torch.randperm(data.shape[0])
#                 output = model(mm * data + (1 - mm) * data[perm],pos) 
#             else:
#                     output = model(data,pos)
#             if mixup:
#                 loss = mm * criterion(output, target) + (1 - mm) * criterion(output, target[perm])
#             else:
#                 loss = criterion(output, target)
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
#             scaler.step(optimizer)
#             scale = scaler.get_scale()
#             scaler.update()
#             losses.append(loss.item())        
#             skip_lr_sched = (scale != scaler.get_scale())
#             if not skip_lr_sched and warmup:
#                 warmup_scheduler.step()
#     return None#np.mean(scores)
# import torch.nn.functional as F

# def focal_loss(inputs, targets, alpha=0.8, gamma=0.7):
#     p = torch.sigmoid(inputs)
#     ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#     p_t = p * targets + (1 - p) * (1 - targets)
#     loss = ce_loss * ((1 - p_t) ** gamma)

#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss
#         loss = loss.mean()
#     return loss.mean()

# def train(epoch, model, criterion, optimizer, train_loader, mixup = False,device='cuda',warmup_scheduler = None,warmup = False,scaler=None):
#     losses, scores = [], []
#     model.train()      
#     device_type = 'cuda' if 'cuda' in device else 'cpu'
#     len_dl = len(train_loader)
#     for batch_idx, batch_data in enumerate(train_loader):
#         with torch.autocast(device_type=device_type,enabled=True,dtype= torch.float16):
#             data,target,pos = batch_data
#             data, target,pos = data.to(device,non_blocking=True), target.long().to(device,non_blocking=True),pos.to(device,non_blocking=True)
#             optimizer.zero_grad()
#             if mixup:
#                 mm = random.random()
#                 perm = torch.randperm(data.shape[0])
#                 output = model(mm * data + (1 - mm) * data[perm],pos) 
#             else:
#                     output = model(data,pos)
#             if mixup:
#                 loss = mm * criterion(output, target) + (1 - mm) * criterion(output, target[perm])
#             else:
#                 #loss = criterion(output, target)
#                 loss = focal_loss(output, torch.nn.functional.one_hot(target, num_classes=2).float())
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
#             scaler.step(optimizer)
#             scale = scaler.get_scale()
#             scaler.update()
#             losses.append(loss.item())        
#             skip_lr_sched = (scale != scaler.get_scale())
#             if not skip_lr_sched and warmup:
#                 warmup_scheduler.step()
#     return None#np.mean(scores)

    
def train(epoch, model, criterion, optimizer, train_loader, mixup = False,device='cuda',warmup_scheduler = None,warmup = False,scaler=None):
    losses, scores = [], []
    model.train()      
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    len_dl = len(train_loader)
    for batch_idx, batch_data in enumerate(train_loader):
        with torch.autocast(device_type=device_type,enabled=True,dtype= torch.float16):
            data,target,pos = batch_data
            data, target,pos = data.to(device,non_blocking=True), target.long().to(device,non_blocking=True),pos.to(device,non_blocking=True)
            optimizer.zero_grad()
            if mixup:
                mm = random.random()
                perm = torch.randperm(data.shape[0])
                output = model(mm * data + (1 - mm) * data[perm],pos) 
            else:
                    output = model(data,pos)
            if mixup:
                loss = mm * criterion(output, target) + (1 - mm) * criterion(output, target[perm])
            else:
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            losses.append(loss.item())        
            skip_lr_sched = (scale != scaler.get_scale())
            if not skip_lr_sched and warmup:
                warmup_scheduler.step()
    return None#np.mean(scores)

def test(epoch, model, test_loader,device='cuda',binary=False):
    score, count = 0, 0 
    model.eval()
    y_decisions = []
    y_targets = []
    y_probs = []
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            with torch.autocast(device_type=device_type,enabled=True,dtype= torch.float16):
                (data, target,pos) = batch_data
                data, target,pos = data.to(device,non_blocking=True), target.to(device,non_blocking=True),pos.to(device,non_blocking=True)
                output = model(data,pos)
                decisions = torch.argmax(output, dim = 1)
                score += (decisions == target).int().sum().item()
                count += target.shape[0]       
                y_decisions.append(decisions)
                y_targets.append(target)
                y_probs.append(output)
           
    gt = torch.cat(y_targets).cpu().numpy()
    pr = torch.cat(y_decisions).cpu().numpy()
    pr_probs = torch.cat(y_probs).cpu().numpy()
    acc = (score / count)
    balanced_acc = balanced_accuracy_score(gt,pr)
    cohen_kappa = cohen_kappa_score(gt,pr)
    f1 = f1_score(gt,pr,average='weighted')
    if binary:
        auroc = roc_auc_score(gt,pr_probs[:,1])
        auc_pr = average_precision_score(gt,pr_probs[:,1])
        return acc,balanced_acc,cohen_kappa,f1,auroc,auc_pr
    else:
        return acc,balanced_acc,cohen_kappa,f1,0,0



# def test(epoch, model, test_loader,bn,device='cuda'):
#     score, count = 0, 0 
#     model.eval()
#     device_type = 'cuda' if 'cuda' in device else 'cpu'
#     with torch.no_grad():
#         for batch_idx, batch_data in enumerate(test_loader):
#             with torch.autocast(device_type=device_type,enabled=True,dtype= torch.float16):
#                 (data, target,pos) = batch_data
#                 data, target,pos = data.to(device,non_blocking=True), target.to(device,non_blocking=True),pos.to(device,non_blocking=True)
#                 output = model(data,pos)
#                 decisions = torch.argmax(output, dim = 1)
#                 score += (decisions == target).int().sum().item()
#                 count += target.shape[0]        
#     return (score / count)



def instanciate_models(args,model_state_dict,n_classes):
    encoder_model = TransformerEncoder(patch_size=args.patch_size,
                        overlap_size=args.overlap_size,
                        noise_ratio=args.noise_ratio,
                        embed_dim=args.embed_dim,
                        depth=args.depth,
                        heads=args.heads,
                        mlp_dim_ratio=args.mlp_dim_ratio,
                        dim_head=args.dim_head,
                        use_flash=args.use_flash,
                        geglu=args.geglu,
                        )
    mae = MAE(encoder = encoder_model,
        masking_ratio = args.masking_ratio,   
        decoder_dim = args.decoder_dim,    
        decoder_depth = args.decoder_depth,
        decoder_heads= args.decoder_heads,
        use_flash =  args.use_flash,
        geglu=args.geglu,
        token_avg = args.token_avg,
        token_avg_lambda = 0.1
        )

    if args.load_model:
        print(1)
        checkpoint = torch.load(model_state_dict,weights_only=False,map_location='cpu')
        checkpoint['model'] = OrderedDict((key.replace('module.', ''), value) for key, value in checkpoint['model'].items())
        mae.load_state_dict(checkpoint['model'])
    
    ## list_mod = []
    ## for i in range(21):
    ##     list_mod.append('/nasbrain/yass_models/tiny_toxic1_48_18_512_model_'+str(i)+'.pth')
    ## for i in range(23):
    ##     list_mod.append('/nasbrain/yass_models/tiny_toxic2_48_18_512_model_'+str(i)+'.pth')
    ## for i in range(36):
    ##     list_mod.append('/nasbrain/yass_models/tiny_toxic3_48_18_512_model_'+str(i)+'.pth')
    # model_state_dict_list = ['/nasbrain/yass_models/tiny_toxic2_48_18_512_model_15.pth',
    #                          '/nasbrain/yass_models/tiny_toxic1_48_18_512_model_0.pth',
    #                          '/nasbrain/yass_models/tiny_toxic3_48_18_512_model_0.pth',]
    # model_state_dict_list = [torch.load(path,weights_only=False) for path in model_state_dict_list]
    
    # uniform_soup = None
    # NUM_MODELS = len(model_state_dict_list)
    
    # for state_dict in model_state_dict_list:
    #     state_dict['model'] = OrderedDict((key.replace('module.', ''), value) for key, value in state_dict['model'].items())
    #     if uniform_soup is None:
    #         uniform_soup = {k: v.clone() * (1. / NUM_MODELS) for k, v in state_dict['model'].items()}
    #     else:
    #         for k, v in state_dict['model'].items():
    #             uniform_soup[k] += v * (1. / NUM_MODELS)
    
    # mae.load_state_dict(uniform_soup)
    
    
    
    encoder_ = mae.encoder
    if args.token_avg and not args.reset_token_avg:
        model = FTViT(encoder_,n_classes,mae.cls_query_token,args.last_pooling,args.classic_pooling)
    else:
        model = FTViT(encoder_,n_classes,None,args.last_pooling,args.classic_pooling)
    
    if args.init_megatron:
        config_megatron = ConfigInit(hidden_size = args.embed_dim,
                         num_hidden_layers = args.depth)
        init_ft(model,config_megatron,args.reset_token_avg)
    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.linear_head.parameters():
        param.requires_grad = True
    model.cls_query_token.requires_grad = True
    
    #model.encoder.to_patch_embedding.requires_grad = True
    #model.encoder.mlp4d.requires_grad = True
    
def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
        
        
def instanciate_soup(args,model_state_dict_list,n_classes,model_):
    model = model_
    uniform_soup = None
    NUM_MODELS = len(model_state_dict_list)
    
    for state_dict in model_state_dict_list:
        if uniform_soup is None:
            uniform_soup = {k: v.clone() * (1. / NUM_MODELS) for k, v in state_dict.items()}
        else:
            for k, v in state_dict.items():
                uniform_soup[k] += v * (1. / NUM_MODELS)
    
    model.load_state_dict(uniform_soup)
    return model


