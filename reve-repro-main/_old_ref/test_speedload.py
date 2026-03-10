import warnings

import pandas as pd


warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
import logging

# from scripts_ft import load_data,train_test_ft
import math
import os
import time

import torch
from accelerate import Accelerator, logging

from args import parse_args
from models.mae_eeg import MAE
from models.vit import SimpleViT
from prepro_scripts.scripts_dataloader import (
    return_loaders_eegnas,  # compute_groups_segs,EEGDataset,GroupedSampler,MultiEpochsDataLoader
)
from utils.optim import get_lr_scheduler, get_optimizer


logger = logging.get_logger(__name__)
args = parse_args()
# logging.basicConfig(filename=pjoin(args.logs_path,str(timestamp)+'_log.txt'), level=logging.INFO,
#                     format='%(message)s',filemode='w')
# logger = logging.getLogger()


def main():
    init_time = time.time()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.acc_steps, log_with="wandb"
    )  # ,mixed_precision='bf16',log_with='wandb')
    accelerator.even_batches = False
    timestamp = int(time.time())

    if accelerator.is_main_process:
        if "lustre" in args.data_path:
            os.environ["WANDB_DIR"] = args.wandb_path  #'/'.join(args.data_path.split('/')[:3])
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_DIR"] = "/".join(args.data_path.split("/")[:3])
        accelerator.init_trackers(project_name="foundation_EEG", init_kwargs={"entity": "brain-imt"}, config=vars(args))

    if args.save_model:
        save_path = os.path.join(args.save_model_path, str(timestamp))
        os.makedirs(save_path, exist_ok=True)
        print("timestamp folder", timestamp)

    train_loader, val_loader, len_train, len_val, len_train_sampler, len_val_sampler = return_loaders_eegnas(args)
    accelerator.print("total segments:", len_train + len_val, "train segments:", len_train, "test segments:", len_val)
    accelerator.print("N GPUS:", args.n_gpus, "Train iterations:", len_train_sampler, "acc steps:", args.acc_steps)

    len_train_sampler = len_train_sampler // args.acc_steps
    n_iter_per_train = round(len_train_sampler // args.n_gpus) * args.acc_steps
    n_iter_per_val = round(len_val_sampler // args.n_gpus) - 1
    # n_iter_per_val = len_val // args.batch_size

    model = SimpleViT(
        patch_size=args.patch_size,
        overlap_size=args.overlap_size,
        noise_ratio=args.noise_ratio,
        embed_method=args.embed_method,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dim_head=args.dim_head,
        cls_token=args.cls_token,
        use_flash=args.use_flash,
        geglu=args.geglu,
        num_classes=args.num_classes,
    )

    mae = MAE(
        encoder=model,
        masking_ratio=args.masking_ratio,  # the paper recommended 75% masked patches
        decoder_dim=args.decoder_dim,  # paper showed good results with just 512
        decoder_depth=args.depth,  # anywhere from 1 to 8
        decoder_heads=args.heads,
        tcr_lambda=args.TCR_loss,
        tcr_epsilon=args.TCR_epsilon,
        use_flash=args.use_flash,
        geglu=args.geglu,
    )

    # if  args.init_patch or args.init_mlp or args.init_out or args.init_transformer:
    #    initialize_model_weights(mae,args.init_patch,args.init_mlp,args.init_out,args.init_transformer)
    optimizer = get_optimizer(mae.parameters(), args)
    scheduler = get_lr_scheduler(optimizer, args, len_train_sampler)
    best_score, best_epoch = 10, 0
    patience_counter = 0

    if args.scheduler != "trapezoid":

        def cosine_warmup_lambda(step):  # TODO prendre en compte acc_steps
            if step < total_steps:
                return 0.5 * (1 - math.cos(math.pi * step / total_steps))  # Increases from 0 to 1
            return 1.0

        if args.warmup_epochs > 0:
            total_steps = (len_train * args.warmup_epochs) // args.batch_size
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warmup_lambda)
        train_loader, val_loader, mae, optimizer, scheduler, warmup_scheduler = accelerator.prepare(
            train_loader, val_loader, mae, optimizer, scheduler, warmup_scheduler
        )
    else:
        train_loader, val_loader, mae, optimizer, scheduler = accelerator.prepare(
            train_loader, val_loader, mae, optimizer, scheduler
        )

    start = time.time()
    for epoch in range(args.train_epochs):
        start = time.time()
        elem, batch_idx = 0, 0
        loss_ema, recon_ema, recon_pos_ema, tcr_loss_ema = 0, 0, 0, 0
        # model.train()
        mae.train()
        for i, (x, pos, b_m, b_u) in enumerate(train_loader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    elem += x.shape[0]
                    optimizer.zero_grad()
                    # loss,recon_loss, tcr_loss,recon_pos_loss = mae(x,pos,b_m,b_u)
                    # accelerator.backward(loss)
                    # if args.grad_clip and accelerator.sync_gradients:
                    #    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    # optimizer.step()
            # tcr_loss_g = accelerator.gather(tcr_loss).mean().item()
            # loss_g = accelerator.gather(loss).mean().item() # pb ici
            # recon_g = accelerator.gather(recon_loss).mean().item()
            # recon_pos_g = accelerator.gather(recon_pos_loss).mean().item()
            # tcr_loss_g = accelerator.gather(tcr_loss).mean().item()
            # loss_ema = loss_g if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_g
            # recon_ema = recon_g if recon_ema is None else 0.95 * recon_ema + 0.05 * recon_g
            # recon_pos_ema = recon_pos_g if recon_pos_ema is None else 0.95 * recon_pos_ema + 0.05 * recon_pos_g
            # tcr_loss_ema = tcr_loss_g if tcr_loss_ema is None else 0.95 * tcr_loss_ema + 0.05 * tcr_loss_g
            batch_idx += 1
            if args.scheduler == "trapezoid":
                scheduler.step()
            elif epoch < args.warmup_epochs:
                warmup_scheduler.step()
            recon_g = 0
            loss_g = 0
            accelerator.print(
                "\r Epoch {:3d} (it. {:3d}/{:3d}) Loss EMA:{:3.3f} Recon (EMA/loss):{:3.3f}/{:3.3f} pos:{:3.3f} tcr:{:3.3f} LR {:.8f}, patience: {:3d}/{:3d}, shape {:3d}, time {:3.1f} ".format(
                    epoch,
                    batch_idx,
                    n_iter_per_train,
                    loss_ema,
                    recon_ema,
                    recon_g,
                    recon_pos_ema,
                    tcr_loss_ema,
                    optimizer.param_groups[0]["lr"],
                    patience_counter,
                    args.patience + 1,
                    x.shape[0],
                    time.time() - start,
                ),
                end="",
            )
            accelerator.log(
                {"epoch": epoch, "it": batch_idx, "loss_ema": loss_ema, "loss": loss_g, "pos_loss": recon_pos_ema}
            )
        accelerator.print("")
        total_val_loss, batch_idx = 0, 0
        mae.eval()
        for i, (x, pos) in enumerate(val_loader):
            with torch.no_grad():
                with accelerator.autocast():
                    loss, _, _, _ = mae(x, pos, eval=True)
                    total_val_loss += accelerator.gather_for_metrics(loss).mean().item()
                accelerator.print(
                    "\r Epoch {:3d} (it. {:3d}/{:3d}) Val Loss total:{:3.3f}, patience: {:3d}/{:3d}                                                 Val Loss total:{:3.3f}".format(
                        epoch,
                        batch_idx,
                        n_iter_per_val,
                        total_val_loss / (batch_idx + 1),
                        patience_counter,
                        args.patience + 1,
                        total_val_loss / (batch_idx + 1),
                    ),
                    end="",
                )
                batch_idx += 1

        accelerator.print("")
        val_score = total_val_loss / (batch_idx)
        accelerator.log(
            {
                "epoch": epoch,
                "val_loss": val_score,
                "best_val": best_score if epoch > 0 else val_score,
                "best_epoch": best_epoch,
            }
        )

        ####
        #  if best_score>val_score:
        #     if args.save_model:
        #         torch.save({'epoch':epoch,'model':mae.state_dict(),'optim':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'ft_score':best_score,}, os.path.join(save_path,f"mae.pth"))
        if best_score > val_score:
            best_score = val_score
            best_epoch = epoch
            patience_counter = 0

            # accelerator.save_state("/homes/y17eloua/Projets/found/checkpoints/best_model")
            # accelerator.save_model(mae, "checkpoints/best_model.pth")
            accelerator.save(
                {
                    "epoch": epoch,
                    "model": mae.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ft_score": best_score,
                },
                "checkpoints/best_model_10.pth",
            )  # os.path.join(save_path,f"mae.pth"))
        else:
            patience_counter += 1
            if patience_counter > args.patience:
                print("Early stopping")
                break

        if args.scheduler != "trapezoid":
            if epoch > args.warmup_epochs:
                scheduler.step(val_score)
            else:
                patience_counter = 0

    accelerator.print(
        "final best score:", best_score, "best_epoch:", best_epoch, "total time:", time.time() - init_time
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
