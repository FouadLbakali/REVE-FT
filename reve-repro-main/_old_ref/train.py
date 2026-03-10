import warnings

import pandas as pd


warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
import os
import time
from datetime import timedelta
from os.path import join as pjoin

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from args import parse_args
from initialize import ConfigInit, init_modules
from models.mae_eeg import MAE
from models.transformer_eeg import TransformerEncoder
from prepro_scripts.scripts_dataloader import return_train_val_loaders
from utils.optim import get_lr_scheduler, get_optimizer

from models.tiling import tile_init

args = parse_args()
if args.debug:
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


def main():
    init_time = time.time()
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=30 * 60))
    accelerator = (
        Accelerator(gradient_accumulation_steps=args.acc_steps, kwargs_handlers=[timeout], log_with="wandb")
        if args.log_wandb
        else Accelerator(gradient_accumulation_steps=args.acc_steps, kwargs_handlers=[timeout])
    )
    accelerator.even_batches = False
    accelerator.step_scheduler_with_optimizer=False
    if accelerator.is_main_process and args.log_wandb:
        if "lustre" in args.data_path:
            os.environ["WANDB_DIR"] = args.wandb_path
            os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_DIR"] = "/".join(args.data_path.split("/")[:3])
        accelerator.init_trackers(
            project_name=args.wandbProjectName, init_kwargs={"entity": args.entity}, config=vars(args)
        )
    accelerator.print(args)

    train_loader, val_loader, len_train, len_val, len_train_sampler, len_val_sampler = return_train_val_loaders(
        args, return_val=True
    )
    accelerator.print("total segments:", len_train + len_val, "train segments:", len_train, "test segments:", len_val)
    accelerator.print("N GPUS:", args.n_gpus, "Train iterations:", len_train_sampler, "acc steps:", args.acc_steps)

    n_iter_per_train = len_train_sampler // (args.n_gpus * args.n_nodes)
    n_iter_per_val = (len_val_sampler // (args.n_gpus * args.n_nodes)) - 1

    if args.decoder_dim != args.embed_dim:
        print("dim decoder != dim encoder")

    model = TransformerEncoder(
        patch_size=args.patch_size,
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

    mae = MAE(
        encoder=model,
        masking_ratio=args.masking_ratio,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        use_flash=args.use_flash,
        geglu=args.geglu,
        token_avg = args.token_avg,
        token_avg_lambda = args.token_avg_lambda
    )

    if args.load_model:
        checkpoint = torch.load(args.checkpoint_path, weights_only=False,map_location="cpu")
        from collections import OrderedDict

        checkpoint["model"] = OrderedDict(
            (key.replace("module.", ""), value) for key, value in checkpoint["model"].items()
        )
        mae.load_state_dict(checkpoint["model"])
        accelerator.print("loaded")
        if args.tiling:
            accelerator.print('!!!!WARNING!!! LOAD MODEL AND TILING')
    elif args.init_megatron:
        config_megatron = ConfigInit(hidden_size=args.embed_dim, num_hidden_layers=args.depth + args.decoder_depth)
        init_modules(mae, config_megatron)
    
    if args.tiling:
        encoder_small = model = TransformerEncoder(
        patch_size=args.patch_size,overlap_size=args.overlap_size,noise_ratio=args.noise_ratio,mlp_dim_ratio=args.mlp_dim_ratio,dim_head=args.dim_head,use_flash=args.use_flash,geglu=args.geglu,
        embed_dim=512,depth=4,heads=8,
        )
        mae_small =  MAE(
            encoder=encoder_small,use_flash=args.use_flash,geglu=args.geglu,token_avg = args.token_avg,token_avg_lambda = args.token_avg_lambda,masking_ratio=args.masking_ratio,
            decoder_dim=512,decoder_depth=1,decoder_heads=8,
                )
        checkpoints_tiling = torch.load(args.tiling_model_path, weights_only=False)
        mae_small.load_state_dict(checkpoints_tiling["model"])
        tile_init(mae_small,mae)
        accelerator.print("Tiling done")
        
    optimizer = get_optimizer(mae.parameters(), args)
    if args.load_model:
        optimizer.load_state_dict(checkpoint["optim"])
    scheduler = get_lr_scheduler(optimizer, args, n_iter_per_train)

    best_score, best_epoch, patience_counter = 10, 0, 0
    train_loader, val_loader, mae, optimizer, scheduler = accelerator.prepare(
        train_loader, val_loader, mae, optimizer, scheduler
    )
    start = time.time()
    saved_19 = False
    for epoch in range(args.train_epochs):
        epoch_save = str(epoch) if args.save_all_epochs else ""
        start = time.time()
        batch_idx = 0
        loss_ema = None
        mae.train()
        for i, (x, pos, b_m, b_u) in enumerate(train_loader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    optimizer.zero_grad()
                    loss = mae(x, pos, b_m, b_u)
                    accelerator.backward(loss)
                    if args.grad_clip and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    optimizer.step()
            loss_g = loss.item()  # accelerator.gather(loss).mean().item()
            loss_ema = loss_g if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_g
            batch_idx += 1
            scheduler.step()
            accelerator.print(
                "\r Epoch {:3d} (it. {:3d}/{:3d}) Loss (EMA/loss):{:3.3f}/{:3.3f} LR {:.8f}, patience: {:3d}/{:3d}, shape {:3d}, time {:3.1f} ".format(
                    epoch,
                    batch_idx,
                    n_iter_per_train,
                    loss_ema,
                    loss_g,
                    optimizer.param_groups[0]["lr"],
                    patience_counter,
                    args.patience + 1,
                    x.shape[0],
                    time.time() - start,
                ),
                end="",
            )
            if args.log_wandb:
                accelerator.log({"epoch": epoch, "it": batch_idx, "loss_ema": loss_ema, "loss": loss_g})
            if (time.time() - start > (19 * 3600 + 30 * 60)) and (not saved_19):
                accelerator.save(
                    {
                        "args": args,
                        "epoch": epoch,
                        "model": accelerator.unwrap_model(mae).state_dict(),
                        "optim": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "ft_score": best_score,
                    },
                    pjoin(args.save_model_path, args.comment + "_model_" + epoch_save + "19h.pth"),
                )
                saved_19 = True
        accelerator.print("")
        total_val_loss, batch_idx = 0, 0
        mae.eval()
        for i, (x, pos, b_m, b_u) in enumerate(val_loader):
            with torch.no_grad():
                with accelerator.autocast():
                    loss = mae(x, pos, b_m, b_u)
                    total_val_loss += accelerator.gather_for_metrics(loss).mean().item()
                accelerator.print(
                    "\r Epoch {:3d} (it. {:3d}/{:3d}) Val Loss total:{:3.3f}, patience: {:3d}/{:3d}                                            Val Loss total:{:3.3f}".format(
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
        if args.log_wandb:
            accelerator.log(
                {
                    "epoch": epoch,
                    "val_loss": val_score,
                    "best_val": best_score if epoch > 0 else val_score,
                    "best_epoch": best_epoch,
                }
            )

        if best_score > val_score:
            best_score = val_score
            best_epoch = epoch
            patience_counter = 0
            # accelerator.save_state("/homes/y17eloua/Projets/found/checkpoints/best_model")
            # accelerator.save_model(mae, "checkpoints/best_model.pth")
            accelerator.save(
                {
                    "args": args,
                    "epoch": epoch,
                    "model": accelerator.unwrap_model(mae).state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ft_score": best_score,
                },
                pjoin(args.save_model_path, args.comment + "_model_" + epoch_save + ".pth"),
            )
        else:
            patience_counter += 1
            if patience_counter > args.patience:
                print("Early stopping")
                break

    accelerator.print(
        "final best score:", best_score, "best_epoch:", best_epoch, "total time:", time.time() - init_time
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
