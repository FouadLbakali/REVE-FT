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
from prepro_scripts.scripts_dataloader_fix import return_train_val_loaders
from utils.optim import get_lr_scheduler, get_optimizer
import numpy as np

def log_scale_int_points(K, N=10):
    points = np.logspace(start=0, stop=np.log10(K), num=N)
    log_int_points = np.unique(np.round(points).astype(int))
    return   np.concatenate([log_int_points[:-1] ,[log_int_points[-1]//2, log_int_points[-1]]])[3:]

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
    train_loader, len_train, len_train_sampler = return_train_val_loaders(args, return_val=False)
    accelerator.print("total segments:", len_train)
    accelerator.print("N GPUS:", args.n_gpus, "Train iterations:", len_train_sampler, "acc steps:", args.acc_steps)

    n_iter_per_train = len_train_sampler // (args.n_gpus * args.n_nodes)
    steps_save = log_scale_int_points(n_iter_per_train, N=10)
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
        mae.load_state_dict(checkpoint["model"],map_location="cpu")
        accelerator.print("loaded")
    elif args.init_megatron:
        config_megatron = ConfigInit(hidden_size=args.embed_dim, num_hidden_layers=args.depth + args.decoder_depth)
        init_modules(mae, config_megatron)
    optimizer = get_optimizer(mae.parameters(), args)
    if args.load_model:
        optimizer.load_state_dict(checkpoint["optim"])
    scheduler = get_lr_scheduler(optimizer, args, n_iter_per_train)

    best_score, best_epoch, patience_counter = 10, 0, 0
    train_loader, mae, optimizer, scheduler = accelerator.prepare(train_loader, mae, optimizer, scheduler)
    start = time.time()
    saved_19 = False
    counter_save = 0
    list_log = []
    for epoch in range(args.train_epochs):
        epoch_save = str(epoch) if args.save_all_epochs else ""
        start = time.time()
        batch_idx = 0
        loss_ema = None
        mae.train()
        if epoch ==1:
            np.save(args.comment+'_loss.npy',np.array(list_log))
            break
        if epoch == 0:
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
                break
        for i, (x, pos, b_m, b_u) in enumerate(train_loader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    optimizer.zero_grad()
                    loss = mae(x, pos, b_m, b_u)
                    accelerator.backward(loss)
                    if args.grad_clip and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    optimizer.step()
            loss_g = accelerator.gather(loss).mean().item()
            list_log.append(loss_g)
            loss_ema = loss_g if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_g
            batch_idx += 1
            if i >200:
                break
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
                        "ft_score": loss_ema,
                    },
                    pjoin(args.save_model_path, args.comment + "_model_" + epoch_save + "19h.pth"),
                )
                saved_19 = True
        accelerator.print("")
        accelerator.save(
            {
                "args": args,
                "epoch": epoch,
                "model": accelerator.unwrap_model(mae).state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ft_score": loss_ema,
            },
            pjoin(args.save_model_path, args.comment + "_model_" + epoch_save + ".pth"),
        )

    accelerator.print(
        "final best score:", best_score, "best_epoch:", best_epoch, "total time:", time.time() - init_time
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
