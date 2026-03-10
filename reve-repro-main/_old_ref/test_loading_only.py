import warnings

import pandas as pd


warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
import logging
import time

from accelerate import Accelerator, logging

from args import parse_args
from prepro_scripts.scripts_dataloader import (
    return_loaders_eegnas,  # compute_groups_segs,EEGDataset,GroupedSampler,MultiEpochsDataLoader
)


# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

logger = logging.get_logger(__name__)
args = parse_args()


def main():
    init_time = time.time()
    accelerator = Accelerator(gradient_accumulation_steps=args.acc_steps)  # ,mixed_precision='bf16',log_with='wandb')
    accelerator.even_batches = False
    timestamp = int(time.time())

    train_loader, val_loader, len_train, len_val, len_train_sampler, len_val_sampler, bs_scheduler_loader = (
        return_loaders_eegnas(args)
    )
    accelerator.print("total segments:", len_train + len_val, "train segments:", len_train, "test segments:", len_val)
    accelerator.print("N GPUS:", args.n_gpus, "Train iterations:", len_train_sampler, "acc steps:", args.acc_steps)

    len_train_sampler = len_train_sampler // args.acc_steps
    n_iter_per_train = len_train_sampler // (args.n_gpus * args.n_nodes * args.acc_steps)
    # n_iter_per_val = len_val // args.batch_size

    train_loader = accelerator.prepare(train_loader)
    start = time.time()
    epoch = 0
    for i, (x, pos, b_m, b_u) in enumerate(train_loader):
        batch_idx += 1
        accelerator.print(
            "\r Epoch {:3d} (it. {:3d}/{:3d}) shape {:3d}, time {:3.1f} ".format(
                epoch, batch_idx, n_iter_per_train, x.shape[0], time.time() - start
            ),
            end="",
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
