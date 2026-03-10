import torch

from utils.stable_adamw import StableAdamW


def get_optimizer(parameters, args):
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            [{"params": parameters, "weight_decay": args.weight_decay}], lr=args.lr, momentum=0.9
        )  # , nesterov=True)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": parameters,
                    "weight_decay": args.adamw_weight_decay,
                    "betas": (args.adamw_beta1, args.adamw_beta2),
                    "eps": args.adamw_epsilon,
                }
            ],
            lr=args.lr,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam([{"params": parameters, "weight_decay": args.weight_decay}], lr=args.lr)
    elif args.optimizer == "lars":
        optimizer = torch.optim.SGD(
            [{"params": parameters, "weight_decay": args.weight_decay}], lr=args.lr, momentum=0.9, nesterov=True
        )

    elif args.optimizer == "stableadamw":
        optimizer = StableAdamW(
            [
                {
                    "params": parameters,
                    "weight_decay": args.adamw_weight_decay,
                    "betas": (args.adamw_beta1, args.adamw_beta2),
                    "eps": args.adamw_epsilon,
                }
            ],
            lr=args.lr,
        )
    return optimizer

class CyclicTrapezoidLR(torch.optim.lr_scheduler._LRScheduler):
    """
    A cyclic trapezoidal learning rate schedule with distinct start, peak, and end LRs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        plateau_steps: int,
        cooldown_steps: int,
        start_lr: float,
        peak_lr: float,
        end_lr: float,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_steps = warmup_steps
        self.plateau_steps = plateau_steps
        self.cooldown_steps = cooldown_steps
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr

        # Correct total steps in one cycle
        self.steps_per_cycle = warmup_steps + plateau_steps + cooldown_steps

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        step = self.last_epoch
        cycle_step = step % self.steps_per_cycle  
        cycle_start_lr = self.start_lr if step < self.steps_per_cycle else self.end_lr

        if cycle_step < self.warmup_steps:
            # Phase 1: warm-up from start_lr -> peak_lr
            progress = cycle_step / self.warmup_steps
            lr = cycle_start_lr + (self.peak_lr - cycle_start_lr) * progress
        elif cycle_step < self.warmup_steps + self.plateau_steps:
            # Phase 2: plateau at peak_lr
            lr = self.peak_lr
        elif cycle_step < self.steps_per_cycle:
            # Phase 3: cool-down from peak_lr -> end_lr
            cooled_step = cycle_step - (self.warmup_steps + self.plateau_steps)
            progress = cooled_step / self.cooldown_steps
            lr = self.peak_lr - (self.peak_lr - self.end_lr) * progress
        else:
            lr = self.end_lr

        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.get_lr()

    


def get_lr_scheduler(optimizer, args, n_iter):
    #n_iter =  (args.n_gpus * args.n_nodes) * n_iter
    if args.scheduler == "trapezoid":
        assert args.warmup_steps + args.cooldown_steps <= 1
        w_steps = int(args.warmup_steps * n_iter) 
        c_steps = int(args.cooldown_steps * n_iter)
        p_steps = n_iter - w_steps - c_steps
        return CyclicTrapezoidLR(
            optimizer=optimizer,
            warmup_steps=w_steps,
            plateau_steps=p_steps,
            cooldown_steps=c_steps,
            start_lr=args.start_lr,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
        )
    elif args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.patience_plateau
        )
