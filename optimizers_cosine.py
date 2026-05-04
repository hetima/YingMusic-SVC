import torch
from torch.optim import AdamW


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup.

    lr(t):
      0 → warmup_steps:  linear ramp from 0  to lr_0
      warmup_steps → T:  cosine decay from lr_0 to lr_min
    """
    def __init__(self, optimizer, warmup_steps, total_steps, lr_min=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min = lr_min
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            alpha = step / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + (progress * 3.141592653589793) ** 0)  # dummy
            import math
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.lr_min + (base_lr - self.lr_min) * cosine_decay
                    for base_lr in self.base_lrs]


class MultiOptimizer:
    def __init__(self, optimizers=None, schedulers=None):
        self.optimizers = optimizers or {}
        self.schedulers = schedulers or {}
        self.keys = list(self.optimizers.keys())

    def zero_grad(self):
        for opt in self.optimizers.values():
            opt.zero_grad()

    def step(self, key=None, scaler=None):
        if key is not None:
            if scaler is not None:
                scaler.step(self.optimizers[key])
            else:
                self.optimizers[key].step()
        else:
            for k in self.keys:
                if scaler is not None:
                    scaler.step(self.optimizers[k])
                else:
                    self.optimizers[k].step()

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step()
        else:
            for k in self.keys:
                self.schedulers[k].step()

    def state_dict(self):
        optimizer_state = {key: self.optimizers[key].state_dict() for key in self.keys}
        scheduler_state = {key: self.schedulers[key].state_dict() for key in self.keys}
        return {'optimizer': optimizer_state, 'scheduler': scheduler_state}

    def load_state_dict(self, state_dict):
        for key in self.keys:
            if key in state_dict['optimizer']:
                self.optimizers[key].load_state_dict(state_dict['optimizer'][key])
        for key in self.keys:
            if key in state_dict['scheduler']:
                self.schedulers[key].load_state_dict(state_dict['scheduler'][key])

    def scheduler_state_dict(self):
        return {key: self.schedulers[key].state_dict() for key in self.keys}

    def load_scheduler_state_dict(self, state_dict):
        for key in self.keys:
            if key in state_dict:
                self.schedulers[key].load_state_dict(state_dict[key])


def build_optimizer(model_dict, lr, type='AdamW', warmup_steps=500, total_steps=5000, lr_min=1e-6):
    optim = {}
    for key, model in model_dict.items():
        model_parameters = model.parameters()
        if type == 'AdamW':
            optim[key] = AdamW(
                model_parameters,
                lr=lr,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01,
            )
        else:
            raise ValueError('Unknown optimizer type: %s' % type)

    schedulers = {
        key: CosineWarmupScheduler(opt, warmup_steps, total_steps, lr_min)
        for key, opt in optim.items()
    }

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim


def build_single_optimizer(model, lr, warmup_steps=500, total_steps=5000, lr_min=1e-6):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = AdamW(
        model_parameters,
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
    )
    scheduler = CosineWarmupScheduler(optim, warmup_steps, total_steps, lr_min)
    return optim, scheduler
