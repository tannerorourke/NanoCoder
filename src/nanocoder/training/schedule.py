""" Warmup + cosine-annealing LR schedule. """
import math

class WarmupCosineAnnealing:
    def __init__(
        self,
        optimizer,
        min_lr: float, warmup_iters: int, max_iters: int,
        lr_decay_iters: int = 0,
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.base_lr = float(optimizer.param_groups[0]["lr"])
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.lr_decay_iters = max(warmup_iters, lr_decay_iters)

        self.last_lr = self.base_lr
        self._it = 0

    def _get_lr(self, it: int) -> float:
        if it < self.warmup_iters:
            return self.base_lr * (it + 1) / (self.warmup_iters + 1)
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.base_lr - self.min_lr)

    def _apply(self, lr: float):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self):
        self.last_lr = self.optimizer.param_groups[0]["lr"]
        lr = self._get_lr(self._it)
        self._apply(lr)
        self._it += 1
