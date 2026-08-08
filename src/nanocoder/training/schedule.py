import math

""" Warmup + cosine-annealing LR schedule. """
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

        self.set_iter(0)

    # -- place the schedule at 'it' completed iterations; a resume calls this before training
    def set_iter(self, it: int) -> None:
        self._it = it
        self.last_lr = self._get_lr(it)
        self._apply(self.last_lr)

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

    # -- last_lr is the rate the finished iteration ran at; the applied one is for the next.
    def step(self):
        self.last_lr = self.optimizer.param_groups[0]["lr"]
        self._it += 1
        self._apply(self._get_lr(self._it))

"""
Linear warmup -> hold. Used in SFT and RFT.
- annealing wastes the pass and weights examples in the fixed set differently
- warmup is still useful to ease into the new objective
"""
def constant_with_warmup(optimizer, base_lr: float, warmup_steps: int):
    def _apply(step: int) -> float:
        lr = base_lr
        if step < warmup_steps:
            lr = base_lr * (step + 1) / (warmup_steps + 1)
        
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    # first step must run at warmup LR, not the optimizer's default
    _apply(0)
    return _apply
