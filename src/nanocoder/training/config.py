"""
Architecture and training configuration (shared by the model and the trainer).

With vocab 49,152 and block 2048 the model is ~123M.
"""
from dataclasses import dataclass

import torch


@dataclass
class NanoCoderConfig:
    # ~123M params with vocab=49,152 / block=2048
    vocab_size: int
    block_size: int = 2048      # RoPE extrapolates; 2x the old 1024 for real code context
    n_layer: int    = 12
    n_head: int     = 12
    emb_dim: int    = 768
    dropout: float  = 0.0       # dropout=0.0 standard for from-scratch pretraining.
    bias: bool      = False
    rope_base: float = 10000.0


@dataclass
class TrainConfig:
    # ---- runtime
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    # 40GB-safe defaults: 8 * 2048 * 24 accum = ~393k tokens / optimizer step.
    # On an 80GB A100 you have plenty of room -> raise batch_size and lower
    # grad_accum_steps for the same effective batch at higher throughput.
    batch_size: int         = 8
    grad_accum_steps: int   = 24
    # --- logging
    log_interval: int       = 25
    eval_interval: int      = 500
    eval_iters: int         = 20
    # --- optimizers
    max_iters: int          = 6000
    warmup_iters: int       = 300
    base_lr: float          = 6e-4
    min_lr: float           = 6e-5
    lr_decay_iters: int     = 6000
    weight_decay: float     = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float        = 1.0
    # --- tokenizer aug
    # 0.15 still teaches infill but stops FIM from dominating the objective.
    # Any higher and FIM becomes a dominant % of training tokens that sit inside a 
    # FIM-reordered block whose sentinels never appear at inference (we always prompt 
    # left-to-right). That both wastes capacity and teaches the model to emit <|fim_*|> 
    # as ordinary code tokens.
    fim_rate: float         = 0.15
