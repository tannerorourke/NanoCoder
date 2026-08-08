"""
Configuration for each stage.

NanoCoderConfig -> base model config (vocab=49,152, block=2048 -> ~123M params)
TrainConfig     -> pretraining
SFTConfig       -> Supervised Fine-Tuning
RFTConfig       -> Reinforcement Fine-Tuning
DPOConfig       -> Direct Preferece Optimization
"""
from dataclasses import dataclass

import torch


# -- Widest autocast dtype the GPU supports. Ampere and later (A100, L4) do bf16;
#    Volta and Turing (V100, T4) do fp16 but not bf16, and need the GradScaler path.
def resolve_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16 # Ampere and later: wider exponent, no scaler needed
    return torch.float16      # Turing: half throughput of bf16 but 8x fp32


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
    dtype = resolve_dtype()
    # 40GB-safe defaults: 8 * 2048 * 24 accum = ~393k tokens / optimizer step.
    # On an 80GB A100 you have plenty of room -> raise batch_size and lower
    # grad_accum_steps for the same effective batch at higher throughput.
    batch_size: int         = 8
    grad_accum_steps: int   = 24
    # --- logging
    log_interval: int       = 25
    eval_interval: int      = 500
    eval_iters: int         = 20
    # --- checkpointing
    # A checkpoint is ~12 bytes/param (fp32 weights + Adam's two moments), so ~1.5GB here.
    checkpoint_interval: int = 250
    keep_checkpoints: int    = 2
    # --- optimizers
    max_iters: int          = 6000
    warmup_iters: int       = 300
    base_lr: float          = 6e-4
    min_lr: float           = 6e-5
    lr_decay_iters: int     = 6000
    weight_decay: float     = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float        = 1.0
    # -- Enough to teach infill without dominating the objective. Inference always prompts
    #    left-to-right, so a reordered block's sentinels never appear there; raising this
    #    wastes capacity and teaches the model to emit <|fim_*|> as ordinary code.
    fim_rate: float         = 0.15


@dataclass
class SFTConfig:
    dtype = resolve_dtype()
    # 512, not the pretrain 2048. Filtered instruction examples run ~200-500 tokens, so
    # padding to the pretrain block would spend 75-85% of every batch on <|eos|> filler.
    # RoPE extrapolates, so inference at 2048 is unaffected by training at 512.
    block_size: int         = 512
    batch_size: int         = 16
    grad_accum_steps: int   = 4         # ~32k tokens/step, small next to pretrain's 393k
    epochs: int             = 3
    # --- logging
    log_interval: int       = 25
    eval_interval: int      = 250
    eval_batches: int       = 20
    checkpoint_interval: int = 250
    keep_checkpoints: int    = 2
    # --- optimizer
    base_lr: float          = 6e-5      # ~10% of the pretrain base
    warmup_steps: int       = 100
    weight_decay: float     = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float        = 1.0
    # No FIM. The sentinels never appear in an instruction prompt, and pretraining already taught infill
    fim_rate: float         = 0.0


@dataclass
class RFTConfig(SFTConfig):
    """
    Same machinery as SFT, one epoch, lower LR.

    The dataset is small and self-generated, so it is easy to overfit into the narrow slice
    of behaviour the model already had right -a known failure of this method.
    """
    epochs: int             = 1
    base_lr: float          = 2e-5
    warmup_steps: int       = 40
    checkpoint_interval: int = 100      # one short epoch; 250 could never fire


@dataclass
class DPOConfig:
    """Direct Preference Optimization
    
    Note the LR is 2orders of magnitude below SFT's.

    DPO optimises a *relative* quantity, so nothing anchors the policy's absolute
    likelihoods; at an SFT-sized LR it diverges loudly, driving both chosen and rejected
    log-probs down while the margin still looks healthy. beta controls how far the policy
    may drift from the reference - lower is freer, and 0.1 is the standard starting point.
    """
    dtype = resolve_dtype()
    block_size: int         = 512
    batch_size: int         = 4         # chosen and rejected both resident, so half of SFT
    grad_accum_steps: int   = 8
    epochs: int             = 1
    beta: float             = 0.1
    log_interval: int       = 10
    eval_interval: int      = 100
    eval_batches: int       = 20
    checkpoint_interval: int = 100      # short run; matched to eval_interval
    keep_checkpoints: int    = 2
    base_lr: float          = 5e-7
    warmup_steps: int       = 20
    weight_decay: float     = 0.0       # the KL term to the reference is the regulariser
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float        = 1.0
