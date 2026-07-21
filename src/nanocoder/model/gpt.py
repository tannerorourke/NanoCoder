"""
GPT-2 style model.

NanoCoderConfig lives in nanocoder.training.config, shared with the trainer, so the
model imports it from there. Verbatim from the notebook otherwise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanocoder.model.layers import Block, RMSNorm, build_rope_cache


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([Block(cfg.n_head, cfg.emb_dim, cfg.bias, cfg.dropout)
                                      for _ in range(cfg.n_layer)])
        self.norm_f  = RMSNorm(cfg.emb_dim)
        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        # weight tying
        self.tok_emb.weight = self.lm_head.weight

        # RoPE cos/sin cache: recomputable, so non-persistent (kept out of the state_dict,
        # which also keeps checkpoints agnostic to block_size).
        cos, sin = build_rope_cache(cfg.block_size, cfg.emb_dim // cfg.n_head, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if hasattr(m, '_is_residual'):
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, pos=None):
        """
        idx: (B, T) embedded input
        pos: (B,) read positions for the inference path, one per row, default is last column.
        targets: (B, T) for training
        
        Batched generation right-pads prompts to a common width, so each row's next token has
        to be read from its own end rather than a shared one - projecting only those B positions 
        keeps the head off a (B, T, vocab) tensor (gb for batch of any size).
        """
        B, T = idx.size()
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]

        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss

        # inference path, project one position per row
        if pos is None:
            x = x[:, [-1], :]
        else:
            x = x[torch.arange(B, device=x.device), pos].unsqueeze(1)
        return self.lm_head(x), None
