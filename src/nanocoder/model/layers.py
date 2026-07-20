"""
Building blocks for the NanoCoder decoder in pure torch

- RMSNorm: Cheaper than LayerNorm (no mean-subtraction, no bias)
- RoPE
- CSA
- SwiGLU
- Block
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """ Root-mean-square normalization """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


""" --- Rotary Position Embedding (RoPE) --- 

Positions injected by *rotating* q and k in 2D subspaces at a frequency that
depends on the position, rather than added as a learned vector. The dot product
q_m * k_n then depends only on the relative offset (m - n), which is why RoPE
extrapolates to longer contexts far better than a learned position table.
"""
def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (hd/2,)
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)               # (T, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)        # (T, hd)
    return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim) ; cos/sin: (T, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin

""" --- Causal Self-Attention (CSA) --- """
class CausalSelfAttention(nn.Module):
    """ Variant of standard transformer attention that only attends to positions 
        *before* the current one. 
    """
    def __init__(self, n_head: int, emb_dim: int, bias=False, dropout=0.0):
        super().__init__()
        assert emb_dim % n_head == 0
        self.n_head, self.emb_dim, self.dropout = n_head, emb_dim, dropout
        self.head_dim = emb_dim // n_head

        self.c_attn = nn.Linear(emb_dim, 3 * emb_dim, bias=bias)
        self.c_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.c_proj._is_residual = True

        # QK-norm: RMSNorm q and k (per head) before attention. Keeps attention logits
        # from blowing up, which lets us train at a higher LR without instability.
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.emb_dim, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        # RoPE on q, k (not v)                
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)  

        # Flash/SDPA: fused, memory-efficient, causal mask applied internally
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class SwiGLU(nn.Module):
    """ 
    Gated MLP: down(silu(gate(x)) * up(x)). The gate lets the network modulate the
    activation per-feature, which consistently beats a plain GELU block. We size the
    hidden dim to 2/3 * 4 * emb_dim so the 3-matrix SwiGLU is *parameter-matched* to
    the 2-matrix 4x GELU MLP it replaces (a fair swap, not a free capacity bump)
    """
    def __init__(self, emb_dim: int, bias=False, dropout=0.0, mult: int = 4):
        super().__init__()
        hidden = int(2 / 3 * mult * emb_dim)
        hidden = 64 * ((hidden + 63) // 64)                 # round to a hardware-friendly multiple
        self.w_gate = nn.Linear(emb_dim, hidden, bias=bias)
        self.w_up   = nn.Linear(emb_dim, hidden, bias=bias)
        self.w_down = nn.Linear(hidden, emb_dim, bias=bias)
        self.w_down._is_residual = True
        self.drop = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, n_head: int, emb_dim: int, bias=False, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(emb_dim)
        self.attn  = CausalSelfAttention(n_head, emb_dim, bias, dropout)
        self.norm2 = RMSNorm(emb_dim)
        self.mlp   = SwiGLU(emb_dim, bias, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)          # pre-norm residual
        x = x + self.mlp(self.norm2(x))
        return x
