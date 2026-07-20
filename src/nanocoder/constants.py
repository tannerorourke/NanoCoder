"""
Shared constants and the package-level RNG.

Seeding lives here so every module draws from one reproducible source.
"""
import random

SEED = 1738

# Package-level RNG. FIM (data/corpus.py) and any other stochastic step default to
# this so a single seed reproduces a whole run; pass your own Random() to override.
RNG = random.Random(SEED)

# The special / locked keyword vocabulary NanoCoder's tokenizer is seeded with.
# - "special" tokens are added verbatim to the vocab and split on at encode time.
# - "locked" keywords are never split by BPE (see build_split_pattern); they get an id
#   at tokenizer construction so common Python tokens stay whole.
SUPTOK_CONFIG = {
    "indent": "<|indent|>",
    "dedent": "<|dedent|>",
    "eos": "<|eos|>",
    "special": [
        "<|eos|>", "<|indent|>", "<|dedent|>",
        "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
    ],
    "locked": ["```", "\n",
        "if", "else", "elif", "for", "while", "return", "yield", "break", "continue",
        "def", "class", "lambda", "with", "as", "import", "from", "try", "except",
        "self", "None", "True", "False", "and", "or", "not", "is", "in",
        "==", "!=", "<=", ">=", "+=", "-=", "->",
    ],
}


def seed_global(seed: int = SEED) -> None:
    """Seed python / numpy / torch. Call once at the top of an entrypoint script."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(prefer: str | None = None) -> str:
    """'cuda' if available else 'cpu', unless a device is explicitly requested."""
    import torch
    if prefer is not None:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"
