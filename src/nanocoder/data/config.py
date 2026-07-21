from dataclasses import dataclass, field

from nanocoder.constants import SUPTOK_CONFIG

suptok_config = SUPTOK_CONFIG


@dataclass
class DatasetConfig:
    max_samples: int             = 2_237_932    # Derived from landing the model at ~2.2B tokens (~20 tok/param) at below epoch counts
    val_split: float             = 0.05
    tokenizer_train_samples: int = 40_000
    vocab_size: int              = 49152        # 48*1024; < 65535 keeps the uint16 corpus valid

    # Per-source char caps
    max_chars: dict[str, int] = field(default_factory=lambda: {
        "codeparrot":     20_000,  # keeps 89%, cuts the generated-file tail
        "cosmo":           8_000,  # keeps 79% of long-form textbook docs
        "glaive":          3_000,
        "tinycodes":       3_000,
        "magicoder_evol":  3_000,  # keeps 95% of responses
        "codefeedback":    3_000,
        "magicoder_oss":   3_000,
    })

    # Doc-level interleave probabilities.
    # NOTE Weights are solved so the *token* mix lands at ~78% code / ~22% prose with every code 
    # source at <=2 epochs. Probabilities are per-DOCUMENT (mean doc length varies ~4x across sources)
    mix_proportions: dict[str, float] = field(default_factory=lambda: {
        # raw python (~55% of tokens)
        "codeparrot":     0.360,
        # instruct (~26% of tokens)
        "glaive":         0.130,
        "tinycodes":      0.115,
        "magicoder_evol": 0.110,
        "codefeedback":   0.095,
        "magicoder_oss":  0.040,
        # prose (~19% of tokens)
        "cosmo":          0.150,
    })
