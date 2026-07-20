from dataclasses import dataclass, field

from nanocoder.constants import SUPTOK_CONFIG

# backwards-compatible alias matching the notebook variable name
suptok_config = SUPTOK_CONFIG


@dataclass
class DatasetConfig:
    # max_samples is derived, not chosen: it's the doc count that lands the corpus at
    # ~2.2B tokens (~20 tok/param) at the epoch counts below. See the Dataset section
    # for the pool measurements these come from.
    max_samples: int             = 2_237_932
    val_split: float             = 0.05
    tokenizer_train_samples: int = 40_000
    vocab_size: int              = 49152   # 48*1024; < 65535 keeps the uint16 corpus valid

    # Per-source char caps
    max_chars: dict[str, int] = field(default_factory=lambda: {
        "codeparrot":     20_000,  # keeps 89%, cuts the generated-file tail (max seen: 361k chars)
        "cosmo":           8_000,  # keeps 79% of long-form textbook docs
        "glaive":          3_000,
        "tinycodes":       3_000,
        "magicoder_evol":  3_000,  # keeps 95% of responses
        "codefeedback":    3_000,
        "magicoder_oss":   3_000,
    })

    # Doc-level interleave probabilities. NOTE these are per-DOCUMENT, but the model
    # trains on TOKENS - and mean doc length varies ~4x across sources, so the two
    # mixes are not the same. These weights are solved so the *token* mix lands at
    # ~78% code / ~22% prose, with every code source at <=2 epochs.
    # Rebalanced after the first 6k-iter run: prompt->code was the visibly weak axis,
    # so instruction sources gain ~5 doc-pts pulled from prose. Raw code stays the
    # backbone. (Instruction data is still pool-limited - see the training notes for
    # the real lever on instruction-following.)
    mix_proportions: dict[str, float] = field(default_factory=lambda: {
        "codeparrot":     0.360, # raw python (~55% of tokens)
        "glaive":         0.130, # instruct
        "tinycodes":      0.115, # instruct
        "magicoder_evol": 0.110, # instruct
        "codefeedback":   0.095, # instruct
        "magicoder_oss":  0.040, # instruct (instruct total ~26% of tokens)
        "cosmo":          0.150, # prose    (~19% of tokens)
    })
