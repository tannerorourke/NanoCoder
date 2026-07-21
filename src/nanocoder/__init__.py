""" Public API """
from nanocoder.constants import SEED, SUPTOK_CONFIG, seed_global, resolve_device

from nanocoder.tokenizer.engine import SemiSupervisedBPE
from nanocoder.tokenizer.tokenizer import NanoCoderTokenizer
from nanocoder.tokenizer.trainer import train_bpe

from nanocoder.model.gpt import GPT
from nanocoder.model.nanocoder import NanoCoder

from nanocoder.data.config import DatasetConfig, suptok_config
from nanocoder.data.sources import build_dataset
from nanocoder.data.corpus import compile_corpus, apply_token_fim, get_batch
from nanocoder.data.sft import SFTExample, encode_example, build_labels, sft_batches

from nanocoder.data.preference import Pair, encode_pair, pair_batches

from nanocoder.training.config import (NanoCoderConfig, TrainConfig, SFTConfig,
                                       RFTConfig, DPOConfig)
from nanocoder.training.schedule import WarmupCosineAnnealing
from nanocoder.training.loop import train_loop
from nanocoder.training.sft_loop import sft_loop
from nanocoder.training.dpo import dpo_loss, sequence_logprobs, dpo_loop

__all__ = [
    "SEED", "SUPTOK_CONFIG", "seed_global", "resolve_device",
    "SemiSupervisedBPE", "NanoCoderTokenizer", "train_bpe",
    "GPT", "NanoCoder",
    "DatasetConfig", "suptok_config", "build_dataset",
    "compile_corpus", "apply_token_fim", "get_batch",
    "SFTExample", "encode_example", "build_labels", "sft_batches",
    "Pair", "encode_pair", "pair_batches",
    "NanoCoderConfig", "TrainConfig", "SFTConfig", "RFTConfig", "DPOConfig",
    "WarmupCosineAnnealing", "train_loop", "sft_loop",
    "dpo_loss", "sequence_logprobs", "dpo_loop",
]

__version__ = "0.1.0"
