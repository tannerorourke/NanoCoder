"""
Public API for NanoCoder
"""
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

from nanocoder.training.config import NanoCoderConfig, TrainConfig
from nanocoder.training.schedule import WarmupCosineAnnealing
from nanocoder.training.loop import train_loop

__all__ = [
    "SEED", "SUPTOK_CONFIG", "seed_global", "resolve_device",
    "SemiSupervisedBPE", "NanoCoderTokenizer", "train_bpe",
    "GPT", "NanoCoder",
    "DatasetConfig", "suptok_config", "build_dataset",
    "compile_corpus", "apply_token_fim", "get_batch",
    "SFTExample", "encode_example", "build_labels", "sft_batches",
    "NanoCoderConfig", "TrainConfig", "WarmupCosineAnnealing", "train_loop",
]

__version__ = "0.1.0"
