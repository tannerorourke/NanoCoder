"""
Direct Preference Optimization utilities.

A DPO pair: one prompt with two completions, one preferred. 
- The objective compares them, so they must be encoded with same prompt tokens, 
  block size, and completion mask - or the comparison measures the encoding rather 
  than the completions.

Chosen and rejected are batched as one concatenated tensor and split after the forward
pass, guaranteeing identical padding and kernels and halving forward calls.

Container is built on the SFT masking.
"""
from dataclasses import dataclass

import torch

from nanocoder.constants import RNG
from nanocoder.data.sft import SFTExample, build_labels, encode_example


@dataclass
class Pair:
    chosen: SFTExample
    rejected: SFTExample


def encode_pair(tokenizer, prompt: str, chosen: str, rejected: str) -> Pair:
    return Pair(
        chosen=encode_example(tokenizer, prompt, chosen),
        rejected=encode_example(tokenizer, prompt, rejected),
    )


def pair_batches(pairs, block_size: int, batch_size: int, eos_id: int,
                 device: str = 'cpu', shuffle: bool = True, rng=RNG, drop_last: bool = True):
    """
    Yield (x, y, n) where the first n rows are chosen, remaining n are rejected.

    One tensor rather than two: the reference model and the policy each see it once, and
    the two halves are guaranteed to have been padded and attended identically.
    """
    order = list(range(len(pairs)))
    if shuffle:
        rng.shuffle(order)

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        if drop_last and len(idx) < batch_size:
            break
        built = ([build_labels(pairs[i].chosen, block_size, eos_id) for i in idx]
                 + [build_labels(pairs[i].rejected, block_size, eos_id) for i in idx])
        x = torch.stack([b[0] for b in built])
        y = torch.stack([b[1] for b in built])
        if device == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        yield x, y, len(idx)


def load_pairs(tokenizer, rows, block_size: int, desc: str = "pairs"):
    """ Encode dataset rows into Pairs, dropping any whose longer side will not fit """
    from tqdm.auto import tqdm
    kept, dropped = [], 0
    for row in tqdm(rows, desc=desc):
        pair = encode_pair(tokenizer, row["prompt"], row["chosen"], row["rejected"])
        if max(len(pair.chosen), len(pair.rejected)) + 1 > block_size + 1:
            dropped += 1
            continue
        kept.append(pair)
    print(f"{desc}: {len(kept):,} pairs | dropped {dropped:,} over {block_size} tokens")
    return kept
