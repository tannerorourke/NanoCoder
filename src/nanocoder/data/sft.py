"""
Supervised Fine-Tuning utility functions.

Instruction tuning needs the loss to see the completion NOT the prompt, which means per-example structure and a label array carrying -1 in the positions to ignore.

Two easy to get wrong details:
- The prompt/completion boundary is *measured*, not searched for. Each side is encoded
  separately and the lengths concatenated, so the split is exact. Searching the token
  stream for '## Solution' would be approximate - a BPE merge can straddle the marker.
- There is no pad token. Vocab is fixed at tokenizer training and the embedding is tied to
  the LM head, so padding uses <|eos|> with label -1.
"""
from dataclasses import dataclass

import torch

from nanocoder.constants import RNG

IGNORE = -1 # GPT.forward's cross_entropy is built with ignore_index=-1


@dataclass
class SFTExample:
    ids: list[int]          # prompt tokens followed by completion tokens
    prompt_len: int         # index of the first completion token

    def __len__(self) -> int:
        return len(self.ids)


def encode_example(tokenizer, prompt: str, completion: str) -> SFTExample:
    """ Encode both sides separately so the boundary is exact """
    p_ids = tokenizer.encode(prompt)
    c_ids = tokenizer.encode(completion)
    return SFTExample(ids=p_ids + c_ids, prompt_len=len(p_ids))


def build_labels(ex: SFTExample, block_size: int, eos_id: int):
    """
    Turn one example into the (x, y) pair GPT.forward expects.
    - x: prompt, y: completion plus its <|eos|>
    - Targets input-shifted by one (matching get_batch), so y[t] is the token the model
    must predict at position t. 
    - Two spans are masked out:
        - y[:prompt_len - 1], targets are prompt tokens. 1st supervised target 
          is y[prompt_len - 1] = the first completion token, predicted from the whole prompt.
        - everything past the terminating <|eos|>, which is padding.
    """
    full = ex.ids + [eos_id]
    n = len(full)                       # real content, including the terminator
    if n > block_size + 1:
        raise ValueError(f"example of {n} tokens exceeds block_size + 1 ({block_size + 1}); "
                         "filter long examples out rather than truncating")
    full = full + [eos_id] * (block_size + 1 - n)

    x = torch.tensor(full[:block_size], dtype=torch.long)
    y = torch.tensor(full[1:block_size + 1], dtype=torch.long)
    y[:max(0, ex.prompt_len - 1)] = IGNORE
    y[n - 1:] = IGNORE
    return x, y


def sft_batches(examples, block_size: int, batch_size: int, eos_id: int,
                device: str = 'cpu', shuffle: bool = True, rng=RNG, drop_last: bool = True):
    """
    One epoch of batches over a finite, ordered dataset.
    - Every batch is padded to same block_size so torch.compile sees one shape and does not recompile per batch
    - Shuffling draws from the package RNG, so a seed reproduces the epoch order.
    """
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        if drop_last and len(idx) < batch_size:
            break
        pairs = [build_labels(examples[i], block_size, eos_id) for i in idx]
        x = torch.stack([p[0] for p in pairs])
        y = torch.stack([p[1] for p in pairs])
        if device == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        yield x, y


def decode_masked(tokenizer, x, y):
    """
    Pretty print an example's prompt against its supervised span.
    
    Worth calling once before every SFT run. An off-by-one in the mask is silent - the loss
    still falls - and it poisons the whole stage, so the only real check is reading the two
    spans side by side and confirming the supervised one starts at the first completion
    token and ends at <|eos|>.
    """
    sup = (y != IGNORE).nonzero().flatten()
    if len(sup) == 0:
        return {"prompt": tokenizer.decode(x.tolist()), "supervised": "<empty>"}
    lo, hi = int(sup[0]), int(sup[-1])
    return {
        "prompt": tokenizer.decode(x[:lo + 1].tolist()),
        "supervised": tokenizer.decode(y[lo:hi + 1].tolist()),
        "n_supervised": len(sup),
    }
