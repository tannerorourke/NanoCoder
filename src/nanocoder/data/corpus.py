from array import array

import numpy as np
import torch
from tqdm.auto import tqdm

from nanocoder.constants import RNG


def apply_token_fim(ids, fim_prefix_id, fim_middle_id, fim_suffix_id, spm_rate=0.5, rng=RNG):
    """
    Token-level Fill-in-the-Middle (Bavarian et al. 2022).

    Splits the *already tokenized* id stream at two token boundaries and reorders.
    With prob spm_rate we emit SPM form (suffix first), else PSM:
    
        SPM: <pre> <suf> suffix <mid> prefix middle
        PSM: <pre> prefix <suf> suffix <mid> middle
    """
    n = len(ids)
    if n < 4:
        return ids
    a = rng.randint(1, n - 2)
    b = rng.randint(a + 1, n - 1)
    prefix, middle, suffix = ids[:a], ids[a:b], ids[b:]
    if rng.random() < spm_rate:
        return [fim_prefix_id, fim_suffix_id] + suffix + [fim_middle_id] + prefix + middle
    return [fim_prefix_id] + prefix + [fim_suffix_id] + suffix + [fim_middle_id] + middle


def compile_corpus(tokenizer, texts, fim_rate, spm_rate=0.5, add_eos=True, rng=RNG):
    """
    array('H') is a flat C buffer of uint16: ~2 bytes/token. 
    
    CPython interns ints up to 256, our ids run to ~49k, so every token is 
    a heap-allocated object plus a pointer. At ~2B tokens = ~4GB vs ~70GB.
    
    A Python list of ids
    costs ~36 bytes/token, because CPython only interns ints up to 256 and our ids
    run to ~49k - so every token is a heap-allocated object plus a pointer. At ~2B
    tokens that's 4GB vs ~70GB.
    """
    buf = array('H')
    n_tokens = 0
    enc = tokenizer._engine.encoder
    fim_ids = (enc["<|fim_prefix|>"], enc["<|fim_middle|>"], enc["<|fim_suffix|>"])
    eos_id = enc["<|eos|>"]
    for t in tqdm(texts, desc="Compiling"):
        ids = tokenizer.encode(t)
        if fim_rate > 0.0 and rng.random() < fim_rate:
            ids = apply_token_fim(ids, *fim_ids, spm_rate=spm_rate, rng=rng)
        if add_eos:
            ids = ids + [eos_id]
        n_tokens += len(ids)
        buf.extend(ids)
    # zero-copy view over the buffer; 'buf' stays alive as the array's .base
    return np.frombuffer(buf, dtype=np.uint16), n_tokens


def get_batch(ids: np.ndarray, block_size: int, batch_size: int, device: str):
    ix = torch.randint(0, len(ids) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(ids[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(ids[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
