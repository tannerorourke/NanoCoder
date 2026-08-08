import hashlib
import os
from array import array

import numpy as np
import torch
from tqdm.auto import tqdm

from nanocoder.constants import RNG, SEED


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


# -- Identity of a compiled stream. Everything that can change an id feeds the hash, so a
#    retrained tokenizer or a different FIM rate names a new file rather than silently
#    reusing ids that no longer match the model being trained.
def corpus_fingerprint(tokenizer, n_docs: int, **params) -> str:
    h = hashlib.sha256()
    for tok, tid in sorted(tokenizer._engine.encoder.items()):
        h.update(f"{tok}\x00{tid}\x01".encode("utf-8", "surrogatepass"))
    h.update(repr((n_docs, SEED, sorted(params.items()))).encode())
    return h.hexdigest()[:16]


def _cache_store(path: str, ids: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # -- write-then-rename, so a job killed mid-write leaves no truncated cache behind
    tmp = f"{path}.tmp.npy"
    np.save(tmp, ids)
    os.replace(tmp, path)


def compile_corpus(tokenizer, texts, fim_rate, spm_rate=0.5, add_eos=True, rng=RNG,
                   cache_path=None):
    """
    Tokenize all documents into one flat uint16 buffer (~2 bytes/token). A Python list
    costs ~36 bytes/token: CPython interns ints only to 256 and ids run to ~49k, so each
    is a heap object plus a pointer. At ~2B tokens, 4GB vs ~70GB.

    cache_path, when set, reuses a previous compile instead of repeating the pass.
    """
    if cache_path and os.path.exists(cache_path):
        ids = np.load(cache_path)
        print(f"Corpus cache hit: {cache_path} ({len(ids):,} tokens)")
        return ids, len(ids)

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
    ids = np.frombuffer(buf, dtype=np.uint16)
    if cache_path:
        _cache_store(cache_path, ids)
        print(f"Corpus cached to {cache_path} ({ids.nbytes / 1e9:.1f} GB)")
    return ids, n_tokens


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
