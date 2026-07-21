""" Callable for the BPE trainer to learn the merge table from a corpus """
from collections import Counter, defaultdict

from tqdm.auto import tqdm

from nanocoder.tokenizer.engine import SemiSupervisedBPE


def train_bpe(tokenizer: SemiSupervisedBPE, texts: list[str], vocab_size: int):
    """ Perform byte pair merges over a subset of texts until the vocab reaches vocab_size """
    # count pre-token frequencies
    word_freqs = Counter()
    for t in tqdm(texts, desc="Counting frequencies"):
        for m in tokenizer.pat.finditer(t):
            word_freqs[m.group()] += 1
    print(f"  {len(word_freqs)} unique pre-tokens / {sum(word_freqs.values())} total occurrences")

    # maps unique pre-tokens to tuple of byte-encoded chars
    splits: dict[str, list[str]] = {
        w: [tokenizer.byte_encoder[b] for b in w.encode("utf-8")]
        for w in word_freqs
    }

    # count pair frequencies across the corpus
    pair_counts: Counter = Counter()
    pair_to_words: dict = defaultdict(set)
    for w, freq in word_freqs.items():
        pieces = splits[w]
        for j in range(len(pieces) - 1):
            p = (pieces[j], pieces[j + 1])
            pair_counts[p] += freq
            pair_to_words[p].add(w)
    assert pair_counts, "No pairs found"

    def count_adj_pairs(pieces):
        """ Count adjacent pairs in one word (unweighted; caller multiplies by freq) """
        c: Counter = Counter()
        for j in range(len(pieces) - 1):
            c[(pieces[j], pieces[j + 1])] += 1
        return c

    num_merges = vocab_size - len(tokenizer.encoder)
    pbar = tqdm(range(num_merges), desc="Training BPE")
    for _ in pbar:
        # merge the most frequent pair into a new token at next available rank
        best_pair, best_count = max(pair_counts.items(), key=lambda kv: kv[1])
        if best_count <= 0:
            break

        merged = best_pair[0] + best_pair[1]
        current_rank = len(tokenizer.bpe_ranks)
        tokenizer.bpe_ranks[best_pair] = current_rank
        # safeguard: don't duplicate merge on top of locked kw's
        # merge is still recorded in bpe_ranks so the encoding can
        # chain it into longer merges (ex: "self" + "_" + "123")
        if merged not in tokenizer.encoder:
            new_id = len(tokenizer.encoder)
            tokenizer.encoder[merged] = new_id
            tokenizer.decoder[new_id] = merged

        # apply merges to word's containing this pair
        a, b = best_pair
        for w in list(pair_to_words[best_pair]):
            freq = word_freqs[w]
            old_pieces = splits[w]
            old_pc = count_adj_pairs(old_pieces)

            new_pieces, j = [], 0
            while j < len(old_pieces):
                if (j < len(old_pieces) - 1 and old_pieces[j] == a and old_pieces[j + 1] == b):
                    new_pieces.append(merged); j += 2
                else:
                    new_pieces.append(old_pieces[j]); j += 1
            new_pc = count_adj_pairs(new_pieces)

            # update global counts + p2w index
            for pair, c in old_pc.items():
                pair_counts[pair] -= c * freq
                pair_to_words[pair].discard(w)
            for pair, c in new_pc.items():
                pair_counts[pair] += c * freq
                pair_to_words[pair].add(w)

            splits[w] = new_pieces

        # pair is fully consumed
        pair_counts.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)

        if (current_rank + 1) % 500 == 0:
            sample_bytes = bytearray(tokenizer.byte_decoder[c] for c in merged).decode("utf-8", errors="replace")
            pbar.set_postfix(last_merge=repr(sample_bytes), count=best_count)

    return tokenizer
