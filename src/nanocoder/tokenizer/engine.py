"""
A byte-level "Semi-Supervised" BPE engine: bytes, merges, and ids.

This BPE machinery is purposely kept separate from 'NanoCoderTokenizer' (API to
the model and hugging face) to separate the BPE engine from the tokenizer.
"""
from functools import lru_cache

import regex as re


@lru_cache()
def bytes_to_unicode():
    """Per https://github.com/Tenoke/gpt-2/blob/finetuning/src/encoder.py#L9
        - For every byte, produce a printable Unicode character in it's place
    """
    # Start from the bytes that are already printable, then map the remaining 68
    # (control chars, space, etc) into an unused Unicode block. This keeps every
    # byte representable as a character BPE can safely merge and print.
    bs = (list(range(ord("!"), ord("~") + 1)) +  # ASCII
          list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1)))  # Latin-1
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """ Set of adjacent symbol pairs in a tuple of symbols. """
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


def build_split_pattern(locked_kws: list[str] = []):
    """
    Build a split pattern with a prepended "locked keyword" branch, which alternates a
    list of keywords. Each keyword is checked with an optional leading space, and a negative
    lookahead that rejects any trailing letter/number/underscore.

        `(?: ?(?:def|self|return|...))(?![a-zA-Z0-9_])`

    The negative lookahead is word-boundary aware by design. If "self" is a locked keyword:
        "self" matches locked branch -> pre-token "self".
        "selfs": locked branch fails -> SPLIT_PAT picks up whole word "selfs" -> word is passed
                 through as unique pre-token to BPE.
        "self_x": locked branch fails -> SPLIT_PAT splits into ["self", "_", "x"] -> pre-token is
                  emitted for "self" (see bpe()), "_" and "x" are encoded normally.
    """
    SPLIT_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    if not locked_kws:
        return re.compile(SPLIT_PAT, re.IGNORECASE)
    kw_pattern = '|'.join(re.escape(k) for k in locked_kws)
    kw_pattern = f'(?: ?(?:{kw_pattern}))(?![a-zA-Z0-9_])'
    return re.compile(kw_pattern + '|' + SPLIT_PAT, re.IGNORECASE)


class SemiSupervisedBPE:
    def __init__(
        self,
        encoder=None,
        bpe_ranks=None,
        special_tokens: list[str] = [],
        locked_kws: list[str] = [],
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.encoder = encoder if encoder is not None else {
            ch: i for i, ch in enumerate(
                sorted(self.byte_encoder.values(), key=lambda c: self.byte_decoder[c]))
        }
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = bpe_ranks if bpe_ranks is not None else {}
        self.cache: dict[str, str] = {}

        self._special_tokens: list[str] = []
        self._special_ids = set()
        self._locked_kws: list[str] = []
        self.pat = build_split_pattern([])

        self.register_special_keywords(special_tokens or [])
        self.register_locked_keywords(locked_kws or [])

        if not locked_kws:
            self.pat = build_split_pattern([])

    @property
    def vocab_size(self):
        return len(self.encoder)

    def register_special_keywords(self, kws: list[str]):
        if not kws:
            return
        self._special_tokens.extend(kws)
        self.special_pat = re.compile("(" + "|".join(re.escape(s) for s in self._special_tokens) + ")")
        for special_kw in kws:
            if special_kw not in self.encoder:
                new_id = len(self.encoder)
                self.encoder[special_kw] = new_id
                self.decoder[new_id] = special_kw
                self._special_ids.add(new_id)

    def register_locked_keywords(self, kws: list[str]):
        if not kws:
            return
        self._locked_kws.extend(kws)
        self.pat = build_split_pattern(self._locked_kws)
        for kw in kws:
            for variant in [kw, " " + kw]:
                locked_kw = ''.join(self.byte_encoder[b] for b in variant.encode('utf-8'))
                if locked_kw not in self.encoder:
                    new_id = len(self.encoder)
                    self.encoder[locked_kw] = new_id
                    self.decoder[new_id] = locked_kw

    def bpe(self, token: str):
        # If the whole pre-token is known (special or locked tokens), return it,
        # effectively locking the token from being split.
        # locked kws were inserted into the encoder at __init__, so they short-circuit 
        # here before a single merge rule is done (the "supervised" mechanism at encode time)
        if token in self.encoder:
            return token

        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token

        while True:
            # Lowest-rank (earliest learned) merges first. Unknown pairs get float('inf')
            # so they sort last and terminate the loop once nothing merges.
            # Rank order (not frequency order) is what makes encoding deterministic
            # and identical to what train_bpe saw.
            bestpair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bestpair not in self.bpe_ranks:
                break

            first, second = bestpair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str):
        if not self._special_ids:
            segments = [text]
        else:  # split on specials, encode the pieces
            segments = self.special_pat.split(text)

        bpe_tokens = []
        for segment in segments:
            if segment in self.encoder and self.encoder[segment] in self._special_ids:
                bpe_tokens.append(self.encoder[segment])
            elif segment:
                for token in re.findall(self.pat, segment):
                    token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
                    bpe_tokens.extend(self.encoder[bpe_token]
                                      for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        out_bytes = bytearray()
        for tk in tokens:
            # Specials bypass byte_decoder (emitting UTF-8), otherwise BPE as normal
            if tk in self._special_ids:
                out_bytes.extend(self.decoder[tk].encode("utf-8"))
            else:
                out_bytes.extend(self.byte_decoder[c] for c in self.decoder[tk])
        return out_bytes.decode("utf-8", errors="replace")
