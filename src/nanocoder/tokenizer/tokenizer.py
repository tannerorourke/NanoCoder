"""
NanoCoderTokenizer: portable wrapper around SemiSupervisedBPE.

Handles everything specific to how NanoCoder sees text 
- indent/dedent scope markers
- the fenced-code preprocessing
- HF-style save/load.
- FIM is deliberately NOT here; it operates on the tokenized id stream in 'nanocoder.data.corpus``.

As a standalone tokenizer there is no reason to carry a torch.nn.module.
"""
import json
from pathlib import Path

import regex as re
import torch

from nanocoder.tokenizer.engine import SemiSupervisedBPE, build_split_pattern


class NanoCoderTokenizer:
    def __init__(self, engine, config, device: str | None = None):
        self._engine: SemiSupervisedBPE = engine

        self.CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*(.*?)\s*```", flags=re.DOTALL)
        self.default_tab_detect: int = 4
        self.config = config
        self.indent_token_id = self.config.get("indent", "<|indent|>")
        self.dedent_token_id = self.config.get("dedent", "<|dedent|>")
        self.eos_token_id = self.config.get("eos", "<|eos|>")

        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu')

    def _apply_indent(self, src: str) -> str:
        lines = src.split("\n")
        tab_width = None
        out_lines = []
        level = 0
        for line in lines:
            sline = line.lstrip(" \t")
            if not sline:
                out_lines.append(line)  # preserve blank lines
                continue
            # treat \t as 1 char width, same as "space"
            indent_chars = line[:len(line) - len(sline)]
            spaces = len(indent_chars.replace("\t", " "))

            if tab_width is None and spaces > 0:
                tab_width = spaces
            tab_unit = tab_width or self.default_tab_detect

            # calc the scope delta, preserve remainder
            new_level = spaces // tab_unit
            remainder = " " * (spaces % tab_unit)

            prefix = ""
            if new_level > level:
                prefix = self.indent_token_id * (new_level - level)
            elif new_level < level:
                prefix = self.dedent_token_id * (level - new_level)
            out_lines.append(prefix + remainder + sline)
            level = new_level

        # close out remaining open scopes so the document ends at level 0.
        # Fixes `' code fence line inheriting leftover indent state in post process
        if level > 0:
            out_lines.append(self.dedent_token_id * level)
        return "\n".join(out_lines)

    def preprocess(self, text: str, add_eos: bool):
        """
        Process raw input from a training doc or prompt: adds <|indent|>/<|dedent|>
        scope markers inside code fences, and optionally appends <|eos|>.

        FIM is NOT applied here anymore - it now operates on the tokenized id stream
        (token-level, in compile_corpus) rather than on raw characters. See the
        Fill-in-the-Middle section of the base-model notebook. That keeps this method
        a clean, near-invertible transform: encode -> decode round-trips up to <|eos|>.

        Training / Val: preprocess(doc, add_eos=True)
        Inference:      preprocess(prompt, add_eos=False)
        """
        def _fmt_code_block(block):
            inner = block.group(1)
            inner_fmt = self._apply_indent(inner)
            start, end = block.start(1) - block.start(0), block.end(1) - block.start(0)
            full = block.group(0)
            return full[:start] + inner_fmt + full[end:]

        processed = self.CODE_FENCE_RE.sub(_fmt_code_block, text)
        if add_eos:
            processed += "<|eos|>"
        return processed

    def encode(self, text: str, add_eos: bool = False, verbose: bool = False):
        clean = self.preprocess(text, add_eos)
        enc = self._engine.encode(clean)
        if verbose:
            print(f"PREPROCESSED:\n{clean}\n----")
            print(f"ENCODED:\n{enc}\n----")
        return enc

    def decode(self, tokens, tab_width: int = 4, verbose: bool = False) -> str:
        """
        Convert decoded output into display format. Mirrors encode():
            - Truncate at first <|eos|>
            - unpack <|indent|>/<|dedent|>
            - FIM sentinels: stripped (they are structural, never literal output)
        """
        decoded = self._engine.decode(tokens)
        out = self._postprocess(decoded, tab_width)
        if verbose:
            print(f"DECODED:\n{decoded}\n----")
            print(f"POSTPROCESSED:\n{out}\n----")
        return out

    # FIM sentinels are structural markers, never literal output. If the model emits one
    # during plain generation we drop it - leaving it in was the source of the
    # "<|fim_prefix|>" leaks in decoded samples.
    _FIM_RE = re.compile(r"<\|fim_(?:prefix|middle|suffix)\|>")

    def _postprocess(self, decoded: str, tab_width: int):
        text = decoded.split("<|eos|>", 1)[0] if "<|eos|>" in decoded else decoded
        text = self._FIM_RE.sub("", text)
        out_lines = []
        level = 0
        for line in text.split("\n"):
            had_marker = ("<|indent|>" in line) or ("<|dedent|>" in line)
            # Consume indent/dedent markers ANYWHERE on the line, not just the front.
            # The old startswith-only loop left mid-line markers ("    else:<|indent|>")
            # in the output verbatim - that was the "<|in" fragment in the samples.
            while "<|indent|>" in line or "<|dedent|>" in line:
                i_at = line.find("<|indent|>")
                d_at = line.find("<|dedent|>")
                if d_at == -1 or (i_at != -1 and i_at < d_at):
                    level += 1
                    line = line[:i_at] + line[i_at + len("<|indent|>"):]
                else:
                    level = max(0, level - 1)
                    line = line[:d_at] + line[d_at + len("<|dedent|>"):]
            if not line and had_marker:
                continue
            out_lines.append((" " * (level * tab_width) + line) if line else "")
        return "\n".join(out_lines)

    def save_pretrained(self, save_dir: Path | str):
        """ Dump config and tokenizer artifacts to JSON (HF-style directory). """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "tokenizer_config.json", "w") as a:
            json.dump(self.config, a, indent=2)
        with open(save_dir / "tokenizer.json", "w") as b:
            json.dump({
                "vocab": self._engine.encoder,
                # merges in learned-rank order; byte-encoding guarantees no literal
                # spaces inside a token, so join/split on " " round-trips exactly.
                "merges": [" ".join(merge) for merge in self._engine.bpe_ranks.keys()],
                "special_tokens": self._engine._special_tokens,     # raw strings
                "special_ids": sorted(self._engine._special_ids),   # set -> JSON list
                "locked": self._engine._locked_kws,
            }, b, indent=2)
        print(f"Saved tokenizer to {save_dir}")

    @classmethod
    def from_pretrained(cls, load_dir: Path | str):
        """ Reconstructs the BPE engine state from a tokenizer.json file. """
        load_dir = Path(load_dir)
        with open(load_dir / "tokenizer.json", "r") as f:
            t_data = json.load(f)
        with open(load_dir / "tokenizer_config.json", "r") as f:
            config = json.load(f)

        encoder = {k: int(v) for k, v in t_data["vocab"].items()}
        bpe = {tuple(m.split(" ")): i for i, m in enumerate(t_data["merges"])}

        # Build with encoder+ranks only, then restore special/locked state directly.
        # (Passing specials to the ctor would NOT rebuild _special_ids when the ids
        #  already live in the loaded encoder, so we restore them explicitly here.)
        engine = SemiSupervisedBPE(encoder=encoder, bpe_ranks=bpe)
        engine._special_tokens = list(t_data.get("special_tokens", []))
        engine._special_ids = set(t_data.get("special_ids", []))
        engine._locked_kws = list(t_data.get("locked", []))
        if engine._special_tokens:
            engine.special_pat = re.compile(
                "(" + "|".join(re.escape(s) for s in engine._special_tokens) + ")")
        engine.pat = build_split_pattern(engine._locked_kws)
        return cls(engine=engine, config=config)
