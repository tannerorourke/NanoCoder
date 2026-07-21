"""
Portable model wrapper holding the model, config, and tokenizer used for 
model interaction and bundling for HuggingFace.
- save_pretrained writes weights, config, and tokenizer together
"""
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from nanocoder.model.gpt import GPT
from nanocoder.tokenizer.tokenizer import NanoCoderTokenizer
from nanocoder.training.config import NanoCoderConfig, resolve_dtype


class NanoCoder(torch.nn.Module):
    """ Portable wrapper interface for GPT model """
    def __init__(self, model, config, tokenizer, device: str | None = None):
        super().__init__()
        self.model: GPT = model
        self.tokenizer: NanoCoderTokenizer | None = tokenizer
        self.config = config

        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu')

    def save_pretrained(self, save_dir: Path | str):
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.model.state_dict(),
                   os.path.join(save_dir, 'pytorch_model.bin'))

        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(cls, load_dir: Path | str, device: str = 'cpu'):
        with open(os.path.join(load_dir, 'config.json'), "r") as f:
            config_data = json.load(f)

        config = NanoCoderConfig(**config_data)
        model = GPT(config)

        state_dict = torch.load(os.path.join(load_dir, 'pytorch_model.bin'), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        tokenizer = None
        if os.path.exists(os.path.join(load_dir, 'tokenizer.json')):
            tokenizer = NanoCoderTokenizer.from_pretrained(load_dir)
        
        return cls(model=model, config=config, tokenizer=tokenizer, device=device)

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        revision: str | None = None,
        commit_message: str = "Push NanoCoder",
        token: str | None = None,
    ) -> str:
        """ Save to a temp dir and upload the folder """
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        if revision is not None:
            api.create_branch(repo_id, branch=revision, repo_type="model", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp)
            api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="model",
                              commit_message=commit_message, revision=revision)
        return f"https://huggingface.co/{repo_id}"

    def _autocast(self, device):
        """Autocast in whatever half precision the card supports, or not at all on CPU."""
        if not str(device).startswith('cuda'):
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=resolve_dtype())

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int | None = 40,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.15,
        min_new_tokens: int = 8,
        suppress_fim: bool = True,
        return_prompt: bool = False,
        stop_ids: list[int] | None = None,
        device: str | None = None,
    ) -> list[str]:
        """
        Sample many completions in one set of forward passes.

        The eval sweep and the preference sampling run are both m tasks x n samples, and
        one-at-a-time sampling makes that the dominant cost of the whole project. Without
        a KV cache each step still re-reads the full context, but batching amortises that
        across the batch, so 64 completions cost about what one used to.
        
        - top_p (nucleus) trims the long tail top_k alone leaves
        - repetition_penalty divides the logits of tokens already in the row, curbing the
          "function's function's" / "lo = lo" loops a small model falls into.
        - suppress_fim masks the FIM sentinels, which are only meaningful when the prompt
          is itself in FIM format. Letting them fire during a normal left-to-right
          completion is what produces <|fim_*|> garbage mid-answer.
        - stop_ids halt a row on any of these tokens as well as <|eos|>. The stopping token
          is kept, so a closing fence ends up inside the returned text rather than beside
          it. See model/decode.py, which uses this to make trailing prose impossible.
        """
        assert self.tokenizer is not None, "Tokenizer not found"
        if not prompts:
            return []
        device = device or self.device
        enc = self.tokenizer._engine.encoder
        eos = enc.get(self.tokenizer.eos_token_id)
        vocab = self.config.vocab_size
        self.model.eval()

        encoded = [self.tokenizer.encode(p) for p in prompts]
        plens = [len(ids) for ids in encoded]
        B, T0 = len(encoded), max(plens)
        assert T0 + max_new_tokens <= self.config.block_size, (
            f"prompt {T0} + {max_new_tokens} new exceeds block_size "
            f"{self.config.block_size}; shorten the prompt or max_new_tokens"
        )

        # Preallocate the full width once. Filler is <|eos|> and is never read: every
        # row's live span is [0, lens[i]) and only grows.
        fill = eos if eos is not None else 0
        buf = torch.full((B, T0 + max_new_tokens), fill, dtype=torch.long, device=device)
        for i, ids in enumerate(encoded):
            buf[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        lens = torch.tensor(plens, dtype=torch.long, device=device)

        # Suppress FIM per row: a row whose own prompt is a FIM prompt keeps its sentinels.
        fim_ids = [enc[t] for t in ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>")
                   if t in enc]
        fim_rows = torch.tensor([suppress_fim and "<|fim_prefix|>" not in p for p in prompts],
                                device=device)
        fim_cols = (torch.tensor(fim_ids, dtype=torch.long, device=device)
                    if fim_ids and bool(fim_rows.any()) else None)

        rows = torch.arange(B, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        gen_len = torch.zeros(B, dtype=torch.long, device=device)   # tokens kept per row

        # <|eos|> plus any caller-supplied halt tokens, deduped.
        halts = sorted({*(stop_ids or []), *([eos] if eos is not None else [])})
        halt_t = torch.tensor(halts, device=device) if halts else None

        with self._autocast(device):
            for step in range(max_new_tokens):
                width = T0 + step
                logits, _ = self.model(buf[:, :width], pos=lens - 1)
                logits = logits[:, -1, :].float()

                if repetition_penalty != 1.0:
                    # Scatter only over live positions: park pad columns in a trash
                    # column at index vocab so filler <|eos|> is not penalised, which
                    # would bias short-prompt rows against ever stopping.
                    live = torch.arange(width, device=device)[None, :] < lens[:, None]
                    tok = torch.where(live, buf[:, :width], vocab)
                    seen = torch.zeros(B, vocab + 1, dtype=torch.bool, device=device)
                    seen.scatter_(1, tok, True)
                    seen = seen[:, :vocab]
                    logits = torch.where(seen, logits / repetition_penalty, logits)

                if fim_cols is not None:
                    logits[:, fim_cols] = logits[:, fim_cols].masked_fill(
                        fim_rows[:, None], float('-inf'))
                if eos is not None and step < min_new_tokens:
                    logits[:, eos] = float('-inf')   # don't stop on an empty answer

                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
                if top_p is not None:
                    sl, si = torch.sort(logits, descending=True)
                    probs = F.softmax(sl, dim=-1)
                    mask = torch.cumsum(probs, dim=-1) - probs > top_p  # keep up to top_p mass
                    sl = sl.masked_fill(mask, float('-inf'))
                    logits = torch.full_like(logits, float('-inf')).scatter(1, si, sl)

                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
                if eos is not None:
                    nxt = torch.where(finished, torch.full_like(nxt, eos), nxt)

                buf[rows, lens] = nxt
                lens += 1
                # Counted before the halt check, so the stopping token is kept - a closing
                # fence belongs inside the completion, not after it.
                gen_len += (~finished).long()      # a finished row contributes nothing more
                if halt_t is not None:
                    finished |= torch.isin(nxt, halt_t)
                    if bool(finished.all()):
                        break

        out = []
        for i, plen in enumerate(plens):
            start = 0 if return_prompt else plen
            out.append(self.tokenizer.decode(buf[i, start:plen + int(gen_len[i])].tolist()))
        return out

    def generate_text(self, prompt: str, **kwargs) -> str:
        """One prompt in, prompt + completion out. A batch of one over generate_batch."""
        kwargs.setdefault("return_prompt", True)
        return self.generate_batch([prompt], **kwargs)[0]
