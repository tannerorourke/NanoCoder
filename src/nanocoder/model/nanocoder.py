"""NanoCoder: the portable model wrapper (weights + config + tokenizer).

Bundles the GPT, its config, and the tokenizer into one object that round-trips to
disk and to the Hub. save_pretrained writes weights, config, and tokenizer together,
so any checkpoint loads standalone - no revision depends on another.

Hub access goes through huggingface_hub directly. Dropping transformers' PushToHubMixin
removed the package's last dependency on transformers, which a from-scratch model has
no reason to carry.
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
from nanocoder.training.config import NanoCoderConfig


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

        return cls(model=model, config=config, tokenizer=tokenizer)

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        revision: str | None = None,
        commit_message: str = "Push NanoCoder",
        token: str | None = None,
    ) -> str:
        """
        Save to a temp dir and upload the folder.

        Replaces transformers' PushToHubMixin: that was the package's only dependency on
        transformers, for a method that is a create_repo + upload_folder underneath.
        'revision' targets a branch, which is how the post-trained stages are published
        side by side on one repo.
        """
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

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        min_new_tokens: int = 8,
        suppress_fim: bool = True,
        device: str | None = None,
    ) -> str:
        """ Autoregressive sampling.

        - top_p (nucleus) trims the long tail top_k alone leaves; 
        - repetition_penalty divides the logits of already-generated tokens to 
          curb the "function's function's" / "lo = lo" loops a small model falls into. 
        - suppress_fim masks the FIM sentinels: they're only meaningful when the 
          *prompt* is in FIM format, letting them fire during a normal left-to-right 
          completion is what produces <|fim_*|> garbage mid-answer.
        """
        assert self.tokenizer is not None, "Tokenizer not found"
        device = device or self.device
        enc = self.tokenizer._engine.encoder

        self.model.eval()
        ids = self.tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        eos = enc.get(self.tokenizer.eos_token_id)
        b_size = self.config.block_size

        # Only suppress FIM if the prompt itself isn't a FIM prompt.
        fim_ids = []
        if suppress_fim and "<|fim_prefix|>" not in prompt:
            fim_ids = [enc[t] for t in ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>")
                       if t in enc]

        actx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                if str(device).startswith('cuda') else nullcontext())
        with actx:
            for step in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= b_size else idx[:, -b_size:]
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :]

                # repetition penalty over tokens already in the context
                if repetition_penalty != 1.0:
                    for t in set(idx[0].tolist()):
                        logits[0, t] /= repetition_penalty

                if fim_ids:
                    logits[:, fim_ids] = float('-inf')
                if eos is not None and step < min_new_tokens:
                    logits[:, eos] = float('-inf')   # don't stop on an empty answer

                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
                if top_p is not None:
                    sl, si = torch.sort(logits, descending=True)
                    cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    mask = cum - F.softmax(sl, dim=-1) > top_p   # keep tokens up to top_p mass
                    sl[mask] = float('-inf')
                    logits = torch.full_like(logits, float('-inf')).scatter(1, si, sl)

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                if eos is not None and idx_next.item() == eos:
                    break

        return self.tokenizer.decode(idx[0].tolist())
