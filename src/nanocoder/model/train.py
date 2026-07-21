"""Entrypoint

Train the NanoCoder base model and push it to Hugging Face.

Requires a write token (HF_TOKEN env var or `huggingface-cli login`)
GPU strongly recommended.

Notebook 3 ("NanoCoder: Base Model") is the narrated walkthrough of this script. 
- Loads the Part-1 tokenizer
- Streams + compiles the Part-1 corpus to uint16 token arrays,
- Builds and train the GPT model
- Pushes the full model (weights + config + tokenizer) to the Hub.

    python -m nanocoder.model.train \
        --tokenizer-repo <user>/NanoCoder-tokenizer \
        --pretrain-repo  <user>/NanoCoder-pretrain \
        --repo-id        <user>/NanoCoder-123M
"""
import argparse
import os
from contextlib import nullcontext

import torch

from nanocoder.constants import resolve_device, seed_global
from nanocoder.data.config import DatasetConfig
from nanocoder.data.corpus import compile_corpus
from nanocoder.data.sources import build_dataset
from nanocoder.model.gpt import GPT
from nanocoder.model.nanocoder import NanoCoder
from nanocoder.tokenizer.tokenizer import NanoCoderTokenizer
from nanocoder.training.config import NanoCoderConfig, TrainConfig
from nanocoder.training.loop import train_loop
from nanocoder.training.schedule import WarmupCosineAnnealing


def build_optimizer(model, tcfg, device):
    """AdamW with decay only on >=2D params; fused kernel on CUDA."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": tcfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tcfg.base_lr, betas=tcfg.betas,
                             fused=(device == "cuda"))


def main():
    ap = argparse.ArgumentParser(description="Train & push the NanoCoder base model.")
    ap.add_argument("--tokenizer-repo", default="torq1/NanoCoder-tokenizer")
    ap.add_argument("--pretrain-repo", default="torq1/NanoCoder-pretrain")  # reserved for a Dataset-backed loader
    ap.add_argument("--repo-id", default="torq1/nano-coder-123M")
    ap.add_argument("--save-dir", default="./nano-coder-export")
    ap.add_argument("--device", default=None, help="Override device (default: auto).")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    seed_global()
    device = resolve_device(args.device)
    is_cuda = device == "cuda"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # --- tokenizer (from Part 1)
    from huggingface_hub import snapshot_download
    tokenizer = NanoCoderTokenizer.from_pretrained(snapshot_download(args.tokenizer_repo))
    print(f"Loaded tokenizer | vocab {tokenizer._engine.vocab_size:,}")

    # --- corpus: stream + compile (token-level FIM happens here)
    dcfg = DatasetConfig()
    tcfg = TrainConfig()
    train_texts, val_texts = build_dataset(dcfg)
    train_ids, ttoks_train = compile_corpus(tokenizer, train_texts, tcfg.fim_rate, add_eos=True)
    val_ids, ttoks_val = compile_corpus(tokenizer, val_texts, fim_rate=0.0, add_eos=True)
    print(f"Train tokens: {ttoks_train:,} | Val tokens: {ttoks_val:,}")

    # --- model
    mcfg = NanoCoderConfig(vocab_size=tokenizer._engine.vocab_size)
    model = GPT(mcfg).to(device)
    nanocoder = NanoCoder(model, mcfg, tokenizer, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params on {device}")

    # --- optimizer / schedule / amp
    optimizer = build_optimizer(model, tcfg, device)
    scheduler = WarmupCosineAnnealing(optimizer, tcfg.min_lr, tcfg.warmup_iters,
                                      tcfg.max_iters, tcfg.lr_decay_iters)
    use_scaler = tcfg.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if is_cuda else None
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=tcfg.dtype)
                    if is_cuda else nullcontext())

    # --- train
    train_loop(
        model, optimizer, scheduler,
        train_ids, val_ids,
        mcfg.block_size, tcfg.batch_size,
        tcfg=tcfg, device=device, autocast_ctx=autocast_ctx,
        scaler=scaler, plot_handle=None,
    )

    # --- export
    nanocoder.save_pretrained(args.save_dir)
    print(f"Saved model to {args.save_dir}")
    if args.no_push:
        print("--no-push set; skipping Hub upload.")
        return
    nanocoder.push_to_hub(args.repo_id, private=args.private)
    print(f"Pushed model to {args.repo_id}")


if __name__ == "__main__":
    main()
