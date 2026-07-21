"""Entrypoint

Supervised fine-tuning: instruction pairs, loss masked to the completion.

    python -m nanocoder.model.train_sft \
        --base-repo <user>/NanoCoder-123M-pretrain \
        --sft-repo  <user>/NanoCoder-sft \
        --repo-id   <user>/NanoCoder-123M-sft

Pretrained base 
-> train on the code-only instruction corpus 
-> push to 'sft' revision on HuggingFace
"""
import argparse
import os
from contextlib import nullcontext

import torch

from nanocoder.constants import resolve_device, seed_global
from nanocoder.data.sft import decode_masked, encode_example, build_labels
from nanocoder.model.nanocoder import NanoCoder
from nanocoder.training.config import SFTConfig
from nanocoder.training.sft_loop import sft_loop


def build_optimizer(model, cfg, device):
    """AdamW with decay only on >=2D params; fused kernel on CUDA. Mirrors train.py """
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.base_lr, betas=cfg.betas,
                             fused=(device == "cuda"))


def encode_split(tokenizer, rows, block_size: int, desc: str):
    """ Tokenize prompt/completion rows, dropping anything that doesn't fit in the block """
    from tqdm.auto import tqdm
    kept, dropped = [], 0
    for row in tqdm(rows, desc=desc):
        ex = encode_example(tokenizer, row["prompt"], row["completion"])
        if len(ex) + 1 > block_size + 1:
            dropped += 1
            continue
        kept.append(ex)
    pct = 100 * dropped / max(1, len(rows))
    print(f"{desc}: {len(kept):,} examples | dropped {dropped:,} ({pct:.1f}%) over {block_size} tokens")
    return kept


def supervision_report(examples, cfg):
    """
    Fraction of trained tokens which carry gradient. Helps determine whether block_size and
    the corpus length filter are matched to each other. low figure means most of the compute
    is being spent on <|eos|> filler.
    """
    total = sup = 0
    for ex in examples[:2000]:
        total += cfg.block_size
        sup += len(ex) - ex.prompt_len + 1
    return sup / max(1, total)


def main():
    ap = argparse.ArgumentParser(description="Instruction-tune NanoCoder.")
    ap.add_argument("--base-repo", default="torq1/NanoCoder-123M-pretrain")
    ap.add_argument("--sft-repo", default="torq1/NanoCoder-sft")
    ap.add_argument("--repo-id", default="torq1/NanoCoder-123M-sft")
    ap.add_argument("--revision", default="sft", help="Branch to push to.")
    ap.add_argument("--save-dir", default="./nano-coder-sft-export")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Cap training rows, for a smoke run.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    seed_global()
    device = resolve_device(args.device)
    is_cuda = device == "cuda"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cfg = SFTConfig()
    for attr, val in (("epochs", args.epochs), ("batch_size", args.batch_size),
                      ("base_lr", args.lr)):
        if val is not None:
            setattr(cfg, attr, val)

    # --- base model (weights + tokenizer travel together)
    from huggingface_hub import snapshot_download
    base = args.base_repo if os.path.isdir(args.base_repo) else snapshot_download(args.base_repo)
    nano = NanoCoder.from_pretrained(base, device=device)
    tokenizer = nano.tokenizer
    assert tokenizer is not None, "Base checkpoint has no tokenizer"
    eos_id = tokenizer._engine.encoder[tokenizer.eos_token_id]
    print(f"Loaded {args.base_repo} | "
          f"{sum(p.numel() for p in nano.model.parameters()) / 1e6:.1f}M params on {device}")

    # --- dataset
    from datasets import load_dataset
    ds = load_dataset(args.sft_repo)
    train_rows = list(ds["train"])[:args.limit] if args.limit else list(ds["train"])
    val_rows = list(ds["validation"])

    train_ex = encode_split(tokenizer, train_rows, cfg.block_size, "train")
    val_ex = encode_split(tokenizer, val_rows, cfg.block_size, "validation")
    print(f"Supervised token fraction: {supervision_report(train_ex, cfg):.1%} "
          f"(the rest is prompt and padding)")

    # --- the check that catches the silent failure
    x, y = build_labels(train_ex[0], cfg.block_size, eos_id)
    view = decode_masked(tokenizer, x, y)
    print("\n--- masking check ---")
    print(f"PROMPT (not supervised):\n{view['prompt']}")
    print(f"\nSUPERVISED SPAN ({view['n_supervised']} tokens):\n{view['supervised']}")
    print("--- end masking check ---\n")

    # --- train
    optimizer = build_optimizer(nano.model, cfg, device)
    use_scaler = cfg.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if is_cuda else None
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=cfg.dtype)
                    if is_cuda else nullcontext())

    sft_loop(nano.model, optimizer, train_ex, val_ex, cfg, eos_id,
             device=device, autocast_ctx=autocast_ctx, scaler=scaler)

    # --- export
    nano.save_pretrained(args.save_dir)
    print(f"Saved model to {args.save_dir}")
    if args.no_push:
        print("--no-push set; skipping Hub upload.")
        return
    nano.push_to_hub(args.repo_id, private=args.private, revision=args.revision,
                     commit_message="Supervised fine-tuning")
    print(f"Pushed model to {args.repo_id} @ {args.revision}")


if __name__ == "__main__":
    main()
