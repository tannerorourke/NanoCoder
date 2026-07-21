"""Entrypoint

Rejection fine-tuning: train on the model's own execution-verified passing samples

    python -m nanocoder.model.train_rft \
        --base-repo <user>/NanoCoder-123M-sft --base-revision sft \
        --rft-repo  <user>/NanoCoder-rft \
        --repo-id   <user>/NanoCoder-123M-sft --revision rft

Same loop, masking, and config shape as SFT. Only the dataset differs.
- the targets are the model's own outputs, filtered by whether they actually ran and passed. 
- RFT's known failure is distribution collapse: training a model on its own successes narrows 
  it toward what it already did well, so pass@1 can rise while pass@5 falls (A pass@1 gain 
  bought entirely out of pass@5 is a model that got more confident, not more capable, at this 
  scale that distinction is the whole ballgame.) 
"""
import argparse
import os
from contextlib import nullcontext

import torch

from nanocoder.constants import resolve_device, seed_global
from nanocoder.model.nanocoder import NanoCoder
from nanocoder.model.train_sft import build_optimizer, encode_split
from nanocoder.training.config import RFTConfig
from nanocoder.training.sft_loop import sft_loop


def main():
    ap = argparse.ArgumentParser(description="Rejection fine-tuning on self-generated wins.")
    ap.add_argument("--base-repo", default="torq1/NanoCoder-123M-sft")
    ap.add_argument("--base-revision", default="sft")
    ap.add_argument("--rft-repo", default="torq1/NanoCoder-rft")
    ap.add_argument("--repo-id", default="torq1/NanoCoder-123M-sft")
    ap.add_argument("--revision", default="rft")
    ap.add_argument("--save-dir", default="./nano-coder-rft-export")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    seed_global()
    device = resolve_device(args.device)
    is_cuda = device == "cuda"

    cfg = RFTConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.base_lr = args.lr

    path = args.base_repo
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(args.base_repo, revision=args.base_revision)
    nano = NanoCoder.from_pretrained(path, device=device)
    tokenizer = nano.tokenizer
    eos_id = tokenizer._engine.encoder[tokenizer.eos_token_id]
    print(f"Loaded {args.base_repo}@{args.base_revision} on {device}")

    from datasets import load_dataset
    ds = load_dataset(args.rft_repo)
    train_ex = encode_split(tokenizer, list(ds["train"]), cfg.block_size, "train")
    val_ex = encode_split(tokenizer, list(ds["validation"]), cfg.block_size, "validation")

    optimizer = build_optimizer(nano.model, cfg, device)
    use_scaler = cfg.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if is_cuda else None
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=cfg.dtype)
                    if is_cuda else nullcontext())

    sft_loop(nano.model, optimizer, train_ex, val_ex, cfg, eos_id,
             device=device, autocast_ctx=autocast_ctx, scaler=scaler)

    nano.save_pretrained(args.save_dir)
    print(f"Saved model to {args.save_dir}")
    if args.no_push:
        print("--no-push set; skipping Hub upload.")
        return
    nano.push_to_hub(args.repo_id, private=args.private, revision=args.revision,
                     commit_message="Rejection fine-tuning")
    print(f"Pushed model to {args.repo_id} @ {args.revision}")


if __name__ == "__main__":
    main()
