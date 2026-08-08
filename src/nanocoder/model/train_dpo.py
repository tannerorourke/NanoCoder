"""Entrypoint

Preference tuning on execution-labelled pairs.

    python -m nanocoder.model.train_dpo \
        --base-repo <user>/NanoCoder-123M-sft --base-revision rft \
        --prefs-repo <user>/NanoCoder-prefs \
        --repo-id <user>/NanoCoder-123M-sft --revision main

Run data/build_prefs.py --probe first.

The policy starts from the **RFT** checkpoint, not the SFT one - DPO refines the last stage,
not a model that stage has already superseded. The reference is a frozen copy of that same
starting point.
"""
import argparse
import os
from contextlib import nullcontext

import torch

from nanocoder.constants import resolve_device, seed_global
from nanocoder.data.preference import load_pairs
from nanocoder.model.nanocoder import NanoCoder
from nanocoder.model.train_sft import build_optimizer
from nanocoder.training.checkpoint import load_checkpoint, resolve_resume
from nanocoder.training.config import DPOConfig
from nanocoder.training.dpo import dpo_loop, freeze_reference


def main():
    ap = argparse.ArgumentParser(description="DPO on execution-labelled preference pairs.")
    ap.add_argument("--base-repo", default="torq1/NanoCoder-123M-sft")
    ap.add_argument("--base-revision", default="rft",
                    help="DPO refines the last stage; default is the RFT checkpoint.")
    ap.add_argument("--prefs-repo", default="torq1/NanoCoder-prefs")
    ap.add_argument("--repo-id", default="torq1/NanoCoder-123M-sft")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--save-dir", default="./nano-coder-dpo-export")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--checkpoint-dir", default="./checkpoints")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint path, or 'auto' for the newest in --checkpoint-dir.")
    args = ap.parse_args()

    seed_global()
    device = resolve_device(args.device)
    is_cuda = device == "cuda"

    cfg = DPOConfig()
    for attr, val in (("beta", args.beta), ("base_lr", args.lr), ("epochs", args.epochs)):
        if val is not None:
            setattr(cfg, attr, val)

    path = args.base_repo
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(args.base_repo, revision=args.base_revision)
    nano = NanoCoder.from_pretrained(path, device=device)
    tokenizer = nano.tokenizer
    eos_id = tokenizer._engine.encoder[tokenizer.eos_token_id]

    ref = freeze_reference(nano.model)
    print(f"Loaded {args.base_repo}@{args.base_revision} as policy and frozen reference")

    from datasets import load_dataset
    ds = load_dataset(args.prefs_repo)
    train_pairs = load_pairs(tokenizer, list(ds["train"]), cfg.block_size, "train")
    val_pairs = load_pairs(tokenizer, list(ds["validation"]), cfg.block_size, "validation")
    if len(train_pairs) < 300:
        print(f"WARNING: {len(train_pairs)} training pairs is below the threshold the probe "
              "was meant to enforce. Treat any result here as indicative at best.")

    optimizer = build_optimizer(nano.model, cfg, device)
    use_scaler = cfg.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if is_cuda else None
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=cfg.dtype)
                    if is_cuda else nullcontext())

    # -- must stay below freeze_reference: the anchor is the base policy, and restoring
    #    first would make the reference a copy of the partly-tuned one, quietly weakening
    #    the KL term that DPO relies on
    start_step = 0
    resume_path = resolve_resume(args.resume, args.checkpoint_dir, "dpo")
    if resume_path:
        start_step, _ = load_checkpoint(resume_path, nano.model, optimizer, scaler)
        print(f"Resumed {resume_path} at step {start_step}")

    dpo_loop(nano.model, ref, optimizer, train_pairs, val_pairs, cfg, eos_id,
             device=device, autocast_ctx=autocast_ctx, scaler=scaler,
             checkpoint_dir=args.checkpoint_dir, start_step=start_step, ckpt_prefix="dpo")

    nano.save_pretrained(args.save_dir)
    print(f"Saved model to {args.save_dir}")
    if args.no_push:
        print("--no-push set; skipping Hub upload.")
        return
    nano.push_to_hub(args.repo_id, private=args.private, revision=args.revision,
                     commit_message="Preference tuning (DPO)")
    print(f"Pushed model to {args.repo_id} @ {args.revision}")


if __name__ == "__main__":
    main()
