"""Entrypoint

Train the base model -> push to Hugging Face (requires a HF write token)

    python -m nanocoder.model.pretrain \
        --tokenizer-repo <user>/NanoCoder-tokenizer \
        --pretrain-repo  <user>/NanoCoder-pretrain \
        --repo-id        <user>/NanoCoder-123M-pretrain

GPU strongly recommended.
"""
import argparse
import os
from contextlib import nullcontext

import torch

from nanocoder.constants import resolve_device, seed_global
from nanocoder.data.config import DatasetConfig
from nanocoder.data.corpus import compile_corpus, corpus_fingerprint
from nanocoder.data.sources import build_dataset
from nanocoder.model.gpt import GPT
from nanocoder.model.nanocoder import NanoCoder
from nanocoder.tokenizer.tokenizer import NanoCoderTokenizer
from nanocoder.training.checkpoint import load_checkpoint, resolve_resume
from nanocoder.training.config import NanoCoderConfig, TrainConfig
from nanocoder.training.loop import CKPT_PREFIX, train_loop
from nanocoder.training.schedule import WarmupCosineAnnealing


# -- Lazy view of one split's text column. compile_corpus only iterates, and datasets keeps
#    the Arrow table memory-mapped, so this never materialises the corpus in RAM. __len__ is
#    what keeps the tqdm total.
class _TextColumn:
    def __init__(self, split):
        self._split = split

    def __len__(self):
        return len(self._split)

    def __iter__(self):
        for row in self._split:
            yield row["text"]


# -- Trains on the documents build_pretrain pushed, not a fresh stream of the seven sources.
def _load_corpus_texts(pretrain_repo: str):
    from datasets import load_dataset
    ds = load_dataset(pretrain_repo)
    return _TextColumn(ds["train"]), _TextColumn(ds["validation"])


# -- AdamW, decay on >=2D params only; fused kernel on CUDA.
def build_optimizer(model, tcfg, device):
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
    ap.add_argument("--pretrain-repo", default="torq1/NanoCoder-pretrain",
                    help="Hub dataset to train on (pushed by build_pretrain).")
    ap.add_argument("--rebuild-corpus", action="store_true",
                    help="Re-stream the seven sources instead of loading --pretrain-repo.")
    ap.add_argument("--repo-id", default="torq1/NanoCoder-123M-pretrain")
    ap.add_argument("--save-dir", default="./nano-coder-export")
    ap.add_argument("--device", default=None, help="Override device (default: auto).")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--checkpoint-dir", default="./checkpoints")
    ap.add_argument("--resume", default=None,
                    help="Checkpoint path, or 'auto' for the newest in --checkpoint-dir. "
                         "'auto' on an empty directory starts from scratch.")
    ap.add_argument("--corpus-cache-dir", default="./corpus-cache",
                    help="Where to keep the compiled token stream (~4GB). Pass '' to disable.")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum-steps", type=int, default=None,
                    help="Pair with --batch-size to hold tokens/step fixed while trading "
                         "micro-batch size against accumulation depth.")
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
    if args.batch_size:
        tcfg.batch_size = args.batch_size
    if args.grad_accum_steps:
        tcfg.grad_accum_steps = args.grad_accum_steps
    if args.rebuild_corpus:
        train_texts, val_texts = build_dataset(dcfg)
    else:
        train_texts, val_texts = _load_corpus_texts(args.pretrain_repo)
    print(f"Corpus: {len(train_texts):,} train / {len(val_texts):,} val docs")

    # -- Tokenizing takes multiple  hours and produces the same ids, so a restart 
    #    should reload it rather than repeat it.
    def cache_for(split: str, n_docs: int, fim_rate: float):
        if not args.corpus_cache_dir:
            return None
        key = corpus_fingerprint(tokenizer, n_docs, split=split, fim_rate=fim_rate)
        return os.path.join(args.corpus_cache_dir, f"{split}-{key}.npy")

    train_ids, ttoks_train = compile_corpus(
        tokenizer, train_texts, tcfg.fim_rate, add_eos=True,
        cache_path=cache_for("train", len(train_texts), tcfg.fim_rate))
    val_ids, ttoks_val = compile_corpus(
        tokenizer, val_texts, fim_rate=0.0, add_eos=True,
        cache_path=cache_for("val", len(val_texts), 0.0))
    print(f"Train tokens: {ttoks_train:,} | Val tokens: {ttoks_val:,}")

    # --- model
    mcfg = NanoCoderConfig(vocab_size=tokenizer._engine.vocab_size)
    model = GPT(mcfg).to(device)
    nanocoder = NanoCoder(model, mcfg, tokenizer, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    # -- tokens/step is the quantity that must stay fixed when batch and accumulation are
    #    retuned for throughput; printing it makes an accidental change to it visible
    print(f"Model: {n_params / 1e6:.1f}M params on {device} | "
          f"{tcfg.batch_size * mcfg.block_size * tcfg.grad_accum_steps:,} tokens/step")

    # --- optimizer / schedule / amp
    optimizer = build_optimizer(model, tcfg, device)
    scheduler = WarmupCosineAnnealing(optimizer, tcfg.min_lr, tcfg.warmup_iters,
                                      tcfg.max_iters, tcfg.lr_decay_iters)
    use_scaler = tcfg.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if is_cuda else None
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=tcfg.dtype)
                    if is_cuda else nullcontext())

    # -- restore after the optimizer exists, so its state loads onto the live parameters
    start_iter = 0
    resume_path = resolve_resume(args.resume, args.checkpoint_dir, CKPT_PREFIX)
    if resume_path:
        start_iter, _ = load_checkpoint(resume_path, model, optimizer, scaler)
        scheduler.set_iter(start_iter)
        print(f"Resumed {resume_path} at iter {start_iter}/{tcfg.max_iters}")

    # --- train
    train_loop(
        model, optimizer, scheduler,
        train_ids, val_ids,
        mcfg.block_size, tcfg.batch_size,
        tcfg=tcfg, device=device, autocast_ctx=autocast_ctx,
        scaler=scaler, plot_handle=None,
        start_iter=start_iter, checkpoint_dir=args.checkpoint_dir,
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
