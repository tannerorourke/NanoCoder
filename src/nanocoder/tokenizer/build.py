"""Entrypoint

Train the NanoCoder tokenizer and push it to the Hub. 

Notebook 2 ("NanoCoder: Tokenizer") is the narrated walkthrough of this script. 
Loads the NanoCoder-pretrain dataset (from Part 1), trains the Semi-Supervised BPE on a
sample of it, verifies the save/load round-trip, and pushes the tokenizer to the Hub.

    python -m nanocoder.tokenizer.build \
        --pretrain-repo <user>/NanoCoder-pretrain \
        --repo-id <user>/NanoCoder-tokenizer

Requires a write token (HF_TOKEN env var or `huggingface-cli login`).
"""
import argparse
import os

from nanocoder.constants import SUPTOK_CONFIG, seed_global
from nanocoder.data.config import DatasetConfig
from nanocoder.tokenizer.engine import SemiSupervisedBPE
from nanocoder.tokenizer.tokenizer import NanoCoderTokenizer
from nanocoder.tokenizer.trainer import train_bpe


def _load_training_texts(pretrain_repo: str, n: int) -> list[str]:
    """Pull the first 'n' train docs from the pushed pretrain dataset (streaming)."""
    from datasets import load_dataset
    ds = load_dataset(pretrain_repo, split="train", streaming=True)
    texts = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        texts.append(row["text"])
    return texts


def main():
    ap = argparse.ArgumentParser(description="Train & push the NanoCoder tokenizer.")
    ap.add_argument("--pretrain-repo", default="torq1/NanoCoder-pretrain",
                    help="Hub dataset to sample training text from (from Part 1).")
    ap.add_argument("--repo-id", default="torq1/NanoCoder-tokenizer",
                    help="Hub repo to push the tokenizer to.")
    ap.add_argument("--save-dir", default="./nano-coder-tokenizer")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    seed_global()
    dcfg = DatasetConfig()

    print(f"Sampling {dcfg.tokenizer_train_samples:,} docs from {args.pretrain_repo} ...")
    texts = _load_training_texts(args.pretrain_repo, dcfg.tokenizer_train_samples)

    # build + train the tokenizer
    engine = SemiSupervisedBPE(
        special_tokens=SUPTOK_CONFIG["special"],
        locked_kws=SUPTOK_CONFIG["locked"],
    )
    tokenizer = NanoCoderTokenizer(engine, SUPTOK_CONFIG)
    tknzr_samples = [tokenizer.preprocess(t, add_eos=True) for t in texts]
    tokenizer._engine = train_bpe(tokenizer._engine, tknzr_samples, dcfg.vocab_size)
    print(f"Trained tokenizer | vocab {tokenizer._engine.vocab_size:,}")

    # verify save/load round-trip before pushing
    tokenizer.save_pretrained(args.save_dir)
    reloaded = NanoCoderTokenizer.from_pretrained(args.save_dir)
    probe = "## Task\nWrite a function that sums a list.\n\n## Solution\n"
    assert reloaded.encode(probe) == tokenizer.encode(probe), "tokenizer encode drift"
    assert reloaded._engine._special_ids == tokenizer._engine._special_ids, "special ids lost"
    assert reloaded._engine.bpe_ranks == tokenizer._engine.bpe_ranks, "merge table drift"
    print("Round-trip verified. Files:", sorted(os.listdir(args.save_dir)))

    if args.no_push:
        print("--no-push set; skipping Hub upload.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=args.save_dir, repo_id=args.repo_id, repo_type="model")
    print(f"Pushed tokenizer to {args.repo_id}")


if __name__ == "__main__":
    main()
