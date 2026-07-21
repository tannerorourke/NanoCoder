"""Entrypoint

Build the NanoCoder-pretrain dataset and push it to HF (requires HF write token)

    python -m nanocoder.data.build_pretrain --repo-id <user>/NanoCoder-pretrain

Notebook 1 ("NanoCoder: Dataset") is the walkthrough of this script
- Streams + filters the sources (nanocoder.data.sources.build_dataset)
- Materialises train/val text
- Pushes a DatasetDict to HuggingFace Datasets
"""
import argparse

from nanocoder.constants import seed_global
from nanocoder.data.config import DatasetConfig
from nanocoder.data.sources import build_dataset


def main():
    ap = argparse.ArgumentParser(description="Build & push the NanoCoder-pretrain dataset.")
    ap.add_argument("--repo-id", default="torq1/NanoCoder-pretrain",
                    help="Hub dataset repo to push to.")
    ap.add_argument("--private", action="store_true", help="Create the repo as private.")
    ap.add_argument("--no-push", action="store_true",
                    help="Build and report only; skip HF push.")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Override DatasetConfig.max_samples (e.g. small smoke run).")
    args = ap.parse_args()

    seed_global()
    dcfg = DatasetConfig()
    if args.max_samples is not None:
        dcfg.max_samples = args.max_samples

    train_texts, val_texts = build_dataset(dcfg)

    if args.no_push:
        print("--no-push set; skipping Hub upload.")
        return

    from datasets import Dataset, DatasetDict
    ds = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts}),
        "validation": Dataset.from_dict({"text": val_texts}),
    })
    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"Pushed NanoCoder-pretrain to {args.repo_id} "
          f"({len(train_texts):,} train / {len(val_texts):,} val docs)")


if __name__ == "__main__":
    main()
