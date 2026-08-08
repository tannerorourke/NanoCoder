"""Entrypoint

Build the NanoCoder-pretrain dataset and push it to HF (requires HF write token)

    python -m nanocoder.data.build_pretrain --repo-id <user>/NanoCoder-pretrain

Notebook 1 ("NanoCoder: Dataset") is the walkthrough of this script
- Streams + filters the sources (nanocoder.data.sources.build_dataset)
- Materialises train/val text
- Pushes a DatasetDict to HuggingFace Datasets
"""
import argparse
import gc

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
    ap.add_argument("--save-dir", default=None,
                    help="save_to_disk here before pushing, so a failed upload costs the "
                         "upload rather than the whole stream.")
    args = ap.parse_args()

    seed_global()
    dcfg = DatasetConfig()
    if args.max_samples is not None:
        dcfg.max_samples = args.max_samples

    train_texts, val_texts = build_dataset(dcfg)

    from datasets import Dataset, DatasetDict, Features, Value

    # -- large_string, not string: Arrow indexes string data with int32 offsets, capping one
    #    array near 2.1GB. The corpus is several times that and overflows on concatenation.
    features = Features({"text": Value("large_string")})

    # -- One split at a time, dropping each source list once converted. The Python strings and
    #    their Arrow copy are both resident during a conversion, which is this script's peak.
    ds_train = Dataset.from_dict({"text": train_texts}, features=features)
    del train_texts; gc.collect()
    ds_val = Dataset.from_dict({"text": val_texts}, features=features)
    del val_texts; gc.collect()
    ds = DatasetDict({"train": ds_train, "validation": ds_val})

    if args.save_dir:
        ds.save_to_disk(args.save_dir)
        print(f"Saved corpus to {args.save_dir}")

    # -- Conversion happens above this exit, so --no-push still exercises it. The Arrow
    #    overflow only appears at full corpus size, and skipping the step would hide it.
    if args.no_push:
        print(f"--no-push set; built {len(ds['train']):,} train / "
              f"{len(ds['validation']):,} val docs.")
        return

    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"Pushed NanoCoder-pretrain to {args.repo_id} "
          f"({len(ds['train']):,} train / {len(ds['validation']):,} val docs)")


if __name__ == "__main__":
    main()
