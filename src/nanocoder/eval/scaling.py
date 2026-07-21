"""Entrypoint

Run the eval harness across the intermediate pretraining checkpoints.

    python -m nanocoder.eval.scaling \
        --tokenizer-repo <user>/NanoCoder-tokenizer \
        --checkpoints checkpoints/nanocoder_2000.pt checkpoints/nanocoder_4000.pt \
                      checkpoints/nanocoder_6000.pt

NanoCoder is knowingly undertrained, and the honest way to present that is to show it 
rather than footnote it. If parse rate and pass@1 are still climbing at the last checkpoint, 
the reader sees what a compute-limited model looks like mid-ascent, and "Chinchilla is the 
wrong yardstick when the size is fixed, not the compute" stops being a citation and becomes a 
measurement made in this repository.

Correctness and parse rate are plotted as separate series on purpose. They come apart at
this scale - form is learned long before function - that gap is the thing the
post-training stages are aimed at.
"""
import argparse
import json
import os
import re

from nanocoder.eval.harness import evaluate, report
from nanocoder.eval.tasks import LOADERS

TOKENS_PER_ITER = 393_216       # batch 8 x block 2048 x 24 accum


def iter_of(path: str) -> int:
    """Recover the iteration count from the checkpoint filename."""
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_checkpoint(path: str, tokenizer, device: str):
    """
    Rebuild a NanoCoder around a raw training checkpoint.

    train_loop saves {'model', 'optimizer'} state dicts, not a save_pretrained directory,
    so the config and tokenizer have to be supplied here rather than read from disk.
    """
    import torch
    from nanocoder.model.gpt import GPT
    from nanocoder.model.nanocoder import NanoCoder
    from nanocoder.training.config import NanoCoderConfig

    cfg = NanoCoderConfig(vocab_size=tokenizer._engine.vocab_size)
    model = GPT(cfg)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device).eval()
    return NanoCoder(model, cfg, tokenizer, device=device)


def plot_curve(points, out_path: str = "scaling_curve.png"):
    """Parse rate and pass@1 against tokens seen. matplotlib lives behind the [plot] extra."""
    import matplotlib.pyplot as plt

    toks = [p["tokens"] / 1e9 for p in points]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(toks, [p["parse_rate"] for p in points], marker="o", label="parse rate")
    ax.plot(toks, [p["pass@1"] for p in points], marker="s", label="pass@1")
    ax.set_xlabel("training tokens (B)")
    ax.set_ylabel("fraction")
    ax.set_title("NanoCoder-123M-pretrain: correctness vs tokens seen")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Eval sweep across pretraining checkpoints.")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="Paths to nanocoder_*.pt")
    ap.add_argument("--tokenizer-repo", default="torq1/NanoCoder-tokenizer")
    ap.add_argument("--benchmark", default="mbpp", choices=sorted(LOADERS))
    ap.add_argument("--limit", type=int, default=200, help="Tasks per checkpoint.")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    missing = [p for p in args.checkpoints if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Missing checkpoint(s): {missing}\n")

    from nanocoder.constants import resolve_device, seed_global
    from nanocoder.tokenizer.tokenizer import NanoCoderTokenizer
    from huggingface_hub import snapshot_download

    seed_global()
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = NanoCoderTokenizer.from_pretrained(snapshot_download(args.tokenizer_repo))

    loader = LOADERS[args.benchmark]
    tasks = (loader(split="test", limit=args.limit) if args.benchmark == "mbpp"
             else loader(limit=args.limit))

    points = []
    for path in sorted(args.checkpoints, key=iter_of):
        it = iter_of(path)
        nano = load_checkpoint(path, tokenizer, device)
        summary = evaluate(
            nano, tasks, n_samples=args.n_samples, batch_size=args.batch_size,
            out_path=os.path.join(args.out_dir, f"scaling_{it}_{args.benchmark}.jsonl"),
        )
        report(f"iter {it} ({it * TOKENS_PER_ITER / 1e9:.2f}B tokens)", summary)
        points.append({"iter": it, "tokens": it * TOKENS_PER_ITER, **summary})
        del nano

    with open(os.path.join(args.out_dir, "scaling_curve.json"), "w") as f:
        json.dump(points, f, indent=2)
    if not args.no_plot:
        plot_curve(points, os.path.join(args.out_dir, "scaling_curve.png"))


if __name__ == "__main__":
    main()
