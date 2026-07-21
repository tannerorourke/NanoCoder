"""Entrypoint

pass@k over a checkpoint, with the failure breakdown that makes the number interpretable.

    python -m nanocoder.eval.harness \
        --model <user>/NanoCoder-123M \
        --benchmark mbpp --n-samples 5 --out results/base_mbpp.jsonl

Reports three things, and the second and third are why this exists:

- pass@1 / pass@5 (Chen et al.'s unbiased estimator, not a naive count).
- parse rate, tracked as its own axis. At 123M these decouple hard from correctness, and
  holding them apart is what separates what SFT contributed from what DPO did.
- the outcome histogram. "pass@1 is 0.02" says nothing actionable; "61% SYNTAX_ERROR" says
  the model has not learned the output shape, which is a data problem, not a capacity one.

Results stream to JSONL one sample at a time. Colab sessions disconnect, and a sweep that
loses everything at 80% is a sweep nobody re-runs - so re-invoking with the same --out
skips whatever is already on disk.
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from nanocoder.eval.sandbox import Outcome, run_candidate
from nanocoder.eval.tasks import LOADERS, extract_code


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k: 1 - C(n-c, k) / C(n, k), the chance k draws from n samples miss all c
    correct ones. Estimating it as "did any of the first k pass" is biased and noisy at the
    sample counts a Colab budget allows.
    """
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _load_done(path: str) -> set[tuple[str, int]]:
    """Sample keys already on disk, so a reconnected session resumes instead of restarting."""
    if not path or not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((rec["task_id"], rec["sample"]))
            except (json.JSONDecodeError, KeyError):
                continue        # a torn last line from a killed session; drop it
    return done


def evaluate(
    nano,
    tasks,
    n_samples: int = 5,
    temperature: float = 0.8,
    max_new_tokens: int = 384,
    batch_size: int = 32,
    out_path: str | None = None,
    timeout: float = 10.0,
    exec_workers: int = 4,
    gen_kwargs: dict | None = None,
) -> dict:
    """
    Sample n completions per task, execute each, and aggregate.

    Work is flattened to (task, sample) pairs and sorted by prompt length before batching.
    Rows in a batch are padded to the longest prompt in it, so length-sorting means short
    tasks are not dragged to a long task's width - a real saving when the harness has no KV
    cache and every step re-reads the whole context.
    """
    done = _load_done(out_path) if out_path else set()
    work = [(t, s) for t in tasks for s in range(n_samples) if (t.task_id, s) not in done]
    work.sort(key=lambda ts: len(ts[0].prompt))
    if done:
        print(f"Resuming: {len(done)} samples already on disk, {len(work)} to go")

    fh = open(out_path, "a") if out_path else None
    records = []
    try:
        for start in tqdm(range(0, len(work), batch_size), desc="sampling"):
            chunk = work[start:start + batch_size]
            completions = nano.generate_batch(
                [t.prompt for t, _ in chunk],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **(gen_kwargs or {}),
            )

            # Execution is subprocess-bound, so threads overlap it despite the GIL.
            codes = [extract_code(c) for c in completions]
            with ThreadPoolExecutor(max_workers=exec_workers) as pool:
                results = list(pool.map(
                    lambda a: run_candidate(a[0], a[1].test_code, a[1].setup_code, timeout),
                    [(code, t) for code, (t, _) in zip(codes, chunk)],
                ))

            for (task, sample), code, res in zip(chunk, codes, results):
                rec = {
                    "task_id": task.task_id,
                    "sample": sample,
                    "outcome": res.outcome.value,
                    "detail": res.detail,
                    # parse rate is its own axis: SYNTAX_ERROR is the only outcome that
                    # implies unparseable, everything else got as far as running
                    "parses": res.outcome is not Outcome.SYNTAX_ERROR,
                    "code": code,
                }
                records.append(rec)
                if fh:
                    fh.write(json.dumps(rec) + "\n")
            if fh:
                fh.flush()
    finally:
        if fh:
            fh.close()

    if out_path:                    # fold in whatever a previous session already wrote
        records = [json.loads(l) for l in open(out_path)]
    return aggregate(records, n_samples)


def aggregate(records, n_samples: int) -> dict:
    """Roll per-sample records into pass@k, parse rate, and the outcome histogram."""
    passes = defaultdict(int)
    counts = defaultdict(int)
    parses = 0
    hist = Counter()
    for r in records:
        counts[r["task_id"]] += 1
        passes[r["task_id"]] += int(r["outcome"] == Outcome.PASS.value)
        parses += int(r["parses"])
        hist[r["outcome"]] += 1

    total = sum(counts.values()) or 1
    ks = [k for k in (1, 5, 10) if k <= n_samples]
    scores = {
        f"pass@{k}": float(np.mean([pass_at_k(counts[t], passes[t], k) for t in counts]))
        for k in ks
    }
    return {
        **scores,
        "parse_rate": parses / total,
        "n_tasks": len(counts),
        "n_samples_total": total,
        "solved_tasks": sum(1 for t in counts if passes[t] > 0),
        "outcomes": {k: v / total for k, v in hist.most_common()},
    }


def report(name: str, summary: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  tasks {summary['n_tasks']} | samples {summary['n_samples_total']} "
          f"| solved {summary['solved_tasks']}")
    for k, v in summary.items():
        if k.startswith("pass@"):
            print(f"  {k:<10} {v:.4f}")
    print(f"  parse_rate {summary['parse_rate']:.4f}")
    print("  outcomes:")
    for outcome, frac in summary["outcomes"].items():
        print(f"    {outcome:<14} {frac:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Execution eval for a NanoCoder checkpoint.")
    ap.add_argument("--model", required=True, help="Hub repo id or a local directory.")
    ap.add_argument("--revision", default=None, help="Hub branch, e.g. sft or rft.")
    ap.add_argument("--benchmark", default="mbpp", choices=sorted(LOADERS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="Cap tasks, for a timing probe.")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--out", default=None, help="JSONL path; also the resume file.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from nanocoder.constants import resolve_device, seed_global
    from nanocoder.model.nanocoder import NanoCoder

    seed_global()
    device = resolve_device(args.device)

    path = args.model
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(args.model, revision=args.revision)
    nano = NanoCoder.from_pretrained(path, device=device)

    loader = LOADERS[args.benchmark]
    tasks = (loader(split=args.split, limit=args.limit) if args.benchmark == "mbpp"
             else loader(limit=args.limit))
    print(f"{args.benchmark}: {len(tasks)} tasks x {args.n_samples} samples on {device}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary = evaluate(
        nano, tasks,
        n_samples=args.n_samples, temperature=args.temperature,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        out_path=args.out, timeout=args.timeout,
    )
    report(f"{args.model}{'@' + args.revision if args.revision else ''} / {args.benchmark}",
           summary)
    if args.out:
        with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
