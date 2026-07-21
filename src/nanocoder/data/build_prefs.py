"""Entrypoint

Run one sampling pass over a task pool to produce two datasets.

    python -m nanocoder.data.build_prefs --model <user>/NanoCoder-123M-sft --probe
    python -m nanocoder.data.build_prefs --model <user>/NanoCoder-123M-sft \
        --rft-repo <user>/NanoCoder-rft --prefs-repo <user>/NanoCoder-prefs

Sample k completions per task, run each against task's tests, and sort the labelled results in 2 ways:
- **NanoCoder-rft** - every passing sample. Needs only wins, so tasks where all k pass still
  contribute. This is the robust half: on-policy targets, verified by execution rather than
  by a proxy.
- **NanoCoder-prefs** - chosen/rejected pairs, from tasks that produced both. Strictly
  fewer tasks qualify, which is the risk the probe exists to measure.

**Run --probe first.** The open question at 123M is not whether the model can pass tests but
whether enough *tasks* land in the both-pass-and-fail regime to make a preference dataset.
"""
import argparse
import ast
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from nanocoder.eval.sandbox import Outcome, run_candidate
from nanocoder.eval.tasks import Task, load_mbpp
from nanocoder.model.decode import constrained_batch

_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)", re.MULTILINE)


# ---------------------------------------------------------------------------- task pool

def harvest_self_testing(sft_repo: str, limit: int = 4_000) -> list[Task]:
    """
    Mine self-testing tasks out of the SFT dataset's held-out 'tasks' split. Competition 
    sets (APPS, CodeContests) deliberately excluded (at this scale, we ain't passing)

    MBPP's train and validation splits total ~464 tasks, thin for a preference dataset.
    Many instruct solutions ship their own asserts; where they do, the task is already 
    test-bearing and costs nothing to harvest. The asserts become the test suite and
    the rest of the solution is discarded.
    """
    from datasets import load_dataset
    ds = load_dataset(sft_repo, split="tasks")
    tasks = []
    for i, row in enumerate(ds):
        if len(tasks) >= limit:
            break
        code = row["completion"].strip("`").replace("python\n", "", 1)
        asserts = [ln for ln in code.splitlines() if ln.strip().startswith("assert ")]
        names = _DEF_RE.findall(code)
        if not asserts or not names:
            continue
        body = "\n".join(ln for ln in code.splitlines() if not ln.strip().startswith("assert "))
        try:
            ast.parse(body)
        except (SyntaxError, ValueError):
            continue
        tasks.append(Task(
            task_id=f"sft/{i}",
            prompt=row["prompt"],
            test_code="\n".join(asserts),
            entry_point=names[-1],
            reference=body,
        ))
    print(f"Harvested {len(tasks):,} self-testing tasks from {sft_repo}:tasks")
    return tasks


def build_pool(sft_repo: str | None, mbpp_only: bool, limit: int | None) -> list[Task]:
    """MBPP train + validation. The test split is reserved for eval and never sampled."""
    pool = load_mbpp(split="train") + load_mbpp(split="validation")
    print(f"MBPP train+validation: {len(pool)} tasks (test split reserved for eval)")
    if sft_repo and not mbpp_only:
        pool += harvest_self_testing(sft_repo)
    return pool[:limit] if limit else pool


# ------------------------------------------------------------------------- sampling run

def _load_done(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["task_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def sample_and_label(nano, tasks, k: int, temperature: float, max_new_tokens: int,
                     batch_size: int, timeout: float, out_path: str | None,
                     exec_workers: int = 4):
    """ Sample k per task, execute all of them, and stream labelled samples to JSONL """
    done = _load_done(out_path) if out_path else set()
    todo = [t for t in tasks if t.task_id not in done]
    if done:
        print(f"Resuming: {len(done)} tasks labelled, {len(todo)} to go")

    fh = open(out_path, "a") if out_path else None
    records = []
    try:
        per_batch = max(1, batch_size // k)      # whole tasks per batch, never a split group
        for start in tqdm(range(0, len(todo), per_batch), desc="sampling"):
            group = todo[start:start + per_batch]
            prompts = [t.prompt for t in group for _ in range(k)]
            codes = constrained_batch(nano, prompts, max_new_tokens=max_new_tokens,
                                      temperature=temperature)

            with ThreadPoolExecutor(max_workers=exec_workers) as pool:
                results = list(pool.map(
                    lambda a: run_candidate(a[0], a[1].test_code, a[1].setup_code, timeout),
                    [(codes[i * k + j], group[i]) for i in range(len(group)) for j in range(k)],
                ))

            for i, task in enumerate(group):
                rec = {
                    "task_id": task.task_id,
                    "prompt": task.prompt,
                    "samples": [
                        {"code": codes[i * k + j],
                         "outcome": results[i * k + j].outcome.value}
                        for j in range(k)
                    ],
                }
                records.append(rec)
                if fh:
                    fh.write(json.dumps(rec) + "\n")
            if fh:
                fh.flush()
    finally:
        if fh:
            fh.close()

    if out_path:
        records = [json.loads(l) for l in open(out_path)]
    return records


# ------------------------------------------------------------------------------- yields

def probe_report(records, k: int, pool_size: int, elapsed: float) -> dict:
    n = len(records) or 1
    any_pass = all_pass = all_fail = both = 0
    for r in records:
        passes = sum(s["outcome"] == Outcome.PASS.value for s in r["samples"])
        total = len(r["samples"])
        any_pass += passes > 0
        all_pass += passes == total
        all_fail += passes == 0
        both += 0 < passes < total

    projected = both / n * pool_size
    out = {
        "tasks_probed": n,
        "rft_yield": any_pass / n,
        "pair_yield": both / n,
        "all_pass": all_pass / n,
        "all_fail": all_fail / n,
        "projected_pairs": projected,
        "seconds": elapsed,
        "projected_hours": elapsed / n * pool_size / 3600,
    }

    print(f"\n=== probe: k={k} over {n} tasks ===")
    print(f"  at least one pass (RFT yield)   {out['rft_yield']:.3f}")
    print(f"  one pass and one fail (pairs)   {out['pair_yield']:.3f}")
    print(f"  all {k} pass                      {out['all_pass']:.3f}")
    print(f"  all {k} fail                      {out['all_fail']:.3f}")
    print(f"  projected pairs over {pool_size} tasks: {projected:.0f}")
    print(f"  projected full-pool wall clock: {out['projected_hours']:.1f}h")

    print("\n--- decision ---")
    if out["rft_yield"] < 0.02:
        print("  RFT yield is near zero. The bottleneck is upstream of this stage: stop and revisit the SFT data")
    elif projected < 300:
        print(f"  {projected:.0f} projected pairs below the ~300 threshold fixed in advance. DEFER DPO. Report results")
    else:
        print(f"  {projected:.0f} projected pairs clears 300 threshold. Proceed with both RFT and DPO.")
    return out


def to_datasets(records):
    """ Split labelled samples into the RFT set and the preference set """
    rft, prefs, seen = [], [], set()
    for r in records:
        passed = [s["code"] for s in r["samples"] if s["outcome"] == Outcome.PASS.value]
        failed = [s["code"] for s in r["samples"] if s["outcome"] != Outcome.PASS.value]

        for code in passed:
            key = (r["task_id"], code.strip())
            if key in seen:
                continue
            seen.add(key)
            rft.append({"prompt": r["prompt"], "completion": f"```python\n{code}\n```"})

        if passed and failed:
            chosen = min(passed, key=len)
            rejected = min(failed, key=lambda c: abs(len(c) - len(chosen)))
            prefs.append({
                "prompt": r["prompt"],
                "chosen": f"```python\n{chosen}\n```",
                "rejected": f"```python\n{rejected}\n```",
            })
    return rft, prefs


def outcome_histogram(records) -> dict:
    hist = defaultdict(int)
    for r in records:
        for s in r["samples"]:
            hist[s["outcome"]] += 1
    total = sum(hist.values()) or 1
    return {k: v / total for k, v in sorted(hist.items(), key=lambda kv: -kv[1])}


def main():
    import time

    ap = argparse.ArgumentParser(description="Sample, execution-label, and build RFT/DPO data.")
    ap.add_argument("--model", required=True, help="Hub repo id or local dir (the SFT model).")
    ap.add_argument("--revision", default="sft")
    ap.add_argument("--sft-repo", default="torq1/NanoCoder-sft",
                    help="Source of extra self-testing tasks.")
    ap.add_argument("--rft-repo", default="torq1/NanoCoder-rft")
    ap.add_argument("--prefs-repo", default="torq1/NanoCoder-prefs")
    ap.add_argument("--probe", action="store_true",
                    help="Sample 100 tasks, report the yields, and stop.")
    ap.add_argument("--probe-tasks", type=int, default=100)
    ap.add_argument("-k", "--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None, help="Cap the task pool.")
    ap.add_argument("--mbpp-only", action="store_true")
    ap.add_argument("--out", default="results/prefs_samples.jsonl", help="JSONL; also resumes.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-push", action="store_true")
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

    pool = build_pool(args.sft_repo, args.mbpp_only, args.limit)

    if args.probe:
        probe_pool = pool[:args.probe_tasks]
        t0 = time.time()
        records = sample_and_label(nano, probe_pool, args.samples, args.temperature,
                                   args.max_new_tokens, args.batch_size, args.timeout,
                                   out_path=None)
        probe_report(records, args.samples, len(pool), time.time() - t0)
        print("\noutcome histogram:", json.dumps(outcome_histogram(records), indent=2))
        return

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    records = sample_and_label(nano, pool, args.samples, args.temperature,
                               args.max_new_tokens, args.batch_size, args.timeout,
                               out_path=args.out)
    rft, prefs = to_datasets(records)
    print(f"\nRFT examples: {len(rft):,} | preference pairs: {len(prefs):,}")
    print("outcome histogram:", json.dumps(outcome_histogram(records), indent=2))

    if not rft:
        raise SystemExit("No passing samples. The bottleneck is upstream - revisit SFT.")
    if len(prefs) < 300:
        print(f"\nWARNING: {len(prefs)} pairs is under the ~300 threshold. DPO on this is "
              "unlikely to produce a trustworthy result; RFT alone is the honest stage.")

    if args.no_push:
        print("--no-push set; skipping Hub upload.")
        return

    from datasets import Dataset, DatasetDict
    n_val = max(1, len(rft) // 20)
    DatasetDict({
        "train": Dataset.from_list(rft[n_val:]),
        "validation": Dataset.from_list(rft[:n_val]),
    }).push_to_hub(args.rft_repo)
    print(f"Pushed {args.rft_repo}")

    if prefs:
        n_val = max(1, len(prefs) // 20)
        DatasetDict({
            "train": Dataset.from_list(prefs[n_val:]),
            "validation": Dataset.from_list(prefs[:n_val]),
        }).push_to_hub(args.prefs_repo)
        print(f"Pushed {args.prefs_repo}")


if __name__ == "__main__":
    main()
