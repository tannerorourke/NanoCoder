"""Entrypoint

Build the NanoCoder-sft dataset: instruction pairs whose completion is code and nothing else.

    python -m nanocoder.data.build_sft --repo-id <user>/NanoCoder-sft

The same five instruction sources the pretrain mix used, re-streamed and filtered far
harder. Pretraining wanted volume; this wants exemplars, and the two goals disagree.

**Prose stripping is the point of this file.** Every source's response is conversational -
preamble, a fenced block, then an explanation. Pretraining supervised all of it, so a large
share of the tokens spent teaching "prompt in, code out" were actually teaching English.
The completion kept here is exactly the fenced block, so effective code supervision rises
at zero compute cost. The discarded fraction is measured and reported per source, because
that number is the entire justification for the approach: if solutions were already 90%
code there is nothing to win, and we should know that before training rather than after.

**On contamination.** The original design excluded SFT documents that appeared in the
pretrain corpus. That check cannot be run, and finding out why is more useful than the check
would have been: pretraining drew max_samples x mix_proportion documents per source, which
against these pool sizes is one to two full epochs of every instruct source. There is no
uncontaminated subset to hold out - the overlap is 100% by construction.

That is fine for what SFT does here. It is not teaching new knowledge; it is re-weighting a
distribution the model has already seen, with the loss masked to the completion and the
prose removed. Re-showing seen data under a different objective is the method, not a leak.

What must be held out is the *benchmark*, and that check is real and runs below. MBPP and
HumanEval problems leak into these instruct pools, and a leaked solution would inflate every
pass@k this project reports.
"""
import argparse
import ast
import hashlib
import re
import warnings
from collections import Counter, defaultdict

from tqdm.auto import tqdm

# Any fenced block, language tag captured. Unterminated fences are deliberately not
# matched: a truncated answer is not an exemplar.
_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)", re.MULTILINE)
_WORD_RE = re.compile(r"[a-z0-9_]+")

PY_LANGS = {"", "python", "py", "python3"}

# The prompt must actually ask for a program. A corpus whose every target is code has no
# place for questions whose honest answer is a paragraph.
_ASKS_CODE = re.compile(
    r"\b(function|code|implement|write|create|program|script|class|method|algorithm|"
    r"def|snippet|generate|return|parse|compute|calculate|sort|convert)\b", re.I)
_CONCEPTUAL = re.compile(
    r"^\s*(what is|what are|why does|why is|explain|describe|when should|"
    r"what's the difference|how does)\b", re.I)

# (repo, config, prompt column, response column, language column or None)
SOURCES = {
    "glaive":         ("glaiveai/glaive-code-assistant", None, "question", "answer", None),
    "tinycodes":      ("nampdn-ai/tiny-codes", None, "prompt", "response", "programming_language"),
    "magicoder_evol": ("ise-uiuc/Magicoder-Evol-Instruct-110K", None, "instruction", "response", None),
    "codefeedback":   ("m-a-p/CodeFeedback-Filtered-Instruction", None, "query", "answer", "lang"),
    "magicoder_oss":  ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "problem", "solution", "lang"),
}

PROMPT_FMT = "## Task\n{task}\n\n## Solution\n"
COMPLETION_FMT = "```python\n{code}\n```"


# --------------------------------------------------------------------------- extraction

def fenced_blocks(text: str) -> list[tuple[str, str]]:
    return [(lang.lower(), body) for lang, body in _FENCE_RE.findall(text)]


def normalized_ast(code: str) -> str | None:
    """
    AST dump with names and constants erased, used as a near-duplicate key.

    The instruct pools are internally repetitive: the same solution reappears with renamed
    variables and a reworded docstring. Hashing the source text misses those; hashing the
    shape catches them. None if the code does not parse.
    """
    try:
        # Scraped code is full of unescaped regex literals; their SyntaxWarnings are noise
        # here, not signal - the code still parses and that is all this asks.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = "_"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.name = "_"
        elif isinstance(node, ast.arg):
            node.arg = "_"
        elif isinstance(node, ast.Constant):
            node.value = 0
        elif isinstance(node, ast.Attribute):
            node.attr = "_"
    try:
        return ast.dump(tree, annotate_fields=False)
    except RecursionError:
        return None


def words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


# ------------------------------------------------------------------------ benchmark guard

def _bare_task(formatted: str) -> str:
    """
    Recover the benchmark's own wording from the '## Task / ## Solution' framing.

    The framing words and the appended assert are shared by every task, so leaving them in
    inflates the overlap between unrelated problems and washes out the signal.
    """
    body = formatted.split("## Task\n", 1)[-1].split("\n\n## Solution", 1)[0]
    return "\n".join(ln for ln in body.splitlines()
                     if not ln.startswith("Your code should satisfy:"))


def build_benchmark_index():
    """
    Map every benchmark entry-point name to the problem statements that use it.

    Keying on the function name prunes cheaply - almost no document defines a colliding
    name - so only the survivors pay for the word-overlap comparison.
    """
    from nanocoder.eval.tasks import load_mbpp, load_humaneval
    index = defaultdict(list)
    tasks = []
    for split in ("train", "test", "validation"):
        tasks += load_mbpp(split=split)
    tasks += load_humaneval()
    for t in tasks:
        index[t.entry_point].append(words(_bare_task(t.prompt)))
    return index


def leaks_benchmark(prompt: str, code: str, index, threshold: float = 0.7) -> bool:
    """
    A document leaks if it defines a benchmark's function and restates its problem.

    Overlap is measured against the *shorter* of the two word sets, not their union.
    Instruct prompts are chatty where MBPP is terse, so a document can restate a benchmark
    problem in full and still share only a third of the combined vocabulary - Jaccard scores
    that as unrelated, which is exactly the leak we would miss.
    """
    names = set(_DEF_RE.findall(code))
    hits = [w for n in names if n in index for w in index[n]]
    if not hits:
        return False
    pw = words(prompt)
    if not pw:
        return False
    return any(len(pw & bw) / min(len(pw), len(bw)) >= threshold for bw in hits)


# ------------------------------------------------------------------------------ filtering

class Stats:
    """Per-source accounting. The drop histogram is a deliverable, not a log line."""
    def __init__(self):
        self.drops = Counter()
        self.kept = 0
        self.seen = 0
        self.response_chars = 0
        self.code_chars = 0

    @property
    def prose_fraction(self) -> float:
        if not self.response_chars:
            return 0.0
        return 1.0 - self.code_chars / self.response_chars


def filter_example(prompt: str, response: str, st: Stats, seen_shapes: set,
                   bench_index, min_code: int, max_code: int,
                   min_prompt: int, max_prompt: int, max_total: int) -> str | None:
    """
    Apply the quality filter in order, returning the stripped code or None.

    Each step is a hypothesis about what makes a bad target. They run cheapest first, so
    the expensive ones see fewer documents.
    """
    st.seen += 1
    prompt, response = prompt.strip(), response.strip()
    if not prompt or not response:
        st.drops["empty"] += 1
        return None

    if not (min_prompt <= len(prompt) <= max_prompt):
        st.drops["prompt_length"] += 1
        return None
    if _CONCEPTUAL.match(prompt) or not _ASKS_CODE.search(prompt):
        st.drops["not_a_code_request"] += 1
        return None

    blocks = fenced_blocks(response)
    if not blocks:
        st.drops["no_fenced_block"] += 1
        return None
    if len(blocks) > 1:
        # Cannot reduce to one completion without guessing which block is the answer.
        st.drops["multi_block"] += 1
        return None

    lang, code = blocks[0]
    if lang not in PY_LANGS:
        st.drops["not_python"] += 1
        return None
    code = code.strip("\n")
    if not (min_code <= len(code) <= max_code):
        st.drops["code_length"] += 1
        return None
    # Prompt and completion share one training block, so the pair's combined length is
    # what has to fit - either side alone passing its own bound is not enough. Sized so
    # nearly everything clears sft_block_size once tokenized; the residual tail is dropped
    # at training time, where the exact token count is known.
    if len(prompt) + len(code) > max_total:
        st.drops["pair_too_long"] += 1
        return None

    shape = normalized_ast(code)
    if shape is None:
        st.drops["unparseable"] += 1
        return None
    key = hashlib.blake2b(shape.encode(), digest_size=16).digest()
    if key in seen_shapes:
        st.drops["near_duplicate"] += 1
        return None

    if bench_index is not None and leaks_benchmark(prompt, code, bench_index):
        st.drops["benchmark_leak"] += 1
        return None

    seen_shapes.add(key)
    st.response_chars += len(response)
    st.code_chars += len(code)
    st.kept += 1
    return code


# ------------------------------------------------------------------------------ streaming

def stream_source(name: str, limit: int, max_chars: int):
    """Yield (prompt, response) from one instruct source, language-filtered at the stream."""
    from datasets import load_dataset
    repo, config, p_col, r_col, lang_col = SOURCES[name]
    ds = load_dataset(repo, config, split="train", streaming=True)
    n = 0
    for ex in ds:
        if n >= limit:
            break
        if lang_col and str(ex.get(lang_col, "")).lower() not in ("python", "python3"):
            continue
        p, r = ex.get(p_col) or "", ex.get(r_col) or ""
        if len(r) > max_chars:
            continue
        n += 1
        yield p, r


def build(per_source: int, max_chars: int, min_code: int, max_code: int,
          min_prompt: int, max_prompt: int, max_total: int, cap: int | None,
          decontaminate: bool, sources=None):
    bench_index = build_benchmark_index() if decontaminate else None
    if bench_index is not None:
        print(f"Benchmark guard: {len(bench_index)} distinct entry points indexed")

    seen_shapes: set = set()        # global, so cross-source duplicates are caught too
    stats: dict[str, Stats] = {}
    records = []

    for name in (sources or SOURCES):
        st = Stats()
        stats[name] = st
        for prompt, response in tqdm(stream_source(name, per_source, max_chars),
                                     total=per_source, desc=f"{name:<15}"):
            if cap and st.kept >= cap:
                st.drops["source_cap"] += 1
                continue
            code = filter_example(prompt, response, st, seen_shapes, bench_index,
                                  min_code, max_code, min_prompt, max_prompt, max_total)
            if code is None:
                continue
            records.append({
                "source": name,
                "prompt": PROMPT_FMT.format(task=prompt.strip()),
                "completion": COMPLETION_FMT.format(code=code),
            })
    return records, stats


def report(stats: dict[str, Stats]) -> None:
    print(f"\n{'source':<16}{'seen':>9}{'kept':>9}{'keep%':>8}{'prose%':>9}   top drop reasons")
    total_seen = total_kept = 0
    for name, st in stats.items():
        total_seen += st.seen
        total_kept += st.kept
        top = ", ".join(f"{k} {v}" for k, v in st.drops.most_common(3))
        print(f"{name:<16}{st.seen:>9,}{st.kept:>9,}"
              f"{100 * st.kept / max(1, st.seen):>7.1f}%{100 * st.prose_fraction:>8.1f}%   {top}")
    print(f"{'TOTAL':<16}{total_seen:>9,}{total_kept:>9,}"
          f"{100 * total_kept / max(1, total_seen):>7.1f}%")

    leaks = sum(st.drops["benchmark_leak"] for st in stats.values())
    print(f"\nBenchmark leaks removed: {leaks}")
    print("prose% is the share of retained responses' characters discarded as non-code - "
          "the supervision this stage recovers for free.")


def main():
    ap = argparse.ArgumentParser(description="Build & push the NanoCoder-sft dataset.")
    ap.add_argument("--repo-id", default="torq1/NanoCoder-sft")
    ap.add_argument("--per-source", type=int, default=60_000,
                    help="Documents to draw per source before filtering.")
    ap.add_argument("--max-chars", type=int, default=3_000, help="Response cap, matching pretrain.")
    ap.add_argument("--min-code", type=int, default=60, help="Drops the one-liner tail.")
    ap.add_argument("--max-code", type=int, default=2_400)
    ap.add_argument("--min-prompt", type=int, default=24)
    ap.add_argument("--max-prompt", type=int, default=1_200)
    ap.add_argument("--max-total", type=int, default=2_000,
                    help="Combined prompt+code chars; keeps the pair inside one 512-token block.")
    ap.add_argument("--val-size", type=int, default=2_000)
    ap.add_argument("--tasks-size", type=int, default=3_000,
                    help="Held out for preference sampling; never trained on.")
    ap.add_argument("--cap-per-source", type=int, default=15_000,
                    help="Ceiling on retained docs per source. Keep rates differ ~3x, so "
                         "without it the most formulaic source dominates the corpus and "
                         "its quirks become the model's. 0 disables.")
    ap.add_argument("--no-decontaminate", action="store_true",
                    help="Skip the MBPP/HumanEval guard. Only for a fast smoke run.")
    ap.add_argument("--save-dir", default=None, help="Write locally instead of pushing.")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    from nanocoder.constants import RNG, seed_global
    seed_global()

    records, stats = build(args.per_source, args.max_chars, args.min_code, args.max_code,
                           args.min_prompt, args.max_prompt, args.max_total,
                           args.cap_per_source or None, not args.no_decontaminate)
    report(stats)
    if not records:
        raise SystemExit("No records survived the filter; loosen the bounds before pushing.")

    RNG.shuffle(records)
    n_tasks, n_val = args.tasks_size, args.val_size
    if len(records) < n_tasks + n_val + 1_000:
        raise SystemExit(f"Only {len(records):,} records for a {n_tasks}+{n_val} holdout; "
                         "raise --per-source or loosen the filter.")

    # 'tasks' comes off the top so preference pairs are never drawn from prompts the SFT
    # model was trained on.
    splits = {
        "tasks": records[:n_tasks],
        "validation": records[n_tasks:n_tasks + n_val],
        "train": records[n_tasks + n_val:],
    }
    print("\n" + " | ".join(f"{k}: {len(v):,}" for k, v in splits.items()))
    print("\n--- example ---")
    print(splits["train"][0]["prompt"] + splits["train"][0]["completion"])

    from datasets import Dataset, DatasetDict
    dd = DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()})
    if args.save_dir:
        dd.save_to_disk(args.save_dir)
        print(f"Saved to {args.save_dir}")
    if args.no_push:
        print("--no-push set; skipping Hub upload.")
        return
    dd.push_to_hub(args.repo_id)
    print(f"Pushed dataset to {args.repo_id}")


if __name__ == "__main__":
    main()
