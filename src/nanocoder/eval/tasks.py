"""
Benchmark loading normalised to (task_id, prompt, test_code, entry_point).
- Prompts are rewritten into the '## Task ... ## Solution' form the model was pretrained on.
- Answers are extracted from the fenced block the model was trained to emit.
"""
import re
from dataclasses import dataclass, field

PROMPT_FMT = "## Task\n{task}\n\n## Solution\n"

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)(?:```|\Z)", re.DOTALL)
_DEF_RE = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_]\w*)", re.MULTILINE)


@dataclass
class Task:
    task_id: str
    prompt: str                 # already in NanoCoder's format, ready to sample from
    test_code: str              # asserts, run after the candidate
    entry_point: str            # the function the tests call
    setup_code: str = ""
    reference: str = ""         # the dataset's own solution, for contrast and for CP4 fallback
    meta: dict = field(default_factory=dict)


def extract_code(completion: str) -> str:
    """ Pull the Python out of a completion """
    m = _FENCE_RE.search(completion)
    if m:
        return m.group(1).strip("\n")
    return completion.strip()


def _entry_point_from(code: str, fallback: str = "solution") -> str:
    names = _DEF_RE.findall(code)
    return names[-1] if names else fallback


def load_mbpp(split: str = "test", limit: int | None = None) -> list[Task]:
    """
    MBPP: a plain-language sentence plus a handful of asserts.

    primary benchmark - short functions specified in English are exactly the distribution 
    the instruction sources trained on, which carries roughly 3x HumanEval's task count and matters 
    more than breadth on a Colab budget.
    """
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    tasks = []
    for ex in ds:
        tests = list(ex["test_list"])
        if not tests:
            continue
        entry = _entry_point_from(ex["code"])
        desc = f"{ex['text'].strip()}\nYour code should satisfy: {tests[0]}"
        tasks.append(Task(
            task_id=f"mbpp/{ex['task_id']}",
            prompt=PROMPT_FMT.format(task=desc),
            test_code="\n".join(tests),
            entry_point=entry,
            setup_code=ex.get("test_setup_code") or "",
            reference=ex["code"],
        ))
        if limit and len(tasks) >= limit:
            break
    return tasks


def load_humaneval(limit: int | None = None) -> list[Task]:
    """ HumanEval (Secondary benchmark): a signature and docstring, tested by a generated check() function """
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    tasks = []
    for ex in ds:
        desc = (f"Complete the following function.\n\n```python\n{ex['prompt'].strip()}\n```")
        tasks.append(Task(
            task_id=ex["task_id"],
            prompt=PROMPT_FMT.format(task=desc),
            test_code=f"{ex['test']}\ncheck({ex['entry_point']})",
            entry_point=ex["entry_point"],
            reference=ex["prompt"] + ex["canonical_solution"],
        ))
        if limit and len(tasks) >= limit:
            break
    return tasks


LOADERS = {"mbpp": load_mbpp, "humaneval": load_humaneval}
