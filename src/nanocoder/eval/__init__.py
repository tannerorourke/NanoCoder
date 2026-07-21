"""
Execution-based evaluation: does the generated code actually run and pass tests?

Validation cross-entropy cannot separate "plausible Python" from "correct Python", and
that separation is the entire claim post-training is meant to move. Everything here
exists to measure functional correctness instead of likelihood.
"""
from nanocoder.eval.sandbox import Outcome, ExecResult, run_candidate
from nanocoder.eval.tasks import Task, load_mbpp, load_humaneval, extract_code
from nanocoder.eval.harness import pass_at_k, evaluate

__all__ = [
    "Outcome", "ExecResult", "run_candidate",
    "Task", "load_mbpp", "load_humaneval", "extract_code",
    "pass_at_k", "evaluate",
]
