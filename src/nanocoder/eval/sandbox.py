"""
Run one generated solution against one test suite, in a process we are willing to lose.

Generated code is hostile by accident: a 123M model writes infinite loops, unbounded
recursion, sys.exit, and allocations that eat the machine. None of that may reach the
harness, so every candidate runs in its own interpreter behind a wall-clock timeout and
an address-space cap.

The return is a typed outcome rather than a bool. What a small model fails *at* is the
diagnostic - SYNTAX_ERROR and FAIL are different problems with different fixes - and the
preference-pair construction later keys off the same distinction.
"""
import ast
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum


class Outcome(str, Enum):
    PASS = "PASS"                   # ran to completion, every assertion held
    FAIL = "FAIL"                   # ran, an assertion did not hold - wrong answer
    TIMEOUT = "TIMEOUT"             # did not terminate in the budget
    SYNTAX_ERROR = "SYNTAX_ERROR"   # not parseable Python at all
    NAME_ERROR = "NAME_ERROR"       # undefined name, usually the entry point was never defined
    ERROR = "ERROR"                 # any other uncaught exception


@dataclass
class ExecResult:
    outcome: Outcome
    detail: str = ""                # last stderr line, for the failure histogram

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASS


# Applied inside the child before exec. RLIMIT_AS is the one that matters: a runaway
# list comprehension otherwise takes the whole session down before the timeout fires.
_LIMIT_SRC = """
import resource, sys
resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
sys.setrecursionlimit(2000)
"""


def _classify(stderr: str) -> tuple[Outcome, str]:
    """Read the exception type off the traceback's last line."""
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    if not lines:
        return Outcome.ERROR, ""
    last = lines[-1].strip()
    exc = last.split(":", 1)[0].strip()
    if exc in ("SyntaxError", "IndentationError", "TabError"):
        return Outcome.SYNTAX_ERROR, last
    if exc == "NameError":
        return Outcome.NAME_ERROR, last
    if exc == "AssertionError":
        return Outcome.FAIL, last
    return Outcome.ERROR, last


def run_candidate(
    code: str,
    test_code: str,
    setup_code: str = "",
    timeout: float = 10.0,
    max_memory_mb: int = 1024,
) -> ExecResult:
    """
    Execute 'code' followed by 'test_code' in a fresh interpreter.

    Syntax is checked in-process first: it is the single most common failure mode at this
    scale and catching it here skips a process spawn per sample, which over an m x n sweep
    is most of the harness's wall-clock.
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        return ExecResult(Outcome.SYNTAX_ERROR, f"SyntaxError: {e.msg}")

    mem = max_memory_mb * 1024 * 1024
    program = "\n".join([
        _LIMIT_SRC.format(mem=mem, cpu=int(timeout) + 1),
        setup_code, code, test_code,
    ])

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=timeout,
                cwd=tmp,                     # anything it writes dies with the temp dir
                env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return ExecResult(Outcome.TIMEOUT, f"exceeded {timeout}s")
        except MemoryError:
            return ExecResult(Outcome.ERROR, "MemoryError spawning child")

    if proc.returncode == 0:
        return ExecResult(Outcome.PASS)
    outcome, detail = _classify(proc.stderr)
    return ExecResult(outcome, detail)
