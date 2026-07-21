"""
Constrained decoding

Three constraints, in increasing order of how much they assume:

- **Forced prefix.** The ```python opening is appended to the context rather than sampled.
  A model that has not fully learned the format cannot fail to open the fence, because it
  is never asked to.
- **Fence stop.** Generation halts on the closing fence as well as <|eos|>. Trailing
  explanation becomes impossible by construction rather than unlikely by training.
- **Parse-filtered best-of-n.** Sample n candidates, discard those that do not ast.parse,
  return the first survivor. This is a verifier that needs no tests, so unlike execution
  feedback it is legitimate at inference time, and it converts the parse-rate gain from
  code-only training directly into pass@1.

Report pass@1 with and without these. The gap is a measurement of how much of the format
lives in the weights versus in the decoder
"""
import ast

SOLUTION_PREFIX = "```python\n"
FENCE = "```"


def fence_token_id(tokenizer) -> int | None:
    """
    The fence token '```', encoded from the tokenizer's locked keyword list
    """
    return tokenizer._engine.encoder.get(FENCE)


def strip_fence(text: str) -> str:
    """Drop the trailing fence the stop token leaves behind."""
    cut = text.find(FENCE)
    return (text[:cut] if cut != -1 else text).strip("\n")


def parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def constrained_batch(nano, prompts: list[str], max_new_tokens: int = 384, **kwargs):
    """
    One constrained completion per prompt returned as bare Python.

    encoding the prompt + prefix together is the context the model would've had if it sampled the prefix itself
    """
    fence = fence_token_id(nano.tokenizer)
    primed = [p + SOLUTION_PREFIX for p in prompts]
    outs = nano.generate_batch(
        primed,
        max_new_tokens=max_new_tokens,
        stop_ids=[fence] if fence is not None else None,
        **kwargs,
    )
    return [strip_fence(o) for o in outs]


def best_of_n(nano, prompts: list[str], n: int = 4, max_new_tokens: int = 384, **kwargs):
    """
    Sample n constrained candidates per prompt, keep the first that parses (default to 1st when none parse)

    All n x len(prompts) sequences go through as one batch - the whole point of batched
    generation is that breadth is nearly free, so best-of-n costs wall-clock proportional
    to the longest sample rather than to n.
    """
    if not prompts:
        return []
    flat = [p for p in prompts for _ in range(n)]
    cands = constrained_batch(nano, flat, max_new_tokens=max_new_tokens, **kwargs)

    out = []
    for i in range(len(prompts)):
        group = cands[i * n:(i + 1) * n]
        out.append(next((c for c in group if parses(c)), group[0]))
    return out
