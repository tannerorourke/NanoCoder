# NanoCoder

Educational Walkthrough implementing a custom GPT2-style LM, BPE tokenizer, and dataset purpose-built from-the ground up for high-performance Pythonic reasoning, framed as a educational walkthrough.

Please provide credit to myself (Tanner O'Rourke, 2026) or reach out if you plan to use this for educational purposes.

Features:

- **A measured data pipeline**: Reason through building a 7-source interleave whose weights are *"solved backwards"* from a target token mix, using survival rates measured against the live datasets.
- **A "semi-supervised" BPE tokenizer**: Design, implement and train a byte-level BPE that seeds Python keywords, handles indentation as scope markers, and uses FIM (fill-in-the-middle).
- **A modern-decoder architecture**: Build a GPT-2 skeleton model upgraded with RoPE, RMSNorm, SwiGLU, and QK-norm.

## Access

The dataset, model, and tokenizer are available on HuggingFace.

- **Datasets:** NanoCoder-corpus-pretrain, NanoCoder-corpus-sft
- **Tokenizer:** NanoCoderTokenizer
- **Models:** NanoCoder-123M-pretrain, NanoCoder-123M-sft (coming soon)

Because NanoCoder is custom (not a `transformers` class), consumers reconstruct it with `NanoCoder.from_pretrained`, which rebuilds both the model and the tokenizer:

```python
from huggingface_hub import snapshot_download
from nanocoder import NanoCoder

nano = NanoCoder.from_pretrained(snapshot_download("<you>/NanoCoder-123M-pretrain"), device="cuda")
print(nano.generate_text("## Task\nWrite a function that reverses a string.\n\n## Solution\n"))
```

## Usage

This repository defines a reusable package (`src/nanocoder`) that is walked through in pieces in the `src/notebooks` modules.

NanoCoder is built as a series of modeling artifacts, each pushed to HuggingFace and consumed by the next. Set a write token first:

```bash
export HF_TOKEN=hf_xxx  # or hf auth login
```

**1 — Build the pretrain dataset** → pushes `NanoCoder-pretrain`

```bash
python -m nanocoder.data.build_pretrain --repo-id <you>/NanoCoder-pretrain
# smoke test first:  --max-samples 2000 --no-push
```

**2 — Train the tokenizer** → pushes `NanoCoder-tokenizer`

```bash
python -m nanocoder.tokenizer.build \
    --pretrain-repo <you>/NanoCoder-pretrain \
    --repo-id       <you>/NanoCoder-tokenizer
```

**3 — Train the base model** → pushes `NanoCoder-123M-pretrain`

```bash
python -m nanocoder.model.train \
    --tokenizer-repo <you>/NanoCoder-tokenizer \
    --repo-id        <you>/NanoCoder-123M-pretrain
```

**4 — Build the SFT dataset** → pushes `NanoCoder-sft`

Instruction pairs only, with every response stripped to its fenced Python block. Held out
against MBPP and HumanEval so the eval stays honest.

```bash
python -m nanocoder.data.build_sft --repo-id <you>/NanoCoder-sft
# smoke test first:  --per-source 2000 --no-push
```

Every entrypoint takes `--help`, `--no-push` (build/verify without uploading), and `--private`.

## Evaluation

Correctness is measured by execution, not by loss. The harness samples $n$ completions per task, runs each against the benchmark's own tests in an isolated subprocess, and reports *pass@k* alongside a parse rate and a failure histogram - parse rate separately because at this scale form is learned well before function.

```bash
python -m nanocoder.eval.harness \
    --model <you>/NanoCoder-123M-pretrain \
    --benchmark mbpp --n-samples 5 --out results/base_mbpp.jsonl
```
