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
- **Models:** NanoCoder-110M-pretrain, NanoCoder-110M-sft (coming soon)

Because NanoCoder is custom (not a `transformers` class), consumers reconstruct it with `NanoCoder.from_pretrained`, which rebuilds both the model and the tokenizer:

```python
from huggingface_hub import snapshot_download
from nanocoder import NanoCoder

nano = NanoCoder.from_pretrained(snapshot_download("<you>/NanoCoder-110M"), device="cuda")
print(nano.generate_text("## Task\nWrite a function that reverses a string.\n\n## Solution\n"))
```

## Files

This repository defines a reusable package (`src/nanocoder`) that is walked through in pieces in the `src/notebooks` modules.
