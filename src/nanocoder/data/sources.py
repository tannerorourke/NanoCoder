""" Stream + Filter the sources into the interleaved pretrain corpus """
import gc

from tqdm.auto import tqdm

from nanocoder.constants import SEED


def build_dataset(dcfg) -> tuple[list[str], list[str]]:
    from datasets import load_dataset, interleave_datasets
    caps = dcfg.max_chars

    # --- language heuristics (only for when a source has no language column)
    non_py_kws = ['public class ', 'std::', 'printf(', 'fn main()', 'namespace ',
                  'extern ', 'await fetch', 'System.out.print', 'using namespace std']
    non_py_langs = ['javascript', 'js ', 'java ', 'cpp', 'c++', 'rust',
                    'c#', 'csharp', 'ruby', 'swift', 'golang', 'sql']

    def looks_python(text: str) -> bool:
        t = text.lower()
        if any(k in t for k in non_py_kws):                 return False
        if any(f"```{l}" in t for l in non_py_langs):        return False
        return ('python' in t) or ('def ' in t and 'import ' in t)

    has_text = lambda ex: ex.get('text')

    # --- maps every source to a single 'text' field
    def as_task(ex, p_col, r_col):
        p, r = (ex.get(p_col) or '').strip(), (ex.get(r_col) or '').strip()
        return {"text": f"## Task\n{p}\n\n## Solution\n{r}" if (p and r) else None}

    def as_fenced_py(ex):
        # Raw .py MUST be fenced: preprocess() only applies <|indent|>/<|dedent|> and
        # FIM inside a `' block, so unfenced source would silently bypass both.
        c = (ex.get('content') or '').strip()
        return {"text": f"```python\n{c}\n```" if c else None}

    def as_cosmo(ex):
        t = (ex.get('text') or '').strip()
        return {"text": t if t else None}

    # --- streams. Each is filtered to Python, capped at its own natural length,
    #     then normalised to 'text'.
    streams = {}

    # Raw Python files, the only source that teaches fluency + FIM + file-shape
    streams["codeparrot"] = (
        load_dataset("codeparrot/codeparrot-clean", split='train', streaming=True)
        .filter(lambda ex: 0 < len(ex['content']) < caps["codeparrot"])
        .map(as_fenced_py).filter(has_text))

    # Instruction sets
    streams["glaive"] = (
        load_dataset("glaiveai/glaive-code-assistant", split='train', streaming=True)
        .filter(lambda ex: len(ex['answer']) < caps["glaive"]
                and looks_python(ex['question'] + ex['answer']))
        .map(lambda ex: as_task(ex, 'question', 'answer')).filter(has_text))

    streams["tinycodes"] = (
        load_dataset("nampdn-ai/tiny-codes", split='train', streaming=True)
        .filter(lambda ex: len(ex['response']) < caps["tinycodes"]
                and str(ex.get('programming_language', '')).lower() == 'python')
        .map(lambda ex: as_task(ex, 'prompt', 'response')).filter(has_text))

    streams["magicoder_evol"] = (
        load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split='train', streaming=True)
        .filter(lambda ex: len(ex['response']) < caps["magicoder_evol"]
                and looks_python(ex['instruction'] + ex['response']))
        .map(lambda ex: as_task(ex, 'instruction', 'response')).filter(has_text))

    streams["codefeedback"] = (
        load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split='train', streaming=True)
        .filter(lambda ex: str(ex.get('lang', '')).lower() == 'python'
                and len(ex['answer']) < caps["codefeedback"])
        .map(lambda ex: as_task(ex, 'query', 'answer')).filter(has_text))

    streams["magicoder_oss"] = (
        load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split='train', streaming=True)
        .filter(lambda ex: str(ex.get('lang', '')).lower() == 'python'
                and len(ex['solution']) < caps["magicoder_oss"])
        .map(lambda ex: as_task(ex, 'problem', 'solution')).filter(has_text))

    # technical language, for the questions whose answer isn't a function
    COSMO_FMT = ['textbook', 'textbook_unconditionned_topic', 'wikihow', 'textbook_narrative',
                     'e-learning_module', 'textbook_academic', 'scientific_article']
    
    streams["cosmo"] = (
        load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split='train', streaming=True)
        .filter(lambda ex: len(ex['text']) < caps["cosmo"]
                and ex['audience'].lower() not in ['children', 'young_children']
                and ex['format'].lower() in COSMO_FMT)
        .map(as_cosmo).filter(has_text))

    # Key off names, not list position, so weights can never silently desync from streams.
    names = list(dcfg.mix_proportions.keys())
    docs = interleave_datasets(
        datasets=[streams[n] for n in names],
        probabilities=[dcfg.mix_proportions[n] for n in names],
        seed=SEED,
        # small pools cycle rather than ending the stream; epochs are budgeted in the config
        stopping_strategy='all_exhausted',
    )
    del streams; _ = gc.collect()

    # --- Materialize, routing every n'th valid doc to val to keep val proportions
    val_stride = max(1, round(1 / dcfg.val_split))
    max_val_docs = int(dcfg.max_samples * dcfg.val_split)

    train_texts: list[str] = []
    val_texts: list[str] = []
    with tqdm(total=dcfg.max_samples, desc="Streaming to interleaved docs") as pbar:
        for _n, doc in enumerate(docs):
            if _n >= dcfg.max_samples:
                break
            is_val_doc = (_n % val_stride == val_stride - 1) and (len(val_texts) < max_val_docs)
            (val_texts if is_val_doc else train_texts).append(doc.get('text'))
            pbar.update(1)

    print(f"\nTrain docs: {len(train_texts):,} | Val docs: {len(val_texts):,}")
    print("\n--- example ---")
    if train_texts:
        print(train_texts[min(1738, len(train_texts) - 1)][:500] + "\n...")

    del docs, load_dataset, interleave_datasets; _ = gc.collect()
    return train_texts, val_texts
