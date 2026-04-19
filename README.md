# custom-llm

Build a small decoder-only Transformer LM from scratch on CPU: **GPT-2 regex pretokenization → byte-level BPE (10k vocab, `<|endoftext|>`) → Transformer → training → sampling**.

## Environment

Use **Python 3.10–3.12** for the virtualenv. **Python 3.13** often has **no PyTorch wheel** on PyPI yet for some macOS/CPU combos, which shows up as `No matching distribution found for torch (from versions: none)`.

```bash
cd "/path/to/custom_LLM"
python3.12 -m venv .venv   # or python3.11 / python3.10
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

On Intel Mac with Homebrew: `brew install python@3.12`, then `"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv`.

Dependencies: PyTorch (CPU), `regex` (Unicode pretokenizer), `datasets` (optional corpus export).

Use **`device="cpu"`** everywhere; no CUDA assumptions.

**NumPy:** The package allows **NumPy 1.26+** (including **2.x**). **Google Colab** needs NumPy 2 for many preinstalled libraries (OpenCV, JAX, …). On **older CPU-only PyTorch + macOS**, if you see NumPy/Torch ABI warnings, try: `pip install "numpy>=1.26,<2"`.

## Milestones (in order)

### 1. Corpus: TinyStories as one plaintext file

Export the **validation** split (~22K stories) for debugging before scaling up:

```bash
python scripts/prepare_tinystories.py --split validation --out data/tinystories_val.txt
```

Documents are separated by blank lines and `<|endoftext|>`.

### 2. Train BPE tokenizer (vocab 10_000)

```bash
python scripts/train_tokenizer.py \
  --corpus data/tinystories_val.txt \
  --out_dir tokenizer \
  --vocab_size 10000
```

This uses the **GPT-2 pretokenizer regex** on each document, then trains **byte-level BPE** merges. The last id `vocab_size - 1` is reserved for `<|endoftext|>`.

### 3. Train the Transformer LM (~17M parameters)

Low-resource shape from the spec:

- `d_model=512`, `d_ff=1344`, `n_layers=4`, `n_heads=16`, `context_length=256`
- Default run: **batch 32 × 5000 steps × 256 tokens** ≈ **40M** token positions (cross-entropy on each position)
- Target validation loss **≤ 2.00** (may require tuning LR / warmup on your machine)

```bash
python -m llm.train \
  --corpus data/tinystories_val.txt \
  --tokenizer_dir tokenizer \
  --out_dir checkpoints \
  --device cpu \
  --batch_size 32 \
  --context_length 256 \
  --steps 5000
```

**Held-out validation (recommended):** prepare `data/tinystories_train.txt` and `data/tinystories_val.txt`, train the tokenizer on **train**, then:

```bash
python -m llm.train \
  --corpus data/tinystories_train.txt \
  --val_corpus data/tinystories_val.txt \
  --tokenizer_dir tokenizer \
  --out_dir checkpoints \
  --device cuda
```

Checkpoints: `checkpoints/best.pt` (lowest **held-out** validation loss if `--val_corpus` is set).

**Google Colab / notebook cells:** `!python -m ...` runs in a **non-TTY** subprocess, so `tqdm`’s in-place bar often **does not update until the process finishes**. The trainer detects this and prints **line-based progress** (flushed) every **`--progress_interval`** steps (default **10**). Use **`--force_tqdm`** only if you want the tqdm bar anyway (often still invisible in `!python`). For other buffering issues, **`python -u`** (unbuffered stdout) can help.

**Longer runs toward lower val loss (e.g. ~2.0):** use the **TinyStories train** split for more data (`prepare_tinystories.py --split train`), **retrain or reuse** a tokenizer trained on that text, increase **`--steps`**, and on GPU use `--device cuda`. Optional **`--cosine_decay`** (with **`--min_lr`**) reduces LR over the run and often helps late training. Validation should use a **held-out** file when possible (e.g. train on `train`, eval corpus = `validation`).

**Continue training on the same corpus (no new data):** after a run, `checkpoints/latest.pt` stores weights, optimizer, and global step. Run again with **`--resume checkpoints/latest.pt`**; **`--steps`** is then the number of **additional** optimizer steps. Example: first run `--steps 5000`, then `--resume checkpoints/latest.pt --steps 5000` goes from step 5000 → 10000 on the same `--corpus`.

### 4. Sample

```bash
python -m llm.sample \
  --checkpoint checkpoints/best.pt \
  --tokenizer_dir tokenizer \
  --prompt "Once upon a time" \
  --device cpu
```

Use `--top_p 0.9 --temperature 0.9` for nucleus sampling.

## Layout

| Module | Role |
|--------|------|
| `llm/gpt2_pretokenize.py` | GPT-2 pretokenizer regex |
| `llm/bpe_trainer.py` | Byte-level BPE merge training |
| `llm/tokenizer.py` | `BPETokenizer` encode/decode + `tokenizer.json` I/O |
| `llm/model.py` | Causal Transformer + RoPE, weight-tied head |
| `llm/data.py` | Random sliding windows over token ids |
| `llm/train.py` | Training loop (CPU or CUDA) |
| `llm/sample.py` | Greedy or top-p generation |
| `llm/pretokenization.py` | Optional parallel chunking for huge files |

## Tests

```bash
PYTHONPATH=. pytest tests/ -q
```

## Notes

- **Runtime**: ~1h 20m on CPU is an estimate; a 2017 dual-core machine may be slower—reduce `--steps` or `--batch_size` for smoke tests.
- **Validation**: The training script evaluates on random windows from the same file; for real generalization, point `--corpus` at train data and add a separate eval file in a later iteration.
