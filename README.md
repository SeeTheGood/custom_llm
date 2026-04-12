# custom_llm

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
python -m custom_llm.train \
  --corpus data/tinystories_val.txt \
  --tokenizer_dir tokenizer \
  --out_dir checkpoints \
  --device cpu \
  --batch_size 32 \
  --context_length 256 \
  --steps 5000
```

Checkpoints: `checkpoints/best.pt` (lowest validation loss seen).

### 4. Sample

```bash
python -m custom_llm.sample \
  --checkpoint checkpoints/best.pt \
  --tokenizer_dir tokenizer \
  --prompt "Once upon a time" \
  --device cpu
```

Use `--top_p 0.9 --temperature 0.9` for nucleus sampling.

## Layout

| Module | Role |
|--------|------|
| `custom_llm/gpt2_pretokenize.py` | GPT-2 pretokenizer regex |
| `custom_llm/bpe_trainer.py` | Byte-level BPE merge training |
| `custom_llm/tokenizer.py` | `BPETokenizer` encode/decode + `tokenizer.json` I/O |
| `custom_llm/model.py` | Causal Transformer + RoPE, weight-tied head |
| `custom_llm/data.py` | Random sliding windows over token ids |
| `custom_llm/train.py` | CPU training loop |
| `custom_llm/sample.py` | Greedy or top-p generation |
| `custom_llm/pretokenization.py` | Optional parallel chunking for huge files |

## Tests

```bash
PYTHONPATH=. pytest tests/ -q
```

## Notes

- **Runtime**: ~1h 20m on CPU is an estimate; a 2017 dual-core machine may be slower—reduce `--steps` or `--batch_size` for smoke tests.
- **Validation**: The training script evaluates on random windows from the same file; for real generalization, point `--corpus` at train data and add a separate eval file in a later iteration.
