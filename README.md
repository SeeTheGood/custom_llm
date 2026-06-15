# custom-llm

A **~15M-parameter** decoder-only Transformer language model built from scratch in PyTorch: **GPT-2 regex pretokenization → byte-level BPE (10k vocab, `<|endoftext|>`) → causal Transformer (RoPE) → training → nucleus sampling**.

The **latest working workflow** is the [Google Colab notebook](https://colab.research.google.com/drive/1wX9ZAf1HWgM73OimfQcr7UYw5wxRVRpj?usp=sharing). This repository holds the Python package the notebook calls (`llm/`, `scripts/`).

## Repository

```bash
git clone https://github.com/SeeTheGood/custom_llm.git
cd custom_llm
```

| Remote | URL |
|--------|-----|
| **origin** | https://github.com/SeeTheGood/custom_llm.git |
| **enterprise** | https://github.com/SeeTheGood/enterprise_llm.git |

Large artifacts (corpora, checkpoints, token caches) are **not** committed. See [`.gitignore`](.gitignore).

## Model

| Setting | Value |
|---------|-------|
| Architecture | Decoder-only Transformer, pre-norm, GELU FFN, causal self-attention |
| Position encoding | RoPE (Llama-style) |
| Output head | Weight-tied with token embedding |
| `vocab_size` | 10,000 (BPE; last id = `<|endoftext|>`) |
| `d_model` | 512 |
| `d_ff` | 1,344 |
| `n_layers` | 4 |
| `n_heads` | 16 |
| `context_length` | 256 |
| Parameters | **~14.8M** (with weight tying) |

Default training target: **batch 32 × 5,000 steps × 256 tokens** on TinyStories GPT-4 exports, with held-out validation loss **≤ 2.0** (may need more steps or `--cosine_decay`).

## Quick start (Google Colab — recommended)

Open the notebook on Colab: **[custom_llm training (Colab)](https://colab.research.google.com/drive/1wX9ZAf1HWgM73OimfQcr7UYw5wxRVRpj?usp=sharing)**

A copy is also in this repo: [`notebooks/custom_llm_colab.ipynb`](notebooks/custom_llm_colab.ipynb) (upload to Colab or open locally in Jupyter).

Typical layout on Colab + Drive:

| Path | Contents |
|------|----------|
| `/content/custom_llm` | This repo (clone or copy from Drive) |
| `/content/drive/MyDrive/building_LLM/data/` | `TinyStoriesV2-GPT4-train.txt`, `TinyStoriesV2-GPT4-valid.txt` |
| `/content/checkpoints` | Checkpoints during training (fast local disk) |
| `/content/cache_train.pt`, `/content/cache_val.pt` | Token-id caches (optional, speeds re-runs) |

**Runtime:** GPU (`Runtime → Change runtime type → GPU`).

### Colab cells (summary)

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content/custom_llm
pip install -e .
```

**Train** (copies corpora from Drive → `/content`, checks CUDA, runs `llm.train`):

```bash
!python scripts/run_colab_train.py
```

Override steps or pass extra flags to `llm.train` after `--`:

```bash
!python scripts/run_colab_train.py --steps 10000 -- \
    --cosine_decay --min_lr 1e-5
```

**Train + backup to Drive + sample smoke test:**

```bash
!python scripts/train_backup_sample.py \
    --drive-backup "/content/drive/MyDrive/building_LLM/run_2025-04-20" \
    --tokenizer-dir tokenizer \
    --checkpoint-dir /content/checkpoints \
    -- \
    --corpus /content/TinyStoriesV2-GPT4-train.txt \
    --val_corpus /content/TinyStoriesV2-GPT4-valid.txt \
    --tokenizer_dir tokenizer \
    --out_dir /content/checkpoints \
    --device cuda \
    --batch_size 32 \
    --context_length 256 \
    --steps 5000
```

### Colab progress output

`!python -m ...` runs in a **non-TTY** subprocess, so `tqdm` may not update live. The trainer prints **line-based progress** every `--progress_interval` steps (default **10**). Use `--force_tqdm` only if you want the bar anyway.

### Updating the notebook in this repo

The Colab notebook is the source of truth for the end-to-end runbook. After you change it in Colab:

1. **File → Download → Download `.ipynb`**
2. Replace `notebooks/custom_llm_colab.ipynb` in this repo
3. Commit and push to GitHub

## Local development (macOS / CPU)

Use **Python 3.10–3.12**. Python 3.13 may lack PyTorch wheels on some macOS setups.

```bash
cd "/path/to/custom_LLM"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

On Intel Mac with Homebrew: `brew install python@3.12`, then `"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv`.

Dependencies: PyTorch, `regex`, `datasets` (corpus export), `tqdm`.

**NumPy:** Colab needs NumPy 2.x for many preinstalled libs. On older CPU-only PyTorch + macOS, if you see ABI warnings: `pip install "numpy>=1.26,<2"`.

## Pipeline (local or Colab)

### 1. Corpus

Export TinyStories to plaintext (documents separated by `<|endoftext|>`):

```bash
python scripts/prepare_tinystories.py --split validation --out data/tinystories_val.txt
python scripts/prepare_tinystories.py --split train --out data/tinystories_train.txt
```

On Colab, use the pre-exported GPT-4 TinyStories `.txt` files on Drive (see table above).

### 2. Train BPE tokenizer

```bash
python scripts/train_tokenizer.py \
  --corpus data/tinystories_train.txt \
  --out_dir tokenizer \
  --vocab_size 10000
```

GPT-2 pretokenizer regex → byte-level BPE merges. Id `vocab_size - 1` is `<|endoftext|>`.

### 3. Train the LM

```bash
python -m llm.train \
  --corpus data/tinystories_train.txt \
  --val_corpus data/tinystories_val.txt \
  --tokenizer_dir tokenizer \
  --out_dir checkpoints \
  --device cpu \
  --batch_size 32 \
  --context_length 256 \
  --steps 5000
```

Use `--device cuda` on GPU. Checkpoints: `checkpoints/best.pt` (lowest held-out val loss), `checkpoints/latest.pt` (resume).

**Resume training:**

```bash
python -m llm.train \
  --resume checkpoints/latest.pt \
  --steps 5000 \
  ...  # same corpus / tokenizer / out_dir as before
```

`--steps` is **additional** optimizer steps after the checkpoint step.

**Longer runs:** increase `--steps`, use the train split, add `--cosine_decay --min_lr 1e-5`.

### 4. Sample

```bash
python -m llm.sample \
  --checkpoint checkpoints/best.pt \
  --tokenizer_dir tokenizer \
  --prompt "Once upon a time" \
  --device cpu
```

Nucleus sampling: `--top_p 0.9 --temperature 0.9`.

## Layout

| Path | Role |
|------|------|
| `llm/model.py` | `TransformerLM` + `TransformerConfig` (RoPE, weight-tied head) |
| `llm/train.py` | Training loop (CPU or CUDA, resume, cosine decay) |
| `llm/sample.py` | Greedy or top-p generation |
| `llm/data.py` | Random sliding windows over token ids |
| `llm/tokenizer.py` | `BPETokenizer` encode/decode + `tokenizer.json` I/O |
| `llm/gpt2_pretokenize.py` | GPT-2 pretokenizer regex |
| `llm/bpe_trainer.py` | Byte-level BPE merge training |
| `llm/pretokenization.py` | Optional parallel chunking for huge files |
| `scripts/run_colab_train.py` | Colab launcher: Drive → `/content`, CUDA check, train |
| `scripts/train_backup_sample.py` | Train, backup checkpoints to Drive, sample |
| `scripts/prepare_tinystories.py` | Export TinyStories to `.txt` |
| `scripts/train_tokenizer.py` | Train and save BPE tokenizer |
| `notebooks/custom_llm_colab.ipynb` | Latest Colab training runbook |

## Tests

```bash
PYTHONPATH=. pytest tests/ -q
```

## Notes

- **Colab vs local:** Train on GPU in Colab for full TinyStories runs; use local CPU for smoke tests (`--steps 100`, small val split).
- **I/O:** `run_colab_train.py` keeps corpora and caches on `/content` (fast) and only backs up checkpoints to Drive — avoids slow Drive I/O during training.
- **Runtime:** ~1h+ on CPU for 5k steps; GPU is much faster.
