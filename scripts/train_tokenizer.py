#!/usr/bin/env python3
"""Train BPE (10k vocab by default) and save ``tokenizer.json``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_llm.bpe_trainer import train_bpe_from_text
from custom_llm.tokenizer import BPETokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Train byte-level BPE + GPT-2 pretokenizer.")
    p.add_argument("--corpus", type=Path, required=True, help="Plaintext with <|endoftext|> between docs")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--vocab_size", type=int, default=10_000)
    p.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    args = p.parse_args()

    print(f"Loading corpus: {args.corpus}", flush=True)
    text = args.corpus.read_text(encoding="utf-8")
    print(f"Loaded {len(text) / 1e9:.2f}B characters — starting BPE...", flush=True)
    eot_id = args.vocab_size - 1
    merges, id_to_bytes = train_bpe_from_text(
        text,
        args.vocab_size,
        eot_id,
        show_progress=not args.no_progress,
    )
    tok = BPETokenizer.from_training(merges, id_to_bytes, args.vocab_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(args.out_dir)
    print(f"Saved tokenizer to {args.out_dir} ({len(merges)} merges)", flush=True)


if __name__ == "__main__":
    main()
