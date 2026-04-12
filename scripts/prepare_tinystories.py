#!/usr/bin/env python3
"""Write TinyStories (validation split by default) to one plaintext file with <|endoftext|> boundaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import load_dataset
from tqdm.auto import tqdm

EOT = "<|endoftext|>"


def main() -> None:
    p = argparse.ArgumentParser(description="Export TinyStories to a single plaintext corpus.")
    p.add_argument(
        "--split",
        default="validation",
        help="HF split name (use 'validation' ~22K docs for debugging; 'train' is large)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/tinystories_val.txt"),
        help="Output file path",
    )
    p.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download/cache the full split before iterating (needs more RAM; can look 'stuck' on Colab). "
        "Default is streaming: stories arrive incrementally.",
    )
    args = p.parse_args()

    streaming = not args.no_streaming
    print(
        f"Loading roneneldan/TinyStories split={args.split!r} streaming={streaming} ...",
        flush=True,
    )
    ds = load_dataset(
        "roneneldan/TinyStories",
        split=args.split,
        streaming=streaming,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"TinyStories {args.split}", unit="doc"):
            text = (row["text"] or "").strip()
            if not text:
                continue
            f.write(text)
            f.write("\n\n")
            f.write(EOT)
            f.write("\n\n")
            n += 1
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes, {n:,} non-empty docs)", flush=True)


if __name__ == "__main__":
    main()
