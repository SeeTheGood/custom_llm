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

EOT = "<|endoftext|>"


def main() -> None:
    p = argparse.ArgumentParser(description="Export TinyStories to a single plaintext corpus.")
    p.add_argument(
        "--split",
        default="validation",
        help="HF split name (use 'validation' ~22K docs for debugging)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/tinystories_val.txt"),
        help="Output file path",
    )
    args = p.parse_args()

    ds = load_dataset("roneneldan/TinyStories", split=args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = (row["text"] or "").strip()
            if not text:
                continue
            f.write(text)
            f.write("\n\n")
            f.write(EOT)
            f.write("\n\n")
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
