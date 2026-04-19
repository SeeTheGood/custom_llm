#!/usr/bin/env python3
"""
Deduplicate and lightly normalize a plaintext corpus that uses <|endoftext|> between documents.

Use this after merging sources or if you suspect repeated stories. Training benefits from:
  - one story per segment (already enforced if you split on EOT only between full stories);
  - fewer near-duplicate documents (this script drops exact duplicates after normalization).

Streams the input file so multi-GB corpora do not need to fit in RAM as one string.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Iterator
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm.bpe_trainer import EOT_STR

EOT = EOT_STR


def normalize_for_dedupe(text: str) -> str:
    """Strip and collapse runs of whitespace so trivial spacing differences still match."""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def doc_fingerprint(text: str) -> bytes:
    n = normalize_for_dedupe(text)
    return hashlib.sha256(n.encode("utf-8")).digest()


def iter_docs_streaming(path: Path) -> Iterator[str]:
    """
    Yield complete document strings (content between <|endoftext|> markers), streaming from disk.
    Same boundary semantics as training encode (strip each doc; skip empty).
    """
    buf = ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(2 * 1024 * 1024)
            if not chunk:
                break
            buf += chunk
            while True:
                i = buf.find(EOT)
                if i == -1:
                    break
                doc = buf[:i].strip()
                buf = buf[i + len(EOT) :]
                if doc:
                    yield doc
        tail = buf.strip()
        if tail:
            yield tail


def main() -> None:
    p = argparse.ArgumentParser(
        description="Remove duplicate stories from an <|endoftext|>-separated corpus (streaming).",
    )
    p.add_argument("--in", dest="in_path", type=Path, required=True, help="Input .txt corpus")
    p.add_argument("--out", dest="out_path", type=Path, required=True, help="Output .txt corpus")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Count duplicates only; do not write --out",
    )
    args = p.parse_args()

    seen: set[bytes] = set()
    n_in = 0
    n_dup = 0
    n_out = 0

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        out_f = args.out_path.open("w", encoding="utf-8")
    try:
        for doc in iter_docs_streaming(args.in_path):
            n_in += 1
            fp = doc_fingerprint(doc)
            if fp in seen:
                n_dup += 1
                continue
            seen.add(fp)
            n_out += 1
            if not args.dry_run:
                out_f.write(doc)
                out_f.write("\n\n")
                out_f.write(EOT)
                out_f.write("\n\n")
    finally:
        if not args.dry_run:
            out_f.close()

    print(
        f"Documents read: {n_in:,}  unique kept: {n_out:,}  duplicates dropped: {n_dup:,}",
        flush=True,
    )
    if not args.dry_run:
        print(f"Wrote {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
