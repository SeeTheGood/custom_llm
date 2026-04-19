#!/usr/bin/env python3
"""Write TinyStories (validation split by default) to one plaintext file with <|endoftext|> boundaries."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import load_dataset
from tqdm.auto import tqdm

EOT = "<|endoftext|>"


def _normalize_for_dedupe(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _doc_fingerprint(text: str) -> bytes:
    return hashlib.sha256(_normalize_for_dedupe(text).encode("utf-8")).digest()


def _apply_resume(ds, resume: int, streaming: bool):
    """Skip the first ``resume`` rows (same index as tqdm / Hub order)."""
    if resume <= 0:
        return ds
    if streaming:
        return ds.skip(resume)
    # Map-style Dataset: iterate with index (skip() exists on Dataset in recent ``datasets``)
    sk = getattr(ds, "skip", None)
    if callable(sk):
        return ds.skip(resume)
    return ds.select(range(resume, len(ds)))


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
    p.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Skip the first N Hub rows and append to --out (use after a network error). "
        "N must match the tqdm doc count when the run stopped (not the 'non-empty' count).",
    )
    p.add_argument(
        "--flush_every",
        type=int,
        default=5000,
        help="Flush the output file every N rows read (0 = flush only at end). Reduces loss on crash.",
    )
    p.add_argument(
        "--dedupe",
        action="store_true",
        help="Skip stories whose normalized text was already written (exact duplicates in this run). "
        "Uses extra RAM (~32 bytes per unique story hash). "
        "Does not read an existing --out file: if you use --resume, run scripts/clean_eot_corpus.py once on the final file.",
    )
    args = p.parse_args()

    streaming = not args.no_streaming
    resume = max(0, args.resume)
    if resume and args.dedupe:
        print(
            "Note: --dedupe only applies to rows in this run; for global dedupe after --resume, "
            "run: python scripts/clean_eot_corpus.py --in <out> --out <out.clean>",
            flush=True,
        )
    print(
        f"Loading roneneldan/TinyStories split={args.split!r} streaming={streaming} resume={resume} ...",
        flush=True,
    )
    ds = load_dataset(
        "roneneldan/TinyStories",
        split=args.split,
        streaming=streaming,
    )
    ds = _apply_resume(ds, resume, streaming)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume > 0 else "w"
    n_non_empty = 0
    n_skipped_dup = 0
    rows_read = resume
    seen_hashes: set[bytes] = set()
    try:
        with args.out.open(mode, encoding="utf-8") as f:
            it = tqdm(
                ds,
                desc=f"TinyStories {args.split}",
                unit="doc",
                initial=resume,
            )
            for row in it:
                text = (row["text"] or "").strip()
                rows_read += 1
                if not text:
                    if args.flush_every and rows_read % args.flush_every == 0:
                        f.flush()
                    continue
                if args.dedupe:
                    fp = _doc_fingerprint(text)
                    if fp in seen_hashes:
                        n_skipped_dup += 1
                        if args.flush_every and rows_read % args.flush_every == 0:
                            f.flush()
                        continue
                    seen_hashes.add(fp)
                f.write(text)
                f.write("\n\n")
                f.write(EOT)
                f.write("\n\n")
                n_non_empty += 1
                if args.flush_every and rows_read % args.flush_every == 0:
                    f.flush()
    except OSError as e:
        raise SystemExit(
            f"Write failed: {e}\nIf this was a download error, re-run with:\n"
            f"  --resume {rows_read}\n"
            f"(use the tqdm row count shown when it stopped; current rows_read={rows_read})"
        ) from e
    except Exception as e:
        err = type(e).__name__
        if "RemoteProtocol" in err or "Connection" in err or "ChunkedEncoding" in err:
            raise SystemExit(
                f"Hub download interrupted ({e!r}).\n"
                f"Re-run the same command and add:\n"
                f"  --resume {rows_read}\n"
                f"so streaming continues from Hub row index {rows_read} (append to {args.out}).\n"
                "Tip: set HF_TOKEN for more stable downloads; consider writing --out on Google Drive."
            ) from e
        raise

    print(
        f"Wrote {args.out} ({args.out.stat().st_size} bytes, {n_non_empty:,} non-empty docs in this run)",
        flush=True,
    )
    if args.dedupe and n_skipped_dup:
        print(f"(dedupe: skipped {n_skipped_dup:,} duplicate stories in Hub order)", flush=True)
    if resume:
        print(f"(skipped first {resume:,} Hub rows; appended)", flush=True)


if __name__ == "__main__":
    main()
