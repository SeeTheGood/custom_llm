"""CLI: print chunk boundaries and optional previews for pretokenization."""

from __future__ import annotations

import argparse

from llm.pretokenization import iter_text_chunks


def main() -> None:
    p = argparse.ArgumentParser(description="Align and preview corpus chunks for pretokenization.")
    p.add_argument("--file", required=True, help="Path to a text corpus file")
    p.add_argument("--chunks", type=int, default=4, help="Target number of parallel chunks")
    p.add_argument(
        "--delimiter",
        default="<|endoftext|>",
        help="Boundary token as UTF-8 (aligned on byte boundaries in file)",
    )
    p.add_argument(
        "--preview-chars",
        type=int,
        default=200,
        help="Characters of each chunk to print (0 to skip)",
    )
    args = p.parse_args()

    delim_bytes = args.delimiter.encode("utf-8")
    for i, (start, end, text) in enumerate(
        iter_text_chunks(args.file, args.chunks, delimiter=delim_bytes)
    ):
        print(f"chunk {i}: bytes [{start}, {end}) length={end - start} chars={len(text)}")
        if args.preview_chars > 0:
            snippet = text[: args.preview_chars].replace("\n", "\\n")
            print(f"  preview: {snippet!r}")
        print()


if __name__ == "__main__":
    main()
