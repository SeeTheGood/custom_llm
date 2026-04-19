"""
Parallel-friendly corpus chunking for pretokenization.

Adapted from the chunk-boundary pattern used in introductory LM assignments:
split large text files on a delimiter (e.g. document boundaries) so each chunk
can be pretokenized independently.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.

    Boundaries are aligned to the next occurrence of ``split_special_token`` after
    each uniform split guess, so chunks do not cut through delimiter tokens.
    May return fewer distinct boundaries if splits collapse.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = max(file_size // desired_num_chunks, 1)

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def iter_text_chunks(
    path: str,
    num_chunks: int,
    delimiter: bytes = b"<|endoftext|>",
    encoding: str = "utf-8",
    errors: str = "ignore",
) -> Iterator[tuple[int, int, str]]:
    """
    Yield (start_byte, end_byte, text) for each chunk of ``path``.

    Serial implementation; ``find_chunk_boundaries`` is structured so you can
    assign (start, end) ranges to worker processes for parallel pretokenization.
    """
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, delimiter)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        with open(path, "rb") as f:
            f.seek(start)
            raw = f.read(end - start)
        yield start, end, raw.decode(encoding, errors=errors)
