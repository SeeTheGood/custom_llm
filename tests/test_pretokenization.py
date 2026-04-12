"""Tests for corpus chunking."""

from pathlib import Path

from custom_llm.pretokenization import find_chunk_boundaries, iter_text_chunks


def test_find_chunk_boundaries_on_delimiter(tmp_path: Path) -> None:
    delim = b"<|endoftext|>"
    parts = [b"aaa", delim, b"bbb", delim, b"ccc"]
    p = tmp_path / "c.txt"
    p.write_bytes(b"".join(parts))

    with p.open("rb") as f:
        bounds = find_chunk_boundaries(f, 4, delim)

    data = p.read_bytes()
    assert bounds[0] == 0
    assert bounds[-1] == len(data)
    assert bounds == sorted(set(bounds))
    # Chunks partition the file without gaps or overlaps
    out = []
    for start, end, _ in iter_text_chunks(str(p), 4, delimiter=delim):
        out.append(data[start:end])
    assert b"".join(out) == data


def test_iter_text_chunks_roundtrip(tmp_path: Path) -> None:
    delim = b"<|eot|>"
    body = b"hello" + delim + b"world" + delim + b"!"
    p = tmp_path / "t.txt"
    p.write_bytes(body)

    chunks = list(iter_text_chunks(str(p), 2, delimiter=delim))
    assert len(chunks) >= 1
    joined = "".join(c[2] for c in chunks)
    assert joined == body.decode("utf-8")
