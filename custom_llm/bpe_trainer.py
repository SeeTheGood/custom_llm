"""Train byte-level BPE merges on pretokenized UTF-8 bytes."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from typing import Iterable

from tqdm.auto import tqdm

from custom_llm.gpt2_pretokenize import pretokenize

EOT_STR = "<|endoftext|>"


def _pairs(word: tuple[int, ...]) -> set[tuple[int, int]]:
    return set(zip(word, word[1:])) if len(word) >= 2 else set()


def _get_stats(
    word_freq: dict[tuple[int, ...], int],
) -> defaultdict[tuple[int, int], int]:
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for word, freq in word_freq.items():
        for a, b in _pairs(word):
            counts[(a, b)] += freq
    return counts


def _merge_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    a, b = pair
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def iter_documents(corpus: str, delimiter: str = EOT_STR) -> Iterable[str]:
    """Split ``corpus`` on ``delimiter`` into document strings."""
    for part in corpus.split(delimiter):
        p = part.strip()
        if p:
            yield p


def build_word_freq_from_corpus(
    corpus: str,
    delimiter: str = EOT_STR,
    *,
    show_progress: bool = False,
) -> Counter[tuple[int, ...]]:
    """Aggregate pretoken byte-tuple counts over all documents."""
    wf: Counter[tuple[int, ...]] = Counter()

    if show_progress:
        # ``str.split`` on multi‑GB strings can take minutes before any iterator runs;
        # split up‑front so we can print status and give tqdm a real total.
        print(
            "Splitting corpus on <|endoftext|> (large files: this step alone can take several minutes)...",
            flush=True,
        )
        parts = [p.strip() for p in corpus.split(delimiter) if p.strip()]
        print(f"Found {len(parts):,} documents — pretokenizing...", flush=True)
        doc_iter: Iterable[str] = tqdm(
            parts,
            desc="Pretokenize corpus",
            unit="doc",
            file=sys.stdout,
            mininterval=0.5,
        )
    else:
        doc_iter = iter_documents(corpus, delimiter)

    for doc in doc_iter:
        for piece in pretokenize(doc):
            if not piece:
                continue
            b = piece.encode("utf-8")
            if b:
                wf[tuple(b)] += 1
    return wf


def train_bpe_from_word_freq(
    word_freq: dict[tuple[int, ...], int],
    vocab_size: int,
    eot_token_id: int,
    *,
    show_progress: bool = False,
) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """
    Train BPE on pretoken byte-tuple frequencies.

    - Ids ``0..255`` are raw bytes.
    - Merge ids run ``256 .. eot_token_id - 1``.
    - ``eot_token_id`` is reserved (typically ``vocab_size - 1``).
    """
    if eot_token_id != vocab_size - 1:
        raise ValueError("eot_token_id must be vocab_size - 1 for this trainer.")
    if vocab_size < 258:
        raise ValueError("vocab_size must be at least 258 (256 bytes + >=1 merge + EOT).")

    num_merges = eot_token_id - 256
    merges: list[tuple[int, int]] = []
    id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    wf: dict[tuple[int, ...], int] = dict(word_freq)

    merge_range = range(num_merges)
    if show_progress:
        merge_range = tqdm(
            merge_range,
            total=num_merges,
            desc="BPE merges",
            unit="merge",
            file=sys.stdout,
            mininterval=0.5,
        )

    for _ in merge_range:
        if next_id >= eot_token_id:
            break
        stats = _get_stats(wf)
        if not stats:
            break
        pair = max(stats, key=stats.get)  # type: ignore[arg-type]
        if stats[pair] == 0:
            break

        merges.append(pair)
        a, b = pair
        id_to_bytes[next_id] = id_to_bytes[a] + id_to_bytes[b]

        new_wf: dict[tuple[int, ...], int] = {}
        for w, c in wf.items():
            new_w = _merge_word(w, pair, next_id)
            new_wf[new_w] = new_wf.get(new_w, 0) + c
        wf = new_wf
        next_id += 1

    id_to_bytes[eot_token_id] = EOT_STR.encode("utf-8")
    return merges, id_to_bytes


def train_bpe_from_text(
    corpus: str,
    vocab_size: int,
    eot_token_id: int,
    *,
    show_progress: bool = False,
) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Train BPE on a corpus that uses ``<|endoftext|>`` between documents."""
    wf = build_word_freq_from_corpus(corpus, show_progress=show_progress)
    return train_bpe_from_word_freq(
        wf, vocab_size, eot_token_id, show_progress=show_progress
    )
