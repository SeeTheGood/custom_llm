"""Byte-level BPE tokenizer with GPT-2 pretokenization and <|endoftext|> special token."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.bpe_trainer import EOT_STR
from llm.gpt2_pretokenize import pretokenize


def _encode_piece(merge_rank: dict[tuple[int, int], int], piece: tuple[int, ...]) -> list[int]:
    """Merge ``piece`` using BPE ranks (lower index = earlier merge = higher priority)."""
    if len(piece) == 0:
        return []
    ids = list(piece)
    while len(ids) >= 2:
        pairs = list(zip(ids, ids[1:]))
        bigram = min(pairs, key=lambda p: merge_rank.get(p, float("inf")))
        if bigram not in merge_rank:
            break
        new_id = 256 + merge_rank[bigram]
        a, b = bigram
        out: list[int] = []
        i = 0
        merged_once = False
        while i < len(ids):
            if (
                not merged_once
                and i < len(ids) - 1
                and ids[i] == a
                and ids[i + 1] == b
            ):
                out.append(new_id)
                merged_once = True
                i += 2
            else:
                out.append(ids[i])
                i += 1
        if out == ids:
            break
        ids = out
    return ids


class BPETokenizer:
    """
    Byte-level BPE after GPT-2 pretokenization.

    - ``vocab_size`` ids: ``0..vocab_size-1``, with ``eot_token_id == vocab_size - 1``.
    - Merge ids are ``256 .. vocab_size-2``; ranks map ``(a,b) -> merge_index`` so token id is ``256 + merge_index``.
    """

    def __init__(
        self,
        merges: list[tuple[int, int]],
        id_to_bytes: dict[int, bytes],
        vocab_size: int,
        eot_token_id: int,
    ) -> None:
        self.merges = merges
        self.id_to_bytes = id_to_bytes
        self.vocab_size = vocab_size
        self.eot_token_id = eot_token_id
        self.merge_rank: dict[tuple[int, int], int] = {
            pair: idx for idx, pair in enumerate(merges)
        }

    @classmethod
    def from_training(
        cls,
        merges: list[tuple[int, int]],
        id_to_bytes: dict[int, bytes],
        vocab_size: int,
    ) -> BPETokenizer:
        eot_token_id = vocab_size - 1
        return cls(merges, id_to_bytes, vocab_size, eot_token_id)

    def encode(self, text: str, *, add_eot_between_docs: bool = True) -> list[int]:
        """
        Encode ``text`` to token ids.

        If ``add_eot_between_docs`` and ``text`` contains ``<|endoftext|>``, inserts
        ``eot_token_id`` between documents (delimiter substrings are not fed to BPE).
        """
        if not add_eot_between_docs or EOT_STR not in text:
            return self._encode_segment(text)

        out: list[int] = []
        parts = text.split(EOT_STR)
        for i, part in enumerate(parts):
            seg = part.strip()
            if seg:
                out.extend(self._encode_segment(seg))
            if i < len(parts) - 1:
                out.append(self.eot_token_id)
        return out

    def _encode_segment(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in pretokenize(text):
            b = piece.encode("utf-8")
            if not b:
                continue
            ids.extend(_encode_piece(self.merge_rank, tuple(b)))
        return ids

    def decode(self, ids: list[int], *, errors: str = "replace") -> str:
        """Decode token ids to a Unicode string (re-inserts ``<|endoftext|>`` markers)."""
        parts: list[str] = []
        buf = bytearray()
        for tid in ids:
            if tid == self.eot_token_id:
                if buf:
                    parts.append(buf.decode("utf-8", errors=errors))
                    buf.clear()
                parts.append(EOT_STR)
                continue
            raw = self.id_to_bytes.get(tid)
            if raw is None:
                continue
            buf.extend(raw)
        if buf:
            parts.append(buf.decode("utf-8", errors=errors))
        return "".join(parts)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        id_bytes_serial = {str(k): list(v) for k, v in sorted(self.id_to_bytes.items())}
        payload: dict[str, Any] = {
            "vocab_size": self.vocab_size,
            "eot_token_id": self.eot_token_id,
            "merges": [list(p) for p in self.merges],
            "id_to_bytes": id_bytes_serial,
        }
        (d / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> BPETokenizer:
        d = Path(directory)
        payload = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
        vocab_size = int(payload["vocab_size"])
        eot_token_id = int(payload["eot_token_id"])
        merges = [tuple(p) for p in payload["merges"]]
        id_to_bytes = {int(k): bytes(v) for k, v in payload["id_to_bytes"].items()}
        return cls(merges, id_to_bytes, vocab_size, eot_token_id)
