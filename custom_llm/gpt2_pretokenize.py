"""GPT-2-style regex pretokenization (Unicode-aware, same pattern as reference stacks)."""

from __future__ import annotations

import regex as re

# OpenAI GPT-2 pretokenizer split pattern (requires the `regex` package for \\p{L}, \\p{N}).
GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> list[str]:
    """Return GPT-2 pretokens for ``text`` (empty strings omitted)."""
    return [p for p in GPT2_SPLIT_PATTERN.findall(text) if p]
