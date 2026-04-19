"""Datasets for next-token prediction on flat token id sequences."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    Random (or fixed) windows of length ``context_length + 1`` for LM training.

    Each item is ``(input_ids [T], target_ids [T])`` with ``target_ids[t] = seq[t+1]``.
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        context_length: int,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if token_ids.dim() != 1:
            raise ValueError("token_ids must be 1-D")
        self.token_ids = token_ids
        self.context_length = context_length
        self.max_start = len(token_ids) - context_length - 1
        if self.max_start < 1:
            raise ValueError("Corpus too short for the requested context length.")
        if num_samples is None:
            num_samples = self.max_start
        self.num_samples = num_samples
        self.generator = generator
        self._starts: torch.Tensor | None = None

    def resample_starts(self) -> None:
        g = self.generator or torch.Generator(device="cpu")
        self._starts = torch.randint(
            0,
            self.max_start,
            (self.num_samples,),
            generator=g,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._starts is None:
            self.resample_starts()
        assert self._starts is not None
        start = int(self._starts[index].item())
        chunk = self.token_ids[start : start + self.context_length + 1]
        x = chunk[:-1].contiguous()
        y = chunk[1:].contiguous()
        return x, y
