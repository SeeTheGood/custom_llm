"""Greedy / top-p sampling from a trained ``TransformerLM`` checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from custom_llm.model import TransformerConfig, TransformerLM
from custom_llm.tokenizer import BPETokenizer


@torch.no_grad()
def sample_greedy(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    ids = list(prompt_ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(x)
        next_id = int(logits[0, -1].argmax())
        ids.append(next_id)
        x = torch.tensor([ids], dtype=torch.long, device=device)
    return ids


@torch.no_grad()
def sample_top_p(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    top_p: float,
    temperature: float,
) -> list[int]:
    ids = list(prompt_ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        s = sorted_probs.sum()
        sorted_probs = sorted_probs / (s + 1e-12)
        choice = torch.multinomial(sorted_probs, num_samples=1)
        next_id = int(sorted_idx[0, choice.item()])
        ids.append(next_id)
        x = torch.tensor([ids], dtype=torch.long, device=device)
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description="Sample from trained LM (CPU).")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer_dir", type=Path, required=True)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--top_p", type=float, default=0.0, help="If >0, use nucleus sampling")
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device(args.device)
    tok = BPETokenizer.load(args.tokenizer_dir)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = TransformerConfig(**ckpt["model_cfg"])
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt_ids = tok.encode(args.prompt, add_eot_between_docs=False)
    if args.top_p > 0:
        out_ids = sample_top_p(
            model,
            prompt_ids,
            args.max_new_tokens,
            device,
            args.top_p,
            args.temperature,
        )
    else:
        out_ids = sample_greedy(model, prompt_ids, args.max_new_tokens, device)
    print(tok.decode(out_ids))


if __name__ == "__main__":
    main()
