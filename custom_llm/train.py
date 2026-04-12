"""Train ``TransformerLM`` on a tokenized corpus (CPU)."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from custom_llm.data import SlidingWindowDataset
from custom_llm.model import TransformerConfig, TransformerLM
from custom_llm.tokenizer import BPETokenizer


def load_corpus_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def encode_corpus(tok: BPETokenizer, text: str) -> torch.Tensor:
    ids = tok.encode(text, add_eot_between_docs=True)
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def main() -> None:
    p = argparse.ArgumentParser(description="Train Transformer LM (CPU).")
    p.add_argument("--corpus", type=Path, required=True, help="Plaintext corpus with <|endoftext|> separators")
    p.add_argument("--tokenizer_dir", type=Path, required=True, help="Directory with tokenizer.json")
    p.add_argument("--out_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tok = BPETokenizer.load(args.tokenizer_dir)
    text = load_corpus_text(args.corpus)
    token_ids = encode_corpus(tok, text)
    if len(token_ids) < args.context_length + 2:
        raise SystemExit("Corpus is too small after tokenization.")

    cfg = TransformerConfig(
        vocab_size=tok.vocab_size,
        context_length=args.context_length,
        d_model=512,
        n_heads=16,
        n_layers=4,
        d_ff=1344,
        dropout=0.1,
    )
    model = TransformerLM(cfg).to(device)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params / 1e6:.2f}M (target ~17M)")

    samples_per_epoch = args.steps * args.batch_size
    train_ds = SlidingWindowDataset(
        token_ids,
        args.context_length,
        num_samples=samples_per_epoch,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_ds.resample_starts()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    eval_ds = SlidingWindowDataset(
        token_ids,
        args.context_length,
        num_samples=args.eval_batches * args.batch_size,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    eval_ds.resample_starts()
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    def lr_at_step(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(args.warmup_steps, 1)
        return args.lr

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    it = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0
    best_val = math.inf

    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at_step(step)

        try:
            x, y = next(it)
        except StopIteration:
            train_ds.resample_starts()
            it = iter(train_loader)
            x, y = next(it)

        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        running_loss += loss.item()
        if step % 50 == 0:
            avg = running_loss / 50
            running_loss = 0.0
            elapsed = time.time() - t0
            print(f"step {step}/{args.steps}  loss {avg:.4f}  lr {opt.param_groups[0]['lr']:.2e}  elapsed {elapsed/60:.1f}m")

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, eval_loader, device)
            print(f"  >> validation loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "model_cfg": cfg.__dict__,
                    "model_state": model.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                }
                torch.save(ckpt, args.out_dir / "best.pt")
            model.train()

    total_time = time.time() - t0
    print(f"Done. Best val loss {best_val:.4f}  total time {total_time/60:.1f}m")


if __name__ == "__main__":
    main()
