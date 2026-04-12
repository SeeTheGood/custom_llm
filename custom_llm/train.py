"""Train ``TransformerLM`` on a tokenized corpus (CPU or CUDA)."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
    *,
    show_progress: bool = False,
) -> float:
    model.eval()
    losses: list[float] = []
    batches = loader
    if show_progress:
        batches = tqdm(
            loader,
            desc="Validation",
            leave=False,
            file=sys.stdout,
            mininterval=0.2,
        )
    for x, y in batches:
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
    p = argparse.ArgumentParser(description="Train Transformer LM (CPU or CUDA).")
    p.add_argument("--corpus", type=Path, required=True, help="Training plaintext with <|endoftext|> separators")
    p.add_argument(
        "--val_corpus",
        type=Path,
        default=None,
        help="Optional held-out plaintext for validation loss (e.g. TinyStories validation split). "
        "If omitted, validation uses random windows from --corpus (not true held-out).",
    )
    p.add_argument("--tokenizer_dir", type=Path, required=True, help="Directory with tokenizer.json")
    p.add_argument("--out_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Training optimizer steps: total steps from scratch, or *additional* steps if --resume is set",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Load weights (and optimizer if saved) from this checkpoint; --steps becomes extra steps to run",
    )
    p.add_argument(
        "--cosine_decay",
        action="store_true",
        help="After warmup, decay LR with cosine schedule to --min_lr",
    )
    p.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="LR floor when --cosine_decay is set (default: 1e-5)",
    )
    p.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable all training progress output except eval lines and every-50-step summary",
    )
    p.add_argument(
        "--progress_interval",
        type=int,
        default=10,
        help="When stdout is not a TTY (e.g. Colab !python), print a line every N steps (default: 10)",
    )
    p.add_argument(
        "--force_tqdm",
        action="store_true",
        help="Use tqdm even when stdout is not a TTY (often invisible in Colab; prefer default)",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tok = BPETokenizer.load(args.tokenizer_dir)
    print(f"Loading training text: {args.corpus}", flush=True)
    text = load_corpus_text(args.corpus)
    print("Encoding training corpus to token ids (large files can take a few minutes)...", flush=True)
    token_ids = encode_corpus(tok, text)
    print(f"Training sequence: {len(token_ids):,} token ids", flush=True)
    if len(token_ids) < args.context_length + 2:
        raise SystemExit("Training corpus is too small after tokenization.")

    eval_token_ids = token_ids
    if args.val_corpus is not None:
        print(f"Loading validation text: {args.val_corpus}", flush=True)
        val_text = load_corpus_text(args.val_corpus)
        print("Encoding validation corpus to token ids...", flush=True)
        eval_token_ids = encode_corpus(tok, val_text)
        if len(eval_token_ids) < args.context_length + 2:
            raise SystemExit("Validation corpus is too small after tokenization.")
        print(f"Validation: held-out file {args.val_corpus} ({len(eval_token_ids)} token ids)")
    else:
        print("Validation: random windows from training corpus (not held-out)")

    cfg = TransformerConfig(
        vocab_size=tok.vocab_size,
        context_length=args.context_length,
        d_model=512,
        n_heads=16,
        n_layers=4,
        d_ff=1344,
        dropout=0.1,
    )

    resume_from = 0
    resume_ckpt: dict | None = None
    if args.resume is not None:
        try:
            resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            resume_ckpt = torch.load(args.resume, map_location=device)
        saved_cfg = TransformerConfig(**resume_ckpt["model_cfg"])
        if saved_cfg.vocab_size != cfg.vocab_size or saved_cfg.context_length != cfg.context_length:
            raise SystemExit(
                "Checkpoint config does not match tokenizer/model settings (vocab_size or context_length)."
            )
        resume_from = int(resume_ckpt.get("step", 0))
        print(f"Resuming from step {resume_from} (will run {args.steps} more steps → up to {resume_from + args.steps})")

    model = TransformerLM(cfg).to(device)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"])
    n_params = model.count_parameters()
    print(f"Parameters: {n_params / 1e6:.2f}M (target ~17M)")

    # One random window index per optimizer step in this segment
    samples_per_segment = args.steps * args.batch_size
    train_ds = SlidingWindowDataset(
        token_ids,
        args.context_length,
        num_samples=samples_per_segment,
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
        eval_token_ids,
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
    if resume_ckpt is not None and "optimizer_state" in resume_ckpt:
        opt.load_state_dict(resume_ckpt["optimizer_state"])
        print("Loaded optimizer state from checkpoint.")

    global_total = resume_from + args.steps

    def lr_at_step(step: int) -> float:
        if resume_from == 0 and step <= args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        if args.cosine_decay:
            if resume_from > 0:
                rel = step - resume_from
                progress = (rel - 1) / max(args.steps - 1, 1) if args.steps > 1 else 1.0
            else:
                rel = step - args.warmup_steps
                denom = max(args.steps - args.warmup_steps, 1)
                progress = min(max(rel / denom, 0.0), 1.0)
            cos_part = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr + (args.lr - args.min_lr) * cos_part
        return args.lr

    best_val = float(resume_ckpt["val_loss"]) if resume_ckpt and "val_loss" in resume_ckpt else math.inf

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    it = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0

    last_val = best_val

    # Colab/Jupyter `!python` runs a non-TTY subprocess: tqdm's \\r updates are often buffered
    # until the process ends. Prefer flushed line prints unless we're in a real terminal.
    _tty = sys.stdout.isatty()
    use_tqdm_loop = (not args.no_progress) and (_tty or args.force_tqdm)
    use_simple_lines = (not args.no_progress) and (not use_tqdm_loop)

    if use_tqdm_loop:
        train_iter: range | tqdm = tqdm(
            range(1, args.steps + 1),
            total=args.steps,
            desc="Training",
            unit="step",
            file=sys.stdout,
            mininterval=0.1,
            dynamic_ncols=True,
        )
    else:
        train_iter = range(1, args.steps + 1)
        if use_simple_lines:
            print(
                f"Progress: printing every {args.progress_interval} steps (non-TTY / Colab-friendly). "
                "Use a real terminal or --force_tqdm for tqdm.",
                flush=True,
            )

    for local_i in train_iter:
        step = resume_from + local_i
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
        if use_tqdm_loop:
            train_iter.set_postfix(
                batch_loss=f"{loss.item():.3f}",
                lr=f"{opt.param_groups[0]['lr']:.1e}",
                step=f"{step}/{global_total}",
            )

        if use_simple_lines and (
            local_i % args.progress_interval == 0
            or local_i == 1
            or local_i == args.steps
        ):
            elapsed = time.time() - t0
            pct = 100.0 * local_i / args.steps
            print(
                f"[train] {local_i}/{args.steps} ({pct:.1f}%)  "
                f"global_step={step}/{global_total}  loss={loss.item():.4f}  "
                f"lr={opt.param_groups[0]['lr']:.2e}  elapsed={elapsed/60:.1f}m",
                flush=True,
            )

        if local_i % 50 == 0:
            avg = running_loss / 50
            running_loss = 0.0
            elapsed = time.time() - t0
            if args.no_progress:
                print(
                    f"step {step}/{global_total}  loss {avg:.4f}  lr {opt.param_groups[0]['lr']:.2e}  "
                    f"elapsed {elapsed/60:.1f}m",
                    flush=True,
                )
            elif use_tqdm_loop:
                train_iter.set_postfix(
                    avg50=f"{avg:.4f}",
                    lr=f"{opt.param_groups[0]['lr']:.2e}",
                    elapsed_min=f"{elapsed/60:.1f}",
                    step=f"{step}/{global_total}",
                )
            else:
                print(
                    f"[train] avg50 @ step {step}: loss={avg:.4f}  lr={opt.param_groups[0]['lr']:.2e}  "
                    f"elapsed={elapsed/60:.1f}m",
                    flush=True,
                )

        if step % args.eval_every == 0 or local_i == args.steps:
            if use_simple_lines:
                print("  (running validation...)", flush=True)
            val_loss = evaluate(
                model,
                eval_loader,
                device,
                show_progress=use_tqdm_loop and (not args.no_progress),
            )
            last_val = val_loss
            val_label = "validation loss (held-out)" if args.val_corpus is not None else "validation loss (train windows)"
            msg = f"  >> {val_label} {val_loss:.4f}"
            print(msg, flush=True)
            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "model_cfg": cfg.__dict__,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                }
                torch.save(ckpt, args.out_dir / "best.pt")
            model.train()

    # Always save last weights for faithful continuation (same corpus, more steps later)
    torch.save(
        {
            "model_cfg": cfg.__dict__,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "step": resume_from + args.steps,
            "val_loss": last_val,
        },
        args.out_dir / "latest.pt",
    )
    print(f"Wrote {args.out_dir / 'latest.pt'} (resume with: --resume {args.out_dir / 'latest.pt'})", flush=True)

    total_time = time.time() - t0
    print(f"Done. Best val loss {best_val:.4f}  total time {total_time/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
