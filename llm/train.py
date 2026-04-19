"""Train ``TransformerLM`` on a tokenized corpus (CPU or CUDA)."""

from __future__ import annotations

import argparse
import array
import gc
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from llm.bpe_trainer import EOT_STR
from llm.data import SlidingWindowDataset
from llm.model import TransformerConfig, TransformerLM
from llm.tokenizer import BPETokenizer


def _use_tqdm_progress(*, no_progress: bool, force_tqdm: bool) -> bool:
    """Use tqdm in real terminals, or in Colab / Jupyter where cell output still updates."""
    if no_progress:
        return False
    if force_tqdm:
        return True
    if sys.stdout.isatty():
        return True
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("JPY_PARENT_PID"):
        return True
    return False


def load_corpus_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def encode_corpus(tok: BPETokenizer, text: str) -> torch.Tensor:
    ids = tok.encode(text, add_eot_between_docs=True)
    return _ids_to_storage_tensor(ids, tok.vocab_size)


def _ids_to_storage_tensor(ids: list[int], vocab_size: int) -> torch.Tensor:
    """Store token ids in the narrowest dtype that fits ``vocab_size`` (saves RAM on huge corpora)."""
    if vocab_size <= 65536:
        # uint16 is enough for default 10k vocab; keeps RAM ~2× lower than int64.
        for tid in ids:
            if tid < 0 or tid >= 65536:
                return torch.tensor(ids, dtype=torch.int32)
        return torch.tensor(ids, dtype=torch.uint16)
    return torch.tensor(ids, dtype=torch.int32)


def encode_corpus_file_streaming(
    path: Path,
    tok: BPETokenizer,
    *,
    label: str = "corpus",
    progress_interval_mb: int = 64,
) -> torch.Tensor:
    """
    Stream a plaintext corpus from disk: never holds the full file as one string or Python list of all ids.

    Documents are separated by ``EOT_STR`` (same as ``prepare_tinystories`` output). Token ids are collected in a
    compact ``array.array`` (type ``H`` when ids fit in uint16), then converted to a tensor.
    """
    out_h = array.array("H")
    buf = ""
    total_bytes = path.stat().st_size
    bytes_read = 0
    docs_seen = 0
    t0 = time.time()
    interval_bytes = max(1, progress_interval_mb) * 1024 * 1024
    next_report = interval_bytes

    def _report(force: bool = False) -> None:
        nonlocal next_report
        if not force and bytes_read < next_report:
            return
        next_report += interval_bytes
        elapsed = time.time() - t0
        pct = (100.0 * bytes_read / total_bytes) if total_bytes > 0 else 100.0
        print(
            f"[encode:{label}] {pct:5.1f}%  {bytes_read/1e9:.2f}GB/{total_bytes/1e9:.2f}GB  "
            f"docs={docs_seen:,}  elapsed={elapsed/60:.1f}m",
            flush=True,
        )

    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(2 * 1024 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk.encode("utf-8", errors="ignore"))
            buf += chunk
            while True:
                i = buf.find(EOT_STR)
                if i == -1:
                    break
                doc = buf[:i].strip()
                buf = buf[i + len(EOT_STR) :]
                if doc:
                    piece = tok.encode(doc, add_eot_between_docs=False)
                    if piece and max(piece) >= 65536:
                        raise SystemExit("Token id >= 65536; use --full_ram_encode or smaller vocab.")
                    out_h.extend(piece)
                out_h.append(tok.eot_token_id)
                docs_seen += 1
            _report()
        tail = buf.strip()
        if tail:
            piece = tok.encode(tail, add_eot_between_docs=False)
            if piece and max(piece) >= 65536:
                raise SystemExit("Token id >= 65536; use --full_ram_encode or smaller vocab.")
            out_h.extend(piece)
            docs_seen += 1

    _report(force=True)

    n_tok = len(out_h)
    print(
        f"[encode:{label}] file read + tokenize done; building torch tensor for {n_tok:,} tokens "
        f"(this step can take several minutes on large corpora; CPU may look idle in monitors)...",
        flush=True,
    )
    t_conv = time.time()
    # Faster and usually lower peak RAM than torch.tensor(out_h) on huge array.array.
    token_ids = torch.frombuffer(memoryview(out_h), dtype=torch.uint16).clone()
    del out_h
    gc.collect()
    print(
        f"[encode:{label}] tensor ready in {time.time() - t_conv:.1f}s "
        f"(dtype={token_ids.dtype}, numel={token_ids.numel():,})",
        flush=True,
    )
    return token_ids


def _load_token_ids_cache(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("token_ids"), torch.Tensor):
        return payload["token_ids"]
    raise SystemExit(f"Unsupported token-id cache format at {path}")


def _save_token_ids_cache(path: Path, token_ids: torch.Tensor, *, source: Path, tok: BPETokenizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_ids": token_ids.cpu(),
        "source_path": str(source),
        "source_size_bytes": source.stat().st_size if source.exists() else None,
        "vocab_size": tok.vocab_size,
        "eot_token_id": tok.eot_token_id,
    }
    torch.save(payload, path)


def _load_or_encode_token_ids(
    corpus_path: Path,
    tok: BPETokenizer,
    *,
    cache_path: Path | None,
    rebuild_cache: bool,
    full_ram_encode: bool,
    encode_progress_interval_mb: int,
    label: str,
) -> torch.Tensor:
    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        print(f"Loading cached {label} token ids: {cache_path}", flush=True)
        token_ids = _load_token_ids_cache(cache_path)
        print(f"Loaded cached {label} token ids: {len(token_ids):,} (dtype={token_ids.dtype})", flush=True)
        return token_ids

    print(f"Loading {label} text: {corpus_path}", flush=True)
    if full_ram_encode:
        text = load_corpus_text(corpus_path)
        print(f"Encoding {label} corpus to token ids (large files can take a few minutes)...", flush=True)
        token_ids = encode_corpus(tok, text)
        del text
        gc.collect()
    else:
        print(
            f"Encoding {label} corpus (streaming from disk, uint16 storage; use --full_ram_encode for legacy path)...",
            flush=True,
        )
        token_ids = encode_corpus_file_streaming(
            corpus_path,
            tok,
            label=label,
            progress_interval_mb=encode_progress_interval_mb,
        )

    if cache_path is not None:
        print(f"Saving {label} token-id cache to: {cache_path}", flush=True)
        _save_token_ids_cache(cache_path, token_ids, source=corpus_path, tok=tok)
    return token_ids


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
        x = x.long().to(device)
        y = y.long().to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def _ascii_progress_bar(current: int, total: int, width: int = 28) -> str:
    """Single-line bar for non-TTY logs (e.g. Colab ``!python``) where tqdm updates poorly."""
    if total <= 0:
        return "[" + "?" * width + "]"
    frac = min(max(current / total, 0.0), 1.0)
    filled = int(round(frac * width))
    filled = min(filled, width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


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
        "--bar_width",
        type=int,
        default=28,
        help="Width of the ASCII [###...] bar in non-TTY progress lines (default: 28)",
    )
    p.add_argument(
        "--force_tqdm",
        action="store_true",
        help="Use tqdm even when stdout is not a TTY (often invisible in Colab; prefer default)",
    )
    p.add_argument(
        "--full_ram_encode",
        action="store_true",
        help="Load each corpus fully into RAM then encode (high memory; default streams from disk for lower RAM).",
    )
    p.add_argument(
        "--encode_progress_interval_mb",
        type=int,
        default=64,
        help="Print streaming encode progress every N MB read (default: 64).",
    )
    p.add_argument(
        "--train_ids_cache",
        type=Path,
        default=None,
        help="Optional path to cache train token ids (.pt). Greatly reduces startup on subsequent runs.",
    )
    p.add_argument(
        "--val_ids_cache",
        type=Path,
        default=None,
        help="Optional path to cache validation token ids (.pt).",
    )
    p.add_argument(
        "--rebuild_ids_cache",
        action="store_true",
        help="Force re-encode corpus and overwrite token-id caches.",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tok = BPETokenizer.load(args.tokenizer_dir)
    token_ids = _load_or_encode_token_ids(
        args.corpus,
        tok,
        cache_path=args.train_ids_cache,
        rebuild_cache=args.rebuild_ids_cache,
        full_ram_encode=args.full_ram_encode,
        encode_progress_interval_mb=args.encode_progress_interval_mb,
        label="train",
    )
    print(f"Training sequence: {len(token_ids):,} token ids (dtype={token_ids.dtype})", flush=True)
    if len(token_ids) < args.context_length + 2:
        raise SystemExit("Training corpus is too small after tokenization.")

    eval_token_ids = token_ids
    if args.val_corpus is not None:
        eval_token_ids = _load_or_encode_token_ids(
            args.val_corpus,
            tok,
            cache_path=args.val_ids_cache,
            rebuild_cache=args.rebuild_ids_cache,
            full_ram_encode=args.full_ram_encode,
            encode_progress_interval_mb=args.encode_progress_interval_mb,
            label="val",
        )
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

    # Prefer tqdm in Colab/Jupyter too (matches notebook cell output like the reference logs).
    use_tqdm_loop = _use_tqdm_progress(no_progress=args.no_progress, force_tqdm=args.force_tqdm)
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
                f"Progress: printing every {args.progress_interval} steps (no TTY / no notebook). "
                "Pass --force_tqdm to use tqdm anyway.",
                flush=True,
            )

    last_avg50: float | None = None
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

        x = x.long().to(device)
        y = y.long().to(device)
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

        if use_simple_lines and (
            local_i % args.progress_interval == 0
            or local_i == 1
            or local_i == args.steps
        ):
            elapsed = time.time() - t0
            pct = 100.0 * local_i / args.steps
            bar = _ascii_progress_bar(local_i, args.steps, width=max(8, args.bar_width))
            print(
                f"[train] {bar} {pct:5.1f}%  {local_i}/{args.steps}  "
                f"global_step={step}/{global_total}  loss={loss.item():.4f}  "
                f"lr={opt.param_groups[0]['lr']:.2e}  elapsed={elapsed/60:.1f}m",
                flush=True,
            )

        if local_i % 50 == 0:
            avg = running_loss / 50
            running_loss = 0.0
            elapsed = time.time() - t0
            last_avg50 = avg
            if args.no_progress:
                print(
                    f"step {step}/{global_total}  loss {avg:.4f}  lr {opt.param_groups[0]['lr']:.2e}  "
                    f"elapsed {elapsed/60:.1f}m",
                    flush=True,
                )
            elif not use_tqdm_loop:
                bar = _ascii_progress_bar(local_i, args.steps, width=max(8, args.bar_width))
                pct = 100.0 * local_i / args.steps
                print(
                    f"[train] {bar} {pct:5.1f}%  avg50 @ step {step}: loss={avg:.4f}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}  elapsed={elapsed/60:.1f}m",
                    flush=True,
                )

        if use_tqdm_loop:
            elapsed = time.time() - t0
            train_iter.set_postfix(
                avg50=f"{last_avg50:.4f}" if last_avg50 is not None else "---",
                elapsed_min=f"{elapsed / 60:.1f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
                step=f"{step}/{global_total}",
                refresh=False,
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
