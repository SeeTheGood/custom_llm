#!/usr/bin/env python3
"""
Run ``python -m llm.train ...``, then copy artifacts to Google Drive, then smoke-test with ``llm.sample``.

Designed for Google Colab after ``drive.mount``:

  %cd /content/custom_llm
  !python scripts/train_backup_sample.py \\
      --drive-backup "/content/drive/MyDrive/building_LLM/run_2025-04-20" \\
      --tokenizer-dir tokenizer \\
      --checkpoint-dir /content/checkpoints \\
      -- \\
      --corpus /content/TinyStoriesV2-GPT4-train.txt \\
      --val_corpus /content/TinyStoriesV2-GPT4-valid.txt \\
      --tokenizer_dir tokenizer \\
      --out_dir /content/checkpoints \\
      --device cuda \\
      --batch_size 32 \\
      --context_length 256 \\
      --steps 5000

Everything after ``--`` is forwarded to ``llm.train`` unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    if "--" not in sys.argv:
        raise SystemExit(
            "Missing ``--`` separator.\n"
            "Example:\n"
            "  python scripts/train_backup_sample.py --drive-backup /path/on/drive/run1 "
            "--tokenizer-dir tokenizer --checkpoint-dir /content/checkpoints -- "
            "--corpus data/train.txt --val_corpus data/val.txt --tokenizer_dir tokenizer "
            "--out_dir /content/checkpoints --device cuda --steps 5000"
        )
    sep = sys.argv.index("--")
    our_argv = sys.argv[1:sep]
    train_argv = sys.argv[sep + 1 :]

    p = argparse.ArgumentParser(description="Train, backup to Drive, sample smoke test.")
    p.add_argument(
        "--drive-backup",
        type=Path,
        required=True,
        help="Folder on Google Drive (mounted) to copy checkpoints into.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/content/checkpoints"),
        help="Directory where llm.train writes checkpoints (default: /content/checkpoints).",
    )
    p.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="Tokenizer directory (for sample step).",
    )
    p.add_argument(
        "--copy-caches",
        nargs="*",
        type=Path,
        default=[],
        help="Optional token-id cache .pt paths to copy next to checkpoints on Drive.",
    )
    p.add_argument(
        "--sample-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for sample (default: best.pt in checkpoint-dir if present, else latest.pt).",
    )
    p.add_argument("--sample-device", type=str, default="cuda")
    p.add_argument("--sample-prompt", type=str, default="Once upon a time")
    p.add_argument("--sample-max-tokens", type=int, default=80)
    p.add_argument("--skip-sample", action="store_true", help="Only train + backup, no llm.sample.")
    args = p.parse_args(our_argv)
    return args, train_argv


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        print(f"[backup] skip missing directory: {src}", flush=True)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        print(f"[backup] {item} -> {target}", flush=True)


def main() -> None:
    args, train_argv = _parse_args()
    if not train_argv:
        raise SystemExit("No arguments after -- for llm.train.")

    drive_dir = args.drive_backup.expanduser().resolve()
    ckpt_dir = args.checkpoint_dir.expanduser().resolve()
    tok_dir = args.tokenizer_dir.expanduser().resolve()

    print("[train] starting: python -m llm.train " + " ".join(train_argv), flush=True)
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "llm.train", *[str(x) for x in train_argv]],
        cwd=_ROOT,
        check=False,
    )
    if r.returncode != 0:
        raise SystemExit(f"llm.train failed with exit code {r.returncode}")

    print(f"[train] finished in {(time.time() - t0) / 60:.1f}m", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_root = drive_dir / f"checkpoints_backup_{stamp}"
    backup_root.mkdir(parents=True, exist_ok=True)
    print(f"[backup] copying {ckpt_dir} -> {backup_root}", flush=True)
    _copy_tree(ckpt_dir, backup_root)

    for c in args.copy_caches:
        c = c.expanduser().resolve()
        if c.is_file():
            dest = backup_root / c.name
            shutil.copy2(c, dest)
            print(f"[backup] cache {c} -> {dest}", flush=True)

    if args.skip_sample:
        print("[sample] skipped (--skip-sample)", flush=True)
        print(f"DONE: backups under {backup_root}", flush=True)
        return

    sample_ckpt = args.sample_checkpoint
    if sample_ckpt is None:
        best = ckpt_dir / "best.pt"
        latest = ckpt_dir / "latest.pt"
        sample_ckpt = best if best.is_file() else latest
    else:
        sample_ckpt = sample_ckpt.expanduser().resolve()

    if not sample_ckpt.is_file():
        raise SystemExit(f"No checkpoint for sample: {sample_ckpt}")

    print(f"[sample] checkpoint={sample_ckpt}", flush=True)
    r2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm.sample",
            "--checkpoint",
            str(sample_ckpt),
            "--tokenizer_dir",
            str(tok_dir),
            "--prompt",
            args.sample_prompt,
            "--max_new_tokens",
            str(args.sample_max_tokens),
            "--device",
            args.sample_device,
        ],
        cwd=_ROOT,
        check=False,
    )
    if r2.returncode != 0:
        raise SystemExit(f"llm.sample failed with exit code {r2.returncode}")

    print(f"DONE: backups under {backup_root}", flush=True)


if __name__ == "__main__":
    main()
