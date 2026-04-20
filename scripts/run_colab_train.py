#!/usr/bin/env python3
"""
Colab-oriented training launcher: copy TinyStories from Drive → /content, sanity-check GPU, run ``llm.train``.

**Before running:** mount Drive in a separate cell::

    from google.colab import drive
    drive.mount("/content/drive")

**Then** from repo root::

    %cd /content/custom_llm
    !python scripts/run_colab_train.py

Override paths, steps, or pass extra ``llm.train`` flags after ``--``::

    !python scripts/run_colab_train.py --steps 10000 -- \\
        --cosine_decay --min_lr 1e-5

Defaults avoid earlier issues: corpora read from fast local disk, token caches and
checkpoints under ``/content`` (not a full Drive), and CUDA is verified before training starts.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DRIVE_DATA = Path("/content/drive/MyDrive/building_LLM/data")
DEFAULT_TRAIN_NAME = "TinyStoriesV2-GPT4-train.txt"
DEFAULT_VAL_NAME = "TinyStoriesV2-GPT4-valid.txt"
DEFAULT_CONTENT = Path("/content")
DEFAULT_CACHE_TRAIN = Path("/content/cache_train.pt")
DEFAULT_CACHE_VAL = Path("/content/cache_val.pt")
DEFAULT_OUT_DIR = Path("/content/checkpoints")


def _split_extra(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    i = argv.index("--")
    return argv[:i], argv[i + 1 :]


def _copy_if_needed(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise SystemExit(
            f"Missing source file: {src}\n"
            "Mount Google Drive first, or fix --drive-data-dir / file names.\n"
            "Open the folder in Drive and match names exactly (case-sensitive)."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        print(f"[copy] skip (already same size): {dst.name}", flush=True)
        return
    print(f"[copy] {src} → {dst} ({src.stat().st_size / 1e9:.2f} GB)", flush=True)
    shutil.copy2(src, dst)


def _cuda_sanity(device: str) -> None:
    if device.lower().startswith("cpu"):
        return
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "You chose --device cuda but torch.cuda.is_available() is False.\n"
            "Colab: Runtime → Change runtime type → GPU, then Runtime → Restart runtime, re-run from the top."
        )
    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except Exception as e:
        raise SystemExit(
            "CUDA sanity check failed (could not allocate on GPU).\n"
            f"Detail: {e!r}\n"
            "Try: Restart runtime, run only this notebook, then `!nvidia-smi` before training."
        ) from e


def main() -> None:
    raw_argv = sys.argv[1:]
    our_argv, train_extra = _split_extra(raw_argv)

    p = argparse.ArgumentParser(
        description="Copy TinyStories GPT4 exports from Drive to /content and run llm.train safely on Colab."
    )
    p.add_argument(
        "--drive-data-dir",
        type=Path,
        default=DEFAULT_DRIVE_DATA,
        help=f"Folder on mounted Drive containing the .txt files (default: {DEFAULT_DRIVE_DATA})",
    )
    p.add_argument("--train-name", default=DEFAULT_TRAIN_NAME)
    p.add_argument("--val-name", default=DEFAULT_VAL_NAME)
    p.add_argument(
        "--content-dir",
        type=Path,
        default=DEFAULT_CONTENT,
        help="Where to copy corpora (default: /content)",
    )
    p.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy from Drive; expect corpora already at --content-dir with the default names.",
    )
    p.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer"))
    p.add_argument("--train-ids-cache", type=Path, default=DEFAULT_CACHE_TRAIN)
    p.add_argument("--val-ids-cache", type=Path, default=DEFAULT_CACHE_VAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--skip-cuda-check", action="store_true")
    args = p.parse_args(our_argv)

    train_local = args.content_dir / args.train_name
    val_local = args.content_dir / args.val_name

    if not args.no_copy:
        _copy_if_needed(args.drive_data_dir / args.train_name, train_local)
        _copy_if_needed(args.drive_data_dir / args.val_name, val_local)
    else:
        for path, label in ((train_local, "train"), (val_local, "val")):
            if not path.is_file():
                raise SystemExit(f"--no-copy but missing {label} corpus: {path}")

    if not args.skip_cuda_check:
        _cuda_sanity(args.device)

    tok_dir = args.tokenizer_dir
    if not tok_dir.is_absolute():
        tok_dir = _ROOT / tok_dir
    if not (tok_dir / "tokenizer.json").is_file():
        raise SystemExit(
            f"No tokenizer.json in {tok_dir}\n"
            "Train a tokenizer first, e.g.:\n"
            f"  python scripts/train_tokenizer.py --corpus {train_local} --out_dir tokenizer --vocab_size 10000"
        )

    train_cmd = [
        sys.executable,
        "-m",
        "llm.train",
        "--corpus",
        str(train_local),
        "--val_corpus",
        str(val_local),
        "--tokenizer_dir",
        str(tok_dir),
        "--train_ids_cache",
        str(args.train_ids_cache),
        "--val_ids_cache",
        str(args.val_ids_cache),
        "--out_dir",
        str(args.out_dir),
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--context_length",
        str(args.context_length),
        "--steps",
        str(args.steps),
        *train_extra,
    ]

    print("[train] " + " ".join(train_cmd[2:]), flush=True)
    r = subprocess.run(train_cmd, cwd=_ROOT)
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
