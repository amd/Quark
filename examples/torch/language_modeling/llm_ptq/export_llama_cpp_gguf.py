#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Export a Quark AWQ checkpoint to llama.cpp GGUF."""

from __future__ import annotations

import argparse
from pathlib import Path

from quark.torch import export_llama_cpp_gguf, list_llama_cpp_export_formats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Quark AWQ checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="GGUF output directory")
    parser.add_argument(
        "--format",
        default="q4_k_m",
        choices=list_llama_cpp_export_formats(),
        help="Target llama.cpp export format",
    )
    parser.add_argument("--name", help="Output name stem")
    parser.add_argument("--llama-cpp-dir", default="/home/l/work/llama.cpp")
    parser.add_argument("--libggml", help="Path to libggml.so")
    parser.add_argument("--tokenizer-source", help="Tokenizer directory override")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--pack-method", choices=["order", "reorder"], default="reorder")
    parser.add_argument("--split-max-size", default="8G")
    parser.add_argument("--max-tensors", type=int, help="Debug partial conversion")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = export_llama_cpp_gguf(
        quark_model_dir=args.model,
        output_dir=args.output_dir,
        export_format=args.format,
        name=args.name,
        llama_cpp_dir=args.llama_cpp_dir,
        libggml=args.libggml,
        tokenizer_source=args.tokenizer_source,
        group_size=args.group_size,
        pack_method=args.pack_method,
        split_max_size=args.split_max_size,
        max_tensors=args.max_tensors,
        dry_run=args.dry_run,
        keep_staging=args.keep_staging,
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
