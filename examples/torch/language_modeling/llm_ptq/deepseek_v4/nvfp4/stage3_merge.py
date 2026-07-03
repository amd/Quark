#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Stage 3: attach calibrated per-expert ``input_scale`` tensors to a DeepSeek-V4-Pro
NVFP4 checkpoint via a sidecar safetensors file and an index update -- no shard
rewriting.

Stage 1 (``stage1_quantize_weight.py``) emits the packed ``weight``, the FP8
per-group ``weight_scale`` and the F32 global ``weight_scale_2`` for every
quantized expert, but it cannot produce the activation ``input_scale`` -- that
value comes from activation calibration (Stage 2,
``stage2_calibrate_input_scale.py``) and is supplied as a separate
safetensors file (one F32 scalar per expert projection, keyed
``<weight_name>.input_scale``).

A HuggingFace checkpoint locates each tensor through ``model.safetensors.index.json``
(a ``tensor_name -> filename`` map); there is no requirement that related tensors
live in the same shard. So instead of rewriting every multi-GB shard to embed the
scalars, this script simply:

  1. copies the ``input_scale`` safetensors file into the checkpoint as a sidecar
     shard (default name ``input_scale.safetensors``);
  2. adds each ``input_scale`` key to ``weight_map`` pointing at that sidecar;
  3. updates ``metadata.total_size``.

This touches only the index and adds one small file. It is idempotent (keys that
already point at the sidecar are left alone) and validates that every
``input_scale`` corresponds to an NVFP4-quantized weight (``.weight`` and
``.weight_scale_2`` present) before wiring it up. The result is the final NVFP4
checkpoint: every quantized expert has ``weight`` / ``weight_scale`` /
``weight_scale_2`` / ``input_scale``.

Usage::

    # Dry run first -- reports coverage, modifies nothing:
    python stage3_merge.py \\
        --checkpoint-path /path/to/output-nvfp4 \\
        --input-scale-path /path/to/input_scale.safetensors \\
        --dry-run

    # Real run:
    python stage3_merge.py \\
        --checkpoint-path /path/to/output-nvfp4 \\
        --input-scale-path /path/to/input_scale.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

SIDECAR_NAME = "input_scale.safetensors"


def _read_safetensors_header(path: str) -> dict:
    """Read the JSON header (tensor metadata) of a safetensors file.

    :param str path: Path to the safetensors file.

    :return: Parsed header dict mapping tensor name -> metadata.
    :rtype: dict
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len))


def main(args: argparse.Namespace) -> None:
    """Attach the ``input_scale`` sidecar to the checkpoint and wire up the index.

    :param argparse.Namespace args: Parsed CLI arguments.

    :return: None
    """
    ckpt = args.checkpoint_path
    index_path = os.path.join(ckpt, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        sys.exit(f"ERROR: index not found: {index_path}")
    if not os.path.exists(args.input_scale_path):
        sys.exit(f"ERROR: input_scale file not found: {args.input_scale_path}")

    sidecar_path = os.path.join(ckpt, args.sidecar_name)

    logger.info(
        "Attach input_scale sidecar to NVFP4 checkpoint\n"
        f"  checkpoint : {ckpt}\n"
        f"  sidecar    : {sidecar_path}\n"
        f"  dry-run    : {args.dry_run}"
    )

    # Read the header from the source file (the sidecar may not exist yet, and a
    # dry run must not write anything into the checkpoint).
    scale_header = _read_safetensors_header(args.input_scale_path)
    scale_keys = [k for k in scale_header if k != "__metadata__"]

    with open(index_path) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    to_wire, unmatched, already = [], 0, 0
    for name in scale_keys:
        if not name.endswith(".input_scale"):
            unmatched += 1
            continue
        base = name[: -len(".input_scale")]
        if base + ".weight" not in weight_map or base + ".weight_scale_2" not in weight_map:
            unmatched += 1
            continue
        if weight_map.get(name) == args.sidecar_name:
            already += 1
            continue
        to_wire.append(name)

    logger.info(
        f"  input_scale entries     : {len(scale_keys)}\n"
        f"  will wire into index    : {len(to_wire)}\n"
        f"  already wired (skipped) : {already}\n"
        f"  unmatched (skipped)     : {unmatched}"
    )
    if unmatched:
        logger.warning(
            f"{unmatched} input_scale entries had no matching NVFP4 weight; they will NOT be wired into the index."
        )
    if args.dry_run:
        logger.info("dry-run: nothing copied, index not modified.")
        return

    # Copy the input_scale file into the checkpoint as the sidecar (unless it is
    # already the sidecar in place).
    if os.path.abspath(args.input_scale_path) != os.path.abspath(sidecar_path):
        shutil.copyfile(args.input_scale_path, sidecar_path)

    for name in to_wire:
        weight_map[name] = args.sidecar_name

    meta = index.setdefault("metadata", {})
    if "total_size" in meta:
        # 4 bytes per F32 scalar added to the addressable tensor set.
        meta["total_size"] = int(meta["total_size"]) + len(to_wire) * 4
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Done. Wired {len(to_wire)} input_scale tensors -> {args.sidecar_name}.\nUpdated index: {index_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed arguments namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Attach calibrated per-expert input_scale tensors to a "
            "DeepSeek-V4-Pro NVFP4 checkpoint via a sidecar safetensors file and "
            "an index update (no shard rewriting)."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        dest="checkpoint_path",
        required=True,
        help="Path to the NVFP4-quantized checkpoint directory (Stage 1 output).",
    )
    parser.add_argument(
        "--input-scale-path",
        dest="input_scale_path",
        required=True,
        help="Local path to the input_scale safetensors file (Stage 2 output); "
        "copied into the checkpoint as the sidecar.",
    )
    parser.add_argument(
        "--sidecar-name",
        dest="sidecar_name",
        default=SIDECAR_NAME,
        help="Filename for the sidecar inside the checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report coverage without modifying the index.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
