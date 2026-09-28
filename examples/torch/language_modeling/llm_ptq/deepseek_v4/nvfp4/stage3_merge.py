#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Stage 3: attach calibrated per-expert ``input_scale`` tensors to a DeepSeek-V4-Pro
NVFP4 checkpoint by appending them into an existing shard -- no separate file added.

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

  1. appends the ``input_scale`` tensors into the last existing shard;
  2. adds each ``input_scale`` key to ``weight_map`` pointing at that shard;
  3. updates ``metadata.total_size``.

This touches only the target shard and the index. It is idempotent (keys that
already point at an existing shard are left alone) and validates that every
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
import struct
import sys

from safetensors.torch import load_file, save_file

from quark.common.utils.log import ScreenLogger
from quark.torch.quantization.config.config import QConfig
from quark.torch.quantization.config.template import QuantizationSchemeCollection

logger = ScreenLogger(__name__)


def _nvfp4_input_tensors_dict() -> list:
    scheme = QuantizationSchemeCollection().get_scheme("nvfp4")
    return QConfig(global_quant_config=scheme.config).to_dict()["global_quant_config"]["input_tensors"]


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
    """Attach the ``input_scale`` tensors to the checkpoint and wire up the index.

    :param argparse.Namespace args: Parsed CLI arguments.

    :return: None
    """
    ckpt = args.checkpoint_path
    index_path = os.path.join(ckpt, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        sys.exit(f"ERROR: index not found: {index_path}")
    if not os.path.exists(args.input_scale_path):
        # Sidecar removed by a prior successful run — nothing left to merge.
        logger.info(f"input_scale file not found at {args.input_scale_path}; assuming already merged. Exiting.")
        return

    logger.info(
        "Attach input_scale tensors into existing checkpoint shard\n"
        f"  checkpoint : {ckpt}\n"
        f"  dry-run    : {args.dry_run}"
    )

    # Read the header from the source file without modifying the checkpoint.
    scale_header = _read_safetensors_header(args.input_scale_path)
    scale_keys = [k for k in scale_header if k != "__metadata__"]

    with open(index_path) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    target_shard_name = sorted(set(weight_map.values()))[-1]
    target_shard_path = os.path.join(ckpt, target_shard_name)

    to_wire, unmatched, already = [], 0, 0
    for name in scale_keys:
        if not name.endswith(".input_scale"):
            unmatched += 1
            continue
        base = name[: -len(".input_scale")]
        if base + ".weight" not in weight_map or base + ".weight_scale_2" not in weight_map:
            unmatched += 1
            continue
        if weight_map.get(name) == target_shard_name:
            already += 1
            continue
        to_wire.append(name)

    logger.info(
        f"  target shard            : {target_shard_name}\n"
        f"  input_scale entries     : {len(scale_keys)}\n"
        f"  will wire into index    : {len(to_wire)}\n"
        f"  already wired (skipped) : {already}\n"
        f"  unmatched (skipped)     : {unmatched}"
    )
    if unmatched:
        logger.warning(
            f"{unmatched} input_scale entries had no matching NVFP4 weight; they will NOT be wired into the index."
        )

    # Reverse check: an NVFP4 weight (has .weight_scale_2) with no input_scale is a
    # broken expert -- quantized weight but no activation scale. This happens when
    # Stage 1 quantized an expert that Stage 2 skipped, i.e. the two stages were run
    # with different --exclude_layers / --keep_original_format_layers lists. Wiring
    # above never surfaces it (it only walks the input_scale entries), so flag it here.
    have_input_scale = {k[: -len(".input_scale")] for k in (*scale_keys, *weight_map) if k.endswith(".input_scale")}
    missing_input_scale = sorted(
        k[: -len(".weight_scale_2")]
        for k in weight_map
        if k.endswith(".weight_scale_2") and k[: -len(".weight_scale_2")] not in have_input_scale
    )
    if missing_input_scale:
        sample = ", ".join(missing_input_scale[:3])
        logger.warning(
            f"{len(missing_input_scale)} NVFP4 weights have no input_scale (e.g. {sample}); the "
            "resulting checkpoint is incomplete. This usually means Stage 1 and Stage 2 were run "
            "with different --exclude_layers / --keep_original_format_layers lists -- re-run Stage 2 "
            "with the SAME two lists used in Stage 1."
        )

    if args.dry_run:
        logger.info("dry-run: nothing copied, index not modified.")
        return

    all_input_scales = load_file(args.input_scale_path)
    new_scales = {name: all_input_scales[name] for name in to_wire}

    if new_scales:
        shard_tensors = load_file(target_shard_path)
        shard_tensors.update(new_scales)
        save_file(shard_tensors, target_shard_path)
        logger.info(f"Appended {len(new_scales)} input_scale tensors to {target_shard_name}")

    for name in to_wire:
        weight_map[name] = target_shard_name

    meta = index.setdefault("metadata", {})
    if "total_size" in meta:
        # 4 bytes per F32 scalar added to the addressable tensor set.
        meta["total_size"] = int(meta["total_size"]) + len(to_wire) * 4
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Done. Wired {len(to_wire)} input_scale tensors -> {target_shard_name}.\nUpdated index: {index_path}")

    try:
        os.remove(args.input_scale_path)
        logger.info(f"Removed input_scale file: {args.input_scale_path}")
    except OSError as e:
        logger.warning(f"Could not remove input_scale file {args.input_scale_path}: {e}. Remove it manually.")

    # Update config.json to reflect that input activations are now quantized.
    # Stage 1 exports input_tensors=null because activation scales are not yet
    # available. After Stage 2+3 attach the per-expert input_scale tensors, the
    # config must be updated so that loaders (vLLM, etc.) know activation
    # quantization is present.
    #
    # Use the shared NVFP4_INPUT_SPEC from nvfp4_config to produce the input_tensors
    # value. This guarantees Stage 3's config.json update is always in sync with
    # Stage 2's calibration config — change it once in nvfp4_config.py, both stages update.
    config_path = os.path.join(ckpt, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            model_config = json.load(f)

        qc = model_config.get("quantization_config")
        if qc is not None:
            gqc = qc.get("global_quant_config")
            if gqc is not None and gqc.get("input_tensors") is None:
                gqc["input_tensors"] = _nvfp4_input_tensors_dict()
                with open(config_path, "w") as f:
                    json.dump(model_config, f, indent=2)
                logger.info(f"Updated quantization_config.global_quant_config.input_tensors in {config_path}")
            else:
                logger.info("config.json input_tensors already set; skipping update.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed arguments namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Attach calibrated per-expert input_scale tensors to a "
            "DeepSeek-V4-Pro NVFP4 checkpoint via an existing shard (no separate file added)."
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
        help="Local path to the input_scale safetensors file (Stage 2 output).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report coverage without modifying the checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
