#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Block-level disk offload for the Stage 2 calibration observer state.

Quark's ``per_block_runner`` offloads block *weights* (which can be re-read from
the checkpoint). It does not manage the *observer / NativeLinear* state created
during calibration, which has no on-disk source. When that state does not fit in
RAM, :func:`install_disk_offload_hooks` registers per-block forward hooks that
spill it to a temp directory between forward calls and read it back on demand.

A block's spilled state is a flat dict written with ``torch.save``; keys use the
``<module_name>@@<attr>`` convention (e.g. ``...@@cpu_native``,
``...@@obs.min_val``) that ``dsv4_collect.collect_input_minmax`` reads back.

The functions detect NativeLinear modules by duck-typing (``cpu_native`` /
``cpu_scale`` attributes) so this module has no dependency on the NativeLinear
class definition.
"""

from __future__ import annotations

import atexit
import gc
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from quark.common.utils.log import ScreenLogger
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

logger = ScreenLogger(__name__)

_SEP = "@@"


def collect_block_state(block: nn.Module) -> dict[str, torch.Tensor]:
    """Collect NativeLinear weights + quantizer buffers/observer attrs from a block.

    After collection, the originals are set to None to free CPU RAM.
    Returns a flat dict suitable for torch.save().
    """
    state: dict[str, torch.Tensor] = {}
    for name, mod in block.named_modules():
        if hasattr(mod, "cpu_native"):
            if mod.cpu_native is not None:
                state[f"{name}{_SEP}cpu_native"] = mod.cpu_native
                mod.cpu_native = None
            if getattr(mod, "cpu_scale", None) is not None:
                state[f"{name}{_SEP}cpu_scale"] = mod.cpu_scale
                mod.cpu_scale = None
        if isinstance(mod, ScaledFakeQuantize):
            for bname, buf in list(mod._buffers.items()):
                if buf is not None and not buf.is_meta:
                    state[f"{name}{_SEP}buf.{bname}"] = buf
                    mod._buffers[bname] = None
            obs = getattr(mod, "observer", None)
            if obs is not None:
                for attr in ("amax", "min_val", "max_val"):
                    v = getattr(obs, attr, None)
                    if isinstance(v, torch.Tensor):
                        state[f"{name}{_SEP}obs.{attr}"] = v
                        setattr(obs, attr, None)
    return state


def restore_block_state(block: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Restore NativeLinear weights + quantizer buffers/observer attrs."""
    mod_cache: dict[str, nn.Module] = dict(block.named_modules())
    for key, tensor in state.items():
        mod_name, attr_key = key.split(_SEP, 1)
        mod = mod_cache.get(mod_name)
        if mod is None:
            continue
        if attr_key == "cpu_native":
            mod.cpu_native = tensor
        elif attr_key == "cpu_scale":
            mod.cpu_scale = tensor
        elif attr_key.startswith("buf."):
            bname = attr_key[4:]
            mod._buffers[bname] = tensor
        elif attr_key.startswith("obs."):
            attr = attr_key[4:]
            obs = getattr(mod, "observer", None)
            if obs is not None:
                setattr(obs, attr, tensor)


def install_disk_offload_hooks(
    model: nn.Module,
    ram_budget_gb: int | None,
    per_block_gb: float = 28.0,
) -> tuple[int, Path | None, int]:
    """Register pre/post hooks that spill block state to disk when RAM is tight.

    Returns (n_disk_blocks, offload_dir, n_ram_blocks) so the caller can read
    back the offloaded observer state (on disk) when exporting input min/max.

    ram_budget_gb is the maximum *total* system RAM the process may use.
    We read current RSS to figure out how much headroom remains for the
    observer/NativeLinear state that will be allocated during calibration.

    Returns the number of blocks that will use disk offload.
    """
    layers = model.layers
    n_blocks = len(layers)

    current_rss_gb = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    current_rss_gb = int(line.split()[1]) / 1048576
                    break
    except Exception:
        pass

    if ram_budget_gb is None:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_gb = int(line.split()[1]) / 1048576
                        ram_budget_gb = int(total_gb * 0.85)
                        break
        except Exception:
            ram_budget_gb = 999999

    headroom_gb = max(0, ram_budget_gb - current_rss_gb)
    total_needed = per_block_gb * n_blocks
    n_ram_blocks = min(n_blocks, max(0, int(headroom_gb / per_block_gb)))
    n_disk_blocks = n_blocks - n_ram_blocks

    logger.info(f"RAM budget: {ram_budget_gb} GB, current RSS: {current_rss_gb:.0f} GB, headroom: {headroom_gb:.0f} GB")
    logger.info(f"Observer state needed: {total_needed:.0f} GB ({n_blocks} blocks x {per_block_gb:.0f} GB)")

    if n_disk_blocks <= 0:
        logger.info(f"Disk offload: not needed — all {n_blocks} blocks fit in RAM")
        return 0, None, n_blocks

    offload_dir = Path(tempfile.mkdtemp(prefix="quark_offload_"))
    atexit.register(lambda: shutil.rmtree(offload_dir, ignore_errors=True))

    logger.info(f"Disk offload: {n_ram_blocks} blocks in RAM, {n_disk_blocks} blocks on disk ({offload_dir})")

    for idx in range(n_ram_blocks, n_blocks):
        block = layers[idx]
        dump_path = offload_dir / f"block_{idx}.pt"

        def _make_hooks(blk, path, blk_idx):
            def pre_hook(mod, inputs):
                if path.exists():
                    st = torch.load(path, map_location="cpu", weights_only=True)
                    restore_block_state(mod, st)
                    del st

            def post_hook(mod, inputs, output):
                st = collect_block_state(mod)
                if st:
                    torch.save(st, path)
                    del st
                    gc.collect()

            return pre_hook, post_hook

        pre_h, post_h = _make_hooks(block, dump_path, idx)
        block.register_forward_pre_hook(pre_h)
        block.register_forward_hook(post_h)

    return n_disk_blocks, offload_dir, n_ram_blocks
