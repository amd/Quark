#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Lazy-loader: stream decoder-block weights one block at a time.

Decoder block weights that are already materialised are kept in CPU RAM and
transferred CPU↔GPU around each block's forward().  This avoids all disk I/O
after the initial load and is fast enough for autoregressive generation.

Blocks whose weights are on the meta device have nothing to keep, so they are
re-read from the safetensors checkpoint on each forward (disk streaming).

Decoder block ``nn.Parameter`` weights are managed lazily. Non-decoder
parameters and live buffers are moved to the target device during ``prepare()`` so
the patched model is runnable immediately.
"""

import json
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger

from .runner import get_module_weight_by_name
from .utils import infer_decoder_layers_path

logger = ScreenLogger(__name__)

_MODEL_ATTR = "_pbr_lazy_state"


# ---------------------------------------------------------------------------
# Parameter swap helpers (shared by both backends)
# ---------------------------------------------------------------------------


def _offload_block_params_to_meta(block: nn.Module) -> None:
    """Replace every parameter in *block* with a meta-device placeholder."""
    for mod in block.modules():
        for name, param in list(mod._parameters.items()):
            if param is not None:
                mod._parameters[name] = nn.Parameter(
                    torch.empty_like(param, device="meta"),
                    requires_grad=param.requires_grad,
                )


def _swap_in_params(block: nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Load tensors from *state_dict* into matching parameter slots in *block*.

    :return: List of relative keys that were loaded.
    """
    loaded: list[str] = []
    for key, tensor in state_dict.items():
        parts = key.split(".")
        sub = block
        try:
            for part in parts[:-1]:
                sub = getattr(sub, part)
            leaf = parts[-1]
        except AttributeError:
            continue
        if hasattr(sub, "_parameters") and leaf in sub._parameters and sub._parameters[leaf] is not None:
            old = sub._parameters[leaf]
            req_grad = old.requires_grad and tensor.is_floating_point()
            new_param = nn.Parameter(tensor, requires_grad=req_grad)
            sub._parameters[leaf] = new_param
            loaded.append(key)
    return loaded


def _reattach_weight_scales(block: nn.Module) -> None:
    """Re-link ``weight.scale`` to the sibling ``scale`` parameter for every submodule
    that carries both, after weights have been swapped in. Quantized linears expect
    ``weight.scale`` to point at the live ``scale`` tensor; swapping in a fresh
    ``nn.Parameter`` for ``weight`` drops that attribute, so it must be restored."""
    for submodule in block.modules():
        if not hasattr(submodule, "_parameters"):
            continue
        weight = submodule._parameters.get("weight")
        scale = submodule._parameters.get("scale")
        if weight is not None and not weight.is_meta and scale is not None and not scale.is_meta:
            weight.scale = scale


def _move_non_decoder_params_and_buffers_to_device(
    model: nn.Module,
    decoder_layers_path: str,
    target_device: torch.device,
) -> None:
    """Move non-decoder parameters and live buffers needed around lazy blocks.

    A tied parameter (``lm_head.weight`` is ``embed_tokens.weight``) is reached twice by
    ``named_modules()``. It is moved once and reused, so the pair stays tied and the
    embedding is not duplicated on ``target_device``.
    """
    # id(original) -> (original, moved). The original is kept alive in the value: once its
    # last module slot is rewritten it could be freed and its id reused by another
    # parameter, aliasing two unrelated weights onto the same tensor.
    moved_params: dict[int, tuple[nn.Parameter, nn.Parameter]] = {}

    for module_name, module in model.named_modules():
        in_decoder_block = module_name == decoder_layers_path or module_name.startswith(f"{decoder_layers_path}.")
        if not in_decoder_block:
            for param_name, param in list(module._parameters.items()):
                if param is None or param.is_meta:
                    continue
                cached = moved_params.get(id(param))
                if cached is None:
                    moved_param = nn.Parameter(
                        param.to(target_device),
                        requires_grad=param.requires_grad,
                    )
                    # Carry over attributes hung on the parameter, e.g. `weight.scale`.
                    moved_param.__dict__.update(param.__dict__)
                    moved_params[id(param)] = (param, moved_param)
                else:
                    moved_param = cached[1]
                module._parameters[param_name] = moved_param
        for buffer_name, buffer in list(module._buffers.items()):
            if buffer is not None and not buffer.is_meta:
                module._buffers[buffer_name] = buffer.to(target_device)


def _swap_out_params_to_meta(block: nn.Module, keys: list[str]) -> None:
    """Replace the given parameters with meta-device placeholders."""
    for key in keys:
        parts = key.split(".")
        sub = block
        try:
            for part in parts[:-1]:
                sub = getattr(sub, part)
            leaf = parts[-1]
        except AttributeError:
            continue
        if hasattr(sub, "_parameters") and leaf in sub._parameters and sub._parameters[leaf] is not None:
            old = sub._parameters[leaf]
            sub._parameters[leaf] = nn.Parameter(
                torch.empty_like(old, device="meta"),
                requires_grad=old.requires_grad,
            )


# ---------------------------------------------------------------------------
# Forward hooks factory
# ---------------------------------------------------------------------------


def _make_forward_hooks(
    block_name: str,
    weight_map: dict[str, str],
    model_dir: str,
    target_device: torch.device,
    state_dict_transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    cpu_cache: dict[str, torch.Tensor] | None = None,
    reattach_scales: Callable[[nn.Module], None] | None = None,
    progress_label: Callable[[], str] = lambda: "",
) -> tuple[Any, Any]:
    """Return (pre_hook, post_hook) for one decoder block.

    If *cpu_cache* is provided (RAM-offload mode), the pre-hook moves tensors
    from CPU RAM to *target_device* and the post-hook moves them back.  No disk
    access occurs.

    If *cpu_cache* is None (disk-streaming mode), the pre-hook reads the block
    weights from the safetensors checkpoint and the post-hook discards them to
    the meta device.
    """
    if cpu_cache is not None:
        # RAM-offload: weights live in cpu_cache between forward calls.
        # The model is in eval mode so weights are never modified during forward —
        # no need to copy them back; just restore from the original cpu_cache.
        def pre_hook(mod: nn.Module, _inputs: tuple[object, ...]) -> None:
            # `tqdm.write` rather than `logger.info`: this fires inside the calibration
            # progress bar's loop, and a plain write would tear the bar on every block.
            tqdm.write(f"Calibrating decoder layer {progress_label()}")
            state_dict = {k: t.to(target_device, non_blocking=True) for k, t in cpu_cache.items()}
            if state_dict_transform is not None:
                state_dict = state_dict_transform(state_dict)
            _swap_in_params(mod, state_dict)
            if reattach_scales is not None:
                reattach_scales(mod)

        def post_hook(mod: nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
            _swap_out_params_to_meta(mod, list(cpu_cache.keys()))
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
    else:
        # Disk-streaming: read from checkpoint each forward, discard after.
        def pre_hook(mod: nn.Module, _inputs: tuple[object, ...]) -> None:
            # `tqdm.write` rather than `logger.info`: this fires inside the calibration
            # progress bar's loop, and a plain write would tear the bar on every block.
            tqdm.write(f"Calibrating decoder layer {progress_label()}")
            state_dict = get_module_weight_by_name(
                block_name,
                weight_map=weight_map,
                model_dir=model_dir,
                device="cpu",
                strip_prefix=block_name,
            )
            if not state_dict:
                mod._pbr_loaded_keys = []  # type: ignore[attr-defined]
                return
            state_dict = {k: t.to(target_device) for k, t in state_dict.items()}
            if state_dict_transform is not None:
                state_dict = state_dict_transform(state_dict)
            mod._pbr_loaded_keys = _swap_in_params(mod, state_dict)  # type: ignore[attr-defined]
            if reattach_scales is not None:
                reattach_scales(mod)

        def post_hook(mod: nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
            keys = getattr(mod, "_pbr_loaded_keys", [])
            if keys:
                _swap_out_params_to_meta(mod, keys)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()

    return pre_hook, post_hook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PerBlockLazyLoader:
    """Streams decoder-block weights so only a few blocks are resident at a time.

    Constructing the loader patches *model* in place; :meth:`finalize` undoes the patch.
    The loader owns the hook state, so callers hold the object instead of probing the model.

    The offload backend for blocks beyond *n_gpu_blocks* is chosen automatically:

    * **CPU RAM** (preferred): blocks whose weights are materialised are held in CPU RAM and
      moved CPU->GPU->CPU around each forward. Costs no extra RAM -- the cache shares storage
      with the parameters it replaces -- and the checkpoint index is never read.
    * **Disk streaming** (fallback): blocks whose weights are on meta are re-read from the
      safetensors shards each forward, which needs ``model.safetensors.index.json``.

    :param nn.Module model: Model with weights already loaded on CPU.
    :param str pretrained_model_name_or_path: Path to the checkpoint directory.
    :param torch.device | str target_device: Device blocks are streamed onto. Default ``"cuda"``.
    :param str | None decoder_layers_path: Dotted path to the ``nn.ModuleList`` of decoder
        blocks. Auto-detected when ``None``.
    :param callable | None state_dict_transform: Optional transform applied to each block's
        raw state dict before swapping params in (disk mode only).
    :param int n_gpu_blocks: Number of leading decoder blocks kept permanently on
        *target_device*. No swap hooks are registered for them. Default ``0``.
    :param bool weights_modified_in_memory: Set when the weights no longer match the
        checkpoint -- an algorithm such as AWQ or GPTQ has edited them in place. Disk
        streaming would read the unmodified tensors back and silently undo that, so it is
        refused rather than selected.
    :raises NotImplementedError: In disk-streaming mode, when the weights have been modified
        in memory, when the checkpoint has no ``model.safetensors.index.json``, or when its
        keys do not match the module paths of the decoder blocks.
    """

    def __init__(
        self,
        model: nn.Module,
        pretrained_model_name_or_path: str,
        target_device: torch.device | str = "cuda",
        decoder_layers_path: str | None = None,
        state_dict_transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
        n_gpu_blocks: int = 0,
        weights_modified_in_memory: bool = False,
        n_batches: int = 0,
    ) -> None:
        self.model = model
        self.model_dir = pretrained_model_name_or_path
        self.target_device = torch.device(target_device)
        self.state_dict_transform = state_dict_transform
        self.weights_modified_in_memory = weights_modified_in_memory
        # Calibration replays every block once per batch, so the progress line carries the batch
        # it belongs to. 0 means the caller could not tell us how many batches there will be.
        self.n_batches = max(0, n_batches)
        self._batch_index = 0
        self.hook_handles: list[Any] = []
        self.cpu_caches: list[dict[str, torch.Tensor] | None] = []
        self.weight_map: dict[str, str] = {}
        self.patched = False
        # Set for real below; defined up front so an unpatched loader is still safe to inspect.
        self.layers_path = ""
        self.block_container = nn.ModuleList()
        self.n_gpu_blocks = 0
        self.use_cpu_ram = False

        layers_path = decoder_layers_path if decoder_layers_path is not None else infer_decoder_layers_path(model)
        block_container = dict(model.named_modules(remove_duplicate=False)).get(layers_path)
        if not isinstance(block_container, nn.ModuleList) or not layers_path:
            logger.warning(
                "lazy_loader: could not locate a root-level nn.ModuleList at path %r. No patches applied.",
                layers_path,
            )
            return

        self.layers_path = layers_path
        self.block_container = block_container
        n_blocks = len(block_container)
        self.n_gpu_blocks = max(0, min(n_gpu_blocks, n_blocks))
        offload_blocks = list(block_container)[self.n_gpu_blocks :]

        self.use_cpu_ram = self._choose_backend(offload_blocks, n_blocks)
        if not self.use_cpu_ram:
            if weights_modified_in_memory:
                raise NotImplementedError(
                    "Per-block disk streaming re-reads block weights from the checkpoint, which "
                    "would discard the in-memory weight modifications made before calibration "
                    "(AWQ, GPTQ, SmoothQuant, rotation). Materialise the decoder block weights so "
                    "they can be cached in CPU RAM, or drop the algorithm."
                )
            # Only disk streaming re-reads the checkpoint, so only it needs the index.
            self.weight_map = self._load_and_validate_weight_map()

        _move_non_decoder_params_and_buffers_to_device(model, layers_path, self.target_device)

        for idx, block in enumerate(block_container):
            if idx < self.n_gpu_blocks:
                self._make_block_resident(block)
                self.cpu_caches.append(None)  # sentinel: block is GPU-resident
                continue
            self._offload_block(block, idx, n_blocks)

        setattr(model, _MODEL_ATTR, self)
        self.patched = True

    def _choose_backend(self, offload_blocks: list[nn.Module], n_blocks: int) -> bool:
        """Return True to hold offloaded blocks in CPU RAM, False to stream them from disk.

        Weights that are already materialised are kept in CPU RAM. Moving them into
        `cpu_caches` costs nothing: `.cpu()` on a CPU tensor returns the tensor itself, so
        the cache shares storage with the parameter it replaces and RSS is unchanged.

        Disk streaming is therefore only for blocks with nothing to keep -- weights on the
        meta device, which have to come from the checkpoint whatever we do.

        The earlier heuristic compared `MemAvailable` against the offloaded bytes, as if the
        cache were a second copy. Since the model is already resident by the time the loader
        runs, that demanded roughly twice the model size in RAM and pushed the exact case
        this flow exists for onto the slow, checkpoint-backed path.
        """
        resident_offload_bytes = sum(
            p.numel() * p.element_size() for block in offload_blocks for p in block.parameters() if not p.is_meta
        )
        use_cpu_ram = resident_offload_bytes > 0

        # Debug, not info: the offload split is diagnostic detail, the per-block progress
        # is what a user watching a long run needs.
        logger.debug(
            "lazy_loader: %d/%d blocks GPU-resident; %d blocks offloaded (%.1f GB) → %s",
            self.n_gpu_blocks,
            n_blocks,
            n_blocks - self.n_gpu_blocks,
            resident_offload_bytes / 1e9,
            "CPU RAM" if use_cpu_ram else "disk",
        )
        return use_cpu_ram

    def _load_and_validate_weight_map(self) -> dict[str, str]:
        """Read the checkpoint index and check it actually describes the decoder blocks."""
        index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise NotImplementedError(
                f"Per-block disk streaming needs a sharded checkpoint index at {index_path}. "
                "Free up RAM to use the index-free CPU RAM backend instead."
            )

        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f).get("weight_map", {})

        # Blocks are looked up by module path; a checkpoint keyed differently (e.g. a
        # multimodal one saved by an older transformers) would load zero tensors per block.
        block_prefix = f"{self.layers_path}."
        if not any(key.startswith(block_prefix) for key in weight_map):
            sample = ", ".join(sorted({key.rsplit(".", 1)[0] for key in list(weight_map)[:64]})[:3])
            raise NotImplementedError(
                f"No key in {index_path} starts with {block_prefix!r} (checkpoint uses e.g. {sample}), "
                "so every block would load zero tensors. Free up RAM to use the CPU RAM backend."
            )
        return weight_map

    def _make_block_resident(self, block: nn.Module) -> None:
        """Move a block to the target device once, with no swap hooks."""
        for mod in block.modules():
            for pname, param in list(mod._parameters.items()):
                if param is not None and not param.is_meta:
                    mod._parameters[pname] = nn.Parameter(
                        param.data.to(self.target_device),
                        requires_grad=param.requires_grad,
                    )
        _reattach_weight_scales(block)

    def _progress_label(self, idx: int, n_blocks: int) -> str:
        """Build the progress label for one block, counting batches off the first hooked block.

        :param int idx: Index of the block in the decoder stack.
        :param int n_blocks: Total number of decoder blocks.
        :return: ``"3/92"``, or ``"3/92 (batch 2/3)"`` when there is more than one batch.
        """
        # GPU-resident blocks are never hooked, so the first hooked block marks a new batch.
        if idx == self.n_gpu_blocks:
            self._batch_index += 1
        label = f"{idx + 1}/{n_blocks}"
        if self.n_batches > 1:
            return f"{label} (batch {self._batch_index}/{self.n_batches})"
        if self.n_batches == 0 and self._batch_index > 1:
            return f"{label} (batch {self._batch_index})"
        return label

    def _offload_block(self, block: nn.Module, idx: int, n_blocks: int) -> None:
        """Offload a block and register the hooks that stream it back in around forward."""
        cpu_cache: dict[str, torch.Tensor] | None = None
        if self.use_cpu_ram:
            # Collect current CPU params into a dict, then offload to meta.
            cpu_cache = {}
            for mod_name, mod in block.named_modules():
                prefix = (mod_name + ".") if mod_name else ""
                for param_name, param in list(mod._parameters.items()):
                    if param is not None and not param.is_meta:
                        cpu_cache[prefix + param_name] = param.data.cpu()
                        mod._parameters[param_name] = nn.Parameter(
                            torch.empty_like(param, device="meta"),
                            requires_grad=param.requires_grad,
                        )
            self.cpu_caches.append(cpu_cache)
        else:
            _offload_block_params_to_meta(block)
            self.cpu_caches.append(None)

        pre_hook, post_hook = _make_forward_hooks(
            f"{self.layers_path}.{idx}",
            self.weight_map,
            self.model_dir,
            self.target_device,
            self.state_dict_transform,
            cpu_cache,
            reattach_scales=_reattach_weight_scales,
            progress_label=lambda: self._progress_label(idx, n_blocks),
        )
        self.hook_handles.append(block.register_forward_pre_hook(pre_hook, with_kwargs=False))
        self.hook_handles.append(block.register_forward_hook(post_hook, with_kwargs=False))

    def finalize(self, target_device: torch.device | str | None = None) -> None:
        """Remove the hooks and bring every decoder block back onto a real device.

        After this call the model behaves like a normally loaded model.

        :param torch.device | str | None target_device: Where to place the restored blocks.
            Defaults to the device the loader was streaming onto.
        """
        if not self.patched:
            return
        if target_device is not None:
            self.target_device = torch.device(target_device)

        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        logger.info("lazy_loader: finalize — reloading all decoder block weights onto %s …", self.target_device)

        for idx, block in enumerate(self.block_container):
            if idx < self.n_gpu_blocks:
                continue  # GPU-resident block: never offloaded
            if self.use_cpu_ram:
                cached = self.cpu_caches[idx]
                if cached is None:
                    continue
                _swap_in_params(block, {k: t.to(self.target_device) for k, t in cached.items()})
            else:
                block_name = f"{self.layers_path}.{idx}"
                state_dict = get_module_weight_by_name(
                    block_name,
                    weight_map=self.weight_map,
                    model_dir=self.model_dir,
                    device="cpu",
                    strip_prefix=block_name,
                )
                if not state_dict:
                    raise RuntimeError(
                        f"Per-block finalize could not read any weight for {block_name} from {self.model_dir}. "
                        "The block would be left on the meta device and exported empty."
                    )
                _swap_in_params(block, {k: t.to(self.target_device) for k, t in state_dict.items()})
            _reattach_weight_scales(block)

        self.patched = False
        if getattr(self.model, _MODEL_ATTR, None) is self:
            delattr(self.model, _MODEL_ATTR)
        logger.info("lazy_loader: finalized — all decoder block weights on %s.", self.target_device)


def prepare(
    model: nn.Module,
    pretrained_model_name_or_path: str,
    target_device: torch.device | str = "cuda",
    decoder_layers_path: str | None = None,
    state_dict_transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    n_gpu_blocks: int = 0,
    weights_modified_in_memory: bool = False,
    n_batches: int = 0,
) -> PerBlockLazyLoader:
    """Patch *model* for lazy per-block execution and return the loader that owns it.

    Thin wrapper over :class:`PerBlockLazyLoader`; see it for the parameters and for the
    backend-selection rules.
    """
    return PerBlockLazyLoader(
        model,
        pretrained_model_name_or_path,
        target_device=target_device,
        decoder_layers_path=decoder_layers_path,
        state_dict_transform=state_dict_transform,
        n_gpu_blocks=n_gpu_blocks,
        weights_modified_in_memory=weights_modified_in_memory,
        n_batches=n_batches,
    )


def finalize(model: nn.Module) -> None:
    """Finalize the loader attached to *model*, if any. No-op when it was never patched.

    :param nn.Module model: Model previously passed to :func:`prepare`.
    """
    loader: PerBlockLazyLoader | None = getattr(model, _MODEL_ATTR, None)
    if loader is not None:
        loader.finalize()
