#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Calibration data collection utilities for diffusers pipelines.

These utilities enable activation-aware quantization of diffusion model
submodules (UNet, Transformer, VAE decoder, etc.) by capturing their
inputs during full pipeline runs.  The captured data is packaged into a
:class:`~torch.utils.data.DataLoader` that can be passed directly to
:meth:`ModelQuantizer.quantize_model` or to algorithm processors such as
:class:`SVDQuantProcessor`.

Each captured sample is a ``dict[str, Any]`` keyed by the target
submodule's forward parameter names. ``ModelQuantizer.quantize_model``
consumes this format directly via ``model(**data)``.

Captured tensors are detached and moved to CPU; calibration data only
moves back to the target device one batch at a time during quantization.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class _InputCapture:
    """Captures inputs to a module during forward passes.

    Registers a ``forward_pre_hook`` that intercepts ``(args, kwargs)``,
    binds them to the module's forward signature so positional arguments
    are mapped to their parameter names, and stores the resulting flat
    dict on CPU for later use as calibration data.
    """

    def __init__(self) -> None:
        self.captured: list[dict[str, Any]] = []
        self._hook: torch.utils.hooks.RemovableHook | None = None
        self._signature: inspect.Signature | None = None

    def register(self, module: nn.Module) -> None:
        signature = inspect.signature(module.forward)
        for param in signature.parameters.values():
            if param.kind.name in ("VAR_POSITIONAL", "VAR_KEYWORD"):
                raise RuntimeError(
                    f"{type(module).__name__}.forward declares parameter "
                    f"'{param.name}' as {param.kind.name}; "
                    f"diffusers calibration requires named parameters only."
                )
        self._signature = signature
        self._hook = module.register_forward_pre_hook(self._capture, with_kwargs=True)

    def remove(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _capture(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Bind ``args`` and ``kwargs`` to the forward signature and store the result on CPU.

        :param torch.nn.Module module: The hooked module (unused; the cached
            signature from :meth:`register` is used instead).
        :param tuple[Any, ...] args: Positional arguments passed to ``forward``.
        :param dict[str, Any] kwargs: Keyword arguments passed to ``forward``.
        """
        assert self._signature is not None  # set by register()
        bound = self._signature.bind_partial(*args, **kwargs)
        entry: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            if torch.is_tensor(value):
                entry[name] = value.detach().cpu()
            elif isinstance(value, dict):
                entry[name] = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in value.items()}
            else:
                entry[name] = value
        self.captured.append(entry)


class _CapturedCalibrationDataset(Dataset):  # type: ignore[type-arg]
    """Wraps captured kwargs dicts as a PyTorch Dataset.

    Each item is a ``dict[str, Any]`` keyed by the target submodule's
    forward parameter names. ``ModelQuantizer._do_calibration`` consumes
    this via ``model(**data)``.

    Tensors are moved to ``device`` on access.
    """

    def __init__(self, captured: list[dict[str, Any]], device: torch.device) -> None:
        self.captured = captured
        self.device = device

    def __len__(self) -> int:
        return len(self.captured)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.captured[idx]
        out: dict[str, Any] = {}
        for name, value in entry.items():
            if torch.is_tensor(value):
                out[name] = value.to(self.device)
            elif isinstance(value, dict):
                out[name] = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in value.items()}
            else:
                out[name] = value
        return out


def _passthrough_collate(batch: list[Any]) -> Any:
    """Collate for ``batch_size=1`` that returns the single item unchanged."""
    if len(batch) != 1:
        raise ValueError(f"Expected batch_size=1, got {len(batch)}")
    return batch[0]


@torch.no_grad()
def get_calib_dataloader(
    pipe: Any,
    target_module: nn.Module,
    prompts: list[str],
    n_steps: int = 20,
    seed: int = 42,
    device: str | torch.device = "cuda",
    **pipe_kwargs: Any,
) -> DataLoader:  # type: ignore[type-arg]
    """Collect calibration data from a diffusers pipeline.

    Hooks into *target_module* (e.g. ``pipe.unet``, ``pipe.transformer``),
    runs the pipeline over *prompts*, captures the submodule's inputs,
    and returns a :class:`~torch.utils.data.DataLoader`.

    Each batch is a ``dict[str, Any]`` keyed by the target submodule's
    forward parameter names, consumed by
    :meth:`ModelQuantizer.quantize_model` via ``model(**data)``.

    :param Any pipe: A diffusers pipeline instance (e.g.
        ``DiffusionPipeline``, ``FluxPipeline``).

    :param torch.nn.Module target_module: The submodule whose inputs to
        capture.  Must be a module that is called during ``pipe(...)``.

    :param list[str] prompts: Calibration prompts.  Each prompt triggers
        one pipeline run; with ``n_steps`` denoising steps the submodule
        is called ``n_steps`` times per prompt, yielding
        ``len(prompts) * n_steps`` calibration samples.

    :param int n_steps: Number of inference (denoising) steps per prompt.

    :param int seed: Random seed for the generator passed to the pipeline.

    :param str | torch.device device: Device for the returned calibration
        tensors.

    :param Any pipe_kwargs: Additional keyword arguments forwarded to
        ``pipe(...)`` (e.g. ``guidance_scale``, ``height``, ``width``,
        ``max_sequence_length``).

    :return: A dataloader with ``batch_size=1`` yielding ``dict[str, Any]``
        batches keyed by ``target_module.forward`` parameter names.
    :rtype: torch.utils.data.DataLoader
    """
    capturer = _InputCapture()
    capturer.register(target_module)

    logger.info(f"Collecting calibration data: {len(prompts)} prompts x {n_steps} steps")

    for prompt in prompts:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        pipe(
            prompt=[prompt],
            num_inference_steps=n_steps,
            generator=generator,
            **pipe_kwargs,
        )

    capturer.remove()

    n_captured = len(capturer.captured)
    logger.info(f"Captured {n_captured} calibration samples")
    if n_captured == 0:
        raise RuntimeError(
            "No calibration samples were captured. Verify that target_module "
            "is called during pipe(...) and that prompts are non-empty."
        )

    dataset = _CapturedCalibrationDataset(capturer.captured, torch.device(device))
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_passthrough_collate)
