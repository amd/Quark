#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from quark.torch.algorithm.blockwise_tuning.blockwise_tuning import BlockwiseTuningProcessor

_PATCH_BASE = "quark.torch.algorithm.blockwise_tuning.blockwise_tuning"


class TestBlockwiseTuningProcessorApply:
    """Cache handling in ``apply`` for flat and nested ``text_config`` layouts."""

    @staticmethod
    def _config(use_cache: bool, nested: bool) -> SimpleNamespace:
        if nested:
            return SimpleNamespace(text_config=SimpleNamespace(use_cache=use_cache))
        return SimpleNamespace(use_cache=use_cache)

    @staticmethod
    def _get_use_cache(config: SimpleNamespace, nested: bool) -> bool:
        return bool(config.text_config.use_cache if nested else config.use_cache)

    def _make_processor(self, num_layers: int = 1, nested_config: bool = False) -> BlockwiseTuningProcessor:
        proc = BlockwiseTuningProcessor.__new__(BlockwiseTuningProcessor)
        # Set by __init__; these tests bypass it. False keeps the offload path live.
        proc.using_accelerate = False
        # The two models start on different settings so a restore that mixes them up is caught.
        proc.model = SimpleNamespace(config=self._config(use_cache=True, nested=nested_config))
        proc.fp_model = SimpleNamespace(config=self._config(use_cache=False, nested=nested_config))
        proc.model_decoder_layers = "layers"
        proc.device_map = {f"layers.{i}": torch.device("cpu") for i in range(num_layers)}
        proc.module_kwargs = {}
        proc.inps = [torch.tensor([[1.0, 2.0, 3.0, 4.0]])]
        proc.trainable_modules = []
        proc.epochs = 1
        proc.weight_lr = 1e-3
        proc.min_lr_factor = 10.0
        proc.weight_decay = 0.0
        proc.max_grad_norm = 0.3

        proc.modules = [MagicMock(spec=nn.Module) for _ in range(num_layers)]
        proc.modules_fp = [MagicMock(spec=nn.Module) for _ in range(num_layers)]
        return proc

    @pytest.mark.parametrize("nested_config", [False, True])
    def test_apply_disables_and_restores_cache_per_model(self, nested_config: bool) -> None:
        proc = self._make_processor(num_layers=1, nested_config=nested_config)
        model_ref, fp_model_ref = proc.model, proc.fp_model
        seen_during_apply: list[tuple[bool, bool]] = []

        def fake_block_forward(*_args: object, **_kwargs: object) -> list[torch.Tensor]:
            seen_during_apply.append(
                (
                    self._get_use_cache(model_ref.config, nested_config),
                    self._get_use_cache(fp_model_ref.config, nested_config),
                )
            )
            return [torch.tensor([[10.0, 11.0, 12.0, 13.0]])]

        with (
            patch(f"{_PATCH_BASE}.block_forward", side_effect=fake_block_forward),
            patch(f"{_PATCH_BASE}.blockwise_training"),
            patch(f"{_PATCH_BASE}.get_device", return_value=torch.device("cpu")),
            patch(f"{_PATCH_BASE}.move_to_device", side_effect=lambda obj, device: obj),
            patch(f"{_PATCH_BASE}.clear_memory"),
        ):
            proc.apply()

        # Both models run the forward with the cache off ...
        assert seen_during_apply and all(flags == (False, False) for flags in seen_during_apply)
        # ... and each is restored to its own original value, not the other's.
        assert self._get_use_cache(model_ref.config, nested_config) is True
        assert self._get_use_cache(fp_model_ref.config, nested_config) is False
        assert not hasattr(proc, "fp_model")
