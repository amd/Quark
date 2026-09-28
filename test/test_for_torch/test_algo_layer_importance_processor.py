#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from quark.torch.algorithm.depth_pruning.layer_importance import LayerImportancePrunerProcessor


class _Layer(nn.Module):
    def __init__(self, idx: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        # Mimics the Transformers attention/decoder ``layer_idx`` that ``apply`` re-indexes.
        self.layer_idx = idx


class _Model(nn.Module):
    def __init__(self, num_layers: int, nested_config: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(i) for i in range(num_layers)])
        if nested_config:
            self.config = SimpleNamespace(text_config=SimpleNamespace(use_cache=True), num_hidden_layers=num_layers)
        else:
            self.config = SimpleNamespace(use_cache=True, num_hidden_layers=num_layers)


def _get_use_cache(model: _Model, nested_config: bool) -> bool:
    return bool(model.config.text_config.use_cache if nested_config else model.config.use_cache)


class TestLayerImportancePrunerProcessorApply:
    """``apply`` disables the KV cache for evaluation and always restores it."""

    @staticmethod
    def _make_processor(
        model: _Model,
        num_layers: int,
        delete_layers_index: list[int],
        delete_layer_num: int,
    ) -> LayerImportancePrunerProcessor:
        proc = LayerImportancePrunerProcessor.__new__(LayerImportancePrunerProcessor)
        # Set by __init__; these tests bypass it. False keeps the offload path live.
        proc.using_accelerate = False
        proc.model = model
        proc.model_decoder_layers = "layers"
        proc.decode_layers = model.layers
        proc.num_hidden_layers = num_layers
        proc.delete_layers_index = delete_layers_index
        proc.delete_layer_num = delete_layer_num
        proc.min_ppl = torch.tensor(float("inf"))
        proc.best_del_idx = []
        proc.ppl_list = []
        proc.delete_list = []
        return proc

    @pytest.mark.parametrize("nested_config", [False, True])
    def test_apply_searches_layers_and_restores_cache(self, nested_config: bool) -> None:
        model = _Model(num_layers=3, nested_config=nested_config)
        proc = self._make_processor(model, num_layers=3, delete_layers_index=[], delete_layer_num=1)

        # One baseline call, then one per candidate window; the middle layer is the cheapest to drop.
        ppls = [torch.tensor(9.0), torch.tensor(5.0), torch.tensor(2.0), torch.tensor(7.0)]
        seen_use_cache: list[bool] = []

        def fake_eval(_model: nn.Module, **_kwargs: Any) -> torch.Tensor:
            seen_use_cache.append(_get_use_cache(model, nested_config))
            return ppls[len(seen_use_cache) - 1]

        proc.eval_func = fake_eval  # type: ignore[assignment]
        proc.apply()

        assert seen_use_cache == [False, False, False, False]
        assert _get_use_cache(model, nested_config) is True
        assert proc.best_del_idx == [1]
        assert proc.min_ppl.item() == pytest.approx(2.0)
        assert proc.delete_list == [[0], [1], [2]]
        assert [p.item() for p in proc.ppl_list] == pytest.approx([5.0, 2.0, 7.0])
        # Layer 1 is gone and the survivors are re-indexed contiguously.
        assert len(model.layers) == 2
        assert [layer.layer_idx for layer in model.layers] == [0, 1]
        assert model.config.num_hidden_layers == 2

    def test_apply_with_user_assigned_layers_returns_early(self) -> None:
        model = _Model(num_layers=4, nested_config=False)
        proc = self._make_processor(model, num_layers=4, delete_layers_index=[1, 2], delete_layer_num=1)

        calls: list[list[int]] = []

        def fake_eval(_model: nn.Module, remain_layer_idx: list[int] = []) -> torch.Tensor:
            calls.append(remain_layer_idx)
            return torch.tensor(3.0)

        proc.eval_func = fake_eval  # type: ignore[assignment]
        proc.apply()

        assert calls == [[0, 1, 2, 3], [0, 3]]
        assert len(model.layers) == 2
        # The search loop never ran, so no candidate bookkeeping and no config rewrite.
        assert proc.ppl_list == []
        assert proc.best_del_idx == []
        assert model.config.num_hidden_layers == 4
        assert model.config.use_cache is True

    def test_apply_restores_cache_when_eval_raises(self) -> None:
        model = _Model(num_layers=2, nested_config=False)
        proc = self._make_processor(model, num_layers=2, delete_layers_index=[], delete_layer_num=1)

        def failing_eval(_model: nn.Module, **_kwargs: Any) -> torch.Tensor:
            raise RuntimeError("eval blew up")

        proc.eval_func = failing_eval  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="eval blew up"):
            proc.apply()

        assert model.config.use_cache is True
