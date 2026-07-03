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

from quark.experimental.torch.algorithm.blockwise_joint_tuning.quantize.learnable_linear import (
    ExperimentalLearnableQuantizedLinear,
)
from quark.torch.algorithm.blockwise_joint_tuning.processor import (
    BlockwiseJointTuningProcessor,
    _replace_linear_with_learnable_quant_linear,
)
from quark.torch.algorithm.config import BlockwiseJointTuningConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GROUP = 4  # tiny group_size so we can use in_features=4


class _Flat(nn.Module):
    """Single Linear layer."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(_GROUP, _GROUP, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _Nested(nn.Module):
    """Two Linear layers in nested submodules."""

    def __init__(self):
        super().__init__()
        self.sub1 = nn.Linear(_GROUP, _GROUP, bias=False)
        self.sub2 = nn.Sequential(nn.Linear(_GROUP, _GROUP, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sub2(self.sub1(x))


class _NoLinear(nn.Module):
    """Module with no nn.Linear layers."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(_GROUP)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def _make_config(**kwargs) -> BlockwiseJointTuningConfig:
    defaults = dict(
        epochs=1,
        weight_lr=1e-3,
        qparam_lr=5e-4,
        weight_decay=0.0,
        qparam_weight_decay=0.0,
        min_lr_factor=10.0,
        max_grad_norm=0.3,
        model_decoder_layers="layers",
        trainable_modules=[],
        quant_trainable_modules=[],
    )
    defaults.update(kwargs)
    return BlockwiseJointTuningConfig(**defaults)


# Patch targets – functions are imported into the processor module.
_PATCH_BASE = "quark.torch.algorithm.blockwise_joint_tuning.processor"
_PATCHES = {
    "init_device_map": f"{_PATCH_BASE}.init_device_map",
    "init_blockwise_algo": f"{_PATCH_BASE}.init_blockwise_algo",
    "get_model_layers": f"{_PATCH_BASE}.get_model_layers",
    "clear_memory": f"{_PATCH_BASE}.clear_memory",
}


def _mock_heavy_deps(
    mock_init_device_map,
    mock_init_blockwise_algo,
    mock_get_model_layers,
    _mock_clear_memory,
):
    mock_init_device_map.return_value = {}
    # init_blockwise_algo returns (modules, module_kwargs, inputs)
    mock_init_blockwise_algo.return_value = ([], {}, [])
    mock_get_model_layers.return_value = []


# ---------------------------------------------------------------------------
# _replace_linear_with_learnable_quant_linear
# ---------------------------------------------------------------------------


class TestReplaceLinearWithLearnableQuantLinear:
    def test_single_linear_replaced(self):
        layer = _Flat()
        _replace_linear_with_learnable_quant_linear(layer, num_bits=4, group_size=_GROUP)
        assert isinstance(layer.linear, ExperimentalLearnableQuantizedLinear)

    def test_nested_linears_all_replaced(self):
        layer = _Nested()
        _replace_linear_with_learnable_quant_linear(layer, num_bits=4, group_size=_GROUP)
        assert isinstance(layer.sub1, ExperimentalLearnableQuantizedLinear)
        assert isinstance(layer.sub2[0], ExperimentalLearnableQuantizedLinear)

    def test_weight_quant_enabled_after_replacement(self):
        layer = _Flat()
        _replace_linear_with_learnable_quant_linear(layer, num_bits=4, group_size=_GROUP)
        assert layer.linear.weight_quant_enabled is True

    def test_no_linear_no_change(self):
        layer = _NoLinear()
        _replace_linear_with_learnable_quant_linear(layer, num_bits=4, group_size=_GROUP)
        # LayerNorm should remain untouched
        assert isinstance(layer.norm, nn.LayerNorm)

    def test_weight_shape_preserved(self):
        layer = _Flat()
        original_shape = layer.linear.weight.shape
        _replace_linear_with_learnable_quant_linear(layer, num_bits=4, group_size=_GROUP)
        assert layer.linear.weight.shape == original_shape

    def test_weight_data_identical_to_original(self):
        layer = _Flat()
        original_weight = layer.linear.weight.data.clone()
        _replace_linear_with_learnable_quant_linear(layer, num_bits=4, group_size=_GROUP)
        assert torch.equal(layer.linear.weight.data, original_weight)

    def test_bias_preserved(self):
        layer = _Nested()
        _replace_linear_with_learnable_quant_linear(layer, num_bits=4, group_size=_GROUP)
        # sub2[0] had bias=True, sub1 had bias=False
        assert layer.sub2[0].bias is not None
        assert layer.sub1.bias is None

    def test_num_bits_propagated(self):
        layer = _Flat()
        _replace_linear_with_learnable_quant_linear(layer, num_bits=8, group_size=_GROUP)
        assert layer.linear.weight_quantizer.num_bits == 8

    def test_forward_still_runnable(self):
        layer = _Flat()
        _replace_linear_with_learnable_quant_linear(layer, num_bits=4, group_size=_GROUP)
        x = torch.randn(2, _GROUP)
        out = layer(x)
        assert out.shape == (2, _GROUP)

    def test_forward_with_quant_disabled_matches_original(self):
        # With weight_quant disabled the output should equal the plain nn.Linear result
        lin = nn.Linear(_GROUP, _GROUP, bias=False)
        layer = _Flat()
        layer.linear = lin
        x = torch.randn(2, _GROUP)
        expected = layer(x).detach().clone()

        _replace_linear_with_learnable_quant_linear(layer, num_bits=4, group_size=_GROUP)
        layer.linear.disable_weight_quant()  # quant off → should use raw weight
        actual = layer(x)
        assert torch.allclose(actual, expected)


# ---------------------------------------------------------------------------
# BlockwiseJointTuningProcessor.__init__ – config attributes & data_loader split
# (heavy dependencies are patched out so the test runs in milliseconds)
# ---------------------------------------------------------------------------


class TestBlockwiseJointTuningProcessorInit:
    def _build(self, data_loader, config=None):
        """Construct a processor with all heavy deps mocked."""
        if config is None:
            config = _make_config()
        model = MagicMock(spec=nn.Module)
        fp_model = MagicMock(spec=nn.Module)
        with (
            patch(_PATCHES["init_device_map"]) as mock_idm,
            patch(_PATCHES["init_blockwise_algo"]) as mock_iba,
            patch(_PATCHES["get_model_layers"]) as mock_gml,
            patch(_PATCHES["clear_memory"]),
        ):
            _mock_heavy_deps(mock_idm, mock_iba, mock_gml, None)
            proc = BlockwiseJointTuningProcessor(fp_model, model, config, data_loader)
        return proc

    def test_config_attrs_assigned(self):
        cfg = _make_config(
            epochs=3,
            weight_lr=2e-4,
            qparam_lr=1e-4,
            weight_decay=0.01,
            qparam_weight_decay=0.001,
            min_lr_factor=5.0,
            max_grad_norm=0.5,
            model_decoder_layers="model.decoder.layers",
            trainable_modules=["attn"],
            quant_trainable_modules=["scale"],
        )
        loader = MagicMock()
        proc = self._build(loader, cfg)

        assert proc.epochs == 3
        assert proc.weight_lr == pytest.approx(2e-4)
        assert proc.quant_lr == pytest.approx(1e-4)
        assert proc.weight_decay == pytest.approx(0.01)
        assert proc.qparam_weight_decay == pytest.approx(0.001)
        assert proc.min_lr_factor == pytest.approx(5.0)
        assert proc.max_grad_norm == pytest.approx(0.5)
        assert proc.model_decoder_layers == "model.decoder.layers"
        assert proc.trainable_modules == ["attn"]
        assert proc.quant_trainable_modules == ["scale"]

    def test_single_loader_used_for_both_train_and_val(self):
        loader = MagicMock()
        proc = self._build(loader)
        assert proc.traindata_loader is loader
        assert proc.valdata_loader is loader

    def test_tuple_loader_splits_train_val(self):
        train_loader = MagicMock(name="train")
        val_loader = MagicMock(name="val")
        proc = self._build((train_loader, val_loader))
        assert proc.traindata_loader is train_loader
        assert proc.valdata_loader is val_loader

    def test_list_loader_splits_train_val(self):
        train_loader = MagicMock(name="train")
        val_loader = MagicMock(name="val")
        proc = self._build([train_loader, val_loader])
        assert proc.traindata_loader is train_loader
        assert proc.valdata_loader is val_loader

    def test_quant_lr_fallback_to_qparam_lr(self):
        """quant_lr should read from qparam_lr when no explicit quant_lr attribute."""
        cfg = _make_config(qparam_lr=7e-5)
        proc = self._build(MagicMock(), cfg)
        assert proc.quant_lr == pytest.approx(7e-5)

    def test_quant_lr_override_via_quant_lr_attr(self):
        """If algo_config has an explicit quant_lr attribute, it takes priority."""
        cfg = _make_config(qparam_lr=1e-4)
        cfg.quant_lr = 9e-5  # type: ignore[attr-defined]
        proc = self._build(MagicMock(), cfg)
        assert proc.quant_lr == pytest.approx(9e-5)

    def test_modules_and_inputs_stored(self):
        loader = MagicMock()
        proc = self._build(loader)
        assert isinstance(proc.modules, list)
        assert isinstance(proc.train_inputs, list)
        assert isinstance(proc.val_inputs, list)
        assert isinstance(proc.module_kwargs, dict)


# ---------------------------------------------------------------------------
# BlockwiseJointTuningProcessor.apply – control-flow tests with full mocking
# ---------------------------------------------------------------------------


class TestBlockwiseJointTuningProcessorApply:
    def _make_processor(self, num_layers: int = 1) -> BlockwiseJointTuningProcessor:
        proc = BlockwiseJointTuningProcessor.__new__(BlockwiseJointTuningProcessor)
        proc.model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
        proc.fp_model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
        proc.model_decoder_layers = "layers"
        proc.device_map = {f"layers.{i}": torch.device("cpu") for i in range(num_layers)}
        proc.module_kwargs = {}
        proc.train_inputs = [torch.tensor([[1.0, 2.0, 3.0, 4.0]])]
        proc.val_inputs = [torch.tensor([[5.0, 6.0, 7.0, 8.0]])]
        proc.trainable_modules = []
        proc.quant_trainable_modules = []
        proc.epochs = 1
        proc.quant_lr = 1e-3
        proc.weight_lr = 1e-3
        proc.min_lr_factor = 10.0
        proc.weight_decay = 0.0
        proc.qparam_weight_decay = 0.0
        proc.max_grad_norm = 0.3

        proc.modules = [MagicMock(spec=nn.Module) for _ in range(num_layers)]
        proc.modules_fp = [MagicMock(spec=nn.Module) for _ in range(num_layers)]
        proc.modules_fp_val = [MagicMock(spec=nn.Module) for _ in range(num_layers)]
        return proc

    def test_apply_restores_cache_and_invokes_core_steps(self):
        proc = self._make_processor(num_layers=1)
        layer0, fp0, fpv0 = proc.modules[0], proc.modules_fp[0], proc.modules_fp_val[0]
        fp_model_ref = proc.fp_model

        bf_ret_train_fp = [torch.tensor([[10.0, 11.0, 12.0, 13.0]])]
        bf_ret_val_fp = [torch.tensor([[20.0, 21.0, 22.0, 23.0]])]
        bf_ret_train_q = [torch.tensor([[30.0, 31.0, 32.0, 33.0]])]

        with (
            patch(f"{_PATCH_BASE}.tqdm", side_effect=lambda x, **kwargs: x),
            patch(f"{_PATCH_BASE}._replace_linear_with_learnable_quant_linear") as mock_replace,
            patch(
                f"{_PATCH_BASE}.block_forward", side_effect=[bf_ret_train_fp, bf_ret_val_fp, bf_ret_train_q]
            ) as mock_bf,
            patch(f"{_PATCH_BASE}.blockwise_joint_training") as mock_train,
            patch(f"{_PATCH_BASE}.get_dtype", return_value=torch.float16),
            patch(
                f"{_PATCH_BASE}.get_device",
                side_effect=[torch.device("cpu"), torch.device("cpu")],
            ),
            patch(f"{_PATCH_BASE}.move_to_device", side_effect=lambda obj, device: obj) as mock_move,
            patch(f"{_PATCH_BASE}.clear_memory"),
        ):
            proc.apply()

        # cache flag must be restored after apply
        assert proc.model.config.use_cache is True
        assert fp_model_ref.config.use_cache is True
        assert not hasattr(proc, "fp_model")

        # per-layer core steps
        mock_replace.assert_called_once_with(layer0)
        assert mock_bf.call_count == 3
        assert mock_train.call_count == 1

        # block_forward order: fp_train, fp_val, quant_layer_train
        assert mock_bf.call_args_list[0].args[0] is fp0
        assert mock_bf.call_args_list[1].args[0] is fpv0
        assert mock_bf.call_args_list[2].args[0] is layer0

        # layer training receives generated fp outputs and correct index
        assert mock_train.call_args.kwargs["fp_layer_outputs"] is bf_ret_train_fp
        assert mock_train.call_args.kwargs["layer_index"] == 0

        # when layer starts on CPU, it should be moved according to device_map
        assert any(c.args[0] is layer0 and c.args[1] == proc.device_map["layers.0"] for c in mock_move.call_args_list)

        # dtype restore path must execute
        layer0.to.assert_any_call(dtype=torch.float16)

    def test_apply_rolls_train_inputs_between_layers(self):
        proc = self._make_processor(num_layers=2)

        # 2 layers -> block_forward called 6 times:
        #   L0: fp_train, fp_val, quant_train
        #   L1: fp_train, fp_val, quant_train
        l0_fp_train = [torch.tensor([[10.0, 0.0, 0.0, 0.0]])]
        l0_fp_val = [torch.tensor([[20.0, 0.0, 0.0, 0.0]])]
        l0_quant_out = [torch.tensor([[30.0, 0.0, 0.0, 0.0]])]
        l1_fp_train = [torch.tensor([[40.0, 0.0, 0.0, 0.0]])]
        l1_fp_val = [torch.tensor([[50.0, 0.0, 0.0, 0.0]])]
        l1_quant_out = [torch.tensor([[60.0, 0.0, 0.0, 0.0]])]

        with (
            patch(f"{_PATCH_BASE}.tqdm", side_effect=lambda x, **kwargs: x),
            patch(f"{_PATCH_BASE}._replace_linear_with_learnable_quant_linear"),
            patch(
                f"{_PATCH_BASE}.block_forward",
                side_effect=[l0_fp_train, l0_fp_val, l0_quant_out, l1_fp_train, l1_fp_val, l1_quant_out],
            ),
            patch(f"{_PATCH_BASE}.blockwise_joint_training") as mock_train,
            patch(f"{_PATCH_BASE}.get_dtype", return_value=torch.float16),
            patch(
                f"{_PATCH_BASE}.get_device",
                side_effect=[
                    torch.device("cpu"),
                    torch.device("cpu"),
                    torch.device("cpu"),
                    torch.device("cpu"),
                ],
            ),
            patch(f"{_PATCH_BASE}.move_to_device", side_effect=lambda obj, device: obj),
            patch(f"{_PATCH_BASE}.clear_memory"),
        ):
            proc.apply()

        assert mock_train.call_count == 2

        # After layer-0, layer_train_inputs should become layer-0 quant outputs.
        # That rolled list should feed layer-1 training.
        second_call = mock_train.call_args_list[1].kwargs
        assert second_call["layer_inputs"] is l0_quant_out
        assert second_call["fp_layer_outputs"] is l1_fp_train
