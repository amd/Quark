#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from unittest.mock import patch

import pytest
import torch
from torch import nn

from quark.torch.quantization.api import from_float_and_dict
from quark.torch.quantization.config.config import Int4PerGroupSpec, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PlaceholderObserver

INT4_PER_GROUP_SYM_SPEC = Int4PerGroupSpec(ch_axis=1, is_dynamic=False, group_size=128).to_quantization_spec()


def test_set_group_size():
    # Create an instance of QTensorConfig

    # Set group size
    new_group_size = 8
    INT4_PER_GROUP_SYM_SPEC.set_group_size(new_group_size)

    # Assert the group size was set correctly
    assert INT4_PER_GROUP_SYM_SPEC.group_size == new_group_size, "The group size should be updated to the new value"

    # Test with group_size = -1 (valid case)
    INT4_PER_GROUP_SYM_SPEC.set_group_size(-1)
    assert INT4_PER_GROUP_SYM_SPEC.group_size == -1, "The group size should be set to -1"

    # Test with group_size = 0.1 (invalid type)
    with pytest.raises(TypeError, match="group_size must be an integer"):
        INT4_PER_GROUP_SYM_SPEC.set_group_size(0.1)

    # Test with group_size = 0 (invalid value)
    with pytest.raises(ValueError, match="group_size must be a positive integer or -1"):
        INT4_PER_GROUP_SYM_SPEC.set_group_size(0)


def test_per_block_valid_config() -> None:
    """per_block with valid block_size (tuple or list) should create config correctly."""
    config = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_block,
        observer_cls=PlaceholderObserver,
        is_dynamic=False,
        block_size=(128, 128),
    )
    assert config.block_size == (128, 128) and config.qscheme == QSchemeType.per_block


@pytest.mark.parametrize(
    "bad_block_size",
    [None, 128, (128,), (128.0, 128.0)],
    ids=["none", "scalar", "length_1", "float_elements"],
)
def test_per_block_invalid_block_size_raises(bad_block_size) -> None:
    """per_block with invalid block_size should raise ValueError."""
    with pytest.raises(ValueError, match="block_size"):
        QTensorConfig(
            dtype=Dtype.fp8_e4m3,
            qscheme=QSchemeType.per_block,
            observer_cls=PlaceholderObserver,
            is_dynamic=False,
            block_size=bad_block_size,
        )


def test_qlayer_config_invalid_input_tensors_type_raises() -> None:
    """QLayerConfig with non-QTensorConfig, non-list input_tensors should raise TypeError."""
    with pytest.raises(TypeError, match="quantization_spec for 'input_tensors'"):
        QLayerConfig(input_tensors="invalid")  # type: ignore[arg-type]


def _make_from_float_and_dict_args(weight_qspec_override=None):
    """Helper to build arguments for from_float_and_dict with compressed=True."""
    module = nn.Linear(4, 4)
    layer_name = "layer"
    param_dict = {
        "layer.weight": torch.randn(4, 4),
        "layer.weight_scale": torch.ones(4),
        "layer.weight_zero_point": torch.zeros(4),
    }
    quant_info = {"weight": "layer.weight", "weight_quant": {"dummy": True}}
    return module, quant_info, param_dict, layer_name, weight_qspec_override


def test_from_float_and_dict_non_qtensorconfig_raises() -> None:
    """from_float_and_dict should raise TypeError when weight_qspec is not QTensorConfig."""
    module, quant_info, param_dict, layer_name, _ = _make_from_float_and_dict_args()
    with (
        patch("quark.torch.quantization.api.QTensorConfig.from_dict", return_value="not_a_config"),
        pytest.raises(TypeError, match="weight_qspec must be a QTensorConfig instance"),
    ):
        from_float_and_dict(module, quant_info, param_dict, layer_name, compressed=True)


def test_from_float_and_dict_invalid_qscheme_raises() -> None:
    """from_float_and_dict should raise TypeError when weight_qspec.qscheme is not QSchemeType."""
    module, quant_info, param_dict, layer_name, _ = _make_from_float_and_dict_args()
    fake_qspec = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
    )
    fake_qspec.qscheme = "invalid_qscheme"  # type: ignore[assignment]
    with (
        patch("quark.torch.quantization.api.QTensorConfig.from_dict", return_value=fake_qspec),
        pytest.raises(TypeError, match="weight_qspec.qscheme must be a QSchemeType instance"),
    ):
        from_float_and_dict(module, quant_info, param_dict, layer_name, compressed=True)


def test_from_float_and_dict_invalid_dtype_raises() -> None:
    """from_float_and_dict should raise TypeError when weight_qspec.dtype is not Dtype."""
    module, quant_info, param_dict, layer_name, _ = _make_from_float_and_dict_args()
    fake_qspec = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
    )
    fake_qspec.dtype = "invalid_dtype"  # type: ignore[assignment]
    with (
        patch("quark.torch.quantization.api.QTensorConfig.from_dict", return_value=fake_qspec),
        pytest.raises(TypeError, match="weight_qspec.dtype must be a Dtype instance"),
    ):
        from_float_and_dict(module, quant_info, param_dict, layer_name, compressed=True)
