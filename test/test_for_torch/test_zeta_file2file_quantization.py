# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import concurrent.futures
import importlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from huggingface_hub import hf_hub_download

from quark.torch.quantization.config.config import FP8E4M3PerTensorSpec, QConfig, QLayerConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType
from quark.torch.quantization.file2file_quantization import (
    _apply_weight_converters,
    _build_exclude_aware_quant_config,
    _collect_mxfp4_source_module_names,
    _collect_rotation_tensor_names,
    _collect_tensor_names_matching_quark_exclude,
    _convert_linear_weight_tensor_name_to_module_name,
    _convert_linear_weight_tensor_names_to_module_names,
    _export_config,
    _export_quant_config,
    _get_layer_quant_config_by_tensor_name,
    _get_model_dtype_from_hf_model_config,
    _get_non_quantized_tensor_names_from_model_safetensors,
    _is_mxfp4_source_pattern,
    _load_safetensor_with_recover,
    _quantize_and_save_safetensor_shard,
    _recover_compressed_tensors_weights,
    _recover_fp8_weights,
    _resolve_legacy_positional_device_arg,
    _single_stage_quantize_weight,
    quantize_model_per_safetensor,
)
from quark.torch.quantization.weight_convert import Chunk, WeightConverter


class _FakeSlice:
    def __init__(self, dtype_name: str, shape: tuple[int, ...] | None = None) -> None:
        self._dtype_name = dtype_name
        self._shape = shape

    def get_dtype(self) -> str:
        return self._dtype_name

    def get_shape(self) -> list[int]:
        """Return the configured shape; raises if no shape was set at construction."""
        if self._shape is None:
            raise AttributeError("get_shape was not configured on this _FakeSlice")
        return list(self._shape)


class _FakeSafeOpen:
    def __init__(
        self,
        tensor_map: dict[str, torch.Tensor],
        dtype_name_map: dict[str, str] | None = None,
    ) -> None:
        self.tensor_map = tensor_map
        self.dtype_name_map = dtype_name_map or {}

    def __enter__(self) -> "_FakeSafeOpen":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def keys(self) -> list[str]:
        return list(self.tensor_map.keys())

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        return self.tensor_map[tensor_name]

    def get_slice(self, tensor_name: str) -> _FakeSlice:
        # Shape is sourced from the real tensor when available so MXFP4-pattern
        # detection (which peeks shape via ``_peek_shape``) works in tests
        # without callers having to supply a separate shape map.
        shape = tuple(self.tensor_map[tensor_name].shape) if tensor_name in self.tensor_map else None
        return _FakeSlice(self.dtype_name_map[tensor_name], shape=shape)


def _build_minimal_quant_config(exclude: list[str] | None = None) -> QConfig:
    return QConfig(global_quant_config=QLayerConfig(), exclude=exclude or [])


def _weight_map_no_cross_shard(shards: list[str]) -> dict[str, str]:
    """Build a weight_map where each shard holds one self-contained weight, so
    _weight_map_has_cross_shard_dependency is False and the multi-device path runs."""
    return {f"model.layers.{i}.self_attn.q_proj.weight": shard for i, shard in enumerate(shards)}


def test_get_model_dtype_from_hf_model_config_prefers_nested_text_dtype() -> None:
    model_dtype = _get_model_dtype_from_hf_model_config(
        {
            "text_config": {"dtype": "bfloat16", "torch_dtype": "float16"},
            "torch_dtype": "float16",
            "dtype": "float32",
        }
    )
    assert model_dtype == torch.bfloat16


def test_get_model_dtype_from_hf_model_config_falls_back_to_explicit_bfloat16_for_auto() -> None:
    assert _get_model_dtype_from_hf_model_config({"torch_dtype": "auto"}) == torch.bfloat16


def test_get_model_dtype_from_hf_model_config_defaults_when_config_is_missing() -> None:
    """Verify that a missing Hugging Face config falls back to the explicit file-to-file default dtype."""
    assert _get_model_dtype_from_hf_model_config(None) == torch.bfloat16


def test_get_model_dtype_from_hf_model_config_prefers_declared_dtype_over_default() -> None:
    """A declared dtype always wins; the bf16 default only covers missing/auto configs."""
    assert _get_model_dtype_from_hf_model_config({"torch_dtype": "float16"}) == torch.float16
    assert _get_model_dtype_from_hf_model_config({"text_config": {"torch_dtype": "float16"}}) == torch.float16


def test_weight_dequant_fp8_always_returns_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that FP8 dequantization always returns fp32 so downstream quantizers
    receive full-precision input regardless of model storage dtype."""
    import quark.common.utils.import_utils as import_utils
    import quark.torch.quantization.file2file_quantization as file2file_quantization

    fake_triton = types.ModuleType("triton")
    fake_tl = types.ModuleType("triton.language")
    fake_triton.__path__ = []
    fake_triton.cdiv = lambda dividend, divisor: (dividend + divisor - 1) // divisor
    fake_triton.jit = lambda function: function
    fake_triton.language = fake_tl
    fake_tl.constexpr = object()

    captured: dict[str, object] = {}

    class _FakeKernel:
        def __getitem__(self, grid):  # type: ignore[no-untyped-def]
            def launcher(
                x: torch.Tensor,
                _scale_inv: torch.Tensor,
                y: torch.Tensor,
                m_dim: int,
                n_dim: int,
                *,
                BLOCK_SIZE: int,
            ) -> None:
                captured["grid"] = grid({"BLOCK_SIZE": BLOCK_SIZE})
                captured["shape"] = (m_dim, n_dim)
                captured["dtype"] = y.dtype
                y.copy_(x.to(y.dtype))

            return launcher

    try:
        with monkeypatch.context() as context:
            context.setattr(import_utils, "is_triton_available", lambda: True)
            context.setitem(sys.modules, "triton", fake_triton)
            context.setitem(sys.modules, "triton.language", fake_tl)
            importlib.reload(file2file_quantization)
            context.setattr(file2file_quantization, "_weight_dequant_kernel", _FakeKernel())

            output = file2file_quantization._weight_dequant_fp8(
                torch.ones((2, 2), dtype=torch.float16).contiguous(),
                torch.ones((1, 1), dtype=torch.float32).contiguous(),
            )
    finally:
        importlib.reload(file2file_quantization)

    assert output.dtype == torch.float32
    assert captured["dtype"] == torch.float32
    assert captured["grid"] == (1, 1)
    assert captured["shape"] == (2, 2)


def test_single_stage_quantize_weight_promotes_scale_type_to_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_scale_type: ScaleType | None = None
    quantized_tensors: dict[str, torch.Tensor] = {}

    class _FakeQuantizer:
        def __init__(self, scale_type: ScaleType | None) -> None:
            self.scale = torch.tensor([1.0], dtype=torch.float32)
            self.zero_point = torch.tensor([0], dtype=torch.int32)
            self.quant_min = -448
            self.quant_max = 448
            self.scale_type = scale_type

        def enable_observer(self) -> None:
            return None

        def disable_fake_quant(self) -> None:
            return None

        def __call__(self, _tensor: torch.Tensor) -> torch.Tensor:
            return _tensor

    class _FakePackMethod:
        def pack(self, tensor: torch.Tensor, _reorder: bool) -> torch.Tensor:
            return tensor

    def fake_get_fake_quantize(weight_config):  # type: ignore[no-untyped-def]
        nonlocal recorded_scale_type
        recorded_scale_type = weight_config.scale_type
        return _FakeQuantizer(weight_config.scale_type)

    def fake_scaled_real_quantize(*args, **kwargs):  # type: ignore[no-untyped-def]
        return args[1]

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.FakeQuantizeBase.get_fake_quantize",
        staticmethod(fake_get_fake_quantize),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.quark.torch.kernel.scaled_real_quantize",
        fake_scaled_real_quantize,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.create_pack_method",
        lambda **_kwargs: _FakePackMethod(),
    )

    weight_config = FP8E4M3PerTensorSpec(
        observer_method="min_max", scale_type="float", is_dynamic=False
    ).to_quantization_spec()
    _single_stage_quantize_weight(
        tensor=torch.ones((2, 2), dtype=torch.bfloat16),
        tensor_name="model.layers.0.q_proj.weight",
        layer_name="model.layers.0.q_proj",
        weight_config=weight_config,
        quantized_tensors=quantized_tensors,
        safetensor_filename="model-00001.safetensors",
    )

    assert weight_config.scale_type == ScaleType.float
    assert recorded_scale_type == ScaleType.float32
    assert quantized_tensors["model.layers.0.q_proj.weight_scale"].dtype == torch.float32


@pytest.mark.parametrize(
    ("scale_type_name", "expected_scale_type", "expected_scale_dtype"),
    [
        pytest.param("float16", ScaleType.float16, torch.float16, id="float16"),
        pytest.param("bfloat16", ScaleType.bfloat16, torch.bfloat16, id="bfloat16"),
    ],
)
def test_single_stage_quantize_weight_preserves_explicit_low_precision_scale_type(
    monkeypatch: pytest.MonkeyPatch,
    scale_type_name: str,
    expected_scale_type: ScaleType,
    expected_scale_dtype: torch.dtype,
) -> None:
    recorded_scale_type: ScaleType | None = None

    class _FakeQuantizer:
        def __init__(self, scale_type: ScaleType | None) -> None:
            self.scale = torch.tensor([1.0], dtype=expected_scale_dtype)
            self.zero_point = torch.tensor([0], dtype=torch.int32)
            self.quant_min = -448
            self.quant_max = 448
            self.scale_type = scale_type

        def enable_observer(self) -> None:
            return None

        def disable_fake_quant(self) -> None:
            return None

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

    class _FakePackMethod:
        def pack(self, tensor: torch.Tensor, _reorder: bool) -> torch.Tensor:
            return tensor

    def fake_get_fake_quantize(weight_config):  # type: ignore[no-untyped-def]
        nonlocal recorded_scale_type
        recorded_scale_type = weight_config.scale_type
        return _FakeQuantizer(weight_config.scale_type)

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.FakeQuantizeBase.get_fake_quantize",
        staticmethod(fake_get_fake_quantize),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.quark.torch.kernel.scaled_real_quantize",
        lambda *args, **kwargs: args[1],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.create_pack_method",
        lambda **_kwargs: _FakePackMethod(),
    )

    weight_config = FP8E4M3PerTensorSpec(
        observer_method="min_max", scale_type=scale_type_name, is_dynamic=False
    ).to_quantization_spec()
    quantized_tensors: dict[str, torch.Tensor] = {}
    _single_stage_quantize_weight(
        tensor=torch.ones((2, 2), dtype=torch.bfloat16),
        tensor_name="model.layers.0.q_proj.weight",
        layer_name="model.layers.0.q_proj",
        weight_config=weight_config,
        quantized_tensors=quantized_tensors,
        safetensor_filename="model-00001.safetensors",
    )

    assert weight_config.scale_type == expected_scale_type
    assert recorded_scale_type == expected_scale_type
    assert quantized_tensors["model.layers.0.q_proj.weight_scale"].dtype == expected_scale_dtype


def test_convert_linear_weight_name_helpers_cover_module_name_conversion() -> None:
    assert _convert_linear_weight_tensor_name_to_module_name("model.layers.0.q_proj.weight") == "model.layers.0.q_proj"
    module_names = _convert_linear_weight_tensor_names_to_module_names(
        [
            "model.layers.0.q_proj.weight",
            "model.layers.1.norm.weight",
            "model.embed_tokens.weight",
        ]
    )
    assert module_names == {"model.layers.0.q_proj"}


def test_get_layer_quant_config_by_tensor_name_skips_1d_weight() -> None:
    """1-D weights (RMSNorm/LayerNorm) must be rejected even when their name passes the heuristic.

    Covers the ``tensor_loaded.ndim < 2`` guard introduced to fix the
    mtp.pre_fc_norm_hidden false-positive: the module name ends with "hidden"
    rather than "norm", so the name heuristic alone cannot catch it.
    """
    qconfig = _build_minimal_quant_config()
    # 1-D norm weight whose name passes _is_linear_weight_tensor → must return None
    assert (
        _get_layer_quant_config_by_tensor_name(
            "mtp.pre_fc_norm_hidden.weight",
            qconfig,
            tensor_loaded=torch.ones(4096),
        )
        is None
    )
    # 2-D linear weight → must return a config (not None)
    assert (
        _get_layer_quant_config_by_tensor_name(
            "model.layers.0.q_proj.weight",
            qconfig,
            tensor_loaded=torch.ones(128, 128),
        )
        is not None
    )


def test_collect_tensor_names_matching_quark_exclude_skips_1d_norm_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_peek_shape guard must filter 1-D norm weights from the exclude-scan path.

    Covers the inline ``_shape is not None and len(_shape) < 2`` guard in
    ``_get_safetensor_excluded_module_names``.
    """
    fake_tensors = {
        "model.layers.0.q_proj.weight": torch.ones(128, 128),  # 2-D linear
        "mtp.pre_fc_norm_hidden.weight": torch.ones(128),  # 1-D norm
    }
    fake_dtype_map = dict.fromkeys(fake_tensors, "BF16")

    def fake_get_safetensor_files(_path: str) -> list[str]:
        return ["dummy.safetensors"]

    def fake_safe_open(_path: str, framework: str) -> _FakeSafeOpen:
        return _FakeSafeOpen(fake_tensors, dtype_name_map=fake_dtype_map)

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        fake_get_safetensor_files,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.safe_open",
        fake_safe_open,
    )

    # Wildcard exclude matches both tensors by name; shape guard must drop the 1-D one
    excluded = _collect_tensor_names_matching_quark_exclude("dummy_path", _build_minimal_quant_config(exclude=["*"]))
    assert "model.layers.0.q_proj.weight" in excluded
    assert "mtp.pre_fc_norm_hidden.weight" not in excluded


def test_recover_compressed_tensors_weights_respects_keep_original_tensor_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeQuantizationArgs:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

    class _FakeQuantizationScheme:
        def __init__(self, targets: list[str], weights: _FakeQuantizationArgs) -> None:
            self.targets = targets
            self.weights = weights

    class _FakeCompressorCls:
        @classmethod
        def decompress(
            cls,
            state_dict: dict[str, torch.Tensor],
            scheme: _FakeQuantizationScheme,
        ) -> dict[str, torch.Tensor]:
            return {"weight": state_dict["weight_scale"] * 42}

    fake_tensor_map = {
        "model.layers.0.q_proj.weight_scale": torch.tensor([1.0]),
        "model.layers.0.q_proj.weight": torch.tensor([2.0]),
        "model.layers.0.q_proj.bias": torch.tensor([3.0]),
        "model.layers.1.k_proj.weight_scale": torch.tensor([4.0]),
        "model.layers.1.k_proj.weight": torch.tensor([5.0]),
    }

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        assert framework == "pt"
        assert isinstance(device, str)
        return _FakeSafeOpen(fake_tensor_map)

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.QuantizationArgs",
        _FakeQuantizationArgs,
        raising=False,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.QuantizationScheme",
        _FakeQuantizationScheme,
        raising=False,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.BaseCompressor",
        type("_FakeBaseCompressor", (), {"get_value_from_registry": staticmethod(lambda _fmt: _FakeCompressorCls)}),
        raising=False,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.is_package_lower_or_equal",
        lambda *_args, **_kwargs: False,
        raising=False,
    )
    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)

    recovered_tensors = _recover_compressed_tensors_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={
            "quant_method": "compressed-tensors",
            "format": "dense",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "tensor",
                        "group_size": 128,
                        "symmetric": True,
                    }
                }
            },
        },
        device="cpu",
        keep_excluded_layers_as_original_model_state=True,
        keep_original_model_state_tensor_names_set={"model.layers.0.q_proj.weight"},
    )
    # q_proj is excluded — kept as-is (both weight and scale pass through as non-quantized)
    assert "model.layers.0.q_proj.weight_scale" in recovered_tensors
    assert "model.layers.0.q_proj.weight" in recovered_tensors
    assert recovered_tensors["model.layers.0.q_proj.bias"].item() == 3.0
    # k_proj is NOT excluded — decompressed via the fake compressor (scale * 42)
    assert recovered_tensors["model.layers.1.k_proj.weight"].item() == 4.0 * 42
    assert "model.layers.1.k_proj.weight_scale" not in recovered_tensors


def test_get_non_quantized_tensor_names_skips_scale_and_ignores_failed_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_safetensor_files(_path: str) -> list[str]:
        return ["ok.safetensors", "bad.safetensors"]

    def fake_safe_open(file_path: str, framework: str):  # type: ignore[no-untyped-def]
        assert framework == "pt"
        if file_path == "bad.safetensors":
            raise RuntimeError("broken shard")
        return _FakeSafeOpen(
            tensor_map={
                "model.layers.0.q_proj.weight": torch.tensor([1.0]),
                "model.layers.0.q_proj.weight_scale": torch.tensor([1.0]),
            },
            dtype_name_map={
                "model.layers.0.q_proj.weight": "F16",
                "model.layers.0.q_proj.weight_scale": "F16",
            },
        )

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files", fake_get_safetensor_files
    )
    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)

    tensor_names = _get_non_quantized_tensor_names_from_model_safetensors("unused")
    assert tensor_names == {"model.layers.0.q_proj.weight"}


def test_collect_tensor_names_matching_quark_exclude_ignores_failed_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    quant_config = _build_minimal_quant_config(exclude=["*.q_proj"])

    def fake_get_safetensor_files(_path: str) -> list[str]:
        return ["ok.safetensors", "bad.safetensors"]

    def fake_safe_open(file_path: str, framework: str):  # type: ignore[no-untyped-def]
        assert framework == "pt"
        if file_path == "bad.safetensors":
            raise RuntimeError("broken shard")
        return _FakeSafeOpen(
            tensor_map={
                "model.layers.0.q_proj.weight": torch.tensor([1.0]),
                "model.layers.0.q_proj.bias": torch.tensor([1.0]),
            }
        )

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files", fake_get_safetensor_files
    )
    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)

    excluded_tensors = _collect_tensor_names_matching_quark_exclude("unused", quant_config)
    assert excluded_tensors == {"model.layers.0.q_proj.weight"}


def test_recover_fp8_weights_validates_quant_method() -> None:
    with pytest.raises(ValueError, match="Expected FP8 quantization config"):
        _recover_fp8_weights(
            safetensor_path="dummy.safetensors",
            quant_config=_build_minimal_quant_config(),
            hf_quant_config_dict={"quant_method": "awq"},
            device="cpu",
            keep_excluded_layers_as_original_model_state=False,
            model_dtype=torch.float16,
        )


def test_recover_fp8_weights_keeps_excluded_layers_in_original_format(monkeypatch: pytest.MonkeyPatch) -> None:
    tensor_map = {
        "model.layers.0.q_proj.weight": torch.ones((2, 2), dtype=torch.float16),
        "model.layers.0.q_proj.weight_scale_inv": torch.full((2, 2), 2.0, dtype=torch.float16),
    }

    def fake_safe_open(_path: str, framework: str, device: str):  # type: ignore[no-untyped-def]
        assert framework == "pt"
        assert isinstance(device, str)
        return _FakeSafeOpen(tensor_map=tensor_map)

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)

    recovered_tensors = _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=True,
        model_dtype=torch.float16,
        keep_original_model_state_tensor_names_set={"model.layers.0.q_proj.weight"},
    )
    assert "model.layers.0.q_proj.weight" in recovered_tensors
    assert "model.layers.0.q_proj.weight_scale" in recovered_tensors


def test_recover_fp8_weights_dequantizes_same_file_scale_inv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that FP8 recovery dequantizes weights when ``_scale_inv`` lives in the same shard."""
    weight_name = "model.layers.0.q_proj.weight"
    scale_inv_name = f"{weight_name}_scale_inv"
    expected_weight = torch.full((2, 2), 4.0, dtype=torch.float16)
    tensor_map = {
        weight_name: torch.ones((2, 2), dtype=torch.float16),
        scale_inv_name: torch.full((2, 2), 2.0, dtype=torch.float16),
    }
    recorded_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    emptied_devices: list[str] = []

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        assert framework == "pt"
        assert isinstance(device, str)
        return _FakeSafeOpen(tensor_map=tensor_map)

    def fake_weight_dequant_fp8(
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
        **_kwargs: Any,
    ) -> torch.Tensor:
        assert block_size == 128
        recorded_calls.append((weight, scale_inv))
        return expected_weight

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)
    monkeypatch.setattr("quark.torch.quantization.file2file_quantization._weight_dequant_fp8", fake_weight_dequant_fp8)
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda device: emptied_devices.append(str(device)),
    )

    recovered_tensors = _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
    )

    assert len(recorded_calls) == 1
    assert torch.equal(recorded_calls[0][0], tensor_map[weight_name])
    assert torch.equal(recorded_calls[0][1], tensor_map[scale_inv_name])
    assert torch.equal(recovered_tensors[weight_name], expected_weight)
    assert emptied_devices == ["cpu"]


def test_recover_fp8_weights_uses_cross_file_scale_inv_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that FP8 recovery dequantizes weights with a preloaded cross-file ``_scale_inv`` cache."""
    weight_name = "model.layers.0.q_proj.weight"
    scale_inv_name = f"{weight_name}_scale_inv"
    expected_weight = torch.full((2, 2), 5.0, dtype=torch.float16)
    tensor_map = {
        weight_name: torch.ones((2, 2), dtype=torch.float16),
    }
    cached_scale_inv = torch.full((2, 2), 3.0, dtype=torch.float16)
    recorded_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    emptied_devices: list[str] = []

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        assert framework == "pt"
        assert isinstance(device, str)
        return _FakeSafeOpen(tensor_map=tensor_map)

    def fake_weight_dequant_fp8(
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
        **_kwargs: Any,
    ) -> torch.Tensor:
        assert block_size == 128
        recorded_calls.append((weight, scale_inv))
        return expected_weight

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)
    monkeypatch.setattr("quark.torch.quantization.file2file_quantization._weight_dequant_fp8", fake_weight_dequant_fp8)
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda device: emptied_devices.append(str(device)),
    )

    recovered_tensors = _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
        scale_inv_cache={scale_inv_name: cached_scale_inv},
    )

    assert len(recorded_calls) == 1
    assert torch.equal(recorded_calls[0][0], tensor_map[weight_name])
    assert torch.equal(recorded_calls[0][1], cached_scale_inv)
    assert torch.equal(recovered_tensors[weight_name], expected_weight)
    assert emptied_devices == ["cpu"]


def test_recover_fp8_weights_dequantizes_dsv4_sibling_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that FP8 recovery dequantizes DeepSeek-V4 weights using the sibling .scale tensor."""
    weight_name = "model.layers.0.ffn.experts.0.w1.weight"
    sibling_scale_name = "model.layers.0.ffn.experts.0.w1.scale"
    expected_weight = torch.full((2, 2), 6.0, dtype=torch.float16)
    tensor_map = {
        weight_name: torch.ones((2, 2), dtype=torch.float8_e4m3fn),
        sibling_scale_name: torch.full((2, 2), 3.0, dtype=torch.float32),
    }
    recorded_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    emptied_devices: list[str] = []

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        assert framework == "pt"
        assert isinstance(device, str)
        return _FakeSafeOpen(tensor_map=tensor_map)

    def fake_weight_dequant_fp8(
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
        **_kwargs: Any,
    ) -> torch.Tensor:
        assert block_size == 128
        recorded_calls.append((weight, scale_inv))
        return expected_weight

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)
    monkeypatch.setattr("quark.torch.quantization.file2file_quantization._weight_dequant_fp8", fake_weight_dequant_fp8)
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda device: emptied_devices.append(str(device)),
    )

    recovered_tensors = _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
    )

    assert len(recorded_calls) == 1
    # FP8 tensors do not support torch.equal; compare via uint8 view
    assert torch.equal(recorded_calls[0][0].view(torch.uint8), tensor_map[weight_name].view(torch.uint8))
    assert torch.equal(recorded_calls[0][1], tensor_map[sibling_scale_name])
    assert torch.equal(recovered_tensors[weight_name], expected_weight)
    # Sibling scale tensor must NOT appear in recovered output (it is consumed, not stored)
    assert sibling_scale_name not in recovered_tensors
    assert emptied_devices == ["cpu"]


def test_recover_fp8_weights_keeps_excluded_dsv4_layers_with_sibling_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify excluded DeepSeek-V4 layers are kept in original FP8 format with their sibling scale."""
    weight_name = "model.layers.0.ffn.experts.0.w1.weight"
    sibling_scale_name = "model.layers.0.ffn.experts.0.w1.scale"
    tensor_map = {
        weight_name: torch.ones((2, 2), dtype=torch.float8_e4m3fn),
        sibling_scale_name: torch.full((2, 2), 4.0, dtype=torch.float32),
    }

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        assert framework == "pt"
        assert isinstance(device, str)
        return _FakeSafeOpen(tensor_map=tensor_map)

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)

    recovered_tensors = _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=True,
        model_dtype=torch.float16,
        keep_original_model_state_tensor_names_set={weight_name},
    )

    assert weight_name in recovered_tensors
    expected_quark_scale_name = f"{weight_name}_scale"
    assert expected_quark_scale_name in recovered_tensors
    assert torch.equal(recovered_tensors[expected_quark_scale_name], tensor_map[sibling_scale_name])


def test_recover_fp8_weights_excluded_layer_uses_cross_file_scale_inv_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Excluded FP8 layers must source their scale from scale_inv_cache when the
    _scale_inv tensor lives in a different shard (BUG-013).

    The excluded-layer branch has two in-shard lookup tiers (_scale_inv name, sibling
    .scale name) but was missing the third tier: scale_inv_cache.  When the _scale_inv
    lives in a different shard and scale_inv_cache is provided, the scale is silently
    dropped — the output checkpoint contains the raw FP8 bytes but no scale, making it
    unusable for downstream inference.

    This test FAILS with current code and passes after adding the cache-lookup elif
    before the terminal ``continue`` in ``_recover_fp8_weights``.
    """
    weight_name = "model.layers.0.q_proj.weight"
    scale_inv_name = f"{weight_name}_scale_inv"
    cached_scale = torch.full((1, 1), 0.5, dtype=torch.float32)

    # Shard contains the weight but NOT the _scale_inv (it lives in a different shard).
    # The cross-shard scale was pre-loaded into scale_inv_cache by _build_cross_file_scale_inv_cache.
    tensor_map = {weight_name: torch.ones((2, 2), dtype=torch.float16)}

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        assert framework == "pt"
        return _FakeSafeOpen(tensor_map=tensor_map)

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)

    recovered_tensors = _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=True,
        model_dtype=torch.float16,
        keep_original_model_state_tensor_names_set={weight_name},
        scale_inv_cache={scale_inv_name: cached_scale},
    )

    assert weight_name in recovered_tensors
    quark_scale_name = f"{weight_name}_scale"
    assert quark_scale_name in recovered_tensors, (
        f"BUG-013: scale_inv_cache was not consulted in the excluded-layer branch; "
        f"'{quark_scale_name}' is absent from recovered_tensors even though "
        f"'{scale_inv_name}' was present in scale_inv_cache"
    )
    assert torch.equal(recovered_tensors[quark_scale_name], cached_scale)


@pytest.mark.parametrize(
    ("quant_method", "expected_dispatch"),
    [
        pytest.param("fp8", "fp8", id="dispatch-fp8"),
        pytest.param("compressed-tensors", "compressed", id="dispatch-compressed-tensors"),
    ],
)
def test_load_safetensor_with_recover_dispatches_by_quant_method(
    monkeypatch: pytest.MonkeyPatch, quant_method: str, expected_dispatch: str
) -> None:
    dispatch_record: dict[str, str] = {}

    def fake_get_quantization_config(_hf_model_config: dict) -> dict[str, str]:
        return {"quant_method": quant_method}

    def fake_recover_fp8(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        dispatch_record["method"] = "fp8"
        return {"fp8.weight": torch.tensor([1.0])}

    def fake_recover_compressed(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        dispatch_record["method"] = "compressed"
        return {"compressed.weight": torch.tensor([1.0])}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        fake_get_quantization_config,
    )
    monkeypatch.setattr("quark.torch.quantization.file2file_quantization._recover_fp8_weights", fake_recover_fp8)
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._recover_compressed_tensors_weights",
        fake_recover_compressed,
    )

    _ = _load_safetensor_with_recover(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
        hf_model_config={"quantization_config": {"quant_method": quant_method}},
    )
    assert dispatch_record["method"] == expected_dispatch


def test_quantize_and_save_safetensor_shard_handles_preexisting_packed_and_scale_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded_tensors = {
        "model.layers.0.q_proj.weight_packed": torch.tensor([0], dtype=torch.uint8),
        "model.layers.0.q_proj.weight_shape": torch.tensor([1], dtype=torch.int32),
        "model.layers.0.q_proj.weight_scale": torch.tensor([1.0], dtype=torch.float32),
        "model.layers.0.q_proj.weight": torch.tensor([[2.0]], dtype=torch.float32),
        "model.layers.0.q_proj.bias": torch.tensor([3.0], dtype=torch.float32),
    }
    saved_payload: dict[str, torch.Tensor] = {}

    def fake_load_safetensor_with_recover(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return loaded_tensors

    def fake_get_layer_quant_config_by_tensor_name(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return None

    def fake_empty_cache_if_cuda(_device: str) -> None:
        return None

    def fake_save_file(tensor_map: dict[str, torch.Tensor], _output_path: str) -> None:
        saved_payload.update(tensor_map)

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_safetensor_with_recover",
        fake_load_safetensor_with_recover,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_layer_quant_config_by_tensor_name",
        fake_get_layer_quant_config_by_tensor_name,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda", fake_empty_cache_if_cuda
    )
    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.save_file", fake_save_file)
    monkeypatch.setattr("os.path.getsize", lambda _path: 1024)

    _quantize_and_save_safetensor_shard(
        safetensor_path="model-00001.safetensors",
        export_path=str(tmp_path),
        quant_config=_build_minimal_quant_config(),
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
    )

    assert "model.layers.0.q_proj.weight_packed" not in saved_payload
    assert "model.layers.0.q_proj.weight_shape" not in saved_payload
    assert "model.layers.0.q_proj.weight_scale" in saved_payload
    assert "model.layers.0.q_proj.weight" in saved_payload
    assert "model.layers.0.q_proj.bias" in saved_payload


def test_quantize_and_save_safetensor_shard_applies_weight_converters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify shard processing applies weight converters before quantization/copying."""
    loaded_tensors = {
        "model.layers.0.q_proj.weight": torch.arange(8, dtype=torch.float16).reshape(4, 2),
    }
    saved_payload: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_safetensor_with_recover",
        lambda *_args, **_kwargs: loaded_tensors,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_layer_quant_config_by_tensor_name",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda _device: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.save_file",
        lambda tensor_map, _output_path: saved_payload.update(tensor_map),
    )
    monkeypatch.setattr("os.path.getsize", lambda _path: 1024)

    _quantize_and_save_safetensor_shard(
        safetensor_path="model-00001.safetensors",
        export_path=str(tmp_path),
        quant_config=_build_minimal_quant_config(),
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
        weight_converters=[
            WeightConverter(
                "q_proj.weight",
                ["q_a.weight", "q_b.weight"],
                operations=[Chunk(dim=0)],
            )
        ],
    )

    assert set(saved_payload) == {"model.layers.0.q_a.weight", "model.layers.0.q_b.weight"}
    torch.testing.assert_close(
        saved_payload["model.layers.0.q_a.weight"], loaded_tensors["model.layers.0.q_proj.weight"][:2]
    )
    torch.testing.assert_close(
        saved_payload["model.layers.0.q_b.weight"], loaded_tensors["model.layers.0.q_proj.weight"][2:]
    )


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="torch.float8_e8m0fnu requires torch >= 2.5",
)
def test_quantize_and_save_safetensor_shard_pass_through_preserves_original_state_dtype(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pass-through export must widen only recovered fp32 weights, never the
    original source state kept for excluded layers. The float8 weight and its
    float8 scale carry the quantized wire format Stage-4 dequant depends on, and
    an excluded layer whose source is genuinely fp32 must also survive intact --
    the decision is keyed on the recorded excluded-name set, not on dtype alone.
    """
    excluded_fp8_weight = "model.layers.0.q_proj.weight"
    excluded_fp8_scale = "model.layers.0.q_proj.weight_scale"
    excluded_fp32_weight = "model.layers.2.q_proj.weight"
    recovered_fp32_weight = "model.layers.1.q_proj.weight"

    loaded_tensors = {
        excluded_fp8_weight: torch.zeros((2, 2), dtype=torch.float8_e4m3fn),
        excluded_fp8_scale: torch.full((2, 1), 0x7F, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        excluded_fp32_weight: torch.ones((2, 2), dtype=torch.float32),
        recovered_fp32_weight: torch.ones((2, 2), dtype=torch.float32),
    }
    saved_payload: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_safetensor_with_recover",
        lambda *_args, **_kwargs: loaded_tensors,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_layer_quant_config_by_tensor_name",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda _device: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.save_file",
        lambda tensor_map, _output_path: saved_payload.update(tensor_map),
    )
    monkeypatch.setattr("os.path.getsize", lambda _path: 1024)

    _quantize_and_save_safetensor_shard(
        safetensor_path="model-00001.safetensors",
        export_path=str(tmp_path),
        quant_config=_build_minimal_quant_config(),
        device="cpu",
        keep_excluded_layers_as_original_model_state=True,
        model_dtype=torch.bfloat16,
        keep_original_model_state_tensor_names_set={excluded_fp8_weight, excluded_fp32_weight},
    )

    # Preserved originals keep their exact source dtype (never widened).
    assert saved_payload[excluded_fp8_weight].dtype is torch.float8_e4m3fn
    assert saved_payload[excluded_fp8_scale].dtype is torch.float8_e8m0fnu
    assert saved_payload[excluded_fp32_weight].dtype is torch.float32
    # Recovered fp32 (not an excluded original) is downcast to model_dtype.
    assert saved_payload[recovered_fp32_weight].dtype is torch.bfloat16


def _patch_shard_io(
    monkeypatch: pytest.MonkeyPatch,
    loaded_tensors: dict[str, torch.Tensor],
    saved_payload: dict[str, torch.Tensor],
) -> None:
    """Stub out shard-level IO so only the pass-through dtype policy is exercised."""
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_safetensor_with_recover",
        lambda *_args, **_kwargs: loaded_tensors,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_layer_quant_config_by_tensor_name",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda _device: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.save_file",
        lambda tensor_map, _output_path: saved_payload.update(tensor_map),
    )
    monkeypatch.setattr("os.path.getsize", lambda _path: 1024)


def test_quantize_and_save_safetensor_shard_preserves_excluded_source_floating_dtypes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Excluded weights that are floating point in the source keep that dtype.

    A single global model_dtype cannot represent mixed excluded dtypes, such as
    BF16 non-router projections plus an FP32 router, so a weight that never went
    through dequantization must survive verbatim.
    """
    excluded_bf16_weight = "model.layers.0.feed_forward.pooler.weight"
    excluded_fp32_weight = "model.layers.0.feed_forward.router.weight"
    recovered_fp16_weight = "model.layers.0.self_attn.q_proj.weight"

    loaded_tensors = {
        excluded_bf16_weight: torch.ones((2, 2), dtype=torch.bfloat16),
        excluded_fp32_weight: torch.ones((2, 2), dtype=torch.float32),
        recovered_fp16_weight: torch.ones((2, 2), dtype=torch.float16),
    }
    saved_payload: dict[str, torch.Tensor] = {}
    _patch_shard_io(monkeypatch, loaded_tensors, saved_payload)

    _quantize_and_save_safetensor_shard(
        safetensor_path="model-00001.safetensors",
        export_path=str(tmp_path),
        quant_config=_build_minimal_quant_config(),
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float32,
        keep_original_model_state_tensor_names_set=set(),
        excluded_source_floating_tensor_names={excluded_bf16_weight, excluded_fp32_weight},
    )

    assert saved_payload[excluded_bf16_weight].dtype is torch.bfloat16
    assert saved_payload[excluded_fp32_weight].dtype is torch.float32
    # Not in the source-floating set: it was recovered, so it is normalized.
    assert saved_payload[recovered_fp16_weight].dtype is torch.float32


def test_quantize_and_save_safetensor_shard_casts_recovered_excluded_linear_to_model_dtype(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An excluded weight dequantized from an FP8 source still lands on model_dtype.

    ``_weight_dequant_fp8`` always produces fp32; leaving it there would double the
    tensor on disk relative to the checkpoint's declared dtype.
    """
    recovered_excluded_weight = "model.layers.0.self_attn.q_proj.weight"

    loaded_tensors = {recovered_excluded_weight: torch.ones((2, 2), dtype=torch.float32)}
    saved_payload: dict[str, torch.Tensor] = {}
    _patch_shard_io(monkeypatch, loaded_tensors, saved_payload)

    _quantize_and_save_safetensor_shard(
        safetensor_path="model-00001.safetensors",
        export_path=str(tmp_path),
        quant_config=_build_minimal_quant_config(exclude=["model.layers.*.self_attn.q_proj"]),
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.bfloat16,
        keep_original_model_state_tensor_names_set=set(),
        # Quantized in the source, so absent from the source-floating set.
        excluded_source_floating_tensor_names=set(),
    )

    assert saved_payload[recovered_excluded_weight].dtype is torch.bfloat16


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch.float8_e4m3fn requires torch >= 2.1",
)
def test_excluded_tensor_names_split_by_source_dtype_on_fp8_checkpoint(tmp_path: Path) -> None:
    """An FP8 checkpoint can hold excluded weights of both kinds at once.

    The router is stored fp32 and was never quantized; the projection is stored fp8
    with a scale_inv. Both match ``exclude``, and the two must be classified apart so
    the router keeps fp32 while the projection follows the recover-then-cast path.

    ``rescale_proj`` is the third case: floating point, but with "scale" in its name.
    The split is keyed on each weight's own dtype precisely so a module named after a
    scale is not mistaken for a companion scale tensor.
    """
    import safetensors.torch

    fp32_router = "model.layers.0.mlp.gate.weight"
    fp32_rescale_proj = "mtp.layers.0.mlp.rescale_proj.weight"
    fp8_proj = "model.layers.0.self_attn.q_proj.weight"
    shard_path = str(tmp_path / "model-00001.safetensors")
    safetensors.torch.save_file(
        {
            fp32_router: torch.ones((4, 4), dtype=torch.float32),
            fp32_rescale_proj: torch.ones((4, 4), dtype=torch.float32),
            fp8_proj: torch.ones((4, 4), dtype=torch.float8_e4m3fn),
            f"{fp8_proj}_scale_inv": torch.ones((1, 1), dtype=torch.float32),
        },
        shard_path,
    )

    quant_config = _build_minimal_quant_config(
        exclude=["model.layers.*.mlp.gate", "mtp.layers.*.mlp.rescale_proj", "model.layers.*.self_attn.q_proj"]
    )
    source_floating: set[str] = set()
    excluded = _collect_tensor_names_matching_quark_exclude(
        str(tmp_path), quant_config, source_floating_names=source_floating
    )

    assert excluded == {fp32_router, fp32_rescale_proj, fp8_proj}
    assert source_floating == {fp32_router, fp32_rescale_proj}
    # The name-substring scan cannot make this call: it drops rescale_proj as a companion.
    assert fp32_rescale_proj not in _get_non_quantized_tensor_names_from_model_safetensors(str(tmp_path))


def test_quantize_and_save_safetensor_shard_follows_excluded_dtype_through_a_converter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A converter renames the excluded weight, and its source dtype follows the rename.

    ``excluded_source_floating_tensor_names`` is resolved against the source checkpoint,
    but converters run before the pass-through loop. Without propagating the rename, both
    halves of the split would be cast to ``model_dtype``.
    """
    excluded_fp32_weight = "model.layers.0.self_attn.q_proj.weight"

    loaded_tensors = {excluded_fp32_weight: torch.ones((4, 4), dtype=torch.float32)}
    saved_payload: dict[str, torch.Tensor] = {}
    _patch_shard_io(monkeypatch, loaded_tensors, saved_payload)

    _quantize_and_save_safetensor_shard(
        safetensor_path="model-00001.safetensors",
        export_path=str(tmp_path),
        quant_config=_build_minimal_quant_config(exclude=["model.layers.*.self_attn.q_proj"]),
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.bfloat16,
        keep_original_model_state_tensor_names_set=set(),
        excluded_source_floating_tensor_names={excluded_fp32_weight},
        weight_converters=[WeightConverter("q_proj.weight", ["q_a.weight", "q_b.weight"], operations=[Chunk(dim=0)])],
    )

    assert set(saved_payload) == {
        "model.layers.0.self_attn.q_a.weight",
        "model.layers.0.self_attn.q_b.weight",
    }
    assert saved_payload["model.layers.0.self_attn.q_a.weight"].dtype is torch.float32
    assert saved_payload["model.layers.0.self_attn.q_b.weight"].dtype is torch.float32


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch.float8_e4m3fn requires torch >= 2.1",
)
def test_quantize_and_save_safetensor_shard_follows_preserved_original_through_a_converter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A preserved prequantized weight keeps its wire format across a converter rename.

    ``keep_original_model_state_tensor_names_set`` is resolved against the source
    checkpoint just like the source-floating set, so a converter takes its outputs out
    of it too. Without propagating the rename, both halves of the split would be cast
    to ``model_dtype`` and the preserved wire format would be lost.
    """
    preserved_fp8_weight = "model.layers.0.self_attn.q_proj.weight"

    loaded_tensors = {preserved_fp8_weight: torch.ones((4, 4), dtype=torch.float8_e4m3fn)}
    saved_payload: dict[str, torch.Tensor] = {}
    _patch_shard_io(monkeypatch, loaded_tensors, saved_payload)

    _quantize_and_save_safetensor_shard(
        safetensor_path="model-00001.safetensors",
        export_path=str(tmp_path),
        quant_config=_build_minimal_quant_config(exclude=["model.layers.*.self_attn.q_proj"]),
        device="cpu",
        keep_excluded_layers_as_original_model_state=True,
        model_dtype=torch.bfloat16,
        keep_original_model_state_tensor_names_set={preserved_fp8_weight},
        excluded_source_floating_tensor_names=set(),
        weight_converters=[WeightConverter("q_proj.weight", ["q_a.weight", "q_b.weight"], operations=[Chunk(dim=0)])],
    )

    assert set(saved_payload) == {
        "model.layers.0.self_attn.q_a.weight",
        "model.layers.0.self_attn.q_b.weight",
    }
    assert saved_payload["model.layers.0.self_attn.q_a.weight"].dtype is torch.float8_e4m3fn
    assert saved_payload["model.layers.0.self_attn.q_b.weight"].dtype is torch.float8_e4m3fn


def test_quantize_and_save_safetensor_shard_pass_through_widens_only_linear_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The widen-to-model_dtype step targets recovered Linear weights only.
    Non-Linear tensors that never go through quantization -- embeddings and
    norms -- must pass through with their source dtype untouched even when it
    differs from model_dtype, so they are never accidentally rewritten. A
    recovered Linear weight whose source dtype differs is still widened.
    """
    linear_weight = "model.layers.0.q_proj.weight"
    embed_weight = "model.embed_tokens.weight"
    norm_weight = "model.layers.0.input_layernorm.weight"

    loaded_tensors = {
        linear_weight: torch.ones((2, 2), dtype=torch.float16),
        embed_weight: torch.ones((2, 2), dtype=torch.float16),
        norm_weight: torch.ones((2,), dtype=torch.float16),
    }
    saved_payload: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_safetensor_with_recover",
        lambda *_args, **_kwargs: loaded_tensors,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_layer_quant_config_by_tensor_name",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda _device: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.save_file",
        lambda tensor_map, _output_path: saved_payload.update(tensor_map),
    )
    monkeypatch.setattr("os.path.getsize", lambda _path: 1024)

    _quantize_and_save_safetensor_shard(
        safetensor_path="model-00001.safetensors",
        export_path=str(tmp_path),
        quant_config=_build_minimal_quant_config(),
        device="cpu",
        keep_excluded_layers_as_original_model_state=True,
        model_dtype=torch.bfloat16,
        keep_original_model_state_tensor_names_set=set(),
    )

    # Recovered Linear weight is widened to model_dtype.
    assert saved_payload[linear_weight].dtype is torch.bfloat16
    # Non-Linear tensors keep their source dtype, never widened.
    assert saved_payload[embed_weight].dtype is torch.float16
    assert saved_payload[norm_weight].dtype is torch.float16


def test_apply_weight_converters_rejects_unsized_source_patterns() -> None:
    """Verify invalid source pattern containers fail before conversion starts."""
    converter = type(
        "BadConverter",
        (),
        {
            "source_patterns": object(),
            "target_patterns": ["target.weight"],
            "operations": [],
        },
    )()

    with pytest.raises(ValueError, match="source_patterns must be a str or sequence"):
        _apply_weight_converters({"layer.source.weight": torch.ones(1)}, [converter])


def test_resolve_legacy_positional_device_rejects_ambiguous_arguments() -> None:
    """Verify compatibility helper rejects ambiguous legacy positional device usage."""
    with pytest.raises(TypeError, match="at most one positional argument"):
        _resolve_legacy_positional_device_arg(("cpu", "cuda"), None, "test_api")

    with pytest.raises(TypeError, match="specified both positionally and by keyword"):
        _resolve_legacy_positional_device_arg(("cpu",), "cuda", "test_api")


def test_quantize_model_per_safetensor_builds_keep_original_tensor_name_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    quant_config = _build_minimal_quant_config(exclude=["*.q_proj"])
    recorded_keep_set: set[str] = set()
    recorded_floating_set: set[str] = set()
    recorded_model_dtype: torch.dtype | None = None

    def fake_get_hf_model_config(_path: str) -> dict:
        return {"torch_dtype": "float16"}

    def fake_get_quantization_config(_hf_model_config: dict) -> None:
        return None

    def fake_get_safetensor_files(_path: str) -> list[str]:
        return ["model.safetensors-00091-of-00094.safetensors"]

    def fake_collect_excluded(_path: str, _config: QConfig, source_floating_names: set[str] | None = None) -> set[str]:
        if source_floating_names is not None:
            source_floating_names.add("model.layers.0.q_proj.weight")
        return {"model.layers.0.q_proj.weight", "model.layers.1.q_proj.weight"}

    def fake_quantize_shard(**kwargs) -> dict[str, str]:
        nonlocal recorded_model_dtype
        recorded_keep_set.update(kwargs["keep_original_model_state_tensor_names_set"])
        recorded_floating_set.update(kwargs["excluded_source_floating_tensor_names"])
        recorded_model_dtype = kwargs["model_dtype"]
        return {}

    def fake_build_exclude_aware_quant_config(
        _pretrained_model_path: str,
        input_quant_config: QConfig,
        _hf_model_config: dict,
        _keep_excluded_layers_as_original_model_state: bool,
    ) -> QConfig:
        return input_quant_config

    def fake_export_config(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config", fake_get_hf_model_config
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        fake_get_quantization_config,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files", fake_get_safetensor_files
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_tensor_names_matching_quark_exclude",
        fake_collect_excluded,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_and_save_safetensor_shard",
        fake_quantize_shard,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        fake_build_exclude_aware_quant_config,
    )
    monkeypatch.setattr("quark.torch.quantization.file2file_quantization._export_config", fake_export_config)

    quantize_model_per_safetensor(
        pretrained_model_path="unused-pretrained-path",
        quant_config=quant_config,
        save_path=str(tmp_path / "export"),
        keep_excluded_layers_as_original_model_state=True,
        device="cpu",
    )
    assert recorded_keep_set == {"model.layers.1.q_proj.weight"}
    assert recorded_floating_set == {"model.layers.0.q_proj.weight"}
    assert recorded_model_dtype == torch.float16


def test_quantize_model_per_safetensor_accepts_legacy_positional_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify legacy positional ``device`` still works when ``weight_converters`` is keyword-only."""
    quantize_call_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: ["model-00001-of-00001.safetensors"],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_and_save_safetensor_shard",
        lambda **kwargs: quantize_call_kwargs.update(kwargs) or {},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _pretrained_model_path, input_quant_config, _hf_model_config, _keep_original: input_quant_config,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_config",
        lambda *_args, **_kwargs: None,
    )

    quantize_model_per_safetensor(
        "unused-pretrained-path",
        _build_minimal_quant_config(),
        str(tmp_path / "export"),
        False,
        "cpu",
        weight_converters=["converter"],
    )

    assert quantize_call_kwargs["device"] == "cpu"
    assert quantize_call_kwargs["weight_converters"] == ["converter"]


def test_direct_quantize_checkpoint_accepts_legacy_positional_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ``ModelQuantizer.direct_quantize_checkpoint`` keeps legacy positional ``device`` support."""
    import quark.torch.quantization.api as api_mod

    forwarded_args: list[object] = []
    forwarded_kwargs: dict[str, object] = {}
    dummy_quantizer = type("DummyQuantizer", (), {"config": _build_minimal_quant_config()})()

    def fake_quantize(*args: object, **kwargs: object) -> None:
        forwarded_args.extend(args)
        forwarded_kwargs.update(kwargs)

    monkeypatch.setattr(api_mod, "quantize_model_per_safetensor", fake_quantize)

    api_mod.ModelQuantizer.direct_quantize_checkpoint(
        dummy_quantizer,
        "unused-pretrained-path",
        "unused-export-path",
        False,
        "cpu",
        weight_converters=["converter"],
    )

    # Legacy positional device is threaded through positionally, keeping old call sites working.
    assert "unused-pretrained-path" in forwarded_args
    assert "unused-export-path" in forwarded_args
    assert "cpu" in forwarded_args
    assert forwarded_kwargs["weight_converters"] == ["converter"]


def test_quantize_model_per_safetensor_loads_and_cleans_fp8_scale_inv_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify that FP8 file-to-file quantization preloads and releases the cross-file scale_inv cache."""
    quant_config = _build_minimal_quant_config()
    fake_weight_map = {
        "model.layers.0.q_proj.weight": "model-00091-of-00092.safetensors",
        "model.layers.0.q_proj.weight_scale_inv": "model-00092-of-00092.safetensors",
    }
    fake_scale_inv_cache = {
        "model.layers.0.q_proj.weight_scale_inv": torch.ones((2, 2), dtype=torch.float16),
    }
    quantize_call_kwargs: dict[str, object] = {}
    export_call_kwargs: dict[str, object] = {}
    emptied_devices: list[str] = []

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: {"quant_method": "fp8"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_weight_map",
        lambda _path: fake_weight_map,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_cross_file_scale_inv_cache",
        lambda _path, weight_map, device: (
            fake_scale_inv_cache if weight_map is fake_weight_map and str(device) == "cuda:0" else {}
        ),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: ["model-00091-of-00092.safetensors"],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_and_save_safetensor_shard",
        lambda **kwargs: quantize_call_kwargs.update(kwargs) or {},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _pretrained_model_path, input_quant_config, _hf_model_config, _keep_original: input_quant_config,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_config",
        lambda pretrained_model_path, exported_quant_config, save_path, weight_map, hf_model_config: (
            export_call_kwargs.update(
                {
                    "pretrained_model_path": pretrained_model_path,
                    "quant_config": exported_quant_config,
                    "save_path": save_path,
                    "weight_map": weight_map,
                    "hf_model_config": hf_model_config,
                }
            )
        ),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda device: emptied_devices.append(str(device)),
    )

    quantize_model_per_safetensor(
        pretrained_model_path="unused-pretrained-path",
        quant_config=quant_config,
        save_path=str(tmp_path / "export"),
        keep_excluded_layers_as_original_model_state=False,
        device="cuda:0",
    )

    assert quantize_call_kwargs["source_weight_map"] is fake_weight_map
    assert quantize_call_kwargs["scale_inv_cache"] is fake_scale_inv_cache
    assert quantize_call_kwargs["model_dtype"] == torch.float16
    assert export_call_kwargs["pretrained_model_path"] == "unused-pretrained-path"
    assert export_call_kwargs["quant_config"] is quant_config
    assert export_call_kwargs["save_path"] == str(tmp_path / "export")
    assert export_call_kwargs["hf_model_config"] == {"torch_dtype": "float16"}
    assert emptied_devices == ["cuda:0"]


def test_quantize_model_per_safetensor_does_not_mutate_default_dtype_on_fp8_cpu_early_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_default_dtype = torch.get_default_dtype()

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: {"quant_method": "fp8"},
    )

    quantize_model_per_safetensor(
        pretrained_model_path="unused-pretrained-path",
        quant_config=_build_minimal_quant_config(),
        save_path="unused-export-path",
        keep_excluded_layers_as_original_model_state=False,
        device="cpu",
    )
    assert torch.get_default_dtype() == original_default_dtype


def test_quantize_model_per_safetensor_does_not_mutate_default_dtype_when_processing_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_default_dtype = torch.get_default_dtype()

    def fake_get_hf_model_config(_path: str) -> dict[str, str]:
        return {"torch_dtype": "float16"}

    def fake_get_quantization_config(_hf_model_config: dict[str, str]) -> None:
        return None

    def fake_get_safetensor_files(_path: str) -> list[str]:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        fake_get_hf_model_config,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        fake_get_quantization_config,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        fake_get_safetensor_files,
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        quantize_model_per_safetensor(
            pretrained_model_path="unused-pretrained-path",
            quant_config=_build_minimal_quant_config(),
            save_path="unused-export-path",
            keep_excluded_layers_as_original_model_state=False,
            device="cpu",
        )
    assert torch.get_default_dtype() == original_default_dtype


def test_export_quant_config_writes_sorted_exclude_list(tmp_path: Path) -> None:
    quant_config = _build_minimal_quant_config(exclude=["z.block", "a.block"])
    _export_quant_config(
        hf_model_config={"architectures": ["DummyForCausalLM"]},
        quant_config=quant_config,
        save_path=str(tmp_path),
    )
    with open(tmp_path / "config.json", encoding="utf-8") as file:
        output_config = json.load(file)
    assert output_config["quantization_config"]["exclude"] == ["a.block", "z.block"]


def test_build_exclude_aware_quant_config_returns_sorted_excludes_when_not_keeping_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_config = _build_minimal_quant_config(exclude=["*.q_proj"])
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_tensor_names_matching_quark_exclude",
        lambda *_args, **_kwargs: {"model.layers.3.q_proj.weight", "model.layers.1.q_proj.weight"},
    )

    updated_config = _build_exclude_aware_quant_config(
        pretrained_model_path="unused",
        quant_config=quant_config,
        hf_model_config={"quantization_config": {}},
        keep_excluded_layers_as_original_model_state=False,
    )
    assert updated_config.exclude == ["model.layers.1.q_proj", "model.layers.3.q_proj"]


@pytest.mark.parametrize(
    ("hf_quantization_config", "expected_message"),
    [
        pytest.param(
            {"fmt": "unsupported", "activation_scheme": "dynamic", "weight_block_size": [128, 128]},
            "Unsupported quantization format",
            id="unsupported-fmt",
        ),
        pytest.param(
            {"fmt": "e4m3", "activation_scheme": "static", "weight_block_size": [128, 128]},
            "Only dynamic activation scheme is supported",
            id="non-dynamic-activation",
        ),
        pytest.param(
            {"fmt": "e4m3", "activation_scheme": "dynamic"},
            "Only per-block quantization is currently supported",
            id="missing-weight-block-size",
        ),
    ],
)
def test_build_exclude_aware_quant_config_validates_source_quantization_config(
    monkeypatch: pytest.MonkeyPatch,
    hf_quantization_config: dict[str, object],
    expected_message: str,
) -> None:
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_tensor_names_matching_quark_exclude",
        lambda *_args, **_kwargs: {"model.layers.0.q_proj.weight"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_non_quantized_tensor_names_from_model_safetensors",
        lambda *_args, **_kwargs: set(),
    )

    with pytest.raises(ValueError, match=expected_message):
        _build_exclude_aware_quant_config(
            pretrained_model_path="unused",
            quant_config=_build_minimal_quant_config(exclude=["*.q_proj"]),
            hf_model_config={"quantization_config": hf_quantization_config},
            keep_excluded_layers_as_original_model_state=True,
        )


def _patch_source_shards(
    monkeypatch: pytest.MonkeyPatch,
    excluded: set[str],
    non_quantized: set[str],
    mxfp4: set[str] | None = None,
) -> None:
    """Stub the disk-scanning helpers so ``_build_exclude_aware_quant_config`` runs
    without real safetensors files."""
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_tensor_names_matching_quark_exclude",
        lambda *_args, **_kwargs: excluded,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_non_quantized_tensor_names_from_model_safetensors",
        lambda *_args, **_kwargs: non_quantized,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_mxfp4_source_module_names",
        lambda *_args, **_kwargs: mxfp4 or set(),
    )


def test_build_exclude_aware_quant_config_defaults_omitted_fmt_to_e4m3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``quant_method: fp8`` with ``fmt`` omitted (e.g. Qwen3.5-35B-A3B-FP8)
    defaults to ``fp8_e4m3`` instead of raising. (fmt-present paths are covered by
    ``splits_retained_exclude_and_layer_config`` / ``describes_mxfp4...``.)"""
    quant_config = _build_minimal_quant_config(exclude=["*.experts.*"])
    quant_config.layer_quant_config = None  # type: ignore[assignment]
    _patch_source_shards(
        monkeypatch,
        excluded={"model.layers.0.mlp.experts.0.w2.weight"},
        non_quantized=set(),
    )

    updated_config = _build_exclude_aware_quant_config(
        pretrained_model_path="unused",
        quant_config=quant_config,
        hf_model_config={"quantization_config": {"activation_scheme": "dynamic", "weight_block_size": [128, 128]}},
        keep_excluded_layers_as_original_model_state=True,
    )

    assert updated_config.layer_quant_config is not None
    described = updated_config.layer_quant_config["model.layers.0.mlp.experts.0.w2"]
    assert described.weight is not None and not isinstance(described.weight, list)
    assert described.weight.dtype is Dtype.fp8_e4m3


def test_build_exclude_aware_quant_config_honors_explicit_e5m2_fmt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``fmt: e5m2`` overrides the E4M3 default."""
    quant_config = _build_minimal_quant_config(exclude=["*.experts.*"])
    quant_config.layer_quant_config = None  # type: ignore[assignment]
    _patch_source_shards(
        monkeypatch,
        excluded={"model.layers.0.mlp.experts.0.w2.weight"},
        non_quantized=set(),
    )

    updated_config = _build_exclude_aware_quant_config(
        pretrained_model_path="unused",
        quant_config=quant_config,
        hf_model_config={
            "quantization_config": {
                "fmt": "e5m2",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            }
        },
        keep_excluded_layers_as_original_model_state=True,
    )

    assert updated_config.layer_quant_config is not None
    described = updated_config.layer_quant_config["model.layers.0.mlp.experts.0.w2"]
    assert described.weight is not None and not isinstance(described.weight, list)
    assert described.weight.dtype is Dtype.fp8_e5m2


@pytest.mark.parametrize(
    ("fmt_value", "expected_dtype"),
    [
        pytest.param("float8_e4m3fn", "fp8_e4m3", id="float8_e4m3fn"),
        pytest.param("float8_e4m3", "fp8_e4m3", id="float8_e4m3"),
        pytest.param("float8_e5m2", "fp8_e5m2", id="float8_e5m2"),
    ],
)
def test_build_exclude_aware_quant_config_normalizes_torch_style_fmt(
    monkeypatch: pytest.MonkeyPatch,
    fmt_value: str,
    expected_dtype: str,
) -> None:
    """``fmt`` spellings like ``float8_e4m3fn``) are normalized to the
    short HF form instead of raising ``Unsupported format``."""
    quant_config = _build_minimal_quant_config(exclude=["*.experts.*"])
    quant_config.layer_quant_config = None  # type: ignore[assignment]
    _patch_source_shards(
        monkeypatch,
        excluded={"model.layers.0.mlp.experts.0.w2.weight"},
        non_quantized=set(),
    )

    updated_config = _build_exclude_aware_quant_config(
        pretrained_model_path="unused",
        quant_config=quant_config,
        hf_model_config={
            "quantization_config": {
                "fmt": fmt_value,
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            }
        },
        keep_excluded_layers_as_original_model_state=True,
    )

    assert updated_config.layer_quant_config is not None
    described = updated_config.layer_quant_config["model.layers.0.mlp.experts.0.w2"]
    assert described.weight is not None and not isinstance(described.weight, list)
    assert described.weight.dtype is getattr(Dtype, expected_dtype)


def test_build_exclude_aware_quant_config_splits_retained_exclude_and_layer_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_config = _build_minimal_quant_config(exclude=["*.q_proj"])
    quant_config.layer_quant_config = None  # type: ignore[assignment]

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_tensor_names_matching_quark_exclude",
        lambda *_args, **_kwargs: {"model.layers.0.q_proj.weight", "model.layers.1.q_proj.weight"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_non_quantized_tensor_names_from_model_safetensors",
        lambda *_args, **_kwargs: {"model.layers.0.q_proj.weight"},
    )

    updated_config = _build_exclude_aware_quant_config(
        pretrained_model_path="unused",
        quant_config=quant_config,
        hf_model_config={
            "quantization_config": {
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 64],
                "ch_axis": -1,
                "group_size": 64,
                "round_method": "half_even",
                "scale_type": "float",
                "symmetric": True,
            }
        },
        keep_excluded_layers_as_original_model_state=True,
    )

    assert updated_config.exclude == ["model.layers.0.q_proj"]
    assert updated_config.layer_quant_config is not None
    assert "model.layers.1.q_proj" in updated_config.layer_quant_config


def test_build_exclude_aware_quant_config_describes_mxfp4_excluded_layers_as_mxfp4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed-format source (source ``fmt: e4m3`` but some excluded layers stored
    on disk as MXFP4) must describe the MXFP4 layers with the per-group FP4 / e8m0
    scheme, not inherit the source FP8 ``fmt``. Regression for unquantized DeepSeek-V4
    MTP routed experts (MXFP4 in source) being mislabeled ``fp8_e4m3``."""
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_tensor_names_matching_quark_exclude",
        lambda *_args, **_kwargs: {"mtp.0.ffn.experts.0.w2.weight", "mtp.0.ffn.shared_experts.w2.weight"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_non_quantized_tensor_names_from_model_safetensors",
        lambda *_args, **_kwargs: set(),
    )
    # Routed expert is MXFP4 on disk; shared expert is genuine FP8.
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_mxfp4_source_module_names",
        lambda *_args, **_kwargs: {"mtp.0.ffn.experts.0.w2"},
    )

    updated_config = _build_exclude_aware_quant_config(
        pretrained_model_path="unused",
        quant_config=_build_minimal_quant_config(exclude=["mtp.*"]),
        hf_model_config={
            "quantization_config": {
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "scale_fmt": "ue8m0",
            }
        },
        keep_excluded_layers_as_original_model_state=True,
    )

    assert updated_config.layer_quant_config is not None
    routed = updated_config.layer_quant_config["mtp.0.ffn.experts.0.w2"]
    shared = updated_config.layer_quant_config["mtp.0.ffn.shared_experts.w2"]

    assert routed.weight is not None and not isinstance(routed.weight, list)
    assert routed.weight.dtype is Dtype.fp4
    assert routed.weight.qscheme is QSchemeType.per_group
    assert routed.weight.group_size == 32
    assert routed.weight.scale_format == "e8m0"
    assert routed.input_tensors is not None and not isinstance(routed.input_tensors, list)
    assert routed.input_tensors.is_dynamic is True

    # Shared expert keeps the source FP8 description.
    assert shared.weight is not None and not isinstance(shared.weight, list)
    assert shared.weight.dtype is Dtype.fp8_e4m3


def test_collect_mxfp4_source_module_names_detects_only_mxfp4_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The collector must flag only Linear weights from ``excluded_tensor_names``
    whose (weight, sibling ``.scale``) dtype pair matches the MXFP4 wire format
    (U8/I8 packed + F8_E8M0, 1x32 ratio), leaving genuine FP8 layers alone."""
    tensor_map = {
        # MXFP4: U8 [4, 16] packed weight + F8_E8M0 [4, 1] scale (inner ratio 16 -> group 32).
        "routed.w2.weight": torch.zeros((4, 16), dtype=torch.uint8),
        "routed.w2.scale": torch.zeros((4, 1), dtype=torch.uint8),
        # FP8: full-width F8_E4M3 weight + block scale, not MXFP4.
        "shared.w2.weight": torch.zeros((4, 4), dtype=torch.uint8),
        "shared.w2.scale": torch.zeros((1, 1), dtype=torch.uint8),
    }
    dtype_name_map = {
        "routed.w2.weight": "U8",
        "routed.w2.scale": "F8_E8M0",
        "shared.w2.weight": "F8_E4M3",
        "shared.w2.scale": "F8_E4M3",
    }
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda *_args, **_kwargs: ["shard.safetensors"],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.safe_open",
        lambda *_args, **_kwargs: _FakeSafeOpen(tensor_map, dtype_name_map),
    )

    # Pass both as the excluded set; collector should only flag the MXFP4 one.
    excluded_tensor_names = {"routed.w2.weight", "shared.w2.weight"}
    result = _collect_mxfp4_source_module_names("unused", excluded_tensor_names)
    assert result == {"routed.w2"}


def test_collect_mxfp4_source_module_names_skips_weights_without_sibling_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weights in the excluded set that have no sibling ``.scale`` are skipped."""
    tensor_map = {
        # Has weight but no .scale companion
        "no_scale.w1.weight": torch.zeros((4, 16), dtype=torch.uint8),
        # Has both weight and scale, MXFP4 pattern
        "with_scale.w2.weight": torch.zeros((4, 16), dtype=torch.uint8),
        "with_scale.w2.scale": torch.zeros((4, 1), dtype=torch.uint8),
    }
    dtype_name_map = {
        "no_scale.w1.weight": "U8",
        "with_scale.w2.weight": "U8",
        "with_scale.w2.scale": "F8_E8M0",
    }
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda *_args, **_kwargs: ["shard.safetensors"],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.safe_open",
        lambda *_args, **_kwargs: _FakeSafeOpen(tensor_map, dtype_name_map),
    )

    excluded_tensor_names = {"no_scale.w1.weight", "with_scale.w2.weight"}
    result = _collect_mxfp4_source_module_names("unused", excluded_tensor_names)
    # Only the one with a sibling scale is checked and flagged as MXFP4
    assert result == {"with_scale.w2"}


def test_collect_mxfp4_source_module_names_handles_shard_read_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shard read errors are logged and do not crash the scan; it continues
    with the next shard."""

    def fake_safe_open_raises(*args, **kwargs):
        raise OSError("disk read error")

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda *_args, **_kwargs: ["bad_shard.safetensors"],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.safe_open",
        fake_safe_open_raises,
    )

    excluded_tensor_names = {"some.weight"}
    # Should not raise; returns empty set when all shards fail
    result = _collect_mxfp4_source_module_names("unused", excluded_tensor_names)
    assert result == set()


@pytest.mark.parametrize(
    ("source_scale_fmt", "expected_scale_type"),
    [
        pytest.param("ue8m0", "float8_e8m0fnu", id="ue8m0-deepseek-v4-flash"),
        pytest.param("e8m0", "float8_e8m0fnu", id="e8m0-alias"),
        pytest.param(None, "float32", id="absent-deepseek-v3-kimi-minimax"),
        pytest.param("other", "float32", id="unrecognized-falls-back-to-fp32"),
    ],
)
def test_build_exclude_aware_quant_config_maps_source_scale_fmt_to_scale_type(
    monkeypatch: pytest.MonkeyPatch,
    source_scale_fmt: str | None,
    expected_scale_type: str,
) -> None:
    """Verify the ``scale_fmt`` (source HF-FP8 field) to ``scale_type`` (Quark per-layer
    config field) mapping inside ``_build_exclude_aware_quant_config``. DSV4 sets
    ``scale_fmt: "ue8m0"`` and ships F8_E8M0 weight scales on disk; DSV3 / Kimi-K2 /
    MiniMax-M2 omit ``scale_fmt`` and ship F32 weight scales.
    """
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_tensor_names_matching_quark_exclude",
        lambda *_args, **_kwargs: {"model.layers.0.q_proj.weight"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_non_quantized_tensor_names_from_model_safetensors",
        lambda *_args, **_kwargs: set(),
    )

    hf_quantization_config: dict[str, object] = {
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    if source_scale_fmt is not None:
        hf_quantization_config["scale_fmt"] = source_scale_fmt

    updated_config = _build_exclude_aware_quant_config(
        pretrained_model_path="unused",
        quant_config=_build_minimal_quant_config(exclude=["*.q_proj"]),
        hf_model_config={"quantization_config": hf_quantization_config},
        keep_excluded_layers_as_original_model_state=True,
    )

    assert updated_config.layer_quant_config is not None
    per_layer = updated_config.layer_quant_config["model.layers.0.q_proj"]
    assert per_layer.weight is not None and not isinstance(per_layer.weight, list)
    # Weight ``scale_type`` follows the source ``scale_fmt`` (on-disk storage hint).
    assert per_layer.weight.scale_type is ScaleType[expected_scale_type]
    # Input ``scale_type`` stays None regardless of source ``scale_fmt``: dynamic
    # activation scales are runtime-computed, so the observer picks the default.
    assert per_layer.input_tensors is not None and not isinstance(per_layer.input_tensors, list)
    assert per_layer.input_tensors.scale_type is None


def test_scale_type_float8_e8m0fnu_round_trips_and_yields_torch_dtype() -> None:
    """The new ``ScaleType.float8_e8m0fnu`` enum member must (a) serialize as its
    name in ``to_dict``, (b) deserialize back to the same enum member, and
    (c) expose ``torch.float8_e8m0fnu`` via ``to_torch_dtype()`` on torch builds
    that support it. This is what lets downstream applications read the per-layer
    config and allocate the right ``weight_scale`` parameter dtype directly.
    """
    from quark.torch.quantization.config.config import QTensorConfig
    from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType
    from quark.torch.quantization.observer import PerBlock2DMinMaxObserver

    spec = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_block,
        block_size=[128, 128],
        round_method=RoundType.half_even,
        scale_type=ScaleType.float8_e8m0fnu,
        observer_cls=PerBlock2DMinMaxObserver,
        is_dynamic=False,
    )
    serialized = spec.to_dict()
    assert serialized["scale_type"] == "float8_e8m0fnu"
    rebuilt = QTensorConfig.from_dict(serialized)
    assert rebuilt.scale_type is ScaleType.float8_e8m0fnu

    if hasattr(torch, "float8_e8m0fnu"):
        assert ScaleType.float8_e8m0fnu.to_torch_dtype() is torch.float8_e8m0fnu
    else:
        with pytest.raises(RuntimeError, match="requires torch >= 2.5"):
            ScaleType.float8_e8m0fnu.to_torch_dtype()


def _make_quark_source_qc_with_per_layer_e8m0(module_name: str) -> dict[str, object]:
    """Build a minimal Quark-exported ``quantization_config`` dict with one explicit
    per-layer FP8-per-block entry whose ``weight.scale_type == "float8_e8m0fnu"``
    (the DSV4-Flash flavor). Used by the round-trip tests below.
    """
    fp8_per_block_weight = {
        "ch_axis": None,
        "dtype": "fp8_e4m3",
        "group_size": None,
        "block_size": [128, 128],
        "is_dynamic": False,
        "observer_cls": "PerBlock2DMinMaxObserver",
        "qscheme": "per_block",
        "round_method": "half_even",
        "scale_type": "float8_e8m0fnu",
        "symmetric": True,
    }
    fp8_per_group_input = {
        "ch_axis": -1,
        "dtype": "fp8_e4m3",
        "group_size": 128,
        "block_size": None,
        "is_dynamic": True,
        "observer_cls": "PerGroupMinMaxObserver",
        "qscheme": "per_group",
        "round_method": "half_even",
        "scale_type": None,
        "symmetric": True,
    }
    per_layer_config = {
        "input_tensors": fp8_per_group_input,
        "output_tensors": None,
        "weight": fp8_per_block_weight,
        "bias": None,
        "target_device": None,
    }
    return {
        "quant_method": "quark",
        "layer_quant_config": {module_name: per_layer_config},
        "global_quant_config": None,
        "exclude": [],
    }


@pytest.mark.parametrize(
    "source_pattern",
    ["model.layers.0.q_proj", "model.layers.*.q_proj"],
    ids=["exact", "wildcard"],
)
def test_build_exclude_aware_quant_config_quark_source_copies_per_layer_e8m0_entry(
    monkeypatch: pytest.MonkeyPatch, source_pattern: str
) -> None:
    """Quark-source round-trip: when the source ``layer_quant_config`` has a matching
    exact or wildcard entry for an excluded module, the entry must be copied verbatim,
    preserving ``weight.scale_type == ScaleType.float8_e8m0fnu`` end-to-end.
    """
    module_name = "model.layers.0.q_proj"
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_tensor_names_matching_quark_exclude",
        lambda *_args, **_kwargs: {f"{module_name}.weight"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_non_quantized_tensor_names_from_model_safetensors",
        lambda *_args, **_kwargs: set(),
    )

    updated = _build_exclude_aware_quant_config(
        pretrained_model_path="unused",
        quant_config=_build_minimal_quant_config(exclude=["*.q_proj"]),
        hf_model_config={"quantization_config": _make_quark_source_qc_with_per_layer_e8m0(source_pattern)},
        keep_excluded_layers_as_original_model_state=True,
    )

    assert updated.layer_quant_config is not None
    per_layer = updated.layer_quant_config[module_name]
    assert per_layer.weight is not None and not isinstance(per_layer.weight, list)
    # The headline invariant: source ``float8_e8m0fnu`` propagates verbatim.
    assert per_layer.weight.scale_type is ScaleType.float8_e8m0fnu
    assert list(per_layer.weight.block_size) == [128, 128]
    # Input spec is round-tripped too.
    assert per_layer.input_tensors is not None and not isinstance(per_layer.input_tensors, list)
    assert per_layer.input_tensors.scale_type is None
    assert per_layer.input_tensors.is_dynamic is True
    # No leftover entry in ``exclude`` for this quantized module.
    assert module_name not in updated.exclude


def test_build_exclude_aware_quant_config_quark_source_falls_back_to_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the source has no per-layer entry for an excluded module, but does have
    a ``global_quant_config``, that global scheme is copied, preserving its
    ``scale_type``.
    """
    module_name = "model.layers.0.ffn.experts.0.w1"
    mxfp4_common = {
        "dtype": "fp4",
        "qscheme": "per_group",
        "ch_axis": -1,
        "group_size": 32,
        "block_size": None,
        "observer_cls": "PerBlockMXObserver",
        "scale_type": "float",
        "scale_format": "e8m0",
        "round_method": "half_even",
        "scale_calculation_mode": "even",
        "symmetric": None,
    }
    mxfp4_weight = {**mxfp4_common, "is_dynamic": False}
    mxfp4_input = {**mxfp4_common, "is_dynamic": True}
    quark_source_qc = {
        "quant_method": "quark",
        "layer_quant_config": {},
        "global_quant_config": {
            "input_tensors": mxfp4_input,
            "output_tensors": None,
            "weight": mxfp4_weight,
            "bias": None,
            "target_device": None,
        },
        "exclude": [],
    }

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_tensor_names_matching_quark_exclude",
        lambda *_args, **_kwargs: {f"{module_name}.weight"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_non_quantized_tensor_names_from_model_safetensors",
        lambda *_args, **_kwargs: set(),
    )

    updated = _build_exclude_aware_quant_config(
        pretrained_model_path="unused",
        quant_config=_build_minimal_quant_config(exclude=["*.experts.*"]),
        hf_model_config={"quantization_config": quark_source_qc},
        keep_excluded_layers_as_original_model_state=True,
    )

    assert updated.layer_quant_config is not None
    per_layer = updated.layer_quant_config[module_name]
    assert per_layer.weight is not None and not isinstance(per_layer.weight, list)
    assert per_layer.weight.dtype.name == "fp4"
    assert per_layer.weight.scale_format == "e8m0"
    assert per_layer.weight.group_size == 32
    assert module_name not in updated.exclude


@pytest.mark.parametrize(
    "source_pattern",
    ["model.layers.0.q_proj", "model.layers.*.q_proj"],
    ids=["exact", "wildcard"],
)
def test_build_exclude_aware_quant_config_quark_source_preserves_source_exclude(
    monkeypatch: pytest.MonkeyPatch, source_pattern: str
) -> None:
    """Linear modules matching an exact or wildcard source ``exclude`` entry (i.e. unquantized
    in the source) must stay in our output ``exclude``.
    """
    module_name = "model.layers.0.q_proj"
    quark_source_qc = {
        "quant_method": "quark",
        "layer_quant_config": {},
        "global_quant_config": None,
        "exclude": [source_pattern],
    }
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._collect_tensor_names_matching_quark_exclude",
        lambda *_args, **_kwargs: {f"{module_name}.weight"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_non_quantized_tensor_names_from_model_safetensors",
        lambda *_args, **_kwargs: set(),
    )

    updated = _build_exclude_aware_quant_config(
        pretrained_model_path="unused",
        quant_config=_build_minimal_quant_config(exclude=["*.q_proj"]),
        hf_model_config={"quantization_config": quark_source_qc},
        keep_excluded_layers_as_original_model_state=True,
    )

    assert module_name in updated.exclude
    assert updated.layer_quant_config == {} or module_name not in updated.layer_quant_config


def test_export_config_calls_export_quant_config(monkeypatch: pytest.MonkeyPatch) -> None:
    call_record = {"copy_called": False, "index_called": False, "quant_called": False}

    def fake_copy_json_and_py_files(_src: str, _dst: str) -> None:
        call_record["copy_called"] = True

    def fake_export_safetensors_index(_save_path: str, _weight_map: dict[str, str]) -> None:
        call_record["index_called"] = True

    def fake_export_quant_config(_hf_config: dict, _quant_config: QConfig, _save_path: str) -> None:
        call_record["quant_called"] = True

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._copy_json_and_py_files",
        fake_copy_json_and_py_files,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_safetensors_index",
        fake_export_safetensors_index,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_quant_config",
        fake_export_quant_config,
    )

    _export_config(
        pretrained_model_path="input-dir",
        quant_config=_build_minimal_quant_config(),
        save_path="output-dir",
        weight_map={"model.layers.0.q_proj.weight": "model-00001.safetensors"},
        hf_model_config={"architectures": ["DummyForCausalLM"]},
    )
    assert call_record == {"copy_called": True, "index_called": True, "quant_called": True}


def test_recover_compressed_tensors_weights_decompresses_real_model() -> None:
    """Integration test: decompress a real pack-quantized compressed-tensors model."""
    pytest.importorskip("compressed_tensors", reason="compressed_tensors required for compressed-tensors tests")

    model_id = "nm-testing/tinysmokeqwen3moe-W4A16-first-only-CTstable"
    safetensor_path = hf_hub_download(model_id, "model.safetensors")
    config_path = hf_hub_download(model_id, "config.json")

    with open(config_path, encoding="utf-8") as f:
        hf_quant_config_dict = json.load(f)["quantization_config"]

    recovered_tensors = _recover_compressed_tensors_weights(
        safetensor_path=safetensor_path,
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict=hf_quant_config_dict,
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
    )

    # Layer 0 q_proj was quantized (pack-quantized) — must be decompressed to a float .weight tensor.
    decompressed_weight = recovered_tensors["model.layers.0.self_attn.q_proj.weight"]
    assert decompressed_weight.dtype in (torch.float16, torch.float32, torch.bfloat16)
    assert decompressed_weight.shape == (128, 128)

    # Packed / scale / shape artefacts must not appear in the output.
    assert "model.layers.0.self_attn.q_proj.weight_packed" not in recovered_tensors
    assert "model.layers.0.self_attn.q_proj.weight_scale" not in recovered_tensors
    assert "model.layers.0.self_attn.q_proj.weight_shape" not in recovered_tensors

    # Non-quantized tensors (layers 1-5) must pass through unchanged.
    assert "model.layers.1.self_attn.q_proj.weight" in recovered_tensors
    assert recovered_tensors["model.layers.1.self_attn.q_proj.weight"].dtype in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
    )


def test_is_mxfp4_source_pattern_matches_dsv4_expert() -> None:
    """The DSV4 expert wire format (I8/U8 packed bytes + F8_E8M0 sibling
    scale at the 1x32 block ratio) must match the MXFP4 pattern predicate.
    The inner-dim ratio is exactly 16 (16 packed bytes hold 32 FP4 nibbles).
    """
    # I8 container (deepseek-native DSV4 convention)
    assert _is_mxfp4_source_pattern(
        weight_dtype_str="I8",
        scale_dtype_str="F8_E8M0",
        weight_shape=(256, 16),  # logical FP4 inner-dim = 32 -> 32/32 = 1 scale per row
        scale_shape=(256, 1),
    )
    # U8 container (standard Quark/SGLang MXFP4 storage)
    assert _is_mxfp4_source_pattern(
        weight_dtype_str="U8",
        scale_dtype_str="F8_E8M0",
        weight_shape=(512, 32),  # 32 bytes = 64 nibbles = 2 MXFP4 blocks
        scale_shape=(512, 2),
    )


def test_is_mxfp4_source_pattern_rejects_non_mxfp4_inputs() -> None:
    """The predicate must reject (weight, scale) pairs that don't match the
    MXFP4 wire format: wrong container dtype, wrong scale dtype, wrong shape
    ratio, missing shapes, or 1-D inputs."""
    # Wrong weight dtype (F16, not a packed-byte container)
    assert not _is_mxfp4_source_pattern("F16", "F8_E8M0", (256, 16), (256, 1))
    # Wrong scale dtype (F32, not e8m0)
    assert not _is_mxfp4_source_pattern("I8", "F32", (256, 16), (256, 1))
    # Wrong shape ratio: inner ratio is 8 instead of the required 16
    assert not _is_mxfp4_source_pattern("U8", "F8_E8M0", (256, 8), (256, 1))
    # Outer dim mismatch
    assert not _is_mxfp4_source_pattern("U8", "F8_E8M0", (256, 16), (128, 1))
    assert not _is_mxfp4_source_pattern("U8", "F8_E8M0", (2, 256, 16), (3, 256, 1))
    # Missing shapes (peek failed)
    assert not _is_mxfp4_source_pattern("U8", "F8_E8M0", None, (256, 1))
    assert not _is_mxfp4_source_pattern("U8", "F8_E8M0", (256, 16), None)
    # 1-D inputs
    assert not _is_mxfp4_source_pattern("U8", "F8_E8M0", (256,), (256,))


def test_recover_fp8_weights_keeps_excluded_mxfp4_expert_via_case_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An MXFP4 expert weight in the exclude set must be passed through
    with the I8 byte container bit-reinterpreted as U8 (Quark/SGLang
    convention) and its sibling ``.scale`` renamed to ``<weight>_scale``. No
    dequantize kernel is called.
    """
    weight_name = "model.layers.0.ffn.experts.0.w1.weight"
    sibling_scale_name = "model.layers.0.ffn.experts.0.w1.scale"
    weight_bytes = torch.full((256, 16), 42, dtype=torch.int8)  # MXFP4-packed I8
    scale_bytes = torch.full((256, 1), 0x7F, dtype=torch.uint8)  # e8m0 stand-in
    tensor_map = {
        weight_name: weight_bytes,
        sibling_scale_name: scale_bytes,
    }
    dtype_name_map = {
        weight_name: "I8",
        sibling_scale_name: "F8_E8M0",
    }
    dequant_calls: list[str] = []

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        return _FakeSafeOpen(tensor_map=tensor_map, dtype_name_map=dtype_name_map)

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._weight_dequant_fp8",
        lambda *a, **k: dequant_calls.append("called") or torch.empty(0),
    )

    recovered = _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=True,
        model_dtype=torch.bfloat16,
        keep_original_model_state_tensor_names_set={weight_name},
    )

    # Weight present, bit-reinterpreted to U8 (NOT dequantized)
    assert weight_name in recovered
    assert recovered[weight_name].dtype is torch.uint8
    assert torch.equal(recovered[weight_name], weight_bytes.view(torch.uint8))
    # Scale renamed to Quark convention; original sibling key gone
    assert f"{weight_name}_scale" in recovered
    assert torch.equal(recovered[f"{weight_name}_scale"], scale_bytes)
    assert sibling_scale_name not in recovered
    # FP8 dequant must NOT have been invoked
    assert not dequant_calls


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="torch.float8_e8m0fnu requires torch >= 2.5",
)
def test_recover_fp8_weights_case_a_preserves_real_float8_e8m0fnu_scale_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass-through must preserve the source's ``float8_e8m0fnu`` scale
    dtype label on the renamed ``_scale`` output, not silently downcast or
    re-interpret it. The scale output is what differentiates the
    pass-through path from the re-quantize path (which emits uint8-stored
    e8m0);
    """
    weight_name = "model.layers.0.ffn.experts.0.w1.weight"
    sibling_scale_name = "model.layers.0.ffn.experts.0.w1.scale"

    scale_e8m0 = torch.full((256, 1), 0x7F, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    weight_bytes = torch.full((256, 16), 42, dtype=torch.int8)
    tensor_map = {
        weight_name: weight_bytes,
        sibling_scale_name: scale_e8m0,
    }
    dtype_name_map = {
        weight_name: "I8",
        sibling_scale_name: "F8_E8M0",
    }

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        return _FakeSafeOpen(tensor_map=tensor_map, dtype_name_map=dtype_name_map)

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)

    recovered = _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=True,
        model_dtype=torch.bfloat16,
        keep_original_model_state_tensor_names_set={weight_name},
    )

    # Scale dtype label preserved as float8_e8m0fnu (not converted to uint8 / fp32)
    out_scale = recovered[f"{weight_name}_scale"]
    assert out_scale.dtype is torch.float8_e8m0fnu
    # Underlying bytes are byte-identical to source
    assert torch.equal(out_scale.view(torch.uint8), scale_e8m0.view(torch.uint8))


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="torch.float8_e8m0fnu requires torch >= 2.5",
)
def test_weight_dequant_fp8_e8m0_upcast_preserves_scale_values_bitexactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numeric-equivalence test for the e8m0 to fp32 upcast inside
    ``_weight_dequant_fp8``. The triton kernel
    is monkey-patched to capture the ``s`` argument it sees; then verify
    that the captured fp32 tensor's values match exactly what e8m0 to fp32
    direct conversion produces.

    Also asserts the kernel never sees a ``float8_e8m0fnu`` tensor (which
    triton's ``tl.load`` cannot consume).
    """
    import types

    import quark.common.utils.import_utils as import_utils
    import quark.torch.quantization.file2file_quantization as f2f

    # Hand-picked e8m0 byte values spanning the range. e8m0 encodes 2^(b-127);
    # 0x7F = 2^0 = 1.0, 0x80 = 2^1 = 2.0, 0x7E = 2^-1 = 0.5, 0x00 = 2^-127 (subnormal).
    e8m0_bytes = torch.tensor([[0x7F, 0x80, 0x7E, 0x00]], dtype=torch.uint8)
    scale_e8m0 = e8m0_bytes.view(torch.float8_e8m0fnu)
    # The expected fp32 after upcast, what `.to(torch.float32)` should produce
    # for this exact e8m0 input.
    expected_fp32 = scale_e8m0.to(torch.float32)

    captured: dict[str, torch.Tensor] = {}

    # Fake triton stack so _weight_dequant_fp8 takes the triton branch.
    fake_triton = types.ModuleType("triton")
    fake_tl = types.ModuleType("triton.language")
    fake_triton.__path__ = []
    fake_triton.cdiv = lambda a, b: (a + b - 1) // b
    fake_triton.jit = lambda fn: fn
    fake_triton.language = fake_tl
    fake_tl.constexpr = object()

    class _CapturingKernel:
        def __getitem__(self, _grid):  # type: ignore[no-untyped-def]
            def launcher(
                _x: torch.Tensor,
                s: torch.Tensor,
                _y: torch.Tensor,
                _m: int,
                _n: int,
                *,
                BLOCK_SIZE: int,
            ) -> None:
                captured["s"] = s

            return launcher

    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_tl)
    monkeypatch.setattr(import_utils, "is_triton_available", lambda: True)
    # Re-import the module so the `is_triton_available()` gate now picks up
    # the real `_weight_dequant_fp8` triton branch with our patched kernel.
    importlib.reload(f2f)
    monkeypatch.setattr(f2f, "_weight_dequant_kernel", _CapturingKernel(), raising=False)

    # 1×4 weight, 1×4 scale (one-block-per-element to keep things simple);
    weight = torch.zeros((1, 4), dtype=torch.float8_e4m3fn)
    f2f._weight_dequant_fp8(weight, scale_e8m0)

    # The kernel was launched with the upcasted scale, not the original e8m0.
    assert "s" in captured, "triton kernel launcher was not invoked"
    received = captured["s"]
    assert received.dtype is torch.float32, (
        f"kernel must receive fp32 scale (triton cannot load e8m0), got {received.dtype}"
    )
    # Bit-exact value preservation through the upcast.
    assert torch.equal(received, expected_fp32), (
        f"e8m0 to fp32 upcast must preserve values bit-exactly. "
        f"Got {received.tolist()}, expected {expected_fp32.tolist()}"
    )


def test_recover_fp8_weights_dequantizes_mxfp4_expert_via_case_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-excluded MXFP4 expert weight must be routed through
    ``_dequantize_mxfp4_source`` (not the FP8 triton kernel) so the downstream
    re-quantizer sees a bf16/fp16 tensor of the unpacked logical shape.
    """
    weight_name = "model.layers.0.ffn.experts.0.w1.weight"
    sibling_scale_name = "model.layers.0.ffn.experts.0.w1.scale"
    weight_bytes = torch.full((256, 16), 42, dtype=torch.uint8)
    scale_bytes = torch.full((256, 1), 0x7F, dtype=torch.uint8)
    tensor_map = {
        weight_name: weight_bytes,
        sibling_scale_name: scale_bytes,
    }
    dtype_name_map = {
        weight_name: "U8",
        sibling_scale_name: "F8_E8M0",
    }
    dequant_calls: list[tuple[torch.Tensor, torch.Tensor, torch.dtype]] = []
    # Unpacked logical shape: inner dim doubles (2 FP4 nibbles per byte)
    dequant_output = torch.zeros((256, 32), dtype=torch.bfloat16)
    fp8_dequant_calls: list[str] = []

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        return _FakeSafeOpen(tensor_map=tensor_map, dtype_name_map=dtype_name_map)

    def fake_dq_mxfp4(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        dequant_calls.append((weight, scale, dtype))
        return dequant_output

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._weight_dequant_fp8",
        lambda *a, **k: fp8_dequant_calls.append("called") or torch.empty(0),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda _d: None,
    )
    # Patch the kernel module's dq_mxfp4 used by _dequantize_mxfp4_source.
    monkeypatch.setattr("quark.torch.kernel.mx.dq_mxfp4", fake_dq_mxfp4)

    recovered = _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.bfloat16,
    )

    # MXFP4-path dequant called exactly once, FP8-path dequant not at all.
    assert len(dequant_calls) == 1
    assert not fp8_dequant_calls
    # The recovered weight is the kernel output (bf16, unpacked inner dim).
    assert weight_name in recovered
    assert recovered[weight_name].dtype is torch.bfloat16
    assert recovered[weight_name].shape == (256, 32)
    # Sibling scale must NOT leak into the output dict — it was consumed.
    assert sibling_scale_name not in recovered


def test_recover_fp8_weights_routes_real_fp8_attn_through_existing_path_when_mxfp4_experts_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An FP8 attention linear (F8_E4M3 weight + F32 sibling scale)
    in the same shard as MXFP4 expert weights must still flow through the
    existing ``_weight_dequant_fp8`` path. The MXFP4 detector must reject the
    FP8 (weight, scale) pair on dtype grounds.
    """
    attn_weight = "model.layers.0.attn.wq_a.weight"
    attn_scale = "model.layers.0.attn.wq_a.scale"
    expert_weight = "model.layers.0.ffn.experts.0.w1.weight"
    expert_scale = "model.layers.0.ffn.experts.0.w1.scale"

    tensor_map = {
        attn_weight: torch.ones((128, 128), dtype=torch.float8_e4m3fn),
        attn_scale: torch.full((1, 1), 0.5, dtype=torch.float32),
        expert_weight: torch.full((256, 16), 42, dtype=torch.uint8),
        expert_scale: torch.full((256, 1), 0x7F, dtype=torch.uint8),
    }
    dtype_name_map = {
        attn_weight: "F8_E4M3",
        attn_scale: "F32",
        expert_weight: "U8",
        expert_scale: "F8_E8M0",
    }
    fp8_dequant_calls: list[torch.Tensor] = []
    mxfp4_dequant_calls: list[torch.Tensor] = []
    fp8_output = torch.full((128, 128), 3.0, dtype=torch.bfloat16)
    mxfp4_output = torch.zeros((256, 32), dtype=torch.bfloat16)

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        return _FakeSafeOpen(tensor_map=tensor_map, dtype_name_map=dtype_name_map)

    def fake_weight_dequant_fp8(
        weight: torch.Tensor,
        _scale: torch.Tensor,
        block_size: int = 128,
        **_kwargs: Any,
    ) -> torch.Tensor:
        fp8_dequant_calls.append(weight)
        return fp8_output

    def fake_dq_mxfp4(weight: torch.Tensor, _scale: torch.Tensor, _dtype: torch.dtype) -> torch.Tensor:
        mxfp4_dequant_calls.append(weight)
        return mxfp4_output

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)
    monkeypatch.setattr("quark.torch.quantization.file2file_quantization._weight_dequant_fp8", fake_weight_dequant_fp8)
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda _d: None,
    )
    monkeypatch.setattr("quark.torch.kernel.mx.dq_mxfp4", fake_dq_mxfp4)

    recovered = _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.bfloat16,
    )

    # FP8 attn weight went through the FP8 triton path.
    assert len(fp8_dequant_calls) == 1
    assert torch.equal(recovered[attn_weight], fp8_output)
    # MXFP4 expert went through the MX path.
    assert len(mxfp4_dequant_calls) == 1
    assert torch.equal(recovered[expert_weight], mxfp4_output)
    # Neither sibling scale leaks into the output.
    assert attn_scale not in recovered
    assert expert_scale not in recovered


def test_collect_rotation_tensor_names_skips_1d_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    """_collect_rotation_tensor_names must skip 1-D tensors (norms) even when
    the name passes the _is_linear_weight_tensor heuristic."""
    fake_tensors = {
        "model.layers.0.q_proj.weight": torch.ones(128, 128),
        "mtp.pre_fc_norm_hidden.weight": torch.ones(128),
    }
    fake_dtype_map = dict.fromkeys(fake_tensors, "BF16")

    def fake_get_safetensor_files(_path: str) -> list[str]:
        return ["dummy.safetensors"]

    def fake_safe_open(_path: str, framework: str) -> _FakeSafeOpen:
        return _FakeSafeOpen(fake_tensors, dtype_name_map=fake_dtype_map)

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        fake_get_safetensor_files,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.safe_open",
        fake_safe_open,
    )

    names, _bias_names = _collect_rotation_tensor_names("dummy_path")
    assert "model.layers.0.q_proj.weight" in names
    assert "mtp.pre_fc_norm_hidden.weight" not in names


def test_recover_compressed_tensors_weights_returns_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decompressed INT4 weights must be fp32, achieved by upcasting weight_scale
    to fp32 before calling decompress.

    The compressor's dequantize step does `x_q.to(scale.dtype) * scale`, so
    multiply precision is controlled by scale.dtype.  If scale stays bf16/fp16
    the INT4 × scale multiply runs in bf16 and the result is already truncated
    before we even see it.  Upcasting scale to fp32 before decompress is the
    only way to get a full-precision result.  This test fails on code that
    passes the bf16 scale directly to the compressor.
    """
    pytest.importorskip("compressed_tensors", reason="compressed_tensors required")

    import tempfile

    import safetensors.torch

    # Build a minimal pack-quantized INT4 shard manually using pack_to_int32, which is
    # a stable public re-export from compressed_tensors.compressors across all versions.
    # Avoids depending on the compressor class hierarchy (internal paths moved in v0.15+).
    original_weight = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.int8,
    )
    scale = torch.ones(4, 2, dtype=torch.bfloat16)

    from compressed_tensors.compressors import pack_to_int32

    # compressed-tensors >= 0.18 ends pack_to_int32 with a trailing slice
    # (``output[:, :packed_cols]``), which returns a non-contiguous view that
    # safetensors.save_file refuses to serialize.
    weight_packed = pack_to_int32(original_weight, num_bits=4).contiguous()
    weight_shape = torch.tensor(original_weight.shape)

    with tempfile.TemporaryDirectory() as tmp:
        shard_path = str(Path(tmp) / "model.safetensors")
        tensors_to_save = {
            "layer.weight_packed": weight_packed,
            "layer.weight_scale": scale,
            "layer.weight_shape": weight_shape,
            "other.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        }
        safetensors.torch.save_file(tensors_to_save, shard_path)

        hf_quant_config_dict = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "group",
                        "group_size": 4,
                        "symmetric": True,
                    },
                }
            },
        }

        recovered = _recover_compressed_tensors_weights(
            safetensor_path=shard_path,
            quant_config=_build_minimal_quant_config(),
            hf_quant_config_dict=hf_quant_config_dict,
            device="cpu",
            keep_excluded_layers_as_original_model_state=False,
        )

    decompressed = recovered["layer.weight"]
    # Must be fp32: the multiply inside decompress runs at scale.dtype, so
    # upcasting scale to fp32 before the call is the only effective fix.
    assert decompressed.dtype == torch.float32, (
        f"Expected fp32 but got {decompressed.dtype}; weight_scale must be "
        f"upcast to fp32 before calling compressor.decompress."
    )
    # Non-quantized tensor must pass through unchanged.
    assert recovered["other.weight"].dtype == torch.bfloat16


def test_quantize_model_per_safetensor_rejects_empty_devices_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing an empty device list must raise ValueError before any disk access."""
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {},
    )
    with pytest.raises(ValueError, match="'device' list must not be empty"):
        quantize_model_per_safetensor(
            pretrained_model_path="unused",
            quant_config=_build_minimal_quant_config(),
            save_path="unused",
            device=[],
        )


def test_quantize_model_per_safetensor_rejects_multi_device_fp8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-device quantization of an FP8 source model must fail fast at entry
    (issue #6219: FP8 recovery feeds a CPU tensor to Triton under spawn workers).
    A single-device FP8 job must still be accepted, so the guard is specific to
    the multi-device case."""
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: {"quant_method": "fp8"},
    )
    with pytest.raises(ValueError, match="not supported for FP8 source models"):
        quantize_model_per_safetensor(
            pretrained_model_path="unused",
            quant_config=_build_minimal_quant_config(),
            save_path="unused",
            device=["cuda:0", "cuda:1"],
        )


def test_quantize_model_per_safetensor_distributes_shards_across_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With two devices, shards are distributed round-robin and each worker
    receives the correct device assignment."""
    # Record which (device, shard_paths) each worker call receives.
    worker_calls: list[tuple[str, list[str]]] = []

    def fake_quantize_shards_on_device(worker_args: object) -> dict[str, str]:
        from quark.torch.quantization.file2file_quantization import _QuantizeWorkerArgs

        assert isinstance(worker_args, _QuantizeWorkerArgs)
        worker_calls.append((str(worker_args.device), list(worker_args.safetensor_paths)))
        return {
            os.path.basename(shard_path): os.path.basename(shard_path) for shard_path in worker_args.safetensor_paths
        }

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: [
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_weight_map",
        lambda _path: _weight_map_no_cross_shard(
            [
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
                "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors",
            ]
        ),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_shards_on_device",
        fake_quantize_shards_on_device,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _pretrained_model_path, input_quant_config, _hf_model_config, _keep_original: input_quant_config,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_config",
        lambda *_args, **_kwargs: None,
    )

    # ProcessPoolExecutor must be replaced so tests do not spawn real subprocesses.
    class _FakeExecutor:
        def __enter__(self) -> "_FakeExecutor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function: object, worker_args: object) -> concurrent.futures.Future:
            future: concurrent.futures.Future = concurrent.futures.Future()
            future.set_result(function(worker_args))  # type: ignore[operator]
            return future

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.concurrent.futures.ProcessPoolExecutor",
        lambda **_kwargs: _FakeExecutor(),
    )

    quantize_model_per_safetensor(
        pretrained_model_path="unused-pretrained-path",
        quant_config=_build_minimal_quant_config(),
        save_path=str(tmp_path / "export"),
        device=["cpu", "cpu"],
    )

    # Two workers must have been launched (one per device).
    assert len(worker_calls) == 2
    all_assigned_shards = worker_calls[0][1] + worker_calls[1][1]
    assert sorted(all_assigned_shards) == sorted(
        [
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ]
    )
    # Round-robin: worker 0 gets shards 0, 2; worker 1 gets shards 1, 3.
    assert worker_calls[0][1] == [
        "model-00001-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
    ]
    assert worker_calls[1][1] == [
        "model-00002-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ]


def test_quantize_model_per_safetensor_warns_when_devices_exceed_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When more devices are requested than shards exist, a warning must be emitted
    and only as many workers as shards are launched."""
    warnings_emitted: list[str] = []

    class _CapturingLogger:
        def info(self, *_args: object, **_kwargs: object) -> None:
            pass

        def warning(self, msg: str, *args: object, **_kwargs: object) -> None:
            warnings_emitted.append(msg % args if args else msg)

        def error(self, *_args: object, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.logger",
        _CapturingLogger(),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: None,
    )
    # Only 2 shards, but 5 devices requested.
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_weight_map",
        lambda _path: _weight_map_no_cross_shard(
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        ),
    )

    worker_devices: list[str] = []

    def fake_worker(worker_args: object) -> dict[str, str]:
        from quark.torch.quantization.file2file_quantization import _QuantizeWorkerArgs

        assert isinstance(worker_args, _QuantizeWorkerArgs)
        worker_devices.append(str(worker_args.device))
        return {os.path.basename(shard): os.path.basename(shard) for shard in worker_args.safetensor_paths}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_shards_on_device",
        fake_worker,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _p, q, _h, _k: q,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_config",
        lambda *_args, **_kwargs: None,
    )

    class _FakeExecutor:
        def __enter__(self) -> "_FakeExecutor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function: object, worker_args: object) -> concurrent.futures.Future:
            future: concurrent.futures.Future = concurrent.futures.Future()
            future.set_result(function(worker_args))  # type: ignore[operator]
            return future

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.concurrent.futures.ProcessPoolExecutor",
        lambda **_kwargs: _FakeExecutor(),
    )

    quantize_model_per_safetensor(
        pretrained_model_path="unused",
        quant_config=_build_minimal_quant_config(),
        save_path=str(tmp_path / "export"),
        device=["cpu"] * 5,
    )

    assert any("3 device(s) will not be used" in w for w in warnings_emitted), (
        f"Expected unused-device warning, got: {warnings_emitted}"
    )
    assert len(worker_devices) == 2, f"Expected 2 workers, got {len(worker_devices)}"


def test_quantize_model_per_safetensor_cleans_up_partial_files_on_worker_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On any worker failure the output shard files already written to disk must be
    removed so the output directory is not left in a corrupted half-written state."""
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    all_shards = [f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]

    # Pre-create files simulating shards already written by the first worker.
    for shard in all_shards[:2]:
        (export_dir / shard).write_text("")

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: all_shards,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_weight_map",
        lambda _path: _weight_map_no_cross_shard(all_shards),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _p, q, _h, _k: q,
    )

    call_count = 0

    def _failing_worker(worker_args: object) -> dict[str, str]:
        from quark.torch.quantization.file2file_quantization import _QuantizeWorkerArgs

        assert isinstance(worker_args, _QuantizeWorkerArgs)
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated worker crash")
        return {os.path.basename(s): os.path.basename(s) for s in worker_args.safetensor_paths}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_shards_on_device",
        _failing_worker,
    )

    class _FailExecutor:
        def __enter__(self) -> "_FailExecutor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function: object, worker_args: object) -> concurrent.futures.Future:
            future: concurrent.futures.Future = concurrent.futures.Future()
            try:
                future.set_result(function(worker_args))  # type: ignore[operator]
            except Exception as exc:
                future.set_exception(exc)
            return future

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.concurrent.futures.ProcessPoolExecutor",
        lambda **_kwargs: _FailExecutor(),
    )

    with pytest.raises(RuntimeError, match="simulated worker crash"):
        quantize_model_per_safetensor(
            pretrained_model_path="unused",
            quant_config=_build_minimal_quant_config(),
            save_path=str(export_dir),
            device=["cpu", "cpu"],
        )

    # On failure the whole output directory is removed, leaving no partial state behind.
    assert not export_dir.exists(), f"Partial output directory was not cleaned up: {export_dir}"


def test_quantize_model_per_safetensor_uses_spawn_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ProcessPoolExecutor must be launched with a 'spawn' mp_context to avoid
    forking a process that has already initialised a CUDA context."""
    captured_mp_context: list[object] = []

    class _FakeExecutor:
        def __enter__(self) -> "_FakeExecutor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function: object, worker_args: object) -> concurrent.futures.Future:
            future: concurrent.futures.Future = concurrent.futures.Future()
            future.set_result(function(worker_args))  # type: ignore[operator]
            return future

    def fake_executor(**kwargs: object) -> _FakeExecutor:
        captured_mp_context.append(kwargs.get("mp_context"))
        return _FakeExecutor()

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.concurrent.futures.ProcessPoolExecutor",
        fake_executor,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_weight_map",
        lambda _path: _weight_map_no_cross_shard(
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        ),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_shards_on_device",
        lambda worker_args: {os.path.basename(worker_args.safetensor_paths[0]): "x"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _p, q, _h, _k: q,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_config",
        lambda *_args, **_kwargs: None,
    )

    quantize_model_per_safetensor(
        pretrained_model_path="unused",
        quant_config=_build_minimal_quant_config(),
        save_path=str(tmp_path / "export"),
        device=["cpu", "cpu"],
    )

    assert len(captured_mp_context) == 1
    assert captured_mp_context[0].get_start_method() == "spawn", (  # type: ignore[union-attr]
        f"Expected spawn mp_context, got: {captured_mp_context[0]}"
    )


def test_quantize_model_per_safetensor_worker_hf_config_is_picklable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: the ``hf_model_config`` handed to each worker must be a picklable
    plain dict, never the read-only ``MappingProxyType`` returned by
    ``_get_hf_model_config``. Under the 'spawn' start method every worker argument is
    pickled; a MappingProxyType raises 'cannot pickle mappingproxy object' and kills
    the worker. This was invisible to earlier mock tests that stubbed the worker."""
    import pickle
    from types import MappingProxyType

    captured_worker_args: list[object] = []

    # Return a real MappingProxyType, exactly like _get_hf_model_config does.
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: MappingProxyType({"torch_dtype": "float16", "text_config": {"dtype": "float16"}}),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_weight_map",
        lambda _path: _weight_map_no_cross_shard(
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        ),
    )

    def fake_worker(worker_args: object) -> dict[str, str]:
        captured_worker_args.append(worker_args)
        return {os.path.basename(s): os.path.basename(s) for s in worker_args.safetensor_paths}  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_shards_on_device",
        fake_worker,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _p, q, _h, _k: q,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_config",
        lambda *_args, **_kwargs: None,
    )

    class _FakeExecutor:
        def __enter__(self) -> "_FakeExecutor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function: object, worker_args: object) -> concurrent.futures.Future:
            future: concurrent.futures.Future = concurrent.futures.Future()
            future.set_result(function(worker_args))  # type: ignore[operator]
            return future

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.concurrent.futures.ProcessPoolExecutor",
        lambda **_kwargs: _FakeExecutor(),
    )

    quantize_model_per_safetensor(
        pretrained_model_path="unused",
        quant_config=_build_minimal_quant_config(),
        save_path=str(tmp_path / "export"),
        device=["cpu", "cpu"],
    )

    assert captured_worker_args, "worker was never invoked"
    for worker_args in captured_worker_args:
        hf_config = worker_args.hf_model_config  # type: ignore[attr-defined]
        # Must be a plain dict, not a MappingProxyType.
        assert type(hf_config) is dict, f"hf_model_config must be plain dict, got {type(hf_config)}"
        # And it must actually round-trip through pickle (the spawn requirement).
        pickle.loads(pickle.dumps(worker_args))


def test_quantize_model_per_safetensor_returns_early_when_no_safetensor_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the model directory contains no safetensors files, the function must
    log a warning and return without error instead of crashing on active_devices[0]."""
    export_called = False

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: [],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_config",
        lambda *_args, **_kwargs: None,
    )

    # Must not raise
    quantize_model_per_safetensor(
        pretrained_model_path="unused",
        quant_config=_build_minimal_quant_config(),
        save_path=str(tmp_path / "export"),
        device="cpu",
    )
    assert not export_called


def test_quantize_model_per_safetensor_single_element_devices_uses_serial_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """device=['cuda:0'] (single-element list) must use the serial shard loop,
    not ProcessPoolExecutor, so behaviour is identical to device='cuda:0'."""
    executor_launched = False

    class _FakeExecutor:
        def __enter__(self) -> "_FakeExecutor":
            nonlocal executor_launched
            executor_launched = True
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.concurrent.futures.ProcessPoolExecutor",
        lambda **_kwargs: _FakeExecutor(),
    )
    processed_shards: list[str] = []

    def fake_shard(safetensor_path: str, **_kwargs: object) -> dict[str, str]:
        processed_shards.append(os.path.basename(safetensor_path))
        return {os.path.basename(safetensor_path): os.path.basename(safetensor_path)}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_weight_map",
        lambda _path: _weight_map_no_cross_shard(
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        ),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_and_save_safetensor_shard",
        fake_shard,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _p, q, _h, _k: q,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_config",
        lambda *_args, **_kwargs: None,
    )

    quantize_model_per_safetensor(
        pretrained_model_path="unused",
        quant_config=_build_minimal_quant_config(),
        save_path=str(tmp_path / "export"),
        device=["cpu"],
    )

    assert not executor_launched, "ProcessPoolExecutor must not be used for a single-element devices list"
    assert processed_shards == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_direct_quantize_checkpoint_passes_device_list_to_quantize_model_per_safetensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """direct_quantize_checkpoint must forward a ``device`` list unchanged
    to quantize_model_per_safetensor for multi-GPU quantization."""
    import quark.torch.quantization.api as api_mod

    forwarded_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        api_mod,
        "quantize_model_per_safetensor",
        lambda *args, **kwargs: forwarded_kwargs.update(kwargs),
    )

    dummy_quantizer = type("DummyQuantizer", (), {"config": _build_minimal_quant_config()})()
    api_mod.ModelQuantizer.direct_quantize_checkpoint(
        dummy_quantizer,
        "unused-pretrained-path",
        "unused-export-path",
        False,
        device=["cuda:0", "cuda:1"],
    )

    assert forwarded_kwargs.get("device") == ["cuda:0", "cuda:1"]


def test_direct_quantize_checkpoint_multi_device_runs_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling direct_quantize_checkpoint with a ``device`` list drives the real
    quantize_model_per_safetensor multi-device path (only the per-shard worker and
    disk-scanning helpers are stubbed) and completes without error."""
    import quark.torch.quantization.api as api_mod

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_weight_map",
        lambda _path: _weight_map_no_cross_shard(
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        ),
    )

    def fake_worker(worker_args: object) -> dict[str, str]:
        from quark.torch.quantization.file2file_quantization import _QuantizeWorkerArgs

        assert isinstance(worker_args, _QuantizeWorkerArgs)
        return {os.path.basename(s): os.path.basename(s) for s in worker_args.safetensor_paths}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_shards_on_device",
        fake_worker,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _p, q, _h, _k: q,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_config",
        lambda *_args, **_kwargs: None,
    )

    class _FakeExecutor:
        def __enter__(self) -> "_FakeExecutor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function: object, worker_args: object) -> concurrent.futures.Future:
            future: concurrent.futures.Future = concurrent.futures.Future()
            future.set_result(function(worker_args))  # type: ignore[operator]
            return future

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.concurrent.futures.ProcessPoolExecutor",
        lambda **_kwargs: _FakeExecutor(),
    )

    dummy_quantizer = type("DummyQuantizer", (), {"config": _build_minimal_quant_config()})()
    api_mod.ModelQuantizer.direct_quantize_checkpoint(
        dummy_quantizer,
        "unused-pretrained-path",
        str(tmp_path / "export"),
        False,
        device=["cpu", "cpu"],
    )


def test_quantize_shards_on_device_processes_all_shards_and_merges_weight_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker entry point must process every shard assigned to it and merge each
    shard's returned weight_map into a single mapping."""
    from quark.torch.quantization.file2file_quantization import (
        _quantize_shards_on_device,
        _QuantizeWorkerArgs,
    )

    processed: list[str] = []

    def fake_shard(*, safetensor_path: str, **_kwargs: object) -> dict[str, str]:
        name = os.path.basename(safetensor_path)
        processed.append(name)
        return {f"{name}.weight": name, f"{name}.weight_scale": name}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_and_save_safetensor_shard",
        fake_shard,
    )

    worker_args = _QuantizeWorkerArgs(
        safetensor_paths=["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        export_path="unused-export-path",
        quant_config=_build_minimal_quant_config(),
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
        keep_original_model_state_tensor_names_set=set(),
        excluded_source_floating_tensor_names=set(),
        weight_converters=None,
        hf_model_config=None,
        source_weight_map=None,
        scale_inv_cache=None,
        presharded_weights=None,
        rotation_plan=None,
        total_shards=2,
        shard_global_indices=[0, 1],
    )

    merged_weight_map = _quantize_shards_on_device(worker_args)

    assert processed == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    assert merged_weight_map == {
        "model-00001-of-00002.safetensors.weight": "model-00001-of-00002.safetensors",
        "model-00001-of-00002.safetensors.weight_scale": "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors.weight": "model-00002-of-00002.safetensors",
        "model-00002-of-00002.safetensors.weight_scale": "model-00002-of-00002.safetensors",
    }


def test_quantize_model_per_safetensor_falls_back_to_single_device_on_cross_shard_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When a weight and its scale companion live in different shards, multi-device
    quantization must fall back to the single-device serial path so no worker is
    handed a shard whose companion tensor it cannot reach."""
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    # Weight in shard 1, its scale_inv companion in shard 2 -> genuine cross-shard dependency.
    cross_shard_weight_map = {
        "model.layers.0.linear.weight": shards[0],
        "model.layers.0.linear.weight_scale_inv": shards[1],
    }

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: shards,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_weight_map",
        lambda _path: cross_shard_weight_map,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _p, q, _h, _k: q,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_config",
        lambda *_args, **_kwargs: None,
    )

    serial_devices: list[str] = []

    def fake_shard(*, device: str, safetensor_path: str, **_kwargs: object) -> dict[str, str]:
        serial_devices.append(str(device))
        return {os.path.basename(safetensor_path): os.path.basename(safetensor_path)}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_and_save_safetensor_shard",
        fake_shard,
    )

    def _no_multi_worker(_worker_args: object) -> dict[str, str]:
        raise AssertionError("multi-device path must not run when a cross-shard dependency exists")

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_shards_on_device",
        _no_multi_worker,
    )

    quantize_model_per_safetensor(
        pretrained_model_path="unused",
        quant_config=_build_minimal_quant_config(),
        save_path=str(tmp_path / "export"),
        device=["cpu", "cpu"],
    )

    # Both shards processed serially on a single device -> fallback took effect.
    assert len(serial_devices) == 2


def test_quantize_model_per_safetensor_multi_device_cleanup_survives_rmtree_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If a worker fails and the subsequent directory cleanup also fails, the original
    worker error must still propagate (the cleanup OSError is swallowed, not masked)."""
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "float16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _hf_model_config: None,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: shards,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_weight_map",
        lambda _path: _weight_map_no_cross_shard(shards),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _p, q, _h, _k: q,
    )

    def _failing_worker(_worker_args: object) -> dict[str, str]:
        raise RuntimeError("simulated worker crash")

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_shards_on_device",
        _failing_worker,
    )

    def _rmtree_fails(_path: str) -> None:
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.shutil.rmtree", _rmtree_fails)

    class _FailExecutor:
        def __enter__(self) -> "_FailExecutor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function: object, worker_args: object) -> concurrent.futures.Future:
            future: concurrent.futures.Future = concurrent.futures.Future()
            try:
                future.set_result(function(worker_args))  # type: ignore[operator]
            except Exception as exc:
                future.set_exception(exc)
            return future

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.concurrent.futures.ProcessPoolExecutor",
        lambda **_kwargs: _FailExecutor(),
    )

    # The worker error propagates even though rmtree cleanup itself raised OSError.
    with pytest.raises(RuntimeError, match="simulated worker crash"):
        quantize_model_per_safetensor(
            pretrained_model_path="unused",
            quant_config=_build_minimal_quant_config(),
            save_path=str(tmp_path / "export"),
            device=["cpu", "cpu"],
        )
