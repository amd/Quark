# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download

from quark.torch.quantization.config.config import FP8E4M3PerTensorSpec, QConfig, QLayerConfig
from quark.torch.quantization.config.type import ScaleType
from quark.torch.quantization.file2file_quantization import (
    _apply_weight_converters,
    _build_exclude_aware_quant_config,
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


def test_get_model_dtype_from_hf_model_config_prefers_nested_text_dtype() -> None:
    model_dtype = _get_model_dtype_from_hf_model_config(
        {
            "text_config": {"dtype": "bfloat16", "torch_dtype": "float16"},
            "torch_dtype": "float16",
            "dtype": "float32",
        }
    )
    assert model_dtype == torch.bfloat16


def test_get_model_dtype_from_hf_model_config_falls_back_to_explicit_float32_for_auto() -> None:
    assert _get_model_dtype_from_hf_model_config({"torch_dtype": "auto"}) == torch.float32


def test_get_model_dtype_from_hf_model_config_defaults_when_config_is_missing() -> None:
    """Verify that a missing Hugging Face config falls back to the explicit file-to-file default dtype."""
    assert _get_model_dtype_from_hf_model_config(None) == torch.float32


def test_weight_dequant_fp8_allocates_output_in_requested_model_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that FP8 dequantization allocates the output buffer in the requested model dtype."""
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
                model_dtype=torch.bfloat16,
            )
    finally:
        importlib.reload(file2file_quantization)

    assert output.dtype == torch.bfloat16
    assert captured["dtype"] == torch.bfloat16
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
        output_weight_map=None,
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
        output_weight_map=None,
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
    recorded_calls: list[tuple[torch.Tensor, torch.Tensor, torch.dtype]] = []
    emptied_devices: list[str] = []

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        assert framework == "pt"
        assert isinstance(device, str)
        return _FakeSafeOpen(tensor_map=tensor_map)

    def fake_weight_dequant_fp8(
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
        *,
        model_dtype: torch.dtype,
    ) -> torch.Tensor:
        assert block_size == 128
        recorded_calls.append((weight, scale_inv, model_dtype))
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
    assert recorded_calls[0][2] == torch.float16
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
    recorded_calls: list[tuple[torch.Tensor, torch.Tensor, torch.dtype]] = []
    emptied_devices: list[str] = []

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        assert framework == "pt"
        assert isinstance(device, str)
        return _FakeSafeOpen(tensor_map=tensor_map)

    def fake_weight_dequant_fp8(
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
        *,
        model_dtype: torch.dtype,
    ) -> torch.Tensor:
        assert block_size == 128
        recorded_calls.append((weight, scale_inv, model_dtype))
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
    assert recorded_calls[0][2] == torch.float16
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
    recorded_calls: list[tuple[torch.Tensor, torch.Tensor, torch.dtype]] = []
    emptied_devices: list[str] = []

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        assert framework == "pt"
        assert isinstance(device, str)
        return _FakeSafeOpen(tensor_map=tensor_map)

    def fake_weight_dequant_fp8(
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
        *,
        model_dtype: torch.dtype,
    ) -> torch.Tensor:
        assert block_size == 128
        recorded_calls.append((weight, scale_inv, model_dtype))
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
    assert recorded_calls[0][2] == torch.float16
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

    output_weight_map: dict[str, str] = {}
    _quantize_and_save_safetensor_shard(
        safetensor_path="model-00001.safetensors",
        export_path=str(tmp_path),
        quant_config=_build_minimal_quant_config(),
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
        output_weight_map=output_weight_map,
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
        "model.layers.0.q_proj.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
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
    recorded_model_dtype: torch.dtype | None = None

    def fake_get_hf_model_config(_path: str) -> dict:
        return {"torch_dtype": "float16"}

    def fake_get_quantization_config(_hf_model_config: dict) -> None:
        return None

    def fake_get_safetensor_files(_path: str) -> list[str]:
        return ["model.safetensors-00091-of-00094.safetensors"]

    def fake_get_non_quantized(_path: str) -> set[str]:
        return {"model.layers.0.q_proj.weight"}

    def fake_collect_excluded(_path: str, _config: QConfig) -> set[str]:
        return {"model.layers.0.q_proj.weight", "model.layers.1.q_proj.weight"}

    def fake_quantize_shard(**kwargs) -> None:  # type: ignore[no-untyped-def]
        nonlocal recorded_model_dtype
        recorded_keep_set.update(kwargs["keep_original_model_state_tensor_names_set"])
        recorded_model_dtype = kwargs["model_dtype"]

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
        "quark.torch.quantization.file2file_quantization._get_non_quantized_tensor_names_from_model_safetensors",
        fake_get_non_quantized,
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
        lambda **kwargs: quantize_call_kwargs.update(kwargs),
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

    forwarded_kwargs: dict[str, object] = {}
    dummy_quantizer = type("DummyQuantizer", (), {"config": _build_minimal_quant_config()})()

    monkeypatch.setattr(api_mod, "quantize_model_per_safetensor", lambda **kwargs: forwarded_kwargs.update(kwargs))

    api_mod.ModelQuantizer.direct_quantize_checkpoint(
        dummy_quantizer,
        "unused-pretrained-path",
        "unused-export-path",
        False,
        "cpu",
        weight_converters=["converter"],
    )

    assert forwarded_kwargs["pretrained_model_path"] == "unused-pretrained-path"
    assert forwarded_kwargs["save_path"] == "unused-export-path"
    assert forwarded_kwargs["keep_excluded_layers_as_original_model_state"] is False
    assert forwarded_kwargs["device"] == "cpu"
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
        lambda **kwargs: quantize_call_kwargs.update(kwargs),
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._build_exclude_aware_quant_config",
        lambda _pretrained_model_path, input_quant_config, _hf_model_config, _keep_original: input_quant_config,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._export_config",
        lambda pretrained_model_path,
        exported_quant_config,
        save_path,
        weight_map,
        hf_model_config: export_call_kwargs.update(
            {
                "pretrained_model_path": pretrained_model_path,
                "quant_config": exported_quant_config,
                "save_path": save_path,
                "weight_map": weight_map,
                "hf_model_config": hf_model_config,
            }
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
    output_config["quantization_config"]["exclude"].sort()
    assert output_config["quantization_config"]["exclude"] == ["a.block", "a.block.*", "z.block", "z.block.*"]


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
            {"activation_scheme": "dynamic", "weight_block_size": [128, 128]},
            "The 'fmt' field is missing",
            id="missing-fmt",
        ),
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


def test_build_exclude_aware_quant_config_quark_source_copies_per_layer_e8m0_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quark-source round-trip: when the source ``layer_quant_config`` already has a
    per-layer entry for an excluded module, the entry must be copied verbatim,
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
        hf_model_config={"quantization_config": _make_quark_source_qc_with_per_layer_e8m0(module_name)},
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


def test_build_exclude_aware_quant_config_quark_source_preserves_source_exclude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linear modules already in the source's ``exclude`` list (i.e. unquantized in
    source) must stay in our output ``exclude``
    """
    module_name = "model.layers.0.q_proj"
    quark_source_qc = {
        "quant_method": "quark",
        "layer_quant_config": {},
        "global_quant_config": None,
        "exclude": [module_name],
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
    f2f._weight_dequant_fp8(weight, scale_e8m0, model_dtype=torch.float32)

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
        *,
        model_dtype: torch.dtype,
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
