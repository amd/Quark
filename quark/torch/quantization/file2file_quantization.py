#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2023 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
File-to-file quantization utilities for large language models.

This module provides memory-efficient weight-only quantization by processing safetensors
files one at a time, without loading the full model into GPU memory. This is particularly
useful for quantizing very large models that exceed available GPU memory.
"""

import copy
import fnmatch
import json
import os
import shutil
from collections.abc import Iterable
from types import MappingProxyType
from typing import Any

import torch

import quark
from quark.common.config import BaseQLayerConfig
from quark.common.profiler import ProfileStep, profile_scope
from quark.common.utils.import_utils import (
    _compressed_tensors_version,
    is_compressed_tensors_available,
    is_package_lower_or_equal,
    is_safetensors_available,
    is_triton_available,
)
from quark.common.utils.log import ScreenLogger
from quark.torch.kernel import mx as _quark_mx
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, ScaleType
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.utils.llm.config import get_quantization_config
from quark.torch.utils.numerics import to_e8m0_uint8
from quark.torch.utils.pack import create_pack_method

logger = ScreenLogger(__name__)

_FILE2FILE_DEFAULT_MODEL_DTYPE = torch.float32

if is_triton_available():
    import triton  # type: ignore[import-not-found, import-untyped]
    import triton.language as tl  # type: ignore[import-not-found, import-untyped]
if is_compressed_tensors_available():
    try:
        from compressed_tensors.compressors.base import BaseCompressor  # type: ignore[import-untyped,import-not-found]
        from compressed_tensors.quantization import (  # type: ignore[import-untyped,import-not-found]
            QuantizationArgs,
            QuantizationScheme,
        )
    except ImportError as e:
        logger.warning(
            "CompressedTensors may have some compatibility issues with other packages, disable compressed tensors model quantization support. Detailed error: %s",
            e,
        )

if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file


def _empty_cache_if_cuda(device: str | torch.device) -> None:
    """Call ``torch.cuda.empty_cache()`` only when using a CUDA device."""
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def _get_safetensor_files(model_dir_path: str) -> list[str]:
    """
    Recursively find all safetensors files in the given directory.

    :param str model_dir_path: Path to the model directory to search.

    :return: List of absolute paths to all ``.safetensors`` files found.
    :rtype: list[str]
    """
    safetensor_files = []
    for root, _, files in os.walk(model_dir_path):
        for file in files:
            if file.endswith(".safetensors"):
                file_path = os.path.join(root, file)
                safetensor_files.append(file_path)
    return safetensor_files


def _apply_weight_converters(
    tensors: dict[str, torch.Tensor],
    weight_converters: list[Any],
) -> dict[str, torch.Tensor]:
    """
    Apply ``WeightConverter`` rules to transform tensors after precision recovery.

    Each converter uses one source suffix pattern and one or more targets (e.g. ``Chunk``).
    Multi-source converters (e.g. merge) are not supported in file-to-file mode.

    :param dict[str, torch.Tensor] tensors: Input tensor dict (name -> tensor).
    :param list weight_converters: List of ``WeightConverter`` instances. Each must
        have ``source_patterns``, ``target_patterns``, and ``operations`` attributes.

    :return: New tensor dict with matched tensors replaced by converted results.
    :rtype: dict[str, torch.Tensor]
    """
    # File-to-file matches one tensor name at a time; ops like Concatenate need every
    # source present in one step, so only single-source converters are allowed here.
    if weight_converters:
        for idx, conv in enumerate(weight_converters):
            sp = getattr(conv, "source_patterns", None)
            if sp is None:
                raise ValueError("Weight converter missing source_patterns.")
            try:
                # WeightTransform normalizes a lone str to a one-element list.
                n_src = 1 if isinstance(sp, str) else len(sp)
            except TypeError as exc:
                raise ValueError("Weight converter source_patterns must be a str or sequence.") from exc
            if n_src != 1:
                msg = f"File-to-file quantization: one source pattern per converter only (index {idx}, got {n_src})."
                logger.error(msg)
                raise ValueError(msg)
    converted_tensors: dict[str, torch.Tensor] = {}
    for tensor_name, tensor in tensors.items():
        matched = False
        for converter in weight_converters:
            source_pattern = converter.source_patterns[0]
            if tensor_name.endswith(source_pattern):
                prefix = tensor_name[: -len(source_pattern)]
                result: dict[str, Any] = {source_pattern: [tensor]}
                for operation in converter.operations:
                    result = operation.convert(
                        result,
                        source_patterns=converter.source_patterns,
                        target_patterns=converter.target_patterns,
                    )
                for target_key, target_tensor in result.items():
                    full_key = prefix + target_key
                    converted_tensors[full_key] = target_tensor
                logger.info(f"Weight converter: {tensor_name} -> {[prefix + k for k in result]}")
                matched = True
                break
        if not matched:
            converted_tensors[tensor_name] = tensor
    return converted_tensors


def _peek_dtype_str(f: Any, tensor_name: str) -> str | None:
    """Return the safetensors dtype string for ``tensor_name`` without loading
    the tensor, or ``None`` if peeking fails."""
    try:
        return f.get_slice(tensor_name).get_dtype()  # type: ignore[no-any-return]
    except Exception:
        return None


def _peek_shape(f: Any, tensor_name: str) -> tuple[int, ...] | None:
    """Return the safetensors shape for ``tensor_name`` without loading the
    tensor, or ``None`` if peeking fails."""
    try:
        return tuple(f.get_slice(tensor_name).get_shape())  # type: ignore[no-untyped-call]
    except Exception:
        return None


def _is_linear_weight_tensor(tensor_name: str) -> bool:
    """
    Check if a tensor is a Linear layer weight by its name.

    A tensor is considered a Linear weight if:
    - Its parameter name is "weight" (not bias, etc.)
    - Its module name does not end with "norm" (exclude LayerNorm, RMSNorm, etc.)
    - Its module name does not contain "embed"

    :param str tensor_name: The full name of the tensor (e.g., "model.layers.0.self_attn.q_proj.weight").

    :return: True if the tensor is a Linear weight, False otherwise.
    :rtype: bool
    """
    if "." not in tensor_name:
        return False
    module_name, param_name = tensor_name.rsplit(".", 1)
    if param_name != "weight":
        return False
    if module_name.endswith("norm") or "embed" in module_name:
        return False
    return True


def _convert_linear_weight_tensor_name_to_module_name(weight_tensor_name: str) -> str:
    """
    Convert a linear weight tensor name to module name.

    :param str weight_tensor_name: Tensor name ending with ``.weight``.

    :return: Module name without ``.weight`` suffix.
    :rtype: str
    """
    return weight_tensor_name.removesuffix(".weight")


def _convert_linear_weight_tensor_names_to_module_names(weight_tensor_names: Iterable[str]) -> set[str]:
    """
    Convert linear weight tensor names to module names.

    :param Iterable[str] weight_tensor_names: Iterable of tensor names.

    :return: Set of module names for tensors identified as linear weights.
    :rtype: set[str]
    """
    module_names: set[str] = set()
    for tensor_name in weight_tensor_names:
        if _is_linear_weight_tensor(tensor_name):
            module_names.add(_convert_linear_weight_tensor_name_to_module_name(tensor_name))
    return module_names


def _get_layer_quant_config_by_tensor_name(
    tensor_name: str,
    quant_config: QConfig,
    tensor_loaded: "torch.Tensor | None" = None,
) -> BaseQLayerConfig | None:
    """
    Get quantization configuration for a specific tensor by name.

    This function determines if the given tensor is a Linear weight and returns
    the appropriate quantization configuration. Layers matching exclusion patterns
    return None, layer-specific patterns take precedence, and remaining layers
    use the global quantization config.

    :param str tensor_name: The full name of the tensor (e.g., "model.layers.0.self_attn.q_proj.weight").
    :param QConfig quant_config: Quantization configuration containing ``exclude`` patterns,
        ``layer_quant_config`` for layer-specific configs, and ``global_quant_config`` as default.
    :param tensor_loaded: Optional loaded tensor. When provided, 1-D tensors are rejected as
        non-Linear (catches RMSNorm/LayerNorm weights whose names pass the name heuristic).

    :return: The quantization configuration for the layer, or None if the layer should not be quantized.
    :rtype: BaseQLayerConfig | None
    """
    if not _is_linear_weight_tensor(tensor_name):
        return None
    # RMSNorm/LayerNorm weights are 1-D; Linear weights are always >=2-D.
    if tensor_loaded is not None and tensor_loaded.ndim < 2:
        return None

    module_name = _convert_linear_weight_tensor_name_to_module_name(tensor_name)

    if any(fnmatch.fnmatch(module_name, pattern) for pattern in quant_config.exclude):
        return None  # excluded

    # Try layer-specific config first
    for pattern, config in quant_config.layer_quant_config.items():
        if fnmatch.fnmatch(module_name, pattern):
            return config

    # Return global config as default
    return quant_config.global_quant_config


if is_triton_available():

    @triton.jit
    def _weight_dequant_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        s_ptr,
        y_ptr,
        M,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):  # type: ignore[no-untyped-def]
        """
        Triton kernel for dequantizing FP8 weights using scaling factors.

        This kernel is provided by deepseek-ai for efficient FP8 weight dequantization.
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        n = tl.cdiv(N, BLOCK_SIZE)
        offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs = offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        s = tl.load(s_ptr + pid_m * n + pid_n)
        y = x * s
        tl.store(y_ptr + offs, y, mask=mask)

    def _weight_dequant_fp8(
        x: torch.Tensor,
        s: torch.Tensor,
        block_size: int = 128,
        *,
        model_dtype: torch.dtype,
        chunk_rows: int | None = None,
    ) -> torch.Tensor:
        """
        Dequantize FP8 weight tensor using inverse scale with Triton kernel.

        :param chunk_rows: When set, treat ``x`` as the row-wise concatenation
            of ``x.size(0) // chunk_rows`` independently-quantized presharded
            TP chunks and dequantize each chunk with its own scale slice.
            Required for checkpoints whose ``chunk_rows`` is not a multiple
            of ``block_size`` (e.g. MiMo-V2.5-Pro fused QKV with
            ``chunk_rows=3392``, ``block_size=128``) — otherwise scale
            blocks straddling chunk boundaries silently apply the wrong
            chunk's scale to the next chunk's first rows. Default ``None``
            preserves the previous (full-tensor) behavior.
        """
        assert x.is_contiguous(), "Input weight tensor must be contiguous"
        assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
        # Upcast UE8M0 (float8_e8m0fnu) scales to fp32 for the Triton kernel
        # for DSV4 FP8 attention weights that ship with e8m0 sibling scales
        if hasattr(torch, "float8_e8m0fnu") and s.dtype == torch.float8_e8m0fnu:
            s = s.to(torch.float32).contiguous()
        assert s.is_contiguous(), "Scale tensor must be contiguous"
        M, N = x.size()

        if chunk_rows is not None:
            assert M % chunk_rows == 0, (
                f"x rows ({M}) must be a multiple of chunk_rows ({chunk_rows}) for presharded dequant"
            )
            n_chunks = M // chunk_rows
            chunk_scale_rows = (chunk_rows + block_size - 1) // block_size
            assert s.size(0) == n_chunks * chunk_scale_rows, (
                f"scale rows ({s.size(0)}) must equal n_chunks ({n_chunks}) * "
                f"ceil(chunk_rows / block_size) ({chunk_scale_rows}) "
                f"for presharded dequant"
            )
            y = torch.empty_like(x, dtype=model_dtype)

            def chunk_grid(meta: dict[str, int]) -> tuple[int, int]:
                return (
                    triton.cdiv(chunk_rows, meta["BLOCK_SIZE"]),
                    triton.cdiv(N, meta["BLOCK_SIZE"]),
                )

            for ci in range(n_chunks):
                cw = x[ci * chunk_rows : (ci + 1) * chunk_rows].contiguous()
                cs = s[ci * chunk_scale_rows : (ci + 1) * chunk_scale_rows].contiguous()
                cy = torch.empty_like(cw, dtype=model_dtype)
                _weight_dequant_kernel[chunk_grid](cw, cs, cy, chunk_rows, N, BLOCK_SIZE=block_size)
                y[ci * chunk_rows : (ci + 1) * chunk_rows] = cy
            return y

        y = torch.empty_like(x, dtype=model_dtype)

        def grid(meta: dict[str, int]) -> tuple[int, int]:
            return (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))

        _weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
        return y

else:

    def _weight_dequant_fp8(*_args: Any, **_kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        raise ImportError("FP8 dequantization requires Triton; please install Triton to use FP8 inputs.")


def _get_hf_model_config(pretrained_model_path: str) -> MappingProxyType[str, Any]:
    """
    Get model config from model directory.

    Returns an immutable (read-only) view of the config to prevent accidental modification.
    Use ``copy.deepcopy(config)`` if a mutable copy is needed.
    """
    with open(os.path.join(pretrained_model_path, "config.json")) as f:
        config = json.load(f)
    return MappingProxyType(config)


def _get_model_dtype_from_hf_model_config(
    hf_model_config: MappingProxyType[str, Any] | dict[str, Any] | None,
) -> torch.dtype:
    """
    Resolve the model compute dtype from Hugging Face config without mutating global state.

    Priority:

    1. ``text_config["dtype"]`` for nested text backbones.
    2. ``text_config["torch_dtype"]`` for nested text backbones.
    3. Root ``torch_dtype`` for standard Hugging Face checkpoints.
    4. Root ``dtype`` for legacy or alternate checkpoints.

    If the config does not provide a concrete dtype string, or provides ``"auto"``,
    this function falls back to explicit ``torch.float32``.

    :param MappingProxyType[str, Any] | dict[str, Any] | None hf_model_config: Hugging Face
        model config loaded from ``config.json``.

    :return: Dtype to use for local tensor allocations during file-to-file processing.
    :rtype: torch.dtype
    """
    if hf_model_config is None:
        return _FILE2FILE_DEFAULT_MODEL_DTYPE

    text_config = hf_model_config.get("text_config")
    text_config = text_config if isinstance(text_config, dict) else {}
    huggingface_config_dtype_string = (
        text_config.get("dtype")
        or text_config.get("torch_dtype")
        or hf_model_config.get("torch_dtype")
        or hf_model_config.get("dtype")
    )
    if isinstance(huggingface_config_dtype_string, str) and huggingface_config_dtype_string.lower() != "auto":
        return Dtype.from_str(huggingface_config_dtype_string).to_torch_packed_dtype()
    return _FILE2FILE_DEFAULT_MODEL_DTYPE


def _recover_compressed_tensors_weights(
    safetensor_path: str,
    quant_config: QConfig,
    hf_quant_config_dict: dict[str, Any],
    device: str | torch.device,
    keep_excluded_layers_as_original_model_state: bool,
    keep_original_model_state_tensor_names_set: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Recover weights from compressed_tensors format.

    Supports multiple compression formats determined by the ``format`` field
    in the quantization config:

    - **pack-quantized**: weights stored with ``.weight_packed``, ``.weight_scale``,
      ``.weight_shape``, ``.weight_zero_point``, and ``.weight_g_idx`` suffixes.
    - **float-quantized** / **int-quantized** / **naive-quantized**: weights stored as
      ``.weight`` (quantized dtype) with ``.weight_scale``, ``.weight_zero_point``,
      and ``.weight_g_idx`` suffixes.

    Quantized modules are identified by the presence of a ``.weight_scale`` tensor.
    All tensors not belonging to a quantized module are copied as-is.

    :param str safetensor_path: Path to the safetensor file.
    :param QConfig quant_config: Quark quantization configuration used for layer-level
        include/exclude policy during recovery.
    :param dict hf_quant_config_dict: Source model quantization configuration dictionary with
        ``quant_method: "compressed-tensors"``.
    :param str | torch.device device: Device to load tensors onto (e.g., ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
    :param bool keep_excluded_layers_as_original_model_state: If ``True``, excluded layers retain their original
        (pre-quantized) weights during recovery instead of being skipped or zeroed out.
        Defaults to ``True``.
    :param set[str] | None keep_original_model_state_tensor_names_set: Tensor names that should
        keep source model format and bypass decompression. Defaults to ``None``.

    :return: Dictionary of tensor name to tensor with decompressed weights.
    :rtype: dict[str, torch.Tensor]
    """
    if is_package_lower_or_equal("compressed-tensors", "0.14.99"):  # pragma: no cover
        raise ImportError(
            f"compressed-tensors integration requires `compressed-tensors>=0.15` but found the version compressed-tensors=={_compressed_tensors_version} in the environement. Please update compressed-tensors."
        )

    # Determine compression format from config (e.g., "pack-quantized", "float-quantized")
    compression_format = hf_quant_config_dict.get("format", "dense")

    # Build QuantizationArgs from config
    weights_config = hf_quant_config_dict["config_groups"]["group_0"]["weights"]
    quant_args = QuantizationArgs(
        num_bits=weights_config["num_bits"],
        type=weights_config["type"],
        strategy=weights_config["strategy"],
        group_size=weights_config["group_size"],
        symmetric=weights_config["symmetric"],
    )

    compressor_cls = BaseCompressor.get_value_from_registry(compression_format)
    compressor = compressor_cls()

    # Known per-module parameter suffixes for compressed-tensors formats.
    compression_param_suffixes = {
        "weight_packed",
        "weight_scale",
        "weight_zero_point",
        "weight_g_idx",
        "weight_shape",
        "weight_global_scale",
        "weight",
    }

    recovered_tensors: dict[str, torch.Tensor] = {}
    keep_input_model_format_tensor_name_set = keep_original_model_state_tensor_names_set or set()

    device_str = str(device)
    with safe_open(safetensor_path, framework="pt", device=device_str) as f:  # type: ignore[no-untyped-call]
        all_keys = set(f.keys())

        # Identify quantized modules by the presence of weight_scale.
        quantized_module_paths: set[str] = set()
        for key in all_keys:
            if key.endswith(".weight_scale"):
                module_path = key.rsplit(".weight_scale", 1)[0]
                tensor_name = f"{module_path}.weight"
                if (
                    keep_excluded_layers_as_original_model_state
                    and tensor_name in keep_input_model_format_tensor_name_set
                ):
                    continue
                quantized_module_paths.add(module_path)

        # Build per-module state dicts and collect non-quantized tensors.
        module_state_dicts: dict[str, dict[str, torch.Tensor]] = {
            module_path: {} for module_path in quantized_module_paths
        }
        quantized_tensor_keys: set[str] = set()
        for module_path in quantized_module_paths:
            for suffix in compression_param_suffixes:
                full_key = f"{module_path}.{suffix}"
                if full_key in all_keys:
                    quantized_tensor_keys.add(full_key)
                    module_state_dicts[module_path][suffix] = f.get_tensor(full_key)

        for key in all_keys:
            if key not in quantized_tensor_keys:
                recovered_tensors[key] = f.get_tensor(key)

    logger.info(f"Decompressing {len(quantized_module_paths)} compressed_tensors quantized weights...")

    # Decompress each module's state dict individually via the public
    # ``BaseCompressor.decompress`` API.
    scheme = QuantizationScheme(targets=[], weights=quant_args)
    for module_path, module_state_dict in module_state_dicts.items():
        recovered_tensors[f"{module_path}.weight"] = compressor.decompress(state_dict=module_state_dict, scheme=scheme)[
            "weight"
        ]

    logger.info(f"Decompressed {len(quantized_module_paths)} weights, total tensors: {len(recovered_tensors)}")

    return recovered_tensors


def _load_weight_map(model_dir_path: str) -> dict[str, str] | None:
    """
    Load weight_map from model.safetensors.index.json.

    :param str model_dir_path: Path to the model directory.

    :return: Dictionary mapping tensor names to safetensor filenames, or None if not found.
    :rtype: dict[str, str] | None
    """
    index_file = os.path.join(model_dir_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index_data = json.load(f)
        return index_data.get("weight_map", {})
    return None


# Safetensors dtypes recognized as packed-byte containers for FP4 nibbles
# (2 FP4 elements per byte). The standard convention is U8; deepseek-native
# DSV4 uses I8 for its expert weights.
_SAFETENSORS_FP4_PACKED_DTYPES = frozenset({"I8", "U8"})
# MXFP4 block size (OCP MX standard): 32 FP4 elements share one e8m0 scale.
_MXFP4_BLOCK_SIZE = 32


def _is_mxfp4_source_pattern(
    weight_dtype_str: str | None,
    scale_dtype_str: str | None,
    weight_shape: tuple[int, ...] | None,
    scale_shape: tuple[int, ...] | None,
) -> bool:
    """Detect a (weight, scale) pair that matches the MXFP4 wire format:
    I8/U8 packed weight + F8_E8M0 scale with 1x32 block ratio along the inner
    dim. Used to recognize DSV4 expert weights stored in the deepseek
    sibling-``.scale`` convention.

    Logical FP4 width = packed_width * 2 (two nibbles per byte). MXFP4 has one
    scale per 32 FP4 elements, so scale_width == packed_width / 16.
    """
    if weight_dtype_str not in _SAFETENSORS_FP4_PACKED_DTYPES:
        return False
    if scale_dtype_str != "F8_E8M0":
        return False
    if weight_shape is None or scale_shape is None:
        return False
    if len(weight_shape) < 2 or len(scale_shape) < 2:
        return False
    # 1×32 block: outer dim matches
    if weight_shape[-2] != scale_shape[-2]:
        return False
    # 1×32 block: inner ratio is exactly 16 (packed bytes per scale)
    if weight_shape[-1] != scale_shape[-1] * (_MXFP4_BLOCK_SIZE // 2):
        return False
    return True


def _dequantize_mxfp4_source(
    weight_packed: torch.Tensor,
    scale_e8m0: torch.Tensor,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize an MXFP4-packed weight (DSV4 convention) to ``model_dtype``.

    ``weight_packed`` is the raw byte buffer (I8 or U8) holding 2 FP4 nibbles
    per byte. ``scale_e8m0`` is the F8_E8M0 sibling scale at 1×32 block ratio.
    Output shape doubles along the inner dim (unpacked FP4 elements).
    """
    weight_packed = weight_packed.contiguous()
    if weight_packed.dtype == torch.int8:
        weight_packed = weight_packed.view(torch.uint8)
    # The HIP MX kernel does not yet accept ``torch.float8_e8m0fnu``;
    # bit-reinterpret the e8m0 scale to uint8
    scale_e8m0 = scale_e8m0.contiguous()
    if hasattr(torch, "float8_e8m0fnu") and scale_e8m0.dtype == torch.float8_e8m0fnu:
        scale_e8m0 = scale_e8m0.view(torch.uint8)
    return _quark_mx.dq_mxfp4(weight_packed, scale_e8m0, model_dtype)


def _build_cross_file_scale_inv_cache(
    model_dir_path: str,
    weight_map: dict[str, str],
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """
    Pre-load scale_inv tensors that are stored in different files from their weights.

    This function analyzes the weight_map to identify weight/scale_inv pairs that are
    stored in different safetensor files, then pre-loads those scale_inv tensors into
    a cache for efficient access during dequantization.

    :param str model_dir_path: Path to the model directory containing safetensor files.
    :param dict[str, str] weight_map: Dictionary mapping tensor names to safetensor filenames.

    :return: Dictionary mapping scale_inv tensor names to their tensors (only for cross-file cases).
    :rtype: dict[str, torch.Tensor]
    """
    # Find all scale_inv tensors that are in different files from their weights
    cross_file_scale_invs: dict[str, str] = {}  # scale_inv_name -> file_name

    for tensor_name, file_name in weight_map.items():
        if tensor_name.endswith("_scale_inv"):
            # This is a scale_inv tensor, find its corresponding weight
            weight_name = tensor_name[:-10]  # Remove "_scale_inv" suffix
            if weight_name in weight_map:
                weight_file = weight_map[weight_name]
                if weight_file != file_name:
                    # Weight and scale_inv are in different files
                    cross_file_scale_invs[tensor_name] = file_name

    if not cross_file_scale_invs:
        return {}

    logger.info(f"Found {len(cross_file_scale_invs)} scale_inv tensors in different files from their weights")

    # Group scale_inv tensors by their file for efficient loading
    file_to_scale_invs: dict[str, list[str]] = {}
    for scale_inv_name, file_name in cross_file_scale_invs.items():
        if file_name not in file_to_scale_invs:
            file_to_scale_invs[file_name] = []
        file_to_scale_invs[file_name].append(scale_inv_name)

    # Load scale_inv tensors from each file (to CPU to save GPU memory)
    scale_inv_cache: dict[str, torch.Tensor] = {}
    for file_name, scale_inv_names in file_to_scale_invs.items():
        file_path = os.path.join(model_dir_path, file_name)
        logger.info(f"Pre-loading {len(scale_inv_names)} scale_inv tensors from {file_name}")
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
                for scale_inv_name in scale_inv_names:
                    scale_inv_cache[scale_inv_name] = f.get_tensor(scale_inv_name)
        except Exception:
            logger.error(f"Failed to pre-load scale_inv tensors from {file_name}")

    logger.info(f"Pre-loaded {len(scale_inv_cache)} cross-file scale_inv tensors into cache (CPU)")
    return scale_inv_cache


def _get_non_quantized_tensor_names_from_model_safetensors(
    pretrained_model_path: str,
) -> set[str]:
    """
    Collect non-quantized (floating-point) tensor names from all safetensor files in a model directory.

    Scans each ``.safetensors`` file and returns tensor names whose dtype
    is one of ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``, excluding any
    tensor whose name contains ``"scale"``.

    :param str pretrained_model_path: Path to the pretrained model directory.

    :return: A set of non-quantized tensor names across all safetensor shards.
    :rtype: set[str]
    """

    def _get_non_quantized_tensors_from_safetensor(file_path: str) -> list[str] | None:
        """
        Retrieve the names of non-quantized (floating-point) tensors from a safetensor file.

        Iterates over all tensors in the file and returns those whose dtype is one of
        ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``, excluding any
        tensor whose name contains ``"scale"``.

        :param str file_path: Path to the ``.safetensors`` file.

        :return: A list of tensor names that are not quantized, or ``None`` if the file
            could not be read.
        """
        floating_point_dtypes = {"F16", "BF16", "F32"}
        non_quantized_tensors = []
        try:
            with safe_open(file_path, framework="pt") as f:  # type: ignore[no-untyped-call]
                all_tensor_names = f.keys()  # noqa: SIM118 - SafeOpen requires .keys()
                for tensor_name in all_tensor_names:
                    tensor_dtype = f.get_slice(tensor_name).get_dtype()
                    if "scale" not in tensor_name and tensor_dtype in floating_point_dtypes:
                        non_quantized_tensors.append(tensor_name)
        except Exception as e:
            logger.error(f"Error reading safetensor: {e}")
            return None
        return non_quantized_tensors

    safetensor_files = _get_safetensor_files(pretrained_model_path)
    non_quantized_tensor_names: set[str] = set()
    for safetensor_path in safetensor_files:
        shard_result = _get_non_quantized_tensors_from_safetensor(safetensor_path)
        if shard_result is not None:
            non_quantized_tensor_names.update(shard_result)
    return non_quantized_tensor_names


def _collect_tensor_names_matching_quark_exclude(
    pretrained_model_path: str,
    quant_config: QConfig,
) -> set[str]:
    """
    Collect tensor names matching ``quant_config.exclude`` patterns from all safetensor files.

    Scans each ``.safetensors`` file in the model directory and filters Linear
    weight tensor names that match any exclusion pattern specified in the
    quantization configuration.

    :param str pretrained_model_path: Path to the pretrained model directory.
    :param QConfig quant_config: Quantization configuration containing ``exclude`` patterns.

    :return: Tensor names that match the exclusion patterns.
    :rtype: set[str]
    """

    def _get_safetensor_excluded_module_names(
        safetensor_path: str,
        quant_config: QConfig,
    ) -> list[str] | None:
        """
        Get excluded tensor names from a single safetensor file.

        Iterates over all tensors in the file and returns tensor names of Linear
        weight tensors that match any pattern in ``quant_config.exclude``.

        :param str safetensor_path: Path to the ``.safetensors`` file.
        :param QConfig quant_config: Quantization configuration containing ``exclude`` patterns.

        :return: A list of excluded tensor names, or ``None`` if the file could not be read.
        :rtype: list[str] | None
        """
        excluded_tensor_names = []
        try:
            with safe_open(safetensor_path, framework="pt") as f:  # type: ignore[no-untyped-call]
                all_tensor_names = f.keys()  # noqa: SIM118 - SafeOpen requires .keys()
                for tensor_name in all_tensor_names:
                    if not _is_linear_weight_tensor(tensor_name):
                        continue
                    _shape = _peek_shape(f, tensor_name)
                    if _shape is not None and len(_shape) < 2:
                        continue
                    module_name = _convert_linear_weight_tensor_name_to_module_name(tensor_name)
                    if any(fnmatch.fnmatch(module_name, pattern) for pattern in quant_config.exclude):
                        excluded_tensor_names.append(tensor_name)
        except Exception as e:
            logger.error(f"Error reading safetensor: {e}")
            return None
        return excluded_tensor_names

    safetensor_files = _get_safetensor_files(pretrained_model_path)
    excluded_tensor_names: set[str] = set()
    for safetensor_path in safetensor_files:
        shard_excluded = _get_safetensor_excluded_module_names(safetensor_path, quant_config)
        if shard_excluded is not None:
            excluded_tensor_names.update(shard_excluded)
    return excluded_tensor_names


def _resolve_presharded_chunk_rows(
    weight_name: str,
    presharded_weights: dict[str, int] | None,
) -> int | None:
    """Look up the presharded ``chunk_rows`` for ``weight_name``.

    ``presharded_weights`` maps fnmatch-style globs to per-chunk row counts.
    Used to thread layout information into ``_weight_dequant_fp8`` for
    presharded TP checkpoints whose chunk row count is not a multiple of
    the FP8 block size (e.g. MiMo-V2.5-Pro fused QKV: chunk_rows=3392,
    block_size=128, 3392 % 128 = 64 != 0). Without this, ``_weight_dequant_fp8``
    silently corrupts the rows on each side of every chunk boundary.
    """
    if not presharded_weights:
        return None
    for pattern, cr in presharded_weights.items():
        if fnmatch.fnmatch(weight_name, pattern):
            return int(cr)
    return None


def _recover_fp8_weights(
    safetensor_path: str,
    quant_config: QConfig,
    hf_quant_config_dict: dict[str, Any],
    device: str | torch.device,
    keep_excluded_layers_as_original_model_state: bool,
    *,
    model_dtype: torch.dtype,
    weight_map: dict[str, str] | None = None,
    scale_inv_cache: dict[str, torch.Tensor] | None = None,
    keep_original_model_state_tensor_names_set: set[str] | None = None,
    presharded_weights: dict[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Recover weights from FP8 quantized format.

    This function handles the dequantization of weights stored in FP8 format.
    FP8 Linear weights are identified by having a corresponding ``_scale_inv`` tensor
    in the weight_map.

    When ``scale_inv_cache`` is provided, the function can retrieve ``_scale_inv`` tensors
    from the cache if they are not in the same file as the weight.

    Memory efficient: loads and processes tensors one at a time using safe_open,
    avoiding loading the entire file into GPU memory at once.

    :param str safetensor_path: Path to the safetensor file.
    :param QConfig quant_config: Quark quantization configuration used for layer-level
        include/exclude policy during recovery.
    :param dict hf_quant_config_dict: Source model quantization configuration dictionary.
        Present for API consistency with other recovery methods.
    :param dict[str, str] | None weight_map: Dictionary mapping tensor names to safetensor
        filenames. Used to determine if a weight has a corresponding scale_inv tensor.
        Defaults to ``None``.
    :param dict[str, torch.Tensor] | None scale_inv_cache: Pre-loaded cache of scale_inv
        tensors that are stored in different files from their weights. Defaults to ``None``.
    :param str | torch.device device: Device to load tensors onto (e.g., ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
    :param bool keep_excluded_layers_as_original_model_state: If ``True``, excluded layers retain their original
        (pre-quantized) weights during recovery instead of being skipped or zeroed out.
    :param set[str] | None keep_original_model_state_tensor_names_set: Tensor names that must bypass FP8
        dequantization and keep source tensors unchanged. Defaults to ``None``.
    :param torch.dtype model_dtype: Floating-point dtype to use for dequantized FP8
        output tensors. Defaults to explicit ``torch.float32``.
    :param dict[str, int] | None presharded_weights: Optional ``{glob: chunk_rows}``
        map identifying weight tensors stored in a presharded TP layout
        (i.e. row-wise concatenation of ``n_chunks`` chunks of
        ``chunk_rows`` rows each, with each chunk's ``_scale_inv``
        computed independently). Required when ``chunk_rows`` is not a
        multiple of the FP8 block size — without this hint,
        ``_weight_dequant_fp8`` corrupts the rows around every chunk
        boundary. Globs are matched via ``fnmatch.fnmatch`` against the
        full safetensor weight name (e.g.
        ``"*.self_attn.qkv_proj.weight"``).

    :return: Dictionary of tensor name to tensor with dequantized weights.
    :rtype: dict[str, torch.Tensor]
    """
    if hf_quant_config_dict.get("quant_method") != "fp8":
        raise ValueError(
            f"Expected FP8 quantization config, but got quant_method='{hf_quant_config_dict.get('quant_method')}'."
        )

    recovered_tensors: dict[str, torch.Tensor] = {}
    fp8_weight_count = 0
    fp4_weight_count = 0
    device_str = str(device)
    keep_input_model_format_tensor_name_set = keep_original_model_state_tensor_names_set or set()

    # Fall back to a hint embedded in the source model's quantization_config
    # if the caller did not pass one explicitly. This lets model authors
    # advertise presharded layouts via config.json without any API change.
    if presharded_weights is None:
        embedded = hf_quant_config_dict.get("presharded_weights")
        if isinstance(embedded, dict):
            presharded_weights = {str(k): int(v) for k, v in embedded.items()}

    with safe_open(safetensor_path, framework="pt", device=device_str) as f:  # type: ignore[no-untyped-call]
        all_keys = set(f.keys())

        # Build sibling-scale names set: DeepSeek-V4 stores scales as "{base}.scale"
        # (e.g. "layers.0.ffn.experts.0.w1.scale") rather than "{weight}_scale_inv".
        sibling_scale_keys: set[str] = set()
        for tensor_name in all_keys:
            if not tensor_name.endswith("_scale_inv") and not tensor_name.endswith(".weight"):
                parent_module_name, _, leaf_attr = tensor_name.rpartition(".")
                if leaf_attr == "scale" and f"{parent_module_name}.weight" in all_keys:
                    sibling_scale_keys.add(tensor_name)

        for weight_name in all_keys:
            # Skip scale_inv tensors, they are used during dequantization
            if weight_name.endswith("_scale_inv"):
                continue

            # Skip DeepSeek-V4 sibling scale tensors (handled during weight dequantization)
            if weight_name in sibling_scale_keys:
                continue

            scale_inv_name = f"{weight_name}_scale_inv"
            if weight_name.endswith(".weight"):
                weight_base_name = weight_name.removesuffix(".weight")
                sibling_scale_name = f"{weight_base_name}.scale"
            else:
                sibling_scale_name = None

            # Detect MXFP4 source pattern on the (weight, sibling .scale) pair:
            # I8/U8 packed weight + F8_E8M0 scale at the 1×32 block ratio.
            # DSV4 expert weights ship this way, need to avoid feeding
            # the I8 bytes to the FP8 dequant kernel
            is_mxfp4_source = (
                sibling_scale_name is not None
                and sibling_scale_name in all_keys
                and _is_mxfp4_source_pattern(
                    _peek_dtype_str(f, weight_name),
                    _peek_dtype_str(f, sibling_scale_name),
                    _peek_shape(f, weight_name),
                    _peek_shape(f, sibling_scale_name),
                )
            )

            # Keep excluded layers in original model format (e.g., FP8 weight + scale_inv)
            if keep_excluded_layers_as_original_model_state and weight_name in keep_input_model_format_tensor_name_set:
                if is_mxfp4_source:
                    # MXFP4 packed bytes + e8m0 sibling scale:
                    # Bit-reinterpret I8 to U8 so downstream MXFP4 consumers see
                    # the standard U8 container, and rename the scale to
                    # `<weight>_scale` without dequantize.
                    weight_tensor = f.get_tensor(weight_name)
                    if weight_tensor.dtype == torch.int8:
                        weight_tensor = weight_tensor.contiguous().view(torch.uint8)
                    recovered_tensors[weight_name] = weight_tensor
                    quark_scale_name = f"{weight_name}_scale"
                    recovered_tensors[quark_scale_name] = f.get_tensor(sibling_scale_name)
                    continue
                recovered_tensors[weight_name] = f.get_tensor(weight_name)
                if scale_inv_name in all_keys:
                    # DeepSeek-V3 / standard FP8: scale stored as "{weight}_scale_inv"
                    quark_scale_name = f"{weight_name}_scale"
                    recovered_tensors[quark_scale_name] = f.get_tensor(scale_inv_name)
                elif sibling_scale_name is not None and sibling_scale_name in all_keys:
                    # DeepSeek-V4: scale stored as "{base}.scale" alongside "{base}.weight"
                    quark_scale_name = f"{weight_name}_scale"
                    recovered_tensors[quark_scale_name] = f.get_tensor(sibling_scale_name)
                continue
            _w_shape = _peek_shape(f, weight_name)
            if _is_linear_weight_tensor(weight_name) and (_w_shape is None or len(_w_shape) >= 2):
                chunk_rows = _resolve_presharded_chunk_rows(weight_name, presharded_weights)
                # Pass ``chunk_rows`` only when explicitly set, so callers that
                # monkeypatch ``_weight_dequant_fp8`` with the previous
                # signature (no ``chunk_rows`` kwarg) are not broken.
                _extra_kwargs = {"chunk_rows": chunk_rows} if chunk_rows is not None else {}

                if is_mxfp4_source:
                    # Non-excluded MXFP4 source weight: dequantize to ``model_dtype``
                    assert sibling_scale_name is not None
                    weight = f.get_tensor(weight_name)
                    scale = f.get_tensor(sibling_scale_name)
                    recovered_tensors[weight_name] = _dequantize_mxfp4_source(weight, scale, model_dtype)
                    fp4_weight_count += 1
                    del weight, scale
                    _empty_cache_if_cuda(device)
                elif scale_inv_name in all_keys:
                    # Load weight and scale_inv from current file, dequantize
                    weight = f.get_tensor(weight_name)
                    scale_inv = f.get_tensor(scale_inv_name)
                    recovered_tensors[weight_name] = _weight_dequant_fp8(
                        weight,
                        scale_inv,
                        model_dtype=model_dtype,
                        **_extra_kwargs,
                    )
                    fp8_weight_count += 1
                    # Free tensors immediately after dequantization
                    del weight, scale_inv
                    _empty_cache_if_cuda(device)
                elif sibling_scale_name is not None and sibling_scale_name in all_keys:
                    # DeepSeek-V4 naming: scale stored as "{base}.scale" alongside "{base}.weight"
                    weight = f.get_tensor(weight_name)
                    scale_inv = f.get_tensor(sibling_scale_name)
                    recovered_tensors[weight_name] = _weight_dequant_fp8(
                        weight,
                        scale_inv,
                        model_dtype=model_dtype,
                    )
                    fp8_weight_count += 1
                    del weight, scale_inv
                    _empty_cache_if_cuda(device)
                elif scale_inv_cache is not None and scale_inv_name in scale_inv_cache:
                    # Load scale_inv from pre-loaded cache -> target device
                    weight = f.get_tensor(weight_name)
                    scale_inv = scale_inv_cache[scale_inv_name].to(device)
                    recovered_tensors[weight_name] = _weight_dequant_fp8(
                        weight,
                        scale_inv,
                        model_dtype=model_dtype,
                        **_extra_kwargs,
                    )
                    fp8_weight_count += 1
                    del weight, scale_inv
                    _empty_cache_if_cuda(device)
                elif weight_map is not None and scale_inv_name not in weight_map:
                    # scale_inv not in weight_map means this weight is not FP8 quantized
                    # Just copy the weight as-is (e.g., bf16/fp16 linear weight)
                    recovered_tensors[weight_name] = f.get_tensor(weight_name)
                else:
                    # scale_inv should exist (in weight_map) but not found - error
                    raise ValueError(
                        f"FP8 weight '{weight_name}' found in '{os.path.basename(safetensor_path)}' "
                        f"but its scale_inv '{scale_inv_name}' is not in the same file "
                        f"and not found in pre-loaded cache. "
                        f"Please ensure model.safetensors.index.json exists and is correct."
                    )
            else:
                # Copy non-linear tensors as-is
                recovered_tensors[weight_name] = f.get_tensor(weight_name)

    logger.info(
        f"Dequantized {fp8_weight_count} FP8 weights and {fp4_weight_count} MXFP4 weights, "
        f"total tensors: {len(recovered_tensors)}"
    )

    return recovered_tensors


def _load_safetensor_with_recover(
    safetensor_path: str,
    quant_config: QConfig,
    device: str | torch.device,
    keep_excluded_layers_as_original_model_state: bool,
    *,
    model_dtype: torch.dtype,
    hf_model_config: dict[str, Any] | None = None,
    weight_map: dict[str, str] | None = None,
    scale_inv_cache: dict[str, torch.Tensor] | None = None,
    keep_original_model_state_tensor_names_set: set[str] | None = None,
    presharded_weights: dict[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Load tensors from a safetensor file, automatically recovering quantized weights if needed.

    This function supports two quantization formats, determined by ``quant_method`` in hf_model_config:

    1. **compressed_tensors format** (``quant_method: "compressed-tensors"``):
       Weights stored with ``.weight_packed``, ``.weight_scale``, ``.weight_shape``,
       and ``.weight_zero_point`` suffixes. Requires hf_model_config with ``quantization_config``.

    2. **FP8 format** (``quant_method: "fp8"``):
       FP8 weights with ``_scale_inv`` suffix for scale tensors. Dequantized using Triton kernel.

    :param str safetensor_path: Path to the safetensor file.
    :param dict | None hf_model_config: Model config dict containing quantization configuration.
        Supports both ``hf_model_config["quantization_config"]`` (pure LLM) and
        ``hf_model_config["text_config"]["quantization_config"]`` (VLM/multimodal).
        Defaults to ``None``.
    :param dict[str, str] | None weight_map: Dictionary mapping tensor names to safetensor
        filenames. Used to determine if a weight has a corresponding scale_inv tensor.
        Defaults to ``None``.
    :param dict[str, torch.Tensor] | None scale_inv_cache: Pre-loaded cache of scale_inv
        tensors that are stored in different files from their weights. Used for FP8 format
        when weight and scale_inv are in different safetensor files. Defaults to ``None``.
    :param str | torch.device device: Device to load tensors onto (e.g., ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
    :param bool keep_excluded_layers_as_original_model_state: If ``True``, excluded layers retain their original
        (pre-quantized) weights during recovery instead of being skipped or zeroed out.
        Defaults to ``True``.
    :param set[str] | None keep_original_model_state_tensor_names_set: Tensor names that must keep original
        source data and bypass recovery logic. Defaults to ``None``.
    :param torch.dtype model_dtype: Floating-point dtype to use for recovered FP8
        tensors. Defaults to explicit ``torch.float32``.

    :return: Dictionary of tensor name to tensor with decompressed/dequantized weights.
    :rtype: dict[str, torch.Tensor]
    """
    # Try to determine quantization method from hf_model_config
    hf_quant_config_dict = get_quantization_config(hf_model_config)

    if hf_quant_config_dict is not None:
        hf_quant_method = hf_quant_config_dict.get("quant_method")

        if hf_quant_method == "fp8":
            # FP8 format: quant_method == "fp8"
            return _recover_fp8_weights(
                safetensor_path=safetensor_path,
                quant_config=quant_config,
                hf_quant_config_dict=hf_quant_config_dict,
                device=device,
                keep_excluded_layers_as_original_model_state=keep_excluded_layers_as_original_model_state,
                keep_original_model_state_tensor_names_set=keep_original_model_state_tensor_names_set,
                model_dtype=model_dtype,
                weight_map=weight_map,
                scale_inv_cache=scale_inv_cache,
                presharded_weights=presharded_weights,
            )

        elif hf_quant_method == "compressed-tensors":
            # compressed_tensors format: quant_method == "compressed-tensors"
            return _recover_compressed_tensors_weights(
                safetensor_path,
                quant_config,
                hf_quant_config_dict,
                device=device,
                keep_excluded_layers_as_original_model_state=keep_excluded_layers_as_original_model_state,
                keep_original_model_state_tensor_names_set=keep_original_model_state_tensor_names_set,
            )

    # No quantization config or unknown quant_method, load normally
    return load_file(safetensor_path, device=str(device))


def _single_stage_quantize_weight(
    tensor: torch.Tensor,
    tensor_name: str,
    layer_name: str,
    weight_config: QTensorConfig,
    quantized_tensors: dict[str, torch.Tensor],
    output_weight_map: dict[str, str] | None,
    safetensor_filename: str,
) -> None:
    """
    Perform single-stage weight quantization and store the results.

    This function quantizes a weight tensor using a single ``QTensorConfig``,
    packs the quantized weight, and stores both the packed weight and its scale
    into ``quantized_tensors``.

    :param torch.Tensor tensor: The weight tensor to quantize.
    :param str tensor_name: The full tensor name (e.g., ``"model.layers.0.self_attn.q_proj.weight"``).
    :param str layer_name: The layer name (e.g., ``"model.layers.0.self_attn.q_proj"``).
    :param QTensorConfig weight_config: Quantization configuration for the weight.
    :param dict[str, torch.Tensor] quantized_tensors: Output dictionary to store quantized tensors (modified in-place).
    :param dict[str, str] | None output_weight_map: Output weight map to update (modified in-place). Can be ``None``.
    :param str safetensor_filename: The safetensor filename for weight map entries.

    :return: None
    """
    weight_config = copy.deepcopy(weight_config)
    if weight_config.scale_format != "e8m0" and weight_config.scale_type == ScaleType.float:
        weight_config.scale_type = ScaleType.float32

    dtype = weight_config.dtype
    quantizer = FakeQuantizeBase.get_fake_quantize(weight_config)
    quantizer.enable_observer()
    quantizer.disable_fake_quant()
    _ = quantizer(tensor)  # Initialize quantizer parameters
    qscheme = weight_config.qscheme
    round_method = weight_config.round_method
    axis = weight_config.ch_axis
    group_size = weight_config.group_size
    assert qscheme is not None, f"qscheme for {layer_name} must not be None"
    assert round_method is not None, f"round_method for {layer_name} must not be None"
    quantized_weight = quark.torch.kernel.scaled_real_quantize(
        dtype.value,
        tensor,
        quantizer.scale,
        quantizer.zero_point,
        axis,
        group_size,
        quantizer.quant_min,
        quantizer.quant_max,
        round_method.value,
        qscheme.value,
    )
    pack_method = create_pack_method(qscheme=qscheme.value, dtype=dtype.value)
    quantized_tensors[tensor_name] = pack_method.pack(quantized_weight, False)

    if weight_config.scale_format == "e8m0":
        # Convert scale to e8m0 format (MXFP4 standard).
        quantized_tensors[tensor_name + "_scale"] = to_e8m0_uint8(quantizer.scale).contiguous()
    else:
        quantized_tensors[tensor_name + "_scale"] = quantizer.scale.contiguous()

    if output_weight_map is not None:
        output_weight_map[tensor_name + "_scale"] = safetensor_filename


def _scale_quantize_weight(
    tensor: torch.Tensor,
    tensor_name: str,
    layer_name: str,
    weight_config_stages: list[QTensorConfig],
    quantized_tensors: dict[str, torch.Tensor],
    output_weight_map: dict[str, str] | None,
    safetensor_filename: str,
    device: str | torch.device,
) -> None:
    """
    Quantize a weight with a two-stage *scale-quant* spec and store the packed result.

    Unlike progressive (cascaded) quantization, the second stage of a scale-quant
    spec does not re-quantize the weight; it quantizes the first stage's per-group
    scale. This is the NVFP4 wire format used by NVIDIA Model-Optimizer and
    ``amd/Kimi-K2.6-NVFP4``:

    - first stage: FP4 per-group weight quantization (group_size 16, fp32 scale).
    - second stage: that per-group scale is itself quantized to FP8-E4M3, tied
      together by a single per-tensor global scale (scale-of-scale).

    To stay byte-identical with the regular (non file-to-file) export and avoid
    re-deriving the two-level scaling math, this reuses the same export machinery
    the regular path uses: build a :class:`SequentialRealQuantizer` from an
    observed :class:`SequentialQuantize` via :func:`get_real_quantizer`, then call
    ``to_real_quantize_params`` (packs the FP4 weight using the f32 block scale)
    followed by ``maybe_convert_and_transpose_scale`` (quantizes the block scale to
    FP8). This is the same order :meth:`ExportBuilder._real_quantize` uses.

    The output tensors stored are:

    - ``tensor_name``: U8-packed FP4 nibbles (2 codes per byte, inner dim halved).
    - ``tensor_name + "_scale"``: F8_E4M3 per-group block scale (the scale itself,
      quantized to FP8).
    - ``tensor_name + "_scale_2"``: F32 per-tensor global scale (scale-of-scale).

    :param torch.Tensor tensor: The weight tensor to quantize.
    :param str tensor_name: The full tensor name (e.g., ``"layers.0.ffn.experts.0.w1.weight"``).
    :param str layer_name: The layer name (e.g., ``"layers.0.ffn.experts.0.w1"``).
    :param list[QTensorConfig] weight_config_stages: List of exactly 2 ``QTensorConfig`` objects
        from ``ScaleQuantSpec.to_quantization_spec()`` (second stage has ``is_scale_quant=True``).
    :param dict[str, torch.Tensor] quantized_tensors: Output dictionary to store quantized tensors (modified in-place).
    :param dict[str, str] | None output_weight_map: Output weight map to update (modified in-place). Can be ``None``.
    :param str safetensor_filename: The safetensor filename for weight map entries.
    :param str | torch.device device: Device used for tensor operations, needed for cache cleanup.

    :return: None
    """
    # Imported lazily to avoid a circular import (the export module imports from
    # quark.torch.quantization at module load time).
    from quark.torch.export.nn.modules.realquantizer import get_real_quantizer

    assert len(weight_config_stages) == 2, (
        f"Scale-quant quantization for {layer_name} requires exactly 2 stages, got {len(weight_config_stages)}"
    )

    # Observe the weight with Quark's SequentialQuantize so all scales (including
    # the global combined-division scale set up in its __init__) are computed
    # exactly as the regular NVFP4 path does.
    sequential_quantizer = FakeQuantizeBase.get_fake_quantize(weight_config_stages, device=device)
    sequential_quantizer = sequential_quantizer.to(device)
    sequential_quantizer.enable_observer()
    sequential_quantizer.disable_fake_quant()
    with torch.no_grad():
        _ = sequential_quantizer(tensor.to(device))

    # Reuse the regular export path: build a SequentialRealQuantizer from the
    # observed quantizer, then pack the weight and the block scale in the same
    # order ExportBuilder._real_quantize uses.
    real_quantizer = get_real_quantizer(
        qspec=weight_config_stages,
        quantizer=sequential_quantizer,
        reorder=True,
        real_quantized=True,
        float_dtype=tensor.dtype,
        device=device,
    )

    # The global per-tensor scale (scale-of-scale) is the second stage's scale;
    # capture it as F32 before maybe_convert_and_transpose_scale mutates state.
    weight_scale_2 = real_quantizer[1].scale.contiguous()

    # to_real_quantize_params packs the FP4 weight using the still-f32 block scale;
    # maybe_convert_and_transpose_scale then quantizes the block scale to FP8.
    packed_weight = real_quantizer.to_real_quantize_params(tensor.to(device))
    real_quantizer.maybe_convert_and_transpose_scale()
    weight_scale_fp8 = real_quantizer[0].scale

    quantized_tensors[tensor_name] = packed_weight.contiguous()
    quantized_tensors[tensor_name + "_scale"] = weight_scale_fp8.contiguous()
    quantized_tensors[tensor_name + "_scale_2"] = weight_scale_2
    if output_weight_map is not None:
        output_weight_map[tensor_name + "_scale"] = safetensor_filename
        output_weight_map[tensor_name + "_scale_2"] = safetensor_filename

    del packed_weight, weight_scale_fp8, real_quantizer, sequential_quantizer
    _empty_cache_if_cuda(device)


def _progressive_quantize_weight(
    tensor: torch.Tensor,
    tensor_name: str,
    layer_name: str,
    weight_config_stages: list[QTensorConfig],
    quantized_tensors: dict[str, torch.Tensor],
    output_weight_map: dict[str, str] | None,
    safetensor_filename: str,
    device: str | torch.device,
) -> None:
    """
    Perform progressive (two-step) weight quantization and store the results.

    This function implements progressive quantization where the weight tensor is
    quantized in two stages. For example:

    1. First stage: quantize the original weight to FP8 using per-tensor quantization.
    2. Second stage: quantize the FP8 result to INT4 using per-channel quantization.

    The output tensors stored are:

    - ``tensor_name``: packed weight from the second stage.
    - ``tensor_name + "_scale"``: scale from the first stage (e.g., FP8 scale).
    - ``tensor_name + "_scale_2"``: scale from the second stage (e.g., INT4 scale).

    :param torch.Tensor tensor: The weight tensor to quantize.
    :param str tensor_name: The full tensor name (e.g., ``"model.layers.0.self_attn.q_proj.weight"``).
    :param str layer_name: The layer name (e.g., ``"model.layers.0.self_attn.q_proj"``).
    :param list[QTensorConfig] weight_config_stages: List of exactly 2 ``QTensorConfig`` objects
        from ``ProgressiveSpec.to_quantization_spec()``.
    :param dict[str, torch.Tensor] quantized_tensors: Output dictionary to store quantized tensors (modified in-place).
    :param dict[str, str] | None output_weight_map: Output weight map to update (modified in-place). Can be ``None``.
    :param str safetensor_filename: The safetensor filename for weight map entries.
    :param str | torch.device device: Device used for tensor operations, needed for cache cleanup.

    :return: None
    """
    assert len(weight_config_stages) == 2, (
        f"Progressive quantization for {layer_name} requires exactly 2 stages, got {len(weight_config_stages)}"
    )
    first_stage_config = copy.deepcopy(weight_config_stages[0])
    second_stage_config = copy.deepcopy(weight_config_stages[1])
    assert isinstance(first_stage_config, QTensorConfig), (
        f"First stage weight config for {layer_name} must be QTensorConfig"
    )
    assert isinstance(second_stage_config, QTensorConfig), (
        f"Second stage weight config for {layer_name} must be QTensorConfig"
    )

    for stage_config in (first_stage_config, second_stage_config):
        if stage_config.scale_format != "e8m0" and stage_config.scale_type == ScaleType.float:
            stage_config.scale_type = ScaleType.float32

    # Stage 1: Quantize the original weight (e.g., FP16/BF16 -> FP8)
    first_stage_quantizer = FakeQuantizeBase.get_fake_quantize(first_stage_config)
    first_stage_quantizer.enable_observer()
    first_stage_quantizer.disable_fake_quant()
    _ = first_stage_quantizer(tensor)  # Initialize first stage quantizer parameters

    assert first_stage_config.qscheme is not None, f"First stage qscheme for {layer_name} must not be None"
    assert first_stage_config.round_method is not None, f"First stage round_method for {layer_name} must not be None"

    first_stage_quantized_weight = quark.torch.kernel.scaled_real_quantize(
        first_stage_config.dtype.value,
        tensor,
        first_stage_quantizer.scale,
        first_stage_quantizer.zero_point,
        first_stage_config.ch_axis,
        first_stage_config.group_size,
        first_stage_quantizer.quant_min,
        first_stage_quantizer.quant_max,
        first_stage_config.round_method.value,
        first_stage_config.qscheme.value,
    )
    first_stage_scale = first_stage_quantizer.scale.contiguous()

    # Stage 2: Quantize the first stage output (e.g., FP8 -> INT4)
    # Convert to float for the second stage quantization
    first_stage_weight_as_float = first_stage_quantized_weight.float()
    second_stage_quantizer = FakeQuantizeBase.get_fake_quantize(second_stage_config)
    second_stage_quantizer.enable_observer()
    second_stage_quantizer.disable_fake_quant()
    _ = second_stage_quantizer(first_stage_weight_as_float)  # Initialize second stage quantizer parameters

    assert second_stage_config.qscheme is not None, f"Second stage qscheme for {layer_name} must not be None"
    assert second_stage_config.round_method is not None, f"Second stage round_method for {layer_name} must not be None"

    second_stage_quantized_weight = quark.torch.kernel.scaled_real_quantize(
        second_stage_config.dtype.value,
        first_stage_weight_as_float,
        second_stage_quantizer.scale,
        second_stage_quantizer.zero_point,
        second_stage_config.ch_axis,
        second_stage_config.group_size,
        second_stage_quantizer.quant_min,
        second_stage_quantizer.quant_max,
        second_stage_config.round_method.value,
        second_stage_config.qscheme.value,
    )

    # Pack the final quantized weight using second stage settings
    pack_method = create_pack_method(
        qscheme=second_stage_config.qscheme.value,
        dtype=second_stage_config.dtype.value,
    )
    quantized_tensors[tensor_name] = pack_method.pack(second_stage_quantized_weight, True)
    # Store first stage scale (e.g., FP8 scale)
    quantized_tensors[tensor_name + "_scale"] = first_stage_scale
    if output_weight_map is not None:
        output_weight_map[tensor_name + "_scale"] = safetensor_filename

    # Store second stage scale (e.g., INT4 scale)
    quantized_tensors[tensor_name + "_scale_2"] = second_stage_quantizer.scale.contiguous()
    if output_weight_map is not None:
        output_weight_map[tensor_name + "_scale_2"] = safetensor_filename

    # Free intermediate tensors
    del first_stage_quantized_weight, first_stage_weight_as_float, second_stage_quantized_weight
    _empty_cache_if_cuda(device)


def _quantize_and_save_safetensor_shard(
    safetensor_path: str,
    export_path: str,
    quant_config: QConfig,
    device: str | torch.device,
    *,
    keep_excluded_layers_as_original_model_state: bool,
    model_dtype: torch.dtype,
    keep_original_model_state_tensor_names_set: set[str] | None = None,
    weight_converters: list[Any] | None = None,
    output_weight_map: dict[str, str] | None = None,
    input_scale_dict: dict[str, torch.Tensor] | None = None,
    hf_model_config: dict[str, Any] | None = None,
    source_weight_map: dict[str, str] | None = None,
    scale_inv_cache: dict[str, torch.Tensor] | None = None,
    presharded_weights: dict[str, int] | None = None,
) -> None:
    """
    Quantize weights in a single safetensors shard file and save the result.

    This function loads tensors from the input safetensors file, quantizes weight tensors
    according to the quantization configuration, packs them, and exports to the output directory.
    Non-weight tensors (e.g., biases, layernorm parameters) are copied as-is.

    :param str safetensor_path: Path to the input safetensors file.
    :param str export_path: Directory path to export the quantized safetensors file.
    :param QConfig quant_config: Quantization configuration containing ``exclude`` patterns,
        ``layer_quant_config`` for layer-specific configs, and ``global_quant_config`` as default.
    :param bool keep_excluded_layers_as_original_model_state: If ``True``, excluded layers that
        must stay in the input checkpoint format bypass recovery and export rewriting.
    :param dict[str, torch.Tensor] | None input_scale_dict: Optional dictionary mapping
        layer names to their input scales. If provided and a matching key is found,
        the input scale will be exported alongside the weight. Defaults to ``None``.
    :param dict[str, Any] | None hf_model_config: Optional HuggingFace model config dictionary
        containing model metadata (e.g., architecture, hidden_size, num_layers).
        This config is stored for potential future use such as model-specific
        optimizations or metadata export. Defaults to ``None``.
    :param dict[str, str] | None source_weight_map: Dictionary mapping tensor names to safetensor
        filenames. Used to determine if a weight has a corresponding scale_inv tensor.
        Defaults to ``None``.
    :param dict[str, torch.Tensor] | None scale_inv_cache: Pre-loaded cache of scale_inv
        tensors that are stored in different files from their weights. Defaults to ``None``.
    :param torch.dtype model_dtype: Floating-point dtype to use for recovered FP8
        tensors in this shard. Defaults to explicit ``torch.float32``.
    :param set[str] | None keep_original_model_state_tensor_names_set: Tensor names that must keep original
        source data and bypass recovery logic. Defaults to ``None``.
    :param list | None weight_converters: Optional list of ``WeightConverter`` instances
        applied after recovery and before quantization. File-to-file mode supports
        only single-source converters.
    This function writes the quantized shard to disk. If ``output_weight_map`` is provided,
    it will be updated in-place.
    """
    safetensor_filename = os.path.basename(safetensor_path)
    logger.info(f"Loading {safetensor_filename}...")
    tensors = _load_safetensor_with_recover(
        safetensor_path=safetensor_path,
        quant_config=quant_config,
        device=device,
        keep_excluded_layers_as_original_model_state=keep_excluded_layers_as_original_model_state,
        hf_model_config=hf_model_config,
        weight_map=source_weight_map,
        scale_inv_cache=scale_inv_cache,
        keep_original_model_state_tensor_names_set=keep_original_model_state_tensor_names_set,
        model_dtype=model_dtype,
        presharded_weights=presharded_weights,
    )

    if weight_converters:
        tensors = _apply_weight_converters(tensors, weight_converters)

    quantized_tensors: dict[str, torch.Tensor] = {}

    for tensor_name, tensor in tensors.items():
        if tensor_name.endswith((".weight_packed", ".weight_shape")):
            continue
        if tensor_name.endswith((".weight_scale",)):
            # Force convert to quark format, no need to quantize
            quantized_tensors[tensor_name] = tensor
            weight_tensor_name = tensor_name[: -len("_scale")]
            quantized_tensors[weight_tensor_name] = tensors[weight_tensor_name]

        if output_weight_map is not None:
            output_weight_map[tensor_name] = safetensor_filename

        layer_name = ".".join(tensor_name.split(".")[:-1])
        layer_config = _get_layer_quant_config_by_tensor_name(
            tensor_name=tensor_name,
            quant_config=quant_config,
            tensor_loaded=tensor,
        )

        if layer_config is not None:
            weight_config = layer_config.weight

            if isinstance(weight_config, list):
                # Two-stage weight config (a list[QTensorConfig]). Two distinct
                # semantics share this shape, distinguished by is_scale_quant on
                # the second stage:
                #   - scale-quant (NVFP4): the second stage quantizes the first
                #     stage's per-group scale, not the weight. Emits the packed
                #     NVFP4 wire format via _scale_quantize_weight.
                #   - progressive (e.g. FP8 -> INT4): the weight is quantized in
                #     two cascaded stages via _progressive_quantize_weight.
                if len(weight_config) == 2 and getattr(weight_config[1], "is_scale_quant", False):
                    _scale_quantize_weight(
                        tensor=tensor,
                        tensor_name=tensor_name,
                        layer_name=layer_name,
                        weight_config_stages=weight_config,
                        quantized_tensors=quantized_tensors,
                        output_weight_map=output_weight_map,
                        safetensor_filename=safetensor_filename,
                        device=device,
                    )
                else:
                    _progressive_quantize_weight(
                        tensor=tensor,
                        tensor_name=tensor_name,
                        layer_name=layer_name,
                        weight_config_stages=weight_config,
                        quantized_tensors=quantized_tensors,
                        output_weight_map=output_weight_map,
                        safetensor_filename=safetensor_filename,
                        device=device,
                    )
            else:
                # Single-stage quantization
                assert isinstance(weight_config, QTensorConfig), f"weight config for {layer_name} must be QTensorConfig"
                _single_stage_quantize_weight(
                    tensor=tensor,
                    tensor_name=tensor_name,
                    layer_name=layer_name,
                    weight_config=weight_config,
                    quantized_tensors=quantized_tensors,
                    output_weight_map=output_weight_map,
                    safetensor_filename=safetensor_filename,
                )

            # Export input scale if provided (common for both progressive and single-stage)
            if input_scale_dict is not None:
                if layer_name in input_scale_dict:
                    input_scale = input_scale_dict[layer_name]
                    input_scale_key = layer_name + ".input_scale"
                    quantized_tensors[input_scale_key] = input_scale.contiguous()
                    if output_weight_map is not None:
                        output_weight_map[input_scale_key] = safetensor_filename
                else:
                    logger.warning(f"Input scale not found for layer: {layer_name}")
        else:
            quantized_tensors[tensor_name] = tensor

    # Free device memory before saving
    del tensors
    _empty_cache_if_cuda(device)

    output_path = os.path.join(export_path, safetensor_filename)
    save_file(quantized_tensors, output_path)
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Saved {safetensor_filename} ({output_size_mb:.1f}MB)")
    return None


def _resolve_legacy_positional_device_arg(
    legacy_device_args: tuple[str | torch.device, ...],
    device: str | torch.device | None,
    function_name: str,
) -> str | torch.device:
    """
    Preserve older positional ``device`` callers after adding keyword-only options.

    :param tuple[str | torch.device, ...] legacy_device_args: Extra positional args after
        ``keep_excluded_layers_as_original_model_state``.
    :param str | torch.device | None device: Keyword ``device`` value.
    :param str function_name: Name used in error messages.

    :return: Resolved device value.
    :rtype: str | torch.device
    """
    if not legacy_device_args:
        return "cuda" if device is None else device
    if len(legacy_device_args) > 1:
        raise TypeError(
            f"{function_name} accepts at most one positional argument after "
            "'keep_excluded_layers_as_original_model_state'; pass weight_converters and device by keyword."
        )
    if device is not None:
        raise TypeError(f"{function_name} got device specified both positionally and by keyword.")
    return legacy_device_args[0]


@profile_scope(ProfileStep.FILE_TO_FILE_QUANTIZATION)
def quantize_model_per_safetensor(
    pretrained_model_path: str,
    quant_config: QConfig,
    save_path: str,
    keep_excluded_layers_as_original_model_state: bool = False,
    *legacy_device_args: str | torch.device,
    weight_converters: list[Any] | None = None,
    device: str | torch.device | None = None,
    presharded_weights: dict[str, int] | None = None,
) -> None:
    """
    Quantize model weights by processing each safetensors file independently.

    This function enables memory-efficient quantization of large models by processing
    one safetensors shard at a time, rather than loading the entire model into memory.
    The quantized shards and all configuration files (``config.json``,
    ``model.safetensors.index.json``, tokenizer files, etc.) are written to ``save_path``.

    For FP8 models where weight and scale_inv tensors may be stored in different files,
    this function pre-loads the cross-file scale_inv tensors into a cache before processing.

    :param str pretrained_model_path: Path to the pretrained model directory
        containing safetensors files.
    :param QConfig quant_config: Quantization configuration specifying dtype, exclusions,
        and per-layer settings.
    :param str save_path: Directory path to save the quantized safetensors files.
    :param bool keep_excluded_layers_as_original_model_state: If ``True``, tensors already
        quantized in the source checkpoint but excluded from Quark quantization retain
        their original model-state format in the export.
    :param list | None weight_converters: Optional list of ``WeightConverter`` instances
        to transform tensors after precision recovery (e.g., split fused ``gate_up_proj``
        into ``gate_proj`` + ``up_proj``). Applied per-shard before quantization.
        File-to-file mode supports only single-source converters.
        Defaults to ``None``.
    :param str | torch.device device: Device for tensor operations (e.g., ``"cuda"``,
        ``"cuda:0"``, ``"cpu"``). Defaults to ``"cuda"``. Legacy positional callers may
        still pass ``device`` after ``keep_excluded_layers_as_original_model_state``.
    """
    device = _resolve_legacy_positional_device_arg(
        legacy_device_args,
        device,
        "quantize_model_per_safetensor",
    )

    # Pre-load cross-file scale_inv tensors into cache (only for FP8 models)
    source_weight_map: dict[str, str] | None = None
    scale_inv_cache: dict[str, torch.Tensor] | None = None
    hf_model_config = _get_hf_model_config(pretrained_model_path)
    model_dtype = _get_model_dtype_from_hf_model_config(hf_model_config)

    quant_config_dict = get_quantization_config(hf_model_config)
    is_fp8_model = quant_config_dict is not None and quant_config_dict.get("quant_method") == "fp8"

    if is_fp8_model and not str(device).startswith("cuda"):
        logger.error(
            "FP8 model dequantization requires a CUDA device (Triton kernel), "
            f"but got device='{device}'. Please use device='cuda' or 'cuda:<id>'."
        )
        return

    if is_fp8_model:
        source_weight_map = _load_weight_map(pretrained_model_path)
        if source_weight_map is not None:
            logger.info(f"Loaded weight_map with {len(source_weight_map)} entries from model.safetensors.index.json")
            scale_inv_cache = _build_cross_file_scale_inv_cache(pretrained_model_path, source_weight_map, device=device)

    output_weight_map: dict[str, str] = {}
    safetensor_files = _get_safetensor_files(pretrained_model_path)
    logger.info(f"Found {len(safetensor_files)} safetensors files to process")
    os.makedirs(save_path, exist_ok=True)

    keep_original_model_state_tensor_names_set: set[str] = set()
    if keep_excluded_layers_as_original_model_state:
        # Find tensors that are already quantized in the HF model but excluded from Quark quantization,
        # these must be kept as-is from the original model state.
        non_quantized_tensor_names = _get_non_quantized_tensor_names_from_model_safetensors(pretrained_model_path)
        excluded_tensor_names = _collect_tensor_names_matching_quark_exclude(pretrained_model_path, quant_config)
        for excluded_tensor_name in excluded_tensor_names:
            if excluded_tensor_name not in non_quantized_tensor_names:
                keep_original_model_state_tensor_names_set.add(excluded_tensor_name)

    for index, safetensor_path in enumerate(safetensor_files):
        logger.info(f"Processing {index + 1}/{len(safetensor_files)}: {os.path.basename(safetensor_path)}")
        _quantize_and_save_safetensor_shard(
            safetensor_path=safetensor_path,
            export_path=save_path,
            quant_config=quant_config,
            device=device,
            keep_excluded_layers_as_original_model_state=keep_excluded_layers_as_original_model_state,
            model_dtype=model_dtype,
            keep_original_model_state_tensor_names_set=keep_original_model_state_tensor_names_set,
            weight_converters=weight_converters,
            output_weight_map=output_weight_map,
            hf_model_config=hf_model_config,
            source_weight_map=source_weight_map,
            scale_inv_cache=scale_inv_cache,
            presharded_weights=presharded_weights,
        )

    # Free the cache after processing
    if scale_inv_cache is not None:
        del scale_inv_cache
        _empty_cache_if_cuda(device)

    quant_config = _build_exclude_aware_quant_config(
        pretrained_model_path, quant_config, hf_model_config, keep_excluded_layers_as_original_model_state
    )

    _export_config(
        pretrained_model_path,
        quant_config,
        save_path,
        output_weight_map,
        hf_model_config,
    )


def _copy_json_and_py_files(src_dir: str, dst_dir: str) -> None:
    """
    Copy all non-safetensors files and directories from source to destination directory.

    This function copies configuration files (e.g., ``config.json``, ``tokenizer.json``),
    custom Python files (e.g., ``modeling_*.py``), and subdirectories required for model loading.
    It excludes ``.safetensors`` files and README markdown files.

    :param str src_dir: Source directory containing files to copy.
    :param str dst_dir: Destination directory to copy files to. Created if not exists.

    :return: None
    """
    os.makedirs(dst_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        # Handle directories
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            continue
        # Skip .safetensors files
        if filename.endswith(".safetensors"):
            continue
        # Skip readme markdown files
        if filename.lower().endswith(".md") and "readme" in filename.lower():
            continue
        shutil.copy2(src_path, dst_path)


def _export_safetensors_index(save_path: str, weight_map: dict[str, str]) -> None:
    """
    Export the safetensors index file (``model.safetensors.index.json``).

    This function creates the HuggingFace-compatible index file that maps tensor names
    to their corresponding safetensors shard files, along with metadata about total size.

    :param str save_path: Directory path containing the safetensors files.
    :param dict[str, str] weight_map: Dictionary mapping tensor names to safetensors filenames.

    :return: None
    """
    total_size = 0
    for safetensor_file in set(weight_map.values()):
        file_path = os.path.join(save_path, safetensor_file)
        if os.path.exists(file_path):
            total_size += os.path.getsize(file_path)

    index_data = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=4)


def _remove_quantization_config(config: dict[str, Any]) -> None:
    """
    Recursively remove all quantization_config entries regardless of format.

    This removes any key named "quantization_config" found at any level
    of the config dictionary, supporting compressed-tensors, GPTQ, AWQ,
    and any other quantization formats.

    :param dict config: Model config dictionary to modify in-place.
    :return: None
    """
    keys_to_delete = []
    for key, value in config.items():
        if key == "quantization_config":
            keys_to_delete.append(key)
        elif isinstance(value, dict):
            _remove_quantization_config(value)

    for key in keys_to_delete:
        del config[key]


def _export_quant_config(
    hf_model_config: dict[str, Any],
    quant_config: QConfig,
    save_path: str,
) -> None:
    """
    Export the model configuration with quantization settings.

    This function takes the HuggingFace model config, adds quantization configuration
    and export settings, then saves the updated config to the output directory.

    :param dict[str, Any] hf_model_config: HuggingFace model config dictionary (passed from caller).
    :param QConfig quant_config: Quantization configuration to embed in the model config.
    :param str save_path: Directory path to save the updated ``config.json``.

    :return: None
    """
    hf_model_config = copy.deepcopy(dict(hf_model_config))
    _remove_quantization_config(hf_model_config)

    hf_model_config["quantization_config"] = quant_config.to_dict()
    hf_model_config["quantization_config"]["export"] = {
        "kv_cache_group": [],
        "min_kv_scale": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    }

    # Resolve exclude to exact module names for downstream platforms
    # that may not support regex/wildcard patterns.
    hf_model_config["quantization_config"]["exclude"] = sorted(quant_config.exclude)

    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_model_config, f, ensure_ascii=False, indent=4)


def _build_exclude_aware_from_quark_source(
    quant_config: QConfig,
    source_quantization_config: dict[str, Any],
    excluded_module_names: set[str],
    pretrained_model_path: str,
) -> QConfig:
    """Refine ``quant_config`` from a Quark-exported source.

    For each excluded module, look up its description in the source's
    ``layer_quant_config`` / ``global_quant_config`` / ``exclude`` and propagate
    verbatim.

    :param QConfig quant_config: The Quark quant config being refined (mutated in place).
    :param dict[str, Any] source_quantization_config: The source ``quantization_config``
        dict; must have ``quant_method == "quark"``.
    :param set[str] excluded_module_names: Resolved-exact module names that this Quark
        run wants excluded from re-quantization.
    :param str pretrained_model_path: Path to the source checkpoint directory.
    :return: The refined ``quant_config`` with ``exclude`` and ``layer_quant_config``
        updated for each excluded module.
    :rtype: QConfig
    """
    source_layer_qc = source_quantization_config.get("layer_quant_config") or {}
    source_global_qc = source_quantization_config.get("global_quant_config")
    source_exclude = set(source_quantization_config.get("exclude") or [])

    non_quantized_module_names = _convert_linear_weight_tensor_names_to_module_names(
        _get_non_quantized_tensor_names_from_model_safetensors(pretrained_model_path)
    )

    retained_exclude_names: list[str] = []
    layer_quant_config: dict[str, QLayerConfig] = {}

    for module_name in excluded_module_names:
        if module_name in source_exclude or module_name in non_quantized_module_names:
            # Source already kept this module unquantized
            retained_exclude_names.append(module_name)
        elif module_name in source_layer_qc:
            # Source has an explicit per-layer entry; copy verbatim.
            layer_quant_config[module_name] = QLayerConfig.from_dict(source_layer_qc[module_name])
        elif source_global_qc is not None:
            layer_quant_config[module_name] = QLayerConfig.from_dict(source_global_qc)
        else:
            # No description available, fall back to plain exclude (loader sees raw bytes).
            retained_exclude_names.append(module_name)

    quant_config.exclude = sorted(retained_exclude_names)
    if quant_config.layer_quant_config is None:
        quant_config.layer_quant_config = {}
    quant_config.layer_quant_config.update(layer_quant_config)
    return quant_config


def _build_exclude_aware_quant_config(
    pretrained_model_path: str,
    quant_config: QConfig,
    hf_model_config: dict[str, Any],
    keep_excluded_layers_as_original_model_state: bool,
) -> QConfig:
    """
    Refine ``quant_config`` by resolving exclude patterns to exact module names and
    optionally splitting already-quantized excluded layers into ``layer_quant_config``.

    For layers listed in ``quant_config.exclude``, this function first resolves
    regex/wildcard patterns to exact module names. Then, depending on
    ``keep_excluded_layers_as_original_model_state``:

    - If ``True``: already-quantized layers (fp8/int8/etc.) are moved from
      ``quant_config.exclude`` into ``quant_config.layer_quant_config`` so they are
      described with a configuration derived from the model's existing
      ``quantization_config``. Non-quantized layers (fp16/bf16/fp32) remain in
      ``quant_config.exclude``.
    - If ``False``: all excluded layers have been dequantized to fp16/bf16, so they
      all remain in ``quant_config.exclude`` with resolved exact names.

    :param str pretrained_model_path: Path to the pretrained model directory containing
        ``.safetensors`` files and ``config.json``.
    :param QConfig quant_config: The base quantization configuration. A deep copy is made
        internally so the original object is not modified.
    :param dict[str, Any] hf_model_config: HuggingFace model config dictionary (from
        ``config.json``), expected to contain a ``"quantization_config"`` key.
    :param bool keep_excluded_layers_as_original_model_state: If ``True``, already-quantized
        excluded layers are kept in their original format and moved to
        ``layer_quant_config``. If ``False``, all excluded layers stay in ``exclude``.

    :return: A new ``QConfig`` with resolved ``exclude`` and optionally updated
        ``layer_quant_config``.
    :rtype: QConfig
    """
    quant_config = copy.deepcopy(quant_config)

    excluded_tensor_names = _collect_tensor_names_matching_quark_exclude(pretrained_model_path, quant_config)
    excluded_module_names = _convert_linear_weight_tensor_names_to_module_names(excluded_tensor_names)

    if not keep_excluded_layers_as_original_model_state:
        quant_config.exclude = sorted(excluded_module_names)
        return quant_config

    source_quantization_config = hf_model_config.get("quantization_config", {})

    if source_quantization_config.get("quant_method") == "quark":
        return _build_exclude_aware_from_quark_source(
            quant_config, source_quantization_config, excluded_module_names, pretrained_model_path
        )

    SUPPORTED_FMT_TO_DTYPE = {
        "e4m3": "fp8_e4m3",
        "e5m2": "fp8_e5m2",
    }
    source_fmt = source_quantization_config.get("fmt")
    if source_fmt is None:
        raise ValueError(
            "The 'fmt' field is missing in the model's quantization_config. "
            "Cannot determine the quantization dtype for excluded layers."
        )
    dtype_str = SUPPORTED_FMT_TO_DTYPE.get(source_fmt)
    if dtype_str is None:
        raise ValueError(
            f"Unsupported quantization format 'fmt={source_fmt}' in the model's quantization_config. "
            f"Currently supported formats: {sorted(SUPPORTED_FMT_TO_DTYPE.keys())}."
        )

    if source_quantization_config.get("activation_scheme") != "dynamic":
        raise ValueError(
            "Only dynamic activation scheme is supported for exclude-aware config, "
            f"but got activation_scheme='{source_quantization_config.get('activation_scheme')}' "
            "in the model's quantization_config."
        )

    weight_block_size = source_quantization_config.get("weight_block_size", None)
    if weight_block_size is None:
        raise ValueError(
            "Only per-block quantization is currently supported for exclude-aware config, "
            "but 'weight_block_size' is not set in the model's quantization_config."
        )

    # Weight ``scale_type`` describes the on-disk scale storage dtype, so it follows
    # the source (``scale_fmt: "ue8m0"`` → e8m0, else fp32) and lets downstream
    # consumers pick the ``weight_scale`` parameter dtype from the per-layer config.
    # Prefer an explicit source ``scale_type`` if present.
    source_scale_fmt = source_quantization_config.get("scale_fmt")
    derived_weight_scale_type = "float8_e8m0fnu" if source_scale_fmt in ("ue8m0", "e8m0") else "float32"
    weight_scale_type = source_quantization_config.get("scale_type") or derived_weight_scale_type

    weight_tensor_quant_config = {
        "ch_axis": source_quantization_config.get("ch_axis", None),
        "dtype": dtype_str,
        "group_size": source_quantization_config.get("group_size", None),
        "block_size": weight_block_size,
        "is_dynamic": False,
        "observer_cls": "PerBlock2DMinMaxObserver",
        "qscheme": "per_block",
        "round_method": source_quantization_config.get("round_method", "half_even"),
        "scale_type": weight_scale_type,
        "symmetric": source_quantization_config.get("symmetric", True),
    }

    # Activation scales are runtime-computed; leave ``scale_type`` unset so the
    # observer picks its default at runtime (avoids inventing source metadata).
    is_input_dynamic = source_quantization_config.get("activation_scheme") == "dynamic"
    input_tensor_quant_config = {
        "ch_axis": -1,
        "dtype": dtype_str,
        "group_size": weight_block_size[1],
        "block_size": None,
        "is_dynamic": is_input_dynamic,
        "observer_cls": "PerGroupMinMaxObserver",
        "qscheme": "per_group",
        "round_method": source_quantization_config.get("round_method", "half_even"),
        "scale_type": source_quantization_config.get("scale_type"),
        "symmetric": source_quantization_config.get("symmetric", True),
    }

    non_quantized_tensor_names = _get_non_quantized_tensor_names_from_model_safetensors(pretrained_model_path)
    non_quantized_module_names = _convert_linear_weight_tensor_names_to_module_names(non_quantized_tensor_names)

    # Split excluded modules into two groups:
    # - retained_exclude_names: not quantized in HF model and excluded by Quark, keep as exclude.
    # - layer_quant_config: already quantized in HF model but excluded by Quark,
    #   need a per-layer quant config to re-quantize with the source model's scheme.
    retained_exclude_names = []
    layer_quant_config = {}
    for module_name in excluded_module_names:
        if module_name in non_quantized_module_names:
            retained_exclude_names.append(module_name)
        else:
            layer_quant_config[module_name] = QLayerConfig.from_dict(
                {
                    "input_tensors": input_tensor_quant_config,
                    "output_tensors": None,
                    "weight": weight_tensor_quant_config,
                    "bias": None,
                    "target_device": None,
                }
            )

    quant_config.exclude = retained_exclude_names
    if quant_config.layer_quant_config is None:
        quant_config.layer_quant_config = {}
    quant_config.layer_quant_config.update(layer_quant_config)
    return quant_config


def _export_config(
    pretrained_model_path: str,
    quant_config: QConfig,
    save_path: str,
    weight_map: dict[str, str],
    hf_model_config: dict[str, Any],
) -> None:
    """
    Export all configuration files required for the quantized model.

    This function orchestrates the export of all necessary files including JSON/Python
    files from the original model, the safetensors index, and the updated model config
    with quantization settings.

    :param str pretrained_model_path: Path to the original pretrained model directory.
    :param QConfig quant_config: Quantization configuration used during quantization.
    :param str save_path: Directory path to save all configuration files.
    :param dict[str, str] weight_map: Dictionary mapping tensor names to safetensors filenames.
    :param dict[str, Any] hf_model_config: HuggingFace model config dictionary.

    :return: None
    """
    _copy_json_and_py_files(pretrained_model_path, save_path)
    _export_safetensors_index(save_path, weight_map)
    _export_quant_config(hf_model_config, quant_config, save_path)


__all__ = [
    "quantize_model_per_safetensor",
]
