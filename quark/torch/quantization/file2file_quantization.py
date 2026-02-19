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
from types import MappingProxyType
from typing import Any

import torch

from quark.shares.utils.import_utils import (
    is_compressed_tensors_available,
    is_safetensors_available,
    is_triton_available,
)

if is_triton_available():
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
if is_compressed_tensors_available():
    from compressed_tensors import ModelCompressor, QuantizationConfig  # type: ignore[import-untyped,import-not-found]
    from compressed_tensors.quantization import (  # type: ignore[import-untyped,import-not-found]
        QuantizationArgs,
        QuantizationScheme,
    )

if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

import quark
from quark.shares.config import BaseQLayerConfig
from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.config.config import Config, QTensorConfig
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.utils.pack import create_pack_method

logger = ScreenLogger(__name__)


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


def _is_linear_weight_tensor(tensor_name: str) -> bool:
    """
    Check if a tensor is a Linear layer weight by its name.

    A tensor is considered a Linear weight if:
    - Its parameter name is "weight" (not bias, etc.)
    - Its module name does not end with "norm" (exclude LayerNorm, RMSNorm, etc.)

    :param str tensor_name: The full name of the tensor (e.g., "model.layers.0.self_attn.q_proj.weight").

    :return: True if the tensor is a Linear weight, False otherwise.
    :rtype: bool
    """
    if "." not in tensor_name:
        return False
    module_name, param_name = tensor_name.rsplit(".", 1)
    return param_name == "weight" and not module_name.endswith("norm") and "embed" not in module_name


def _get_layer_quant_config_by_tensor_name(
    tensor_name: str,
    quant_config: Config,
    excluded_layer_names: set[str] | None = None,
) -> BaseQLayerConfig | None:
    """
    Get quantization configuration for a specific tensor by name.

    This function determines if the given tensor is a Linear weight and returns
    the appropriate quantization configuration. Layers matching exclusion patterns
    return None, layer-specific patterns take precedence, and remaining layers
    use the global quantization config.

    :param str tensor_name: The full name of the tensor (e.g., "model.layers.0.self_attn.q_proj.weight").
    :param Config quant_config: Quantization configuration containing ``exclude`` patterns,
        ``layer_quant_config`` for layer-specific configs, and ``global_quant_config`` as default.

    :return: The quantization configuration for the layer, or None if the layer should not be quantized.
    :rtype: BaseQLayerConfig | None
    """
    if not _is_linear_weight_tensor(tensor_name):
        return None

    module_name = tensor_name.rsplit(".", 1)[0]

    if any(fnmatch.fnmatch(module_name, pattern) for pattern in quant_config.exclude):
        if excluded_layer_names is not None:
            excluded_layer_names.add(module_name)
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

    def _weight_dequant_fp8(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
        """
        Dequantize FP8 weight tensor using inverse scale with Triton kernel.
        """
        assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
        assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
        M, N = x.size()
        y = torch.empty_like(x, dtype=torch.get_default_dtype())

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
    config = json.load(open(os.path.join(pretrained_model_path, "config.json")))
    return MappingProxyType(config)


def _get_quantization_config(hf_model_config: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Get quantization config from model config.

    Supports both pure LLM format (``hf_model_config["quantization_config"]``) and
    VLM/multimodal format (``hf_model_config["text_config"]["quantization_config"]``).

    :param dict | None hf_model_config: Model config dictionary.

    :return: Quantization config dict or None if not found.
    :rtype: dict[str, Any] | None
    """
    if hf_model_config is None:
        return None

    if "quantization_config" in hf_model_config:
        # Pure LLM format: hf_model_config["quantization_config"]
        return hf_model_config["quantization_config"]
    elif "text_config" in hf_model_config and "quantization_config" in hf_model_config["text_config"]:
        # VLM/multimodal format: hf_model_config["text_config"]["quantization_config"]
        return hf_model_config["text_config"]["quantization_config"]

    return None


def _recover_compressed_tensors_weights(
    safetensor_path: str,
    quant_config_dict: dict[str, Any],
    device: str | torch.device,
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
    :param dict quant_config_dict: Quantization configuration dictionary with
        ``quant_method: "compressed-tensors"``.

    :return: Dictionary of tensor name to tensor with decompressed weights.
    :rtype: dict[str, torch.Tensor]
    """
    quant_config = QuantizationConfig.model_validate(quant_config_dict)
    compressor = ModelCompressor(quantization_config=quant_config)

    # Determine compression format from config (e.g., "pack-quantized", "float-quantized")
    compression_format = quant_config_dict.get("format", "dense")
    if compression_format not in compressor.quantization_compressor:
        raise ValueError(
            f"Unsupported compression format: '{compression_format}'. "
            f"Supported formats: {list(compressor.quantization_compressor.keys())}"
        )
    quantization_compressor = compressor.quantization_compressor[compression_format]
    compression_param_suffixes = set(quantization_compressor.compression_param_names)

    # Build QuantizationArgs from config
    weights_config = quant_config_dict["config_groups"]["group_0"]["weights"]
    quant_args = QuantizationArgs(
        num_bits=weights_config["num_bits"],
        type=weights_config["type"],
        strategy=weights_config["strategy"],
        group_size=weights_config["group_size"],
        symmetric=weights_config["symmetric"],
    )

    # Scan file to identify quantized modules and collect non-quantized tensors
    names_to_scheme: dict[str, QuantizationScheme] = {}
    recovered_tensors: dict[str, torch.Tensor] = {}

    device_str = str(device)
    with safe_open(safetensor_path, framework="pt", device=device_str) as f:  # type: ignore[no-untyped-call]
        all_keys = set(f.keys())

        # Identify quantized modules by the presence of weight_scale
        quantized_module_paths: set[str] = set()
        for key in all_keys:
            if key.endswith(".weight_scale"):
                module_path = key.rsplit(".weight_scale", 1)[0]
                quantized_module_paths.add(module_path)
                names_to_scheme[module_path] = QuantizationScheme(
                    targets=["Linear"],
                    weights=quant_args,
                )

        # Build the full set of tensor keys that belong to quantized modules
        quantized_tensor_keys: set[str] = set()
        for module_path in quantized_module_paths:
            for suffix in compression_param_suffixes:
                quantized_tensor_keys.add(f"{module_path}.{suffix}")

        # Copy non-quantized tensors directly
        for key in all_keys:
            if key not in quantized_tensor_keys:
                recovered_tensors[key] = f.get_tensor(key)

    logger.info(f"Decompressing {len(names_to_scheme)} compressed_tensors quantized weights...")

    # Decompress quantized weights using compressed_tensors library
    for name, tensor_dict in quantization_compressor.decompress(
        safetensor_path,
        names_to_scheme=names_to_scheme,
        device=device_str,
    ):
        weight = tensor_dict["weight"]
        recovered_tensors[name + ".weight"] = weight

    logger.info(f"Decompressed {len(names_to_scheme)} weights, total tensors: {len(recovered_tensors)}")

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


def _recover_fp8_weights(
    safetensor_path: str,
    device: str | torch.device,
    weight_map: dict[str, str] | None = None,
    scale_inv_cache: dict[str, torch.Tensor] | None = None,
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
    :param dict[str, str] | None weight_map: Dictionary mapping tensor names to safetensor
        filenames. Used to determine if a weight has a corresponding scale_inv tensor.
        Defaults to ``None``.
    :param dict[str, torch.Tensor] | None scale_inv_cache: Pre-loaded cache of scale_inv
        tensors that are stored in different files from their weights. Defaults to ``None``.
    :param str | torch.device device: Device to load tensors onto (e.g., ``"cuda"``, ``"cuda:0"``, ``"cpu"``).

    :return: Dictionary of tensor name to tensor with dequantized weights.
    :rtype: dict[str, torch.Tensor]
    """
    recovered_tensors: dict[str, torch.Tensor] = {}
    fp8_weight_count = 0
    device_str = str(device)

    with safe_open(safetensor_path, framework="pt", device=device_str) as f:  # type: ignore[no-untyped-call]
        all_keys = set(f.keys())

        for weight_name in all_keys:
            # Skip scale_inv tensors, they are used during dequantization
            if weight_name.endswith("_scale_inv"):
                continue

            # Check if this is an FP8 Linear weight with corresponding scale_inv
            scale_inv_name = f"{weight_name}_scale_inv"
            if _is_linear_weight_tensor(weight_name):
                if scale_inv_name in all_keys:
                    # Load weight and scale_inv from current file, dequantize
                    weight = f.get_tensor(weight_name)
                    scale_inv = f.get_tensor(scale_inv_name)
                    recovered_tensors[weight_name] = _weight_dequant_fp8(weight, scale_inv)
                    fp8_weight_count += 1
                    # Free tensors immediately after dequantization
                    del weight, scale_inv
                    _empty_cache_if_cuda(device)
                elif scale_inv_cache is not None and scale_inv_name in scale_inv_cache:
                    # Load scale_inv from pre-loaded cache -> target device
                    weight = f.get_tensor(weight_name)
                    scale_inv = scale_inv_cache[scale_inv_name].to(device)
                    recovered_tensors[weight_name] = _weight_dequant_fp8(weight, scale_inv)
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

    logger.info(f"Dequantized {fp8_weight_count} FP8 weights, total tensors: {len(recovered_tensors)}")

    return recovered_tensors


def _load_safetensor_with_recover(
    safetensor_path: str,
    device: str | torch.device,
    hf_model_config: dict[str, Any] | None = None,
    weight_map: dict[str, str] | None = None,
    scale_inv_cache: dict[str, torch.Tensor] | None = None,
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

    :return: Dictionary of tensor name to tensor with decompressed/dequantized weights.
    :rtype: dict[str, torch.Tensor]
    """
    # Try to determine quantization method from hf_model_config
    quant_config_dict = _get_quantization_config(hf_model_config)

    if quant_config_dict is not None:
        quant_method = quant_config_dict.get("quant_method")

        if quant_method == "fp8":
            # FP8 format: quant_method == "fp8"
            return _recover_fp8_weights(safetensor_path, device, weight_map, scale_inv_cache)

        elif quant_method == "compressed-tensors":
            # compressed_tensors format: quant_method == "compressed-tensors"
            return _recover_compressed_tensors_weights(safetensor_path, quant_config_dict, device=device)

    # No quantization config or unknown quant_method, load normally
    return load_file(safetensor_path, device=str(device))


def _quantize_and_save_safetensor_shard(
    safetensor_path: str,
    export_path: str,
    quant_config: Config,
    device: str | torch.device,
    output_weight_map: dict[str, str] | None = None,
    excluded_layer_names: set[str] | None = None,
    input_scale_dict: dict[str, torch.Tensor] | None = None,
    hf_model_config: dict[str, Any] | None = None,  # Optional HuggingFace model config for metadata
    source_weight_map: dict[str, str] | None = None,  # Weight map for cross-file scale_inv lookup
    scale_inv_cache: dict[str, torch.Tensor] | None = None,  # Pre-loaded scale_inv cache
) -> None:
    """
    Quantize weights in a single safetensors shard file and save the result.

    This function loads tensors from the input safetensors file, quantizes weight tensors
    according to the quantization configuration, packs them, and exports to the output directory.
    Non-weight tensors (e.g., biases, layernorm parameters) are copied as-is.

    :param str safetensor_path: Path to the input safetensors file.
    :param str export_path: Directory path to export the quantized safetensors file.
    :param Config quant_config: Quantization configuration containing ``exclude`` patterns,
        ``layer_quant_config`` for layer-specific configs, and ``global_quant_config`` as default.
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
    This function writes the quantized shard to disk. If ``output_weight_map`` is provided,
    it will be updated in-place.
    """
    safetensor_filename = os.path.basename(safetensor_path)
    logger.info(f"Loading {safetensor_filename}...")
    tensors = _load_safetensor_with_recover(
        safetensor_path, device, hf_model_config, source_weight_map, scale_inv_cache
    )
    quantized_tensors: dict[str, torch.Tensor] = {}

    for tensor_name, tensor in tensors.items():
        if tensor_name.endswith((".weight_packed", ".weight_scale", ".weight_shape")):
            continue

        if output_weight_map is not None:
            output_weight_map[tensor_name] = safetensor_filename

        layer_name = ".".join(tensor_name.split(".")[:-1])
        layer_config = _get_layer_quant_config_by_tensor_name(
            tensor_name=tensor_name,
            quant_config=quant_config,
            excluded_layer_names=excluded_layer_names,
        )

        if layer_config is not None:
            weight_config = layer_config.weight
            assert isinstance(weight_config, QTensorConfig), f"weight config for {layer_name} must be QTensorConfig"

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

            if getattr(weight_config, "scale_format", None) == "e8m0":
                # Convert scale to e8m0 format (MXFP4 standard)
                scale_e8m0 = (torch.log2(quantizer.scale).round().to(torch.int16).clamp(-127, 127) + 127).to(
                    torch.uint8
                )
                quantized_tensors[tensor_name + "_scale"] = scale_e8m0.contiguous()
            else:
                quantized_tensors[tensor_name + "_scale"] = quantizer.scale.contiguous()

            if output_weight_map is not None:
                output_weight_map[tensor_name + "_scale"] = safetensor_filename

            # Export input scale if provided
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


def quantize_model_per_safetensor(
    pretrained_model_path: str,
    quant_config: Config,
    save_path: str,
    device: str | torch.device = "cuda",
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
    :param Config quant_config: Quantization configuration specifying dtype, exclusions,
        and per-layer settings.
    :param str save_path: Directory path to save the quantized safetensors files.
    :param str | torch.device device: Device for tensor operations (e.g., ``"cuda"``,
        ``"cuda:0"``, ``"cpu"``).
    """
    # Pre-load cross-file scale_inv tensors into cache (only for FP8 models)
    source_weight_map: dict[str, str] | None = None
    scale_inv_cache: dict[str, torch.Tensor] | None = None
    hf_model_config = _get_hf_model_config(pretrained_model_path)

    # Set the default dtype for the model
    if "torch_dtype" in hf_model_config:
        torch.set_default_dtype(Dtype.from_str(hf_model_config["torch_dtype"]).to_torch_packed_dtype())

    quant_config_dict = _get_quantization_config(hf_model_config)
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
    excluded_layer_names: set[str] = set()
    safetensor_files = _get_safetensor_files(pretrained_model_path)
    logger.info(f"Found {len(safetensor_files)} safetensors files to process")
    os.makedirs(save_path, exist_ok=True)
    for index, safetensor_path in enumerate(safetensor_files):
        logger.info(f"Processing {index + 1}/{len(safetensor_files)}: {os.path.basename(safetensor_path)}")
        _quantize_and_save_safetensor_shard(
            safetensor_path=safetensor_path,
            export_path=save_path,
            quant_config=quant_config,
            output_weight_map=output_weight_map,
            excluded_layer_names=excluded_layer_names,
            hf_model_config=hf_model_config,
            source_weight_map=source_weight_map,
            scale_inv_cache=scale_inv_cache,
            device=device,
        )

    # Free the cache after processing
    if scale_inv_cache is not None:
        del scale_inv_cache
        _empty_cache_if_cuda(device)

    _export_config(
        pretrained_model_path,
        quant_config,
        save_path,
        output_weight_map,
        hf_model_config,
        excluded_layer_names=excluded_layer_names,
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
    quant_config: Config,
    save_path: str,
    excluded_layer_names: set[str] | None = None,
) -> None:
    """
    Export the model configuration with quantization settings.

    This function takes the HuggingFace model config, adds quantization configuration
    and export settings, then saves the updated config to the output directory.

    :param dict[str, Any] hf_model_config: HuggingFace model config dictionary (passed from caller).
    :param Config quant_config: Quantization configuration to embed in the model config.
    :param str save_path: Directory path to save the updated ``config.json``.

    :return: None
    """
    hf_model_config = copy.deepcopy(dict(hf_model_config))
    # Remove any existing quantization configs (compressed-tensors, deepseek-style fp8, etc.)
    _remove_quantization_config(hf_model_config)

    hf_model_config["quantization_config"] = quant_config.to_dict()
    hf_model_config["quantization_config"]["export"] = {
        "kv_cache_group": [],
        "min_kv_scale": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    }

    # Downstream platforms may not support regex/wildcard patterns.
    # If we resolved full excluded layer names during quantization, export exact names.
    hf_model_config["quantization_config"]["exclude"] = sorted(excluded_layer_names)

    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_model_config, f, ensure_ascii=False, indent=4)


def _export_config(
    pretrained_model_path: str,
    quant_config: Config,
    save_path: str,
    weight_map: dict[str, str],
    hf_model_config: dict[str, Any],
    excluded_layer_names: set[str] | None = None,
) -> None:
    """
    Export all configuration files required for the quantized model.

    This function orchestrates the export of all necessary files including JSON/Python
    files from the original model, the safetensors index, and the updated model config
    with quantization settings.

    :param str pretrained_model_path: Path to the original pretrained model directory.
    :param Config quant_config: Quantization configuration used during quantization.
    :param str save_path: Directory path to save all configuration files.
    :param dict[str, str] weight_map: Dictionary mapping tensor names to safetensors filenames.
    :param dict[str, Any] hf_model_config: HuggingFace model config dictionary.

    :return: None
    """
    _copy_json_and_py_files(pretrained_model_path, save_path)
    _export_safetensors_index(save_path, weight_map)
    _export_quant_config(hf_model_config, quant_config, save_path, excluded_layer_names=excluded_layer_names)


__all__ = [
    "quantize_model_per_safetensor",
]
