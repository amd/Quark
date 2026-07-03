#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import itertools
import json
import re
import time
from collections import Counter
from pathlib import Path

from safetensors import safe_open

from quark.common.profiler import GlobalProfiler
from quark.common.utils.import_utils import (
    is_accelerate_available,
    is_compressed_tensors_available,
    is_package_lower_or_equal,
    is_torch_available,
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)
from quark.common.utils.log import ScreenLogger
from quark.torch.utils.llm.config import get_quantization_config
from quark.torch.utils.llm.device_mapping import get_no_split_modules

if is_torch_available():
    import torch
    import torch.nn as nn

if is_transformers_available():
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        CompressedTensorsConfig,
    )

    if is_transformers_version_higher_or_equal("5.0"):
        from transformers.initialization import no_init_weights
    else:  # pragma: no cover
        from transformers.modeling_utils import no_init_weights  # type: ignore[attr-defined, no-redef]

if is_accelerate_available():
    from accelerate import dispatch_model, init_empty_weights
    from accelerate.utils import get_balanced_memory, infer_auto_device_map, set_module_tensor_to_device


if is_compressed_tensors_available():
    import compressed_tensors
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.quantization import apply_quantization_config

logger = ScreenLogger(__name__)


# TODO: Remove all `  # pragma: no cover` in this file once once test/test_for_torch/test_inverse_quantizer.py's test_kimi_k25_quantize_export and test_kimi_k25_nvfp4_quantization_and_export are adapted to support transformers>=5.0 which is used in PR CIs.
def _is_compressed_tensors_model(config: "AutoConfig") -> bool:  # pragma: no cover
    """
    Return True if the config describes an already-compressed compressed-tensors checkpoint.
    """
    quantization_config = get_quantization_config(config)

    if quantization_config is None:
        return False

    if isinstance(quantization_config, dict):
        return (
            quantization_config.get("quant_method") == "compressed-tensors"
            and quantization_config.get("quantization_status") == "compressed"
        )

    return (
        getattr(quantization_config, "quant_method", None) == "compressed-tensors"
        and getattr(quantization_config, "quantization_status", None) == "compressed"
    )


# Regex to detect split-expert parameter keys in Kimi-K2.5 checkpoints:
#   <experts_prefix>.experts.<expert_idx>.<layer_name>.<suffix>
# e.g. "language_model.model.layers.1.mlp.experts.16.down_proj.weight_packed"
_EXPERT_KEY_RE = re.compile(r"^(.+\.\d+\.mlp\.experts)\.(\d+)\.(down_proj|gate_proj|up_proj)\.([^.]+)$")


def _load_fused_experts_and_split(
    model: nn.Module,
    cpu_buffer: dict[tuple[str, str, str], dict[int, torch.Tensor]],
    param_to_device: dict[str, str | int],
    experts_prefix: str,
    layer_name: str,
    suffix: str,
) -> None:  # pragma: no cover
    """Stack all expert tensors for one group on CPU, transfer to GPU, assign subviews."""
    buffered = cpu_buffer.pop((experts_prefix, layer_name, suffix))
    sorted_indices = sorted(buffered.keys())

    # Stack into a single contiguous CPU tensor.
    stacked_cpu = torch.stack([buffered[i] for i in sorted_indices], dim=0).contiguous()

    # Determine target device (same for all experts in the group).
    first_param = f"{experts_prefix}.{sorted_indices[0]}.{layer_name}.{suffix}"
    target_device = param_to_device[first_param]
    if suffix == "weight_shape":
        target_device = "cpu"

    # Single batched transfer.
    stacked_gpu = stacked_cpu.to(target_device)

    # Assign each expert's subview.
    for pos, expert_idx in enumerate(sorted_indices):
        param_name = f"{experts_prefix}.{expert_idx}.{layer_name}.{suffix}"
        set_module_tensor_to_device(model, param_name, target_device, value=stacked_gpu[pos])


def _load_on_device_moe_batched(
    model: nn.Module,
    model_dir: str | Path,
    shard_files: set[str],
    param_to_device: dict[str, str | int],
) -> None:  # pragma: no cover
    """
    Load weights for an MOE model with fused experts transfer from RAM to GPU HBM.

    Expert tensors (``*.experts.<idx>.<layer_name>.<suffix>``) are accumulated on CPU and transferred to GPU in a single batched ``.to()`` call, reducing the number of small GPU allocations. Each group is loaded as soon as all its expert indices are collected.

    This function is used to avoid memory fragmentation issues.

    Non-expert tensors are loaded directly to their target device.
    """
    model_param_names = {name for name, _ in model.named_parameters()} | {name for name, _ in model.named_buffers()}

    # Number of experts.
    n_routed_experts = model.config.text_config.n_routed_experts

    # CPU buffer: (experts_prefix, layer_name, suffix) -> {expert_idx: tensor}
    cpu_buffer: dict[tuple[str, str, str], dict[int, torch.Tensor]] = {}

    # Load shards.
    start_time = time.time()
    for shard_file in sorted(shard_files):
        shard_path = Path(model_dir, shard_file)
        shard_size_gib = shard_path.stat().st_size / 1024**3
        logger.info(
            f"  Loading shard: {shard_file} ({shard_size_gib:.2f} GiB). Elapsed: {time.time() - start_time:.2f} s."
        )
        GlobalProfiler.log_torch_memory()

        with safe_open(shard_path, framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
            for key in f.keys():  # noqa
                if key not in model_param_names:
                    continue

                match = _EXPERT_KEY_RE.match(key)
                if match:
                    # Example:
                    # experts_prefix = language_model.model.layers.1.mlp.experts
                    # expert_idx = 63
                    # layer_name = gate_proj
                    # suffix = weight_packed
                    experts_prefix, expert_idx, layer_name, suffix = (
                        match.group(1),
                        int(match.group(2)),
                        match.group(3),
                        match.group(4),
                    )
                    buffer_key = (experts_prefix, layer_name, suffix)
                    if buffer_key not in cpu_buffer:
                        cpu_buffer[buffer_key] = {}
                    cpu_buffer[buffer_key][expert_idx] = f.get_tensor(key)

                    # Load the group as soon as all experts are collected.
                    if len(cpu_buffer[buffer_key]) == n_routed_experts:
                        _load_fused_experts_and_split(
                            model, cpu_buffer, param_to_device, experts_prefix, layer_name, suffix
                        )
                else:
                    # Non-expert tensor: load directly to target device.
                    target_device = param_to_device[key]
                    if key.endswith(".weight_shape"):
                        target_device = "cpu"
                    set_module_tensor_to_device(model, key, target_device, value=f.get_tensor(key))

    # All expert groups should have been loaded during shard loading.
    if len(cpu_buffer) > 0:
        raise RuntimeError("Expected all experts to be loaded. Please open an issue.")


def _load_from_compressed_tensors(
    model_dir: str | Path,
    config: AutoConfig,
    device_map: str | dict[str, int | str],
    max_memory: dict[str, int | str] | None,
    trust_remote_code: bool,
) -> nn.Module:  # pragma: no cover
    """
    Load an already-compressed model from a compressed-tensors checkpoint,
    bypassing the AutoModelForCausalLM quantizer path.

    Flow:
      1. Build an empty model skeleton on meta device.
      2. Use ModelCompressor to set up quantized module structure
         (CompressedLinear on <=0.14, decompression hook on >=0.15).
      3. Load safetensors weights from disk via accelerate.
      4. Move the model to the target device(s).
    """
    logger.info("Detected compressed-tensors checkpoint; using compressed-tensors loading path.")

    if not is_compressed_tensors_available() or not is_accelerate_available():
        raise ImportError(
            "The packages `compressed-tensors` and `accelerate` are required to load pre-quantized compressed-tensors models. Please install them."
        )

    with no_init_weights(), init_empty_weights():
        model = AutoModelForCausalLM.from_config(  # type: ignore[no-untyped-call]
            config, trust_remote_code=trust_remote_code, attn_implementation="eager"
        )

    if is_package_lower_or_equal("compressed-tensors", "0.14.99"):
        raise ImportError(
            f"inverse_quantizer.py requires compressed-tensors>=0.15, but found compressed-tensors=={compressed_tensors.__version__} in the environment. Please update compressed-tensors."
        )
    else:
        # `ModelCompressor.from_pretrained` API was removed in compressed-tensors==0.15.
        compression_config = CompressedTensorsConfig.from_dict(config.quantization_config)  # type: ignore[no-untyped-call, attr-defined]
        compressor = ModelCompressor.from_compression_config(compression_config)

    if compressor is None:
        raise ValueError(
            f"ModelCompressor could not be created from the quantization config of {model_dir}. Please check that the model has a valid quantization configuration."
        )

    if compressor.quantization_config is None:
        raise ValueError(
            f"ModelCompressor for {model_dir} has no quantization configuration. Only quantized checkpoints are supported."
        )

    logger.info("Setting up quantized module structure via ModelCompressor...")
    apply_quantization_config(model, compressor.quantization_config, run_compressed=True)
    compressor.compress_model(model)

    # Resolve device map.
    # Keep track of whether we need dispatch_model (multi-device) or a simple .to().
    single_device: str | torch.device | None = None
    if isinstance(device_map, torch.device) or (isinstance(device_map, str) and device_map != "auto"):
        single_device = device_map
        param_to_device: dict[str, str | int] = {n: "cpu" for n, _ in model.named_parameters()}
        param_to_device.update({n: "cpu" for n, _ in model.named_buffers()})
    else:
        if device_map == "auto":
            _no_split = get_no_split_modules(model)

            if max_memory is None:
                max_memory = get_balanced_memory(model, no_split_module_classes=_no_split)

            device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=_no_split)

            assert isinstance(device_map, dict)
            disk_modules = [k for k, v in device_map.items() if v == "disk"]
            if len(disk_modules) > 0:
                raise NotImplementedError(
                    f"infer_auto_device_map assigned {len(disk_modules)} module(s) to 'disk', "
                    f"which is not implemented for the moment. Please open an issue."
                )

        # Log device_map distribution and estimated model size for diagnostics.
        assert isinstance(device_map, dict)
        dist = Counter(str(v) for v in device_map.values())
        logger.info(f"device_map distribution: {dist}")
        total_param_bytes = sum(
            p.numel() * torch.tensor([], dtype=p.dtype).element_size() for _, p in model.named_parameters()
        )
        logger.info(f"Estimated model size from meta parameters: {total_param_bytes / 1024**3:.2f} GiB")

        # device_map is now a dict — build param→device lookup.
        param_to_device = {}
        for param_name, _ in itertools.chain(model.named_parameters(), model.named_buffers()):
            parts = param_name.split(".")
            for i in range(len(parts), 0, -1):
                module_key = ".".join(parts[:i])
                if module_key in device_map:
                    param_to_device[param_name] = device_map[module_key]
                    break
            else:
                param_to_device[param_name] = device_map.get("", "cpu")  # type: ignore[union-attr]

    # Load safetensors weights directly to their target devices.
    index_path = Path(model_dir, "model.safetensors.index.json")
    if index_path.exists():
        with open(index_path) as f:
            shard_files: set[str] = set(json.load(f)["weight_map"].values())
    else:
        shard_files = {"model.safetensors"}

    logger.info(f"Loading weights from {len(shard_files)} shard(s)...")

    start_time = time.time()

    if config.model_type == "kimi_k25":  # type: ignore[attr-defined]
        # Kimi-K2.5 has many small expert tensors; batch them to reduce GPU allocations.
        # In case this code path is not called, we hit memory fragmentation issues with OOM
        # despite free memory available (torch.OutOfMemoryError: HIP out of memory. Tried to allocate 20.00 MiB. GPU 3 has a total capacity of 251.98 GiB of which 127.96 GiB is free.), when using transformers==4.57 + torch==2.11 + compressed-tensors==0.14 (and 0.15 as well).
        _load_on_device_moe_batched(model, model_dir, shard_files, param_to_device)
    else:
        model_param_names = {param_name for param_name, _ in model.named_parameters()} | {
            buffer_name for buffer_name, _ in model.named_buffers()
        }
        for shard_file in sorted(shard_files):
            shard_path = Path(model_dir, shard_file)
            shard_size_gib = shard_path.stat().st_size / 1024**3
            logger.info(
                f"  Loading shard: {shard_file} ({shard_size_gib:.2f} GiB). Elapsed: {time.time() - start_time:.2f} s."
            )
            GlobalProfiler.log_torch_memory()
            with safe_open(shard_path, framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
                for key in f.keys():  # noqa
                    if key not in model_param_names:
                        logger.warning(
                            f"The key {key} present in {shard_path} is not in the model parameters/buffers and is ignored."
                        )
                        continue
                    target_device = param_to_device[key]
                    if key.endswith(".weight_shape"):
                        target_device = "cpu"
                    set_module_tensor_to_device(model, key, target_device, value=f.get_tensor(key))

    meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if len(meta_params) > 0:
        raise ValueError(
            f"Parameters: {meta_params} (total {len(meta_params)} parameter(s)) are still on meta device after loading, this should not happen. Please open an issue. "
        )

    # Move to target device(s).
    if single_device is not None:
        model.to(single_device)
    else:
        dispatch_model(model, device_map=device_map)

    return model
