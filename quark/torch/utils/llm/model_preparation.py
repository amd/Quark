#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from quark.common.utils.import_utils import (
    is_psutil_available,
    is_safetensors_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)
from quark.torch.export.utils import get_source_name_or_path
from quark.torch.integrations.compressed_tensors.loading import (
    _is_compressed_tensors_model,
    _is_meta_device_map,
    _load_from_compressed_tensors,
)
from quark.torch.utils.llm.preprocessing import maybe_save_preprocessors

if is_psutil_available():
    import psutil  # type: ignore[import-untyped]

if is_safetensors_available():
    import safetensors

if is_torch_available():
    import torch
    import torch.nn as nn

if is_transformers_available():
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        MllamaForConditionalGeneration,
    )
    from transformers.models.dbrx.modeling_dbrx import DbrxExperts

if is_transformers_available() and is_transformers_version_higher_or_equal("4.51.0"):
    from transformers import Llama4ForConditionalGeneration
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe  # type: ignore[attr-defined]

if is_transformers_available() and is_transformers_version_higher_or_equal("5.0"):
    # Loader handle for the multi-GPU MoE worker throttle (see ``_throttled_hf_loader``).
    import transformers.core_model_loading as _hf_loader_module  # type: ignore[no-redef]
elif is_transformers_available():  # pragma: no cover
    _hf_loader_module = None  # type: ignore[assignment]

if is_transformers_available() and is_transformers_version_higher_or_equal("4.55.1"):
    from transformers import Mxfp4Config  # type: ignore[attr-defined]
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssMLP
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import GraniteMoeHybridMoE

if is_transformers_available() and is_transformers_version_higher_or_equal("4.57.0"):
    from transformers import Qwen3VLMoeForConditionalGeneration  # type: ignore[attr-defined]
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeTextExperts,  # type: ignore[attr-defined]
    )

if is_transformers_available() and is_transformers_version_higher_or_equal("5.2.0"):
    from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration

if is_transformers_available() and is_transformers_version_higher_or_equal("5.16.0"):  # pragma: no cover
    # Unreachable below transformers 5.16, which is where qwen4_exp first ships.
    from transformers import Qwen4ExpForConditionalGeneration  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from quark.common.utils.log import ScreenLogger

from ..torch_utils import setattr_recursive
from .device_mapping import build_skeleton_from_config, create_auto_adjusted_device_map
from .module_replacement import PREPROCESS_REGISTRY
from .module_replacement.dbrx_expert import DbrxExperts_
from .module_replacement.replacement_utils import (
    replace_gptoss_experts_with_linear,
    replace_gptoss_mlp_with_linear_router,
    replace_granite_moe_experts_with_linear,
    replace_llama4_experts_with_sequential,
    replace_qwen3vlmoe_experts_with_linear,
)

logger = ScreenLogger(__name__)


# Throttle the HF loader once the model is estimated to produce more quantized
# linears than this. Calibrated so GLM-5 / GLM-4.7 / Qwen3.5-397B trigger,
# Qwen3-30B / Mixtral-8x7B do not.
_HUGE_MOE_LINEAR_THRESHOLD = 30000
_VM_MAX_MAP_COUNT_SAFE = 1048576
_VM_MAX_MAP_COUNT_RECOMMENDED = 4194304

_GIB = 1024**3
_KIB = 1024

# Activation budget for one per_block calibration forward: a ~250 GiB MI350 minus 100 GiB for the
# resident decoder layer. That layer is ~50 GiB on Qwen3.8-2.4T-A95B (5 TB of weights / 92 layers);
# the 2x reserve covers the fake-quantized weight copies built during the forward, and fragmentation.
_CALIB_RESERVED_GIB = 150
# bf16 activations alive per token inside one decoder layer: 2*(3h + 2*top_k*h + 3*top_k*i) bytes,
# i.e. residual/norm/combine, the MoE gather buffer, and the expert intermediates. ~400 KiB at
# h=8192, moe_intermediate=2048, top_k=8, the largest MoE shape we target; dense models are ~20x under.
_ACT_PER_TOKEN_KIB = 400


def _num_hidden_layers(config: object | None) -> int | None:
    """``num_hidden_layers`` from *config*, unwrapping one level of ``text_config`` for VLMs.

    Returns ``None`` if *config* is ``None`` or doesn't expose a positive ``num_hidden_layers``.
    """
    if config is None:
        return None
    text_config = getattr(config, "text_config", None) or config
    num_layers = getattr(text_config, "num_hidden_layers", None)
    return num_layers if isinstance(num_layers, int) and num_layers > 0 else None


def _estimate_quantized_linears(config: object | None) -> int:
    """Rough count of nn.Linear layers Quark will wrap, derived from HF config.

    Used as a size signal for the HF loader throttle. Probes the common alias
    set (``n_routed_experts`` / ``num_local_experts`` / ``num_experts``).
    Returns 0 if the config doesn't expose ``num_hidden_layers``.
    """
    num_layers = _num_hidden_layers(config)
    if num_layers is None:
        return 0
    text_config = getattr(config, "text_config", None) or config
    num_experts = (
        getattr(text_config, "n_routed_experts", None)
        or getattr(text_config, "num_local_experts", None)
        or getattr(text_config, "num_experts", None)
        or 0
    )
    if not isinstance(num_experts, int) or num_experts < 0:
        num_experts = 0
    first_k_dense = getattr(text_config, "first_k_dense_replace", 0) or 0
    if not isinstance(first_k_dense, int) or first_k_dense < 0:
        first_k_dense = 0
    if num_experts > 0:
        num_dense = min(first_k_dense, num_layers)
        num_moe = num_layers - num_dense
    else:
        num_dense, num_moe = num_layers, 0
    # attn Q/K/V/O (4) + dense MLP gate/up/down (3); MoE adds router (1) + experts × 3.
    return num_dense * 7 + num_moe * (5 + num_experts * 3)


def _warn_vm_max_map_count(num_linears: int) -> None:
    """Warn if ``vm.max_map_count`` is too low for the huge MoE about to be loaded."""
    if not sys.platform.startswith("linux"):
        return
    try:
        with open("/proc/sys/vm/max_map_count") as f:
            current = int(f.read().strip())
    except (OSError, ValueError):
        return
    if current >= _VM_MAX_MAP_COUNT_SAFE:
        return
    logger.warning(
        "vm.max_map_count = %d is too low for huge multi-GPU MoE quantization "
        "(~%d quantized linears). Calibration may abort with misleading OOM / "
        "HSA_STATUS_ERROR_OUT_OF_RESOURCES errors. Fix: 'sudo sysctl -w vm.max_map_count=%d'.",
        current,
        num_linears,
        _VM_MAX_MAP_COUNT_RECOMMENDED,
    )


def _hf_loader_target_workers(is_multi_gpu: bool, config: object | None) -> int | None:
    """Worker count to apply to the HF parallel weight loader, or ``None`` for HF default.

    ``QUARK_HF_LOADER_WORKERS`` overrides everything. Otherwise auto-throttle to 1
    when the model spans >1 GPU and exceeds ``_HUGE_MOE_LINEAR_THRESHOLD`` quantized
    linears (e.g. GLM-5, Qwen3.5-397B); also warns about ``vm.max_map_count``.
    """
    env_override = os.environ.get("QUARK_HF_LOADER_WORKERS")
    if env_override is not None:
        return max(1, int(env_override))
    if not is_multi_gpu:
        return None
    num_linears = _estimate_quantized_linears(config)
    if num_linears <= _HUGE_MOE_LINEAR_THRESHOLD:
        return None
    _warn_vm_max_map_count(num_linears)
    logger.info(
        "Throttling HF loader to 1 worker for huge multi-GPU MoE (~%d linears); "
        "override with QUARK_HF_LOADER_WORKERS=N.",
        num_linears,
    )
    return 1


def _set_hf_loader_workers(target: int | None) -> int | None:
    """Set ``GLOBAL_WORKERS = target`` if applicable; return previous value to restore later.

    No-op (returns ``None``) if ``target`` is ``None``, transformers is too old, or
    the loader module doesn't expose ``GLOBAL_WORKERS``.
    """
    if target is None or _hf_loader_module is None or not hasattr(_hf_loader_module, "GLOBAL_WORKERS"):
        return None
    saved = _hf_loader_module.GLOBAL_WORKERS
    _hf_loader_module.GLOBAL_WORKERS = target
    return saved


def _restore_hf_loader_workers(saved: int | None) -> None:
    """Inverse of ``_set_hf_loader_workers``; safe to call with ``None``."""
    if saved is not None and _hf_loader_module is not None:
        _hf_loader_module.GLOBAL_WORKERS = saved


def get_tokenizer(
    ckpt_path: str, max_seq_len: int = 2048, model_type: str | None = None, trust_remote_code: bool = True
) -> "PreTrainedTokenizerBase":
    logger.info(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, padding_side="left", trust_remote_code=trust_remote_code)  # type: ignore[no-untyped-call]
    if model_type and model_type in ["qwen", "qwen2"]:
        # qwen2 use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError(f"Pad token cannot be resolved (model_type={model_type}, ckpt_path={ckpt_path}).")

    return tokenizer


# TODO: we should implement a proper model patcher in quark namespace, well tested, with well supported cases / modularity. This if/else design is not tractable.
def _legacy_prepare_for_moe_quant(model: nn.Module, reload: bool = False) -> None:
    if model.config.model_type in ["qwen3_5"]:
        raise ValueError(
            f"The model architecture model.config.model_type={model.config.model_type} is not yet supported in Quark."
        )
    elif model.config.model_type in ["dbrx", "llama4"]:
        for name, module in model.named_modules(remove_duplicate=False):
            if isinstance(module, DbrxExperts):
                new_experts = DbrxExperts_.from_float(module)
                setattr_recursive(model, name, new_experts)
                logger.info(f"Module replaced for quantization: {name} ({type(module)} -> {type(new_experts)})")
            elif isinstance(module, Llama4TextMoe):
                replace_llama4_experts_with_sequential(module, model.config.text_config, reload=reload)
    elif model.config.model_type == "gpt_oss":
        for name, module in tqdm(
            model.named_modules(remove_duplicate=False),
            desc="Replacing GptOssExperts implementation to use torch.nn.Linear for quantization",
        ):
            # NOTE: MOE router patching is useful only in case the router is quantized,
            # or in case we use rotation.
            if isinstance(module, GptOssExperts):
                replace_gptoss_experts_with_linear(experts_module=module, reload=reload)

            if isinstance(module, GptOssMLP):
                replace_gptoss_mlp_with_linear_router(mlp=module)
    elif model.config.model_type == "granitemoehybrid":
        for name, module in model.named_modules(remove_duplicate=False):
            if isinstance(module, GraniteMoeHybridMoE):
                replace_granite_moe_experts_with_linear(module)
    elif model.config.model_type == "qwen3_vl_moe":
        for name, module in model.named_modules(remove_duplicate=False):
            if isinstance(module, Qwen3VLMoeTextExperts):
                replace_qwen3vlmoe_experts_with_linear(module)


def _prepare_for_moe_quant(model: nn.Module, reload: bool = False) -> None:
    """
    Internal: Traverse named_modules, when PREPROCESS_REGISTRY is hit, replace module via QuarkXxx.from_hf.
    """
    config = getattr(model, "config", None)
    if config is None:
        logger.warning("Model has no config; skipping MoE preprocess")
        return

    for name, module in model.named_modules(remove_duplicate=False):
        module_type = type(module)
        replacement_class = PREPROCESS_REGISTRY.get(module_type)
        if replacement_class is None:
            continue

        try:
            new_module = replacement_class.from_hf(module, reload=reload)
            setattr_recursive(model, name, new_module)
            logger.debug(f"Preprocessed {name} ({module_type.__name__} -> {type(new_module).__name__})")
        except Exception as e:
            logger.error(f"Preprocess failed for {name} ({module_type.__name__}): {e}")
            raise


def preprocess_for_quantization(model: nn.Module, reload: bool = False) -> None:
    """
    Transform modules in-place for quantization (fused params → per-expert nn.Linear).
    High-level entry point; users should call this.

    Args:
        model: The model to preprocess.
        moe_quant: If True, run MoE prepare logic (unsupported check + registry handlers).
                   If False, no-op.
    """
    if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
        _prepare_for_moe_quant(model, reload)
    else:
        _legacy_prepare_for_moe_quant(model, reload)


def prepare_for_moe_quant(model: nn.Module, reload: bool = False) -> None:
    """
    Deprecated: use `preprocess_for_quantization` instead.
    """
    logger.warning(
        "`prepare_for_moe_quant` is deprecated and will be removed in a future release. "
        "Use `preprocess_for_quantization` instead."
    )
    if is_transformers_version_higher_or_equal("5.0.0"):
        raise ValueError(
            "prepare_for_moe_quant is not supported for transformers v5+. For transformers v5+, use preprocess_for_quantization instead."
        )
    else:
        _legacy_prepare_for_moe_quant(model, reload)


def _module_bytes(module: nn.Module) -> int:
    """Bytes held by *module*'s parameters and buffers, counting shared tensors once."""
    return sum(t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers()))


def _per_block_resident_bytes(model: nn.Module) -> tuple[int, int] | None:
    """Bytes of the largest decoder block, and of everything outside the decoder stack.

    Both numbers come out of the same budget. ``per_block_runner.prepare()`` loads the non-block
    modules (embeddings, final norm, lm_head) onto the target device once and keeps them there for
    the whole run, so they occupy memory the calibration activations cannot use. And block sizes
    are not uniform: MoE stacks front-load dense layers (``first_k_dense_replace``) or interleave
    them, so the block that has to fit is the largest one, not the last one and not an
    ``model_bytes / num_layers`` average that gets diluted by the smaller dense layers.

    :param nn.Module model: Model to inspect.
    :return: ``(largest_block_bytes, non_block_bytes)``, or ``None`` if the decoder block stack
        can't be located.
    """
    try:
        # Imported lazily: per_block_runner pulls in safetensors unconditionally, which is an
        # optional dependency for this module (see is_safetensors_available() above).
        from quark.torch.utils.per_block_runner.utils import infer_decoder_layers_path
    except ImportError:
        return None
    layers_path = infer_decoder_layers_path(model)
    if not layers_path:
        return None
    block_container = dict(model.named_modules(remove_duplicate=False)).get(layers_path)
    if not isinstance(block_container, nn.ModuleList) or len(block_container) == 0:
        return None
    block_bytes = [_module_bytes(block) for block in block_container]
    # Clamped because a tensor shared across blocks is counted once in the model total but once
    # per block in the sum, which would otherwise drive the remainder negative.
    non_block_bytes = max(0, _module_bytes(model) - sum(block_bytes))
    return max(block_bytes), non_block_bytes


def get_per_block_calib_batch_size(
    num_calib_data: int,
    seq_len: int,
    device: torch.device | str | None = None,
    model: nn.Module | None = None,
    n_gpu_resident_blocks: int = 0,
) -> int:
    """Return the largest per_block calibration batch whose activations fit in the reserved budget.

    per_block streams every decoder block CPU->GPU around each forward, so a larger batch means
    fewer transfers but more activation memory. Pinning the batch to *num_calib_data* ignores
    *seq_len* and runs out of memory on long sequences, so derive it from the budget instead.

    When *device* is CUDA and *model*'s decoder block stack can be located, the budget is derived
    from that device's actual free memory (``torch.cuda.mem_get_info``) minus everything
    ``per_block_runner.prepare()`` will pin there afterwards -- the non-block modules (embeddings,
    final norm, lm_head), the ``--gpu_resident_blocks`` permanently resident decoder blocks, and
    headroom for the one block streamed in during the forward -- instead of the static MI350-shaped
    estimate, so it scales with the card and with whatever else is already using it. Falls back to
    the static estimate otherwise, so behavior is unchanged for non-CUDA runs or callers that don't
    pass *model*.

    :param int num_calib_data: Total number of calibration samples; the batch cannot exceed it.
    :param int seq_len: Sequence length of each calibration sample.
    :param device: Device per_block calibration streams decoder blocks onto.
    :param model: Model being calibrated; used to estimate what stays resident on *device*.
    :param int n_gpu_resident_blocks: Decoder blocks kept permanently resident on *device*.
    :return: Batch size in ``[1, num_calib_data]``.
    """
    reserved_bytes = _CALIB_RESERVED_GIB * _GIB
    torch_device = torch.device(device) if device is not None else None
    resident = _per_block_resident_bytes(model) if model is not None else None
    if torch_device is not None and torch_device.type == "cuda" and resident is not None:
        block_bytes, non_block_bytes = resident
        free_bytes, _ = torch.cuda.mem_get_info(torch_device)
        resident_bytes = non_block_bytes + n_gpu_resident_blocks * block_bytes
        # The block streamed in for the current forward is also resident; double it for the
        # fake-quantized weight copy and fragmentation, the same margin the static estimate used.
        reserved_bytes = max(0, free_bytes - resident_bytes - 2 * block_bytes)
    max_tokens = reserved_bytes // (_ACT_PER_TOKEN_KIB * _KIB)
    return max(1, min(num_calib_data, max_tokens // seq_len))


def move_model_to_device_if_it_fits(model: nn.Module, device: torch.device) -> nn.Module:
    """Move *model* to *device* when its weights fit there, otherwise leave it where it is.

    per_block finalizes the model onto CPU, but evaluation runs on a single device, so the whole
    model has to be resident. That is only possible when it fits: weights are fake-quantized in the
    original dtype unless native inference is enabled, so quantization does not shrink them.

    :param nn.Module model: Model to move.
    :param torch.device device: Target device.
    :return: The model, moved only if the weights fit in the device's free memory.
    """
    current_device = getattr(model, "device", None)
    if current_device is not None and current_device.type == device.type:
        if device.index is None or current_device.index == device.index:
            return model

    if device.type != "cuda":
        return model.to(device)

    model_bytes = sum(t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers()))
    free_bytes, _ = torch.cuda.mem_get_info(device)
    # Keep 10% of the free memory for evaluation activations and the KV cache.
    if model_bytes >= free_bytes * 0.9:
        logger.warning(
            f"Keeping the model on CPU: it needs {model_bytes / _GIB:.1f} GiB but {device} only has "
            f"{free_bytes / _GIB:.1f} GiB free. Evaluation will be slow, and kernels that require "
            f"accelerator tensors will fail. Evaluate the exported checkpoint separately instead."
        )
        return model
    logger.info(f"Moving model to {device}.")
    return model.to(device)


def get_model(
    ckpt_path: str,
    data_type: str = "auto",
    device: str = "cuda",
    multi_gpu: str | bool | None = False,
    multi_device: bool = False,
    attn_implementation: str = "eager",
    trust_remote_code: bool = True,
) -> tuple[nn.Module, torch.dtype | None]:
    if data_type == "float16":
        model_dtype = torch.float16
    elif data_type == "bfloat16":
        model_dtype = torch.bfloat16
    elif data_type == "float32":
        model_dtype = torch.float32
    elif data_type == "auto":
        model_dtype = data_type
    else:
        raise ValueError(f"Unsupported data_type={data_type}. Expected one of: float16, bfloat16, float32, auto.")
    config = AutoConfig.from_pretrained(
        ckpt_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
    )

    max_memory = None
    device_map: str | dict[str, int | str] = device
    if multi_device:
        device_map = "auto"
        max_memory = get_device_max_memory()
    if multi_gpu:
        if multi_gpu == "balanced":
            device_map = create_auto_adjusted_device_map(config)
        else:
            device_map = "auto"
            if max_memory is None:
                # Leave real headroom per device (get_device_max_memory reserves
                # ~50% of GPU0, ~87.5% of others) so accelerate cannot greedily
                # cram the whole model onto one device when every visible GPU
                # reports similar free memory (no naturally-constrained device
                # to force a split).
                max_memory = get_device_max_memory()

    # TODO: Remove `_load_from_compressed_tensors` once native Transformers/compressed-tensors/Kimi compatibility is stable and fast.
    # TODO: Remove all `  # pragma: no cover` in this file once once test/test_for_torch/test_inverse_quantizer.py's test_kimi_k25_quantize_export and test_kimi_k25_nvfp4_quantization_and_export are adapted to support transformers>=5.0 which is used in PR CIs.
    # NOTE: As of `compressed-tensors==0.14.0.1`, compressed-tensors model loading was broken using `transformers==4.57.6` along Kimi models (AttributeError: 'CompressedLinear' object has no attribute 'weight' bug, reference: https://huggingface.co/moonshotai/Kimi-K2.5/discussions/39).
    # `compressed-tensors==0.15` + https://huggingface.co/moonshotai/Kimi-K2.6/discussions/22 fixed the loading issue, but native `AutoModelForCausalLM.from_pretrained` is stillsignificantly than this logic to load INT4 compressed-tensors models.
    if _is_compressed_tensors_model(config):  # pragma: no cover
        model = _load_from_compressed_tensors(
            model_dir=ckpt_path,
            config=config,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=trust_remote_code,
        )
        model.config._name_or_path = ckpt_path
        model.eval()
        if data_type == "auto" and not _is_meta_device_map(device_map):
            _restore_fp32_params_from_source(model, ckpt_path)
        return model, None

    # ``--multi_gpu`` and ``--multi_device`` both produce multi-GPU placement; treat
    # either as triggering the huge-MoE HF loader throttle (avoids HIP allocator
    # deadlock / post-load caching_allocator_warmup OOM on GPU 0).
    is_multi_gpu = bool(multi_gpu) or multi_device
    _hf_loader_saved = _set_hf_loader_workers(_hf_loader_target_workers(is_multi_gpu, config))
    load_kwargs = {
        "torch_dtype": model_dtype,
        "device_map": device_map,
        "max_memory": max_memory,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
    }
    try:
        if config.model_type == "mllama":
            model = MllamaForConditionalGeneration.from_pretrained(ckpt_path, **load_kwargs)  # type: ignore[no-untyped-call]
        elif config.model_type == "llama4":
            model = Llama4ForConditionalGeneration.from_pretrained(ckpt_path, **load_kwargs)  # type: ignore[no-untyped-call]
        elif config.model_type == "gpt_oss":
            quantization_config = Mxfp4Config(dequantize=True)  # type: ignore[misc]
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, quantization_config=quantization_config, **load_kwargs
            )  # type: ignore[no-untyped-call]
        elif config.model_type == "minimax_m3_vl":
            model = AutoModelForImageTextToText.from_pretrained(ckpt_path, **load_kwargs)  # type: ignore[no-untyped-call]
            if (
                trust_remote_code
                and hasattr(model, "register_for_auto_class")
                and hasattr(config, "auto_map")
                and "AutoModelForCausalLM" in config.auto_map
            ):
                model.register_for_auto_class("AutoModelForCausalLM")  # type: ignore[no-untyped-call]
        elif config.model_type == "muse_glimmer":
            # VLM wrapper, not registered under AutoModelForCausalLM.
            model = AutoModelForImageTextToText.from_pretrained(ckpt_path, **load_kwargs)  # type: ignore[no-untyped-call]
        elif config.model_type == "qwen3_vl_moe":
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(  # type: ignore[misc]
                ckpt_path, **load_kwargs
            )  # type: ignore[no-untyped-call]
        elif config.model_type == "qwen3_5_moe":
            model = Qwen3_5MoeForConditionalGeneration.from_pretrained(  # type: ignore[misc]
                ckpt_path, **load_kwargs
            )  # type: ignore[no-untyped-call]
        elif config.model_type == "qwen3_5":
            model = Qwen3_5ForConditionalGeneration.from_pretrained(  # type: ignore[misc]
                ckpt_path, **load_kwargs
            )  # type: ignore[no-untyped-call]
        elif config.model_type == "qwen4_exp":
            # The composite (vision + text) wrapper. AutoModelForCausalLM would resolve the nested
            # text config and build the text-only model, silently dropping the vision tower from
            # the exported checkpoint; the text-only path is reached via model_type
            # "qwen4_exp_text" instead.
            model = Qwen4ExpForConditionalGeneration.from_pretrained(  # type: ignore[misc]
                ckpt_path, **load_kwargs
            )  # type: ignore[no-untyped-call]
        elif config.model_type == "deepseek_vl_v2":
            model = AutoModel.from_pretrained(ckpt_path, **load_kwargs)  # type: ignore[no-untyped-call]
        elif config.model_type == "deepseek_v4":
            from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

            # vLLM registers a different config under the same model_type.
            # Transformers' model needs its own complete architecture config.
            config = DeepseekV4Config.from_pretrained(ckpt_path)
            model = AutoModelForCausalLM.from_pretrained(ckpt_path, config=config, **load_kwargs)
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(ckpt_path, **load_kwargs)  # type: ignore[no-untyped-call]
            except Exception:
                # Some models / transformers versions do not accept attn_implementation.
                logger.exception(
                    "AutoModelForCausalLM.from_pretrained failed with attn_implementation=%s, retrying without it.",
                    attn_implementation,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path,
                    device_map=device_map,
                    torch_dtype=model_dtype,
                    max_memory=max_memory,
                    trust_remote_code=trust_remote_code,
                )  # type: ignore[no-untyped-call]
            if (
                trust_remote_code
                and hasattr(model, "register_for_auto_class")
                and hasattr(config, "auto_map")
                and "AutoModelForCausalLM" in config.auto_map
            ):
                model.register_for_auto_class("AutoModelForCausalLM")  # type: ignore[no-untyped-call]
    except Exception:
        # Provide actionable guidance for model loading failures
        version_hint = getattr(config, "transformers_version", None)
        error_msg = "Failed to load model. Suggested resolutions:\n"
        if version_hint:
            error_msg += (
                f"  1. Install a compatible Transformers version: "
                f"pip install transformers=={version_hint} (as specified in model's config.json)\n"
            )
        else:
            error_msg += "  1. Install a Transformers version compatible with this model\n"
        error_msg += (
            "  2. Implement custom model loading instead of using `get_model()`. "
            "Refer to Transformers documentation for model-specific loading procedures."
        )
        logger.exception(error_msg)
        raise
    finally:
        _restore_hf_loader_workers(_hf_loader_saved)
    if multi_device and hasattr(model, "hf_device_map"):
        logger.info(f"hf_device_map: {model.hf_device_map}")
    # For certain models, the attribute model.config._name_or_path is an empty string; enforce the setting here.
    model.config._name_or_path = ckpt_path

    model.eval()  # type: ignore[no-untyped-call]
    model_dtype = next(model.parameters()).dtype

    if data_type == "auto" and not _is_meta_device_map(device_map):
        _restore_fp32_params_from_source(model, ckpt_path)

    return model, model_dtype


def _restore_fp32_params_from_source(model: "nn.Module", ckpt_path: str) -> None:
    """
    Restore parameters/buffers that are float32 in the source checkpoint but were downcast
    during loading (e.g. Kimi-K2.5 missing _keep_in_fp32_modules = ["MoEGate"] causes
    e_score_correction_bias to be silently cast to bfloat16 with torch_dtype="auto").

    Reads only safetensors headers (via lazy tensor slices) to detect mismatches; full
    tensor data is loaded only for the affected parameters.
    """
    if not is_safetensors_available():
        return

    # Resolve ckpt_path: local directory first, then HuggingFace cache.
    source_dir = Path(ckpt_path)
    if not source_dir.exists():
        if not is_transformers_available():
            return
        try:
            from transformers.utils import cached_file  # type: ignore[attr-defined]

            local_config = cached_file(ckpt_path, "config.json", local_files_only=True)
            source_dir = Path(local_config).parent
        except Exception:
            # Best-effort restore: e.g. the model wasn't loaded from a locally cached HF repo.
            # Not finding a source checkpoint to compare against is not an error condition.
            logger.info(
                "[restore_fp32_params] Could not resolve source dir for %s, skipping.",
                ckpt_path,
                allow_duplicate=False,
            )
            return

    index_path = source_dir / "model.safetensors.index.json"
    single_path = source_dir / "model.safetensors"
    if index_path.exists():
        with open(index_path) as f:
            shard_names: list[str] = list(set(json.load(f)["weight_map"].values()))
    elif single_path.exists():
        shard_names = ["model.safetensors"]
    else:
        # E.g. the checkpoint is stored as pytorch_model.bin instead of safetensors.
        logger.info(
            "[restore_fp32_params] No safetensors files found under %s, skipping.",
            source_dir,
            allow_duplicate=False,
        )
        return

    # Collect float32 keys from source checkpoint headers (no tensor data loaded).
    fp32_keys_to_shard_name: dict[str, Path] = {}
    for shard_name in shard_names:
        shard_path = source_dir / shard_name
        try:
            with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if f.get_slice(key).get_dtype() == "F32":
                        fp32_keys_to_shard_name[key] = shard_path
        except (OSError, safetensors.SafetensorError):
            logger.warning("[restore_fp32_params] Could not read header from %s, skipping.", shard_path)
            continue

    model_params = dict(model.named_parameters())
    model_params.update(model.named_buffers())
    downcast_keys = [k for k in fp32_keys_to_shard_name if k in model_params and model_params[k].dtype != torch.float32]
    if not downcast_keys:
        return

    logger.warning(
        "[restore_fp32_params] %d parameter(s) are float32 in source but downcast during loading "
        "(likely missing _keep_in_fp32_modules). Restoring: %s%s",
        len(downcast_keys),
        downcast_keys[:10],
        " ..." if len(downcast_keys) > 10 else "",
    )
    keys_by_shard: dict[Path, list[str]] = defaultdict(list)
    for key in downcast_keys:
        keys_by_shard[fp32_keys_to_shard_name[key]].append(key)
    for shard_path, keys in keys_by_shard.items():
        with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys:
                param = model_params[key]
                # Use .data = to preserve float32 dtype; .copy_() would cast to the existing bfloat16.
                param.data = f.get_tensor(key).to(param.device)


def create_model_skeleton(
    model_dir: str,
    trust_remote_code: bool = False,
    attn_implementation: str = "eager",
) -> nn.Module:
    """
    Create an empty model skeleton from the config of the Transformers model in ``model_dir``.

    Parameters are placed on meta device (zero memory), while buffers (e.g. rotary
    embeddings) are correctly computed on CPU via ``init_empty_weights(include_buffers=False)`` default.

    :param str model_dir: Directory containing ``config.json``.
    :param bool trust_remote_code: Whether to trust remote code for custom models.
    :param str attn_implementation: Attention implementation to use (e.g. ``"eager"``, ``"sdpa"``).
    :return: A model on meta device with no weights loaded.
    """
    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    return build_skeleton_from_config(config, trust_remote_code=trust_remote_code)


# TODO: test this function in CI.
def save_model(model: nn.Module, tokenizer: "PreTrainedTokenizerBase | None", save_dir: str) -> None:
    model.save_pretrained(save_dir, safe_serialization=True)  # type: ignore[attr-defined]

    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)  # type: ignore[attr-defined]
    else:  # pragma: no cover
        name_or_path = get_source_name_or_path(model)
        if name_or_path:
            maybe_save_preprocessors(name_or_path, save_dir)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device_max_memory() -> dict[int | str, str]:
    max_memory: dict[int | str, str] = {}
    for i in range(torch.cuda.device_count()):
        _ = torch.tensor([0], device=i)
        cuda_avail_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}
        cpu_avail_memory = psutil.virtual_memory().available
        for cuda_num, cuda_memory in cuda_avail_memory.items():
            cuda_memory_gb = cuda_memory / (10**9)
            logger.info(f"GPU{cuda_num} cuda_avail_memory: {cuda_memory_gb:.1f}GB")
            if cuda_num == 0:
                # The ratio is an experience value that you can manually adjust yourself.
                gpu0_ratio = 0.5 if cuda_memory_gb > 30 else 0.3
                max_memory[cuda_num] = f"{cuda_memory_gb * gpu0_ratio:.1f}GB"
            else:
                other_ratio = 0.875 if cuda_memory_gb > 30 else 0.7
                max_memory[cuda_num] = f"{cuda_memory_gb * other_ratio:.1f}GB"
        logger.info(f"cpu_avail_memory: {cpu_avail_memory / (10**9):.1f}GB")
        cpu_ratio = 0.875
        max_memory["cpu"] = f"{cpu_avail_memory / (10**9) * cpu_ratio:.1f}GB"
        logger.info(f"final_use_model_kwargs: {max_memory}")
        # max_memory =  {0: '0.1GB', 'cpu': '100GB'}

    return max_memory
