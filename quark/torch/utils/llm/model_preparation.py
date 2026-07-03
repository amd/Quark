#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import random
import sys
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from quark.common.utils.import_utils import (
    is_psutil_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)
from quark.torch.integrations.compressed_tensors.loading import (
    _is_compressed_tensors_model,
    _load_from_compressed_tensors,
)

if is_psutil_available():
    import psutil  # type: ignore[import-untyped]

if is_torch_available():
    import torch
    import torch.nn as nn

if is_transformers_available():
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
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

if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # type: ignore[attr-defined]
        Qwen3MoeExperts,
        Qwen3MoeSparseMoeBlock,
    )
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
    from transformers import Qwen3_5MoeForConditionalGeneration

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
    replace_qwen3_moe_experts_with_linear,
    replace_qwen3moe_sparse_moe_block_with_linear_gate,
    replace_qwen3vlmoe_experts_with_linear,
)

logger = ScreenLogger(__name__)


# Throttle the HF loader once the model is estimated to produce more quantized
# linears than this. Calibrated so GLM-5 / GLM-4.7 / Qwen3.5-397B trigger,
# Qwen3-30B / Mixtral-8x7B do not.
_HUGE_MOE_LINEAR_THRESHOLD = 30000
_VM_MAX_MAP_COUNT_SAFE = 1048576
_VM_MAX_MAP_COUNT_RECOMMENDED = 4194304


def _estimate_quantized_linears(config: object | None) -> int:
    """Rough count of nn.Linear layers Quark will wrap, derived from HF config.

    Used as a size signal for the HF loader throttle. Probes the common alias
    set (``n_routed_experts`` / ``num_local_experts`` / ``num_experts``) and
    unwraps one level of ``text_config`` for VLM/multimodal wrappers.
    Returns 0 if the config doesn't expose ``num_hidden_layers``.
    """
    if config is None:
        return 0
    text_config = getattr(config, "text_config", None) or config
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        return 0
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
    elif model.config.model_type == "qwen3_moe":
        # transformers<5 used split experts and nn.Linear gate - no need to patch in this case.
        if is_transformers_version_higher_or_equal("5.0.0"):
            for name, module in tqdm(
                model.named_modules(remove_duplicate=False), desc="Replacing Qwen3MoeExperts to use nn.Linear"
            ):
                if isinstance(module, Qwen3MoeExperts):
                    replace_qwen3_moe_experts_with_linear(module, reload=reload)

                # NOTE: MOE router patching is useful only in case the router is quantized,
                # or in case we use rotation.
                if isinstance(module, Qwen3MoeSparseMoeBlock):
                    replace_qwen3moe_sparse_moe_block_with_linear_gate(moe_block=module)


def _prepare_for_moe_quant(model: nn.Module, reload: bool = False) -> None:
    """
    Internal: Traverse named_modules, when PREPROCESS_REGISTRY is hit, replace module via QuarkXxx.from_hf.
    """
    config = getattr(model, "config", None)
    if config is None:
        logger.warning("Model has no config; skipping MoE preprocess")
        return
    model_type = config.model_type

    if model_type in ["qwen3_5"]:
        raise ValueError(
            f"The quantization of MoE layer for model architecture {model_type} is not yet supported in Quark."
        )

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
        return model, None

    # ``--multi_gpu`` and ``--multi_device`` both produce multi-GPU placement; treat
    # either as triggering the huge-MoE HF loader throttle (avoids HIP allocator
    # deadlock / post-load caching_allocator_warmup OOM on GPU 0).
    is_multi_gpu = bool(multi_gpu) or multi_device
    _hf_loader_saved = _set_hf_loader_workers(_hf_loader_target_workers(is_multi_gpu, config))
    try:
        if config.model_type == "mllama":
            model = MllamaForConditionalGeneration.from_pretrained(
                ckpt_path,
                device_map=device_map,
                torch_dtype=model_dtype,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )  # type: ignore[no-untyped-call]
        elif config.model_type == "llama4":
            model = Llama4ForConditionalGeneration.from_pretrained(
                ckpt_path,
                device_map=device_map,
                torch_dtype=model_dtype,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )  # type: ignore[no-untyped-call]
        elif config.model_type == "gpt_oss":
            quantization_config = Mxfp4Config(dequantize=True)  # type: ignore[misc]
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map=device_map,
                torch_dtype=model_dtype,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
            )  # type: ignore[no-untyped-call]
            if (
                trust_remote_code
                and hasattr(model, "register_for_auto_class")
                and hasattr(config, "auto_map")
                and "AutoModelForCausalLM" in config.auto_map
            ):
                model.register_for_auto_class("AutoModelForCausalLM")  # type: ignore[no-untyped-call]
        elif config.model_type == "qwen3_vl_moe":
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(  # type: ignore[misc]
                ckpt_path,
                device_map=device_map,
                torch_dtype=model_dtype,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )  # type: ignore[no-untyped-call]
        elif config.model_type == "qwen3_5_moe":
            model = Qwen3_5MoeForConditionalGeneration.from_pretrained(  # type: ignore[misc]
                ckpt_path,
                device_map=device_map,
                torch_dtype=model_dtype,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )  # type: ignore[no-untyped-call]
        elif config.model_type == "deepseek_vl_v2":
            model = AutoModel.from_pretrained(
                ckpt_path,
                device_map=device_map,
                torch_dtype=model_dtype,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
                use_safetensors=True,
            )  # type: ignore[no-untyped-call]
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path,
                    device_map=device_map,
                    torch_dtype=model_dtype,
                    max_memory=max_memory,
                    trust_remote_code=trust_remote_code,
                    attn_implementation=attn_implementation,
                )  # type: ignore[no-untyped-call]
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

    return model, model_dtype


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


def save_model(model: nn.Module, tokenizer: "PreTrainedTokenizerBase | None", save_dir: str) -> None:
    model.save_pretrained(save_dir, safe_serialization=True)  # type: ignore[attr-defined]
    if tokenizer is None and getattr(model.config, "_name_or_path", None):  # type: ignore[attr-defined]
        try:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True)  # type: ignore
            logger.info(f"Saving tokenizer from pretrained: {model.config._name_or_path}")  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Failed to load tokenizer for saving.")
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)  # type: ignore[attr-defined]


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
