#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import random

import numpy as np
from tqdm import tqdm

from quark.shares.utils.import_utils import (
    is_psutil_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)

if is_psutil_available():
    import psutil  # type: ignore[import-untyped]

if is_torch_available():
    import torch
    import torch.nn as nn

if is_transformers_available():
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Llama4ForCausalLM,
        Llama4ForConditionalGeneration,
        MllamaForConditionalGeneration,
    )
    from transformers.models.dbrx.modeling_dbrx import DbrxExperts, DbrxForCausalLM
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe


if is_transformers_available() and is_transformers_version_higher_or_equal("4.55.1"):
    from transformers import Mxfp4Config  # type: ignore[attr-defined]
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssTopKRouter
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import GraniteMoeHybridMoE

if is_transformers_available() and is_transformers_version_higher_or_equal("4.57.0"):
    from transformers import Qwen3VLMoeForConditionalGeneration  # type: ignore[attr-defined]
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts


from ..torch_utils import setattr_recursive
from .module_replacement.dbrx_expert import DbrxExperts_
from .module_replacement.replacement_utils import (
    replace_gptoss_experts_with_linear,
    replace_gptoss_topkrouter_with_linear,
    replace_granite_moe_experts_with_linear,
    replace_llama4_experts_with_sequential,
    replace_qwen3vlmoe_experts_with_linear,
)


def get_tokenizer(
    ckpt_path: str, max_seq_len: int = 2048, model_type: str | None = None, trust_remote_code: bool = True
) -> AutoTokenizer:
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, padding_side="left", trust_remote_code=trust_remote_code)  # type: ignore[no-untyped-call]
    if model_type and model_type in ["qwen", "qwen2"]:
        # qwen2 use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    return tokenizer


# TODO: we should implement a proper model patcher in quark namespace, well tested, with well supported cases / modularity. This if/else design is not tractable.
def prepare_for_moe_quant(model: nn.Module) -> None:
    if isinstance(model, (DbrxForCausalLM, Llama4ForConditionalGeneration, Llama4ForCausalLM)):
        for name, module in model.named_modules(remove_duplicate=False):
            if isinstance(module, DbrxExperts):
                new_experts = DbrxExperts_.from_float(module)
                setattr_recursive(model, name, new_experts)
                print(f"Module {name} has been replaced")
            elif isinstance(module, Llama4TextMoe):
                replace_llama4_experts_with_sequential(module, model.config.text_config)
    elif model.config.model_type == "gpt_oss":
        for name, module in tqdm(
            model.named_modules(remove_duplicate=False),
            desc="Replacing GptOssExperts implementation to use torch.nn.Linear for quantization",
        ):
            if isinstance(module, GptOssExperts):
                replace_gptoss_experts_with_linear(experts_module=module)

            if isinstance(module, GptOssTopKRouter):
                replace_gptoss_topkrouter_with_linear(router=module)
    elif model.config.model_type == "granitemoehybrid":
        for name, module in model.named_modules(remove_duplicate=False):
            if isinstance(module, GraniteMoeHybridMoE):
                replace_granite_moe_experts_with_linear(module)
    elif model.config.model_type == "qwen3_vl_moe":
        for name, module in model.named_modules(remove_duplicate=False):
            if isinstance(module, Qwen3VLMoeTextExperts):
                replace_qwen3vlmoe_experts_with_linear(module)


def revert_model_patching(model: nn.Module) -> None:
    if model.config.model_type == "gpt_oss":
        for name, module in model.named_modules(remove_duplicate=False):
            if isinstance(module, GptOssTopKRouter):
                if not hasattr(module, "linear"):
                    # gpt_oss router should normally be replaced by an nn.Linear, e.g. to quantize it or to apply offline rotations on it. However, it is not strictly required.
                    continue
                else:
                    if not isinstance(module.linear, nn.Linear):
                        # Case where the router is quantized.
                        # We leave the state_dict as `router.linear.weight`, `router.linear.bias`.
                        continue

                weight = module.linear.weight
                bias = module.linear.bias
                device = module.linear.weight.device

                with torch.device(device):
                    router = GptOssTopKRouter(model.config)  # type: ignore[no-untyped-call]
                    router.weight = weight
                    router.bias = bias

                setattr_recursive(model, name, router)


def get_model(
    ckpt_path: str,
    data_type: str = "auto",
    device: str = "cuda",
    multi_gpu: bool = False,
    multi_device: bool = False,
    attn_implementation: str = "eager",
    trust_remote_code: bool = True,
) -> tuple[nn.Module, torch.dtype]:
    if data_type == "float16":
        model_dtype = torch.float16
    elif data_type == "bfloat16":
        model_dtype = torch.bfloat16
    elif data_type == "float32":
        model_dtype = torch.float32
    elif data_type == "auto":
        model_dtype = data_type
    else:
        raise ValueError(f"{data_type} not support for current model")
    config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=trust_remote_code)
    max_memory = None
    if multi_device:
        device = "auto"
        max_memory = get_device_max_memory()
    if multi_gpu:
        device = "auto"
    if config.model_type == "mllama":
        model = MllamaForConditionalGeneration.from_pretrained(
            ckpt_path,
            device_map=device,
            torch_dtype=model_dtype,
            max_memory=max_memory,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )  # type: ignore[no-untyped-call]
    elif config.model_type == "llama4":
        model = Llama4ForConditionalGeneration.from_pretrained(
            ckpt_path,
            device_map=device,
            torch_dtype=model_dtype,
            max_memory=max_memory,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )  # type: ignore[no-untyped-call]
    elif config.model_type == "gpt_oss":
        quantization_config = Mxfp4Config(dequantize=True)  # type: ignore[misc]
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            device_map=device,
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
            device_map=device,
            torch_dtype=model_dtype,
            max_memory=max_memory,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )  # type: ignore[no-untyped-call]
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map=device,
                torch_dtype=model_dtype,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )  # type: ignore[no-untyped-call]
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                device_map=device,
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
    if multi_device and hasattr(model, "hf_device_map"):
        print("device_map:", model.hf_device_map)
    # For certain models, the attribute model.config._name_or_path is an empty string; enforce the setting here.
    model.config._name_or_path = ckpt_path

    model.eval()
    model_dtype = next(model.parameters()).dtype

    return model, model_dtype


def save_model(model: nn.Module, tokenizer: AutoTokenizer | None, save_dir: str) -> None:
    model.save_pretrained(save_dir, safe_serialization=True)  # type: ignore[attr-defined]
    if tokenizer is None and getattr(model.config, "_name_or_path", None):  # type: ignore[attr-defined]
        try:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True)  # type: ignore[attr-defined,no-untyped-call]
            print(f"Save the tokenizer from pretrained: {model.config._name_or_path}")  # type: ignore[attr-defined]
        except Exception as e:
            print(f"An error occurred when loading tokenizer: {e}")
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)  # type: ignore[attr-defined]


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device_max_memory() -> dict[int | str, str]:
    max_memory: dict[int | str, str] = {}
    for i in range(torch.cuda.device_count()):
        _ = torch.tensor([0], device=i)
        cuda_avail_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}
        cpu_avail_memory = psutil.virtual_memory().available
        for cuda_num, cuda_memory in cuda_avail_memory.items():
            cuda_memory_gb = cuda_memory / (10**9)
            print(f"GPU{cuda_num} cuda_avail_memory: {cuda_memory_gb:.1f}GB")
            if cuda_num == 0:
                # The ratio is an experience value that you can manually adjust yourself.
                gpu0_ratio = 0.5 if cuda_memory_gb > 30 else 0.3
                max_memory[cuda_num] = f"{cuda_memory_gb * gpu0_ratio:.1f}GB"
            else:
                other_ratio = 0.875 if cuda_memory_gb > 30 else 0.7
                max_memory[cuda_num] = f"{cuda_memory_gb * other_ratio:.1f}GB"
        print(f"cpu_avail_memory: {cpu_avail_memory / (10**9):.1f}GB")
        cpu_ratio = 0.875
        max_memory["cpu"] = f"{cpu_avail_memory / (10**9) * cpu_ratio:.1f}GB"
        print("final_use_model_kwargs: ", max_memory)
        # max_memory =  {0: '0.1GB', 'cpu': '100GB'}

    return max_memory
