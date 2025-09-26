#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import fnmatch
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate, distribute_tensor  # type: ignore[attr-defined]
from tqdm import tqdm

from quark.shares.utils.import_utils import is_accelerate_available
from quark.shares.utils.log import ScreenLogger
from quark.torch.export.main_export.quant_config_parser import QuantConfigParser, get_layer_quant_config
from quark.torch.export.main_import.pretrained_config import PretrainedConfig
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.export.nn.modules.realquantizer import get_real_quantizer
from quark.torch.quantization.config.config import Config
from quark.torch.quantization.config.type import QSchemeType
from quark.torch.quantization.model_transformation import prepare_for_attention_quant
from quark.torch.quantization.nn.modules import QuantLinear
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, NonScaledFakeQuantize, ScaledFakeQuantize
from quark.torch.utils import getattr_recursive, setattr_recursive
from quark.torch.utils.device import TPDeviceManager, e4m3fn_to_e4m3fnuz

if is_accelerate_available():
    from accelerate.utils.modeling import find_tied_parameters, named_module_tensors

logger = ScreenLogger(__name__)

__all__ = [
    "preprocess_import_info",
    "split_params_for_DbrxExperts",
    "find_patterns_groups",
    "_build_quantized_model",
    "_untie_parameters",
    "_convert_quantized_model",
    "_handle_multi_device_loading",
    "_convert_e4m3fn_to_e4m3fnuz",
]


def preprocess_import_info(
    model_state_dict: dict[str, torch.Tensor], is_kv_cache: bool, kv_layers_name: list[str] | None, custom_mode: str
) -> tuple[dict[str, Any], bool, list[str] | None]:
    """
    Load model weights, preprocess state_dict for some cases such as dbrx split, fp8 kv_cache, tied_parameter, etc.
    """
    # for dbrx
    dbrx_experts_groups: list[list[str]] = []
    dbrx_params_name = [
        ["*ffn.experts.mlp.v1_weight", "*ffn.experts.mlp.v1_weight_scale"],
        ["*ffn.experts.mlp.w1_weight", "*ffn.experts.mlp.w1_weight_scale"],
        ["*ffn.experts.mlp.w2_weight", "*ffn.experts.mlp.w2_weight_scale"],
    ]

    params_name = list(model_state_dict.keys())
    dbrx_experts_groups = find_patterns_groups(dbrx_params_name, params_name)
    if dbrx_experts_groups is not None:
        split_params_for_DbrxExperts(model_state_dict, dbrx_experts_groups)

    # The weight of kv_scale is handled only if custom_mode = fp8
    if custom_mode == "fp8":
        if kv_layers_name is None:
            raise ValueError(
                "we need `kv_layers_name` to restore model_state_dict for reloading, but it is None, please offer it in config.json"
            )
        keys = list(model_state_dict.keys())
        for layer_name in keys:
            # kv_scale is same, only match k_scale
            if fnmatch.fnmatch(layer_name, "*.k_scale"):
                prefix = layer_name.split("k_scale")[0]
                for k_v_name in kv_layers_name:
                    full_scale_name = prefix + k_v_name.split("*")[-1] + ".output_scale"
                    model_state_dict[full_scale_name] = model_state_dict[layer_name]
                del model_state_dict[layer_name]
                del model_state_dict[prefix + "v_scale"]
                is_kv_cache = True
    return model_state_dict, is_kv_cache, kv_layers_name


# TODO: Override state_dict, load_state_dict of dbrx func
def split_params_for_DbrxExperts(model_state_dict: dict[str, Any], dbrx_experts_groups: list[list[str]]) -> None:
    """
    The moe part of dbrx needs special treatment, when loading a model, we do some splitting of that model, so the tensor that is loaded in here, needs to be split as well
    """
    params_name = list(model_state_dict.keys())
    for group in dbrx_experts_groups:
        for name in group:
            if "weight_scale" in name.split(".")[-1]:
                weight_scale_name = name
            else:
                weight_name = name
        mlp_suffix = weight_name.rsplit("_", 1)
        mlp_suffix[-1] = mlp_suffix[-1].replace("weight", "input_scale")
        input_scale_name = "_".join(mlp_suffix)
        input_scale_exist = True if input_scale_name in params_name else False

        mlp_suffix[-1] = mlp_suffix[-1].replace("input_scale", "output_scale")
        output_scale_name = "_".join(mlp_suffix)
        output_scale_exist = True if output_scale_name in params_name else False

        weight_tensor = model_state_dict[weight_name]
        weight_scale_tensor = model_state_dict[weight_scale_name]
        experts_num = weight_scale_tensor.shape[0]
        weight_chunk = torch.chunk(weight_tensor, experts_num)

        mlp_name = weight_name.split(".")[:-1]
        suffix_name = weight_name.split(".")[-1]
        param_name = suffix_name.split("_")[0]

        for i, item in enumerate(weight_chunk):
            weight_name_list = mlp_name + [str(i), param_name, "weight"]
            weight_scale_name_list = mlp_name + [str(i), param_name, "weight_scale"]

            weight_i_name = ".".join(weight_name_list)
            weight_scale_i_name = ".".join(weight_scale_name_list)

            model_state_dict[weight_scale_i_name] = weight_scale_tensor[i]
            if "w2" in suffix_name:
                model_state_dict[weight_i_name] = item.t().contiguous()
            else:
                model_state_dict[weight_i_name] = item

            if input_scale_exist:
                input_scale_name_list = mlp_name + [str(i), param_name, "input_scale"]
                input_scale_i_name = ".".join(input_scale_name_list)
                model_state_dict[input_scale_i_name] = model_state_dict[input_scale_name][i]

            if output_scale_exist:
                output_scale_name_list = mlp_name + [str(i), param_name, "output_scale"]
                output_scale_i_name = ".".join(output_scale_name_list)
                model_state_dict[output_scale_i_name] = model_state_dict[output_scale_name][i]

        model_state_dict.pop(weight_name)
        model_state_dict.pop(weight_scale_name)
        model_state_dict.pop(input_scale_name, None)
        model_state_dict.pop(output_scale_name, None)


def find_patterns_groups(patterns: list[list[str]] | None, layer_names: list[str]) -> list[list[str]]:
    pattern_groups: list[list[str]] = []
    if patterns is None:
        return pattern_groups
    for pattern in patterns:
        pattern0 = pattern[0]
        for key in layer_names:
            if fnmatch.fnmatch(key, pattern0):
                word0 = pattern0.replace("*", "")
                key_list = [key]
                for other in pattern[1:]:
                    other_word = other.replace("*", "")
                    other_key = key.replace(word0, other_word)
                    if other_key in layer_names:
                        key_list.append(other_key)
                if key_list and len(key_list) > 0:
                    pattern_groups.append(key_list)
    return pattern_groups


def _build_quantized_model(
    model: nn.Module, model_config: "PretrainedConfig", model_state_dict: dict[str, Any]
) -> nn.Module:
    """
    Build quantized model with proper module replacement.
    Equivalent to ModelImporter._build_model.
    """
    if model_config.quantization_config is None:
        logger.info("This is a non-quantized model")
        return model

    custom_mode = model_config.quantization_config["quant_method"]
    assert custom_mode in ["fp8", "awq", "quark"], f"Unsupported quantization method: {custom_mode}"

    is_kv_cache = False
    model_state_dict, is_kv_cache, kv_layers_name = preprocess_import_info(
        model_state_dict=model_state_dict,
        is_kv_cache=is_kv_cache,
        kv_layers_name=model_config.kv_layers_name,
        custom_mode=custom_mode,
    )

    # Parse quantization configuration
    if custom_mode != "quark":
        # For AWQ and FP8 custom modes
        is_bias_quantized = any("bias.scales" in key or "bias_scale" in key for key in model_state_dict.keys())
        quantization_config = QuantConfigParser.from_custom_config(
            model_config.quantization_config,
            is_bias_quantized=is_bias_quantized,
            is_kv_cache=is_kv_cache,
            kv_layers_name=kv_layers_name,
        )
    else:
        quantization_config = Config.from_dict(model_config.quantization_config)

    # Determine if using real quantized mode
    is_real_quantized_mode = model_config.weight_format != "fake_quantized"

    # Handle softmax quantization
    if quantization_config.softmax_quant_spec is not None:
        if is_real_quantized_mode:
            get_quantize = partial(
                get_real_quantizer, quantizer=None, reorder=False, real_quantized=False, float_dtype=torch.float32
            )
        else:
            get_quantize = FakeQuantizeBase.get_fake_quantize  # type: ignore
        prepare_for_attention_quant(model, quantization_config, get_quantize)

    logger.info("In-place OPs replacement start.")

    # Replace modules with quantized versions
    if is_real_quantized_mode:
        # TODO: we should not have circular imports.
        from quark.torch.export.api import _map_to_quark

        _map_to_quark(
            model,
            quantization_config,
            model_config.pack_method,  # type: ignore[arg-type]
            custom_mode,
        )
    else:
        # Handle fake quantization mode
        named_modules = dict(model.named_modules(remove_duplicate=False))
        for name, float_module in tqdm(named_modules.items()):
            layer_quantization_config = get_layer_quant_config(quantization_config, type(float_module), name)
            if layer_quantization_config is not None and isinstance(float_module, nn.Linear):
                # Initialize on proper device
                if float_module.weight.device.type == "meta":
                    device = torch.device("cpu")
                else:
                    device = float_module.weight.device

                quant_module = QuantLinear.from_float(float_module, layer_quantization_config, device=device)
                quant_module.register_buffer("export_enabled", torch.tensor([1], dtype=torch.uint8), persistent=False)
                setattr_recursive(model, name, quant_module)

        # Enable observers and fake quantization for dynamic quantization
        for name, module in model.named_modules():
            if isinstance(module, ScaledFakeQuantize):
                if module.is_dynamic and not (module.is_scale_quant and module.qscheme == QSchemeType.per_tensor):
                    module.enable_observer()
                    module.enable_fake_quant()
                else:
                    module.disable_observer()
                    module.enable_fake_quant()
            elif isinstance(module, NonScaledFakeQuantize):
                module.enable_fake_quant()

    logger.info("Converting quantized ops end")
    return model


def _untie_parameters(model: nn.Module, model_state_dict: dict[str, Any]) -> None:
    """
    Some parameters share weights, such as embedding and lm_head, and when exporting with `PretrainedModel.save_pretrained`
    only one of them will be exported, so need to copy the parameters.
    """
    # TODO: Only embedding for now, need to solve other cases, such as encoder-decoder tied
    tied_param_groups = find_tied_parameters(model)
    if len(tied_param_groups) > 0:
        if len(tied_param_groups) > 1 or "lm_head.weight" not in tied_param_groups[0]:
            raise ValueError(
                f"Your have tied_param_groups: {tied_param_groups}, temporarily does not support the case where tied_param is not 'lm_head and embedding'"
            )
        missing_key: list[str] = []
        tied_param_value: torch.Tensor | None = None
        for tied_param_name in tied_param_groups[0]:
            if tied_param_name in model_state_dict.keys():
                tied_param_value = model_state_dict[tied_param_name]
            else:
                missing_key.append(tied_param_name)
        if tied_param_value is not None:
            for tied_param_key in missing_key:
                model_state_dict[tied_param_key] = tied_param_value
        else:
            raise ValueError("Cannot assign a value to tied_params because tied_param_value is None")


def _convert_quantized_model(model: nn.Module, model_config: "PretrainedConfig") -> nn.Module:
    """
    Convert quantized model for specific formats.
    Equivalent to ModelImporter._convert_model.
    """
    if model_config.quantization_config is None:
        return model

    custom_mode = model_config.quantization_config["quant_method"]
    assert custom_mode in ["fp8", "awq", "quark"], f"Unsupported quantization method: {custom_mode}"

    if custom_mode != "fp8":
        return model

    is_real_quantized_mode = model_config.weight_format != "fake_quantized"
    if is_real_quantized_mode and torch.version.hip is not None:
        logger.info("In-place fp8 e4m3fn to e4m3fnuz conversion start.")
        _convert_e4m3fn_to_e4m3fnuz(model)

    return model


def _handle_multi_device_loading(model: nn.Module, checkpoint_weights: dict[str, torch.Tensor]) -> None:
    """
    Handle multi-device loading with accelerate hooks.
    """
    named_modules = dict(model.named_modules(remove_duplicate=False))
    for name, module in tqdm(named_modules.items()):
        # deivce must be meta and can only get the lowest granularity mods.
        if hasattr(module, "_hf_hook") and module._hf_hook.offload is True:
            hook = module._hf_hook
            weight_modules = dict(
                named_module_tensors(module, include_buffers=False, recurse=False, remove_non_persistent=True)
            )

            # If meta, send the value of checkpoint_weight to the hook's "weights_map" and convert the param to meta.
            # unquantized mods like "lm_head", "DeepseekV3RMSNorm" can be done directly like this.
            # Like qparamlinear, weight is handled the same way, but scale, zero should be sent directly to execution_device,
            # "weight_map" doesn't support increasing KV.
            prefix = hook.weights_map.prefix
            weight_keys = []
            for weight_name in weight_modules.keys():
                full_name = prefix + weight_name
                weight_keys.append(full_name)
                hook.weights_map[weight_name].data = checkpoint_weights[full_name]
                # can't del checkpoint_weights[full_name], should move to meta
                checkpoint_weights[full_name] = checkpoint_weights[full_name].to("meta")

            for checkpoint_weights_name in checkpoint_weights.keys():
                if checkpoint_weights_name.startswith(prefix):
                    if checkpoint_weights_name not in weight_keys:  # is scale or zero
                        # how to add kv into weights_map? For OffloadedWeightsLoader and PrefixedDataset, it is not possible to add a k and v.
                        # So scale and zero are buffers that we put directly on the execution_device, while weight is handled by the hook.
                        checkpoint_weights[checkpoint_weights_name] = checkpoint_weights[checkpoint_weights_name].to(
                            hook.execution_device
                        )
        torch.cuda.empty_cache()


def _convert_e4m3fn_to_e4m3fnuz(model: nn.Module) -> None:
    """
    Convert a model with QParamsLinear layers with fp8 weights to hip supported fp8 format.>

    Parameters:
        model (torch.nn.Module): An instance of the original not-quantized model. This model may be on `meta` device, or may have random weights.
    """
    if TPDeviceManager._tp_mesh is None:
        return

    named_modules = dict(model.named_modules(remove_duplicate=False))
    for module_name, float_module in tqdm(named_modules.items()):
        if isinstance(float_module, QParamsLinear):
            qparams_linear = getattr_recursive(model, module_name)
            # Use DTensor to speed up the conversion
            placements = [Replicate()]
            dweight = distribute_tensor(
                qparams_linear.weight.data.to(torch.float16),
                device_mesh=TPDeviceManager._tp_mesh,
                placements=placements,
            )
            dwscale = distribute_tensor(
                qparams_linear.weight_quantizer.scale.data, device_mesh=TPDeviceManager._tp_mesh, placements=placements
            )
            dweight, dwscale = e4m3fn_to_e4m3fnuz(dweight, dwscale)

            # Not always need to copy to CPU, if the GPU memory is enough, this step can be skip to save time.
            if type(dweight) == DTensor and type(dwscale) == DTensor:
                dweight = dweight.to_local().to("cpu")
                dwscale = dwscale.to_local().to("cpu")

            qparams_linear.weight = torch.nn.Parameter(dweight)
            qparams_linear.weight_quantizer.scale = torch.nn.Parameter(dwscale)
            setattr_recursive(model, module_name, qparams_linear)
