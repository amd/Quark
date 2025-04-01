#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import fnmatch
from typing import List, Optional, Dict, Any
import torch


def preprocess_import_info(model_state_dict: Dict[str,
                                                  torch.Tensor], is_kv_cache: bool, kv_layers_name: Optional[List[str]],
                           custom_mode: str) -> tuple[Dict[str, Any], bool, Optional[List[str]]]:
    '''
    Load model weights, preprocess state_dict for some cases such as dbrx split, fp8 kv_cache, tied_parameter, etc.
    '''
    # for dbrx
    dbrx_experts_groups: List[List[str]] = []
    dbrx_params_name = [["*ffn.experts.mlp.v1_weight", "*ffn.experts.mlp.v1_weight_scale"],
                        ["*ffn.experts.mlp.w1_weight", "*ffn.experts.mlp.w1_weight_scale"],
                        ["*ffn.experts.mlp.w2_weight", "*ffn.experts.mlp.w2_weight_scale"]]

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
def split_params_for_DbrxExperts(model_state_dict: Dict[str, Any], dbrx_experts_groups: List[List[str]]) -> None:
    '''
    The moe part of dbrx needs special treatment, when loading a model, we do some splitting of that model, so the tensor that is loaded in here, needs to be split as well
    '''
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


def find_patterns_groups(patterns: Optional[List[List[str]]], layer_names: List[str]) -> List[List[str]]:
    pattern_groups: List[List[str]] = []
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
