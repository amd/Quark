#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
import os
import argparse
import copy
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_utils.model_preparation import MODEL_NAME_KV_LAYERS_MAP, MODEL_NAME_EXCLUDE_LAYERS_MAP, MODEL_NAME_Q_LAYERS_MAP

from quark.torch.quantization import Config, QuantizationConfig, FP8E4M3PerTensorSpec, \
    Int4PerChannelSpec, Int4PerGroupSpec, Uint4PerGroupSpec, MXSpec, \
    load_pre_optimization_config_from_file, load_quant_algo_config_from_file, RotationConfig, AWQConfig, AutoSmoothQuantConfig
from quark.torch.export import ExporterConfig, JsonExporterConfig, OnnxExporterConfig

from customized_configuration import get_global_config, INT8_PER_TENSOR_SPEC, INT8_PER_TOKEN_DYNAMIC_SPEC, INT8_PER_TENSOR_DYNAMIC_SPEC, UINT4_PER_CHANNEL_ASYM_DYNAMIC_SPEC

'''
Instructions for Setting Up Quark Quantization Configuration:
Step 1: Configure `QuantizationSpec` for torch.Tensors. Specify attributes such as dtype, observer_method, etc.
Step 2: Establish `QuantizationConfig` for nn.Module. Define the QuantizationSpec of input_tensors, output_tensors, weight, and bias.
Step 3: Set up the overall `Config` for the model. This includes:
        - global_quant_config (required)
        - layer_type_quant_config
        - layer_quant_config
        - kv_cache_quant_config
        - exclude
        - pre_quant_opt_config
        - algo_config
        - quant_mode
'''

# Step 1: Configure `DataTypeSpec` for torch.Tensors. Specify attributes such as dtype, observer_method, etc. More customer settings refer customized_configuration.py
FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max",
                                           is_dynamic=False).to_quantization_spec()
FP8_PER_TENSOR_SPEC_DYNAMIC = FP8E4M3PerTensorSpec(observer_method="min_max",
                                                   is_dynamic=True).to_quantization_spec()
INT4_PER_CHANNEL_SPEC = Int4PerChannelSpec(symmetric=True,
                                           scale_type="float",
                                           round_method="half_even",
                                           ch_axis=0,
                                           is_dynamic=False).to_quantization_spec()
INT4_PER_GROUP_SYM_SPEC = Int4PerGroupSpec(symmetric=True,
                                           scale_type="float",
                                           round_method="half_even",
                                           ch_axis=1,
                                           is_dynamic=False,
                                           group_size=128).to_quantization_spec()
UINT4_PER_GROUP_ASYM_SPEC = Uint4PerGroupSpec(symmetric=False,
                                              scale_type="float",
                                              round_method="half_even",
                                              ch_axis=1,
                                              is_dynamic=False,
                                              group_size=128).to_quantization_spec()
MX_FP8_PER_GROUP_SYM_SPEC = MXSpec(mx_element_dtype="fp8_e4m3",
                                   ch_axis=-1,
                                   block_size=32,
                                   is_dynamic=True).to_quantization_spec()
MX_FP6E2M3_PER_GROUP_SYM_SPEC = MXSpec(mx_element_dtype="fp6_e2m3",
                                       ch_axis=-1,
                                       block_size=32,
                                       is_dynamic=True).to_quantization_spec()
MX_FP6E3M2_PER_GROUP_SYM_SPEC = MXSpec(mx_element_dtype="fp6_e3m2",
                                       ch_axis=-1,
                                       block_size=32,
                                       is_dynamic=True).to_quantization_spec()

# Step 2: Establish `QuantizationConfig` for nn.Module. Define the QuantizationSpec of input_tensors, output_tensors, weight, and bias. More customer settings refer customized_configuration.py
W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
W_INT4_PER_CHANNEL_CONFIG = QuantizationConfig(weight=INT4_PER_CHANNEL_SPEC)
W_INT4_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=INT4_PER_GROUP_SYM_SPEC)
W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)


# Step 3: Set up the overall `Config` for the model.
def get_config(
    quant_scheme: str,
    group_size: int,
    model_dir: str,
    kv_cache_dtype: Optional[str],
    fp8_attention_quant: bool,
    exclude_layers: Optional[str],
    pre_quantization_optimization: Optional[str],
    pre_optimization_config_file_path: str,
    quant_algo: Optional[str],
    quant_algo_config_file_path: str,
    model_type: str,
    group_size_per_layer: Optional[list[tuple[str, int]]],
) -> Config:

    # Set up `global_quant_config`(required).
    if quant_scheme == 'w_fp8_a_fp8':
        global_quant_config = W_FP8_A_FP8_PER_TENSOR_CONFIG
    elif quant_scheme == 'w_int4_per_channel_sym':
        global_quant_config = W_INT4_PER_CHANNEL_CONFIG
    elif quant_scheme == 'w_int4_per_group_sym':
        global_quant_config = W_INT4_PER_GROUP_SYM_CONFIG
        global_quant_config.weight.set_group_size(group_size)
    elif quant_scheme == 'w_uint4_per_group_asym':
        global_quant_config = W_UINT4_PER_GROUP_CONFIG
        global_quant_config.weight.set_group_size(group_size)
    else:
        global_quant_config = get_global_config(quant_scheme, group_size)

    # Set up `layer_quant_config` and `kv_cache_quant_config`
    layer_quant_config = {}
    kv_cache_quant_config = {}
    if kv_cache_dtype is not None:
        if kv_cache_dtype == "fp8":
            KV_CACHE_SPEC = FP8_PER_TENSOR_SPEC
        elif kv_cache_dtype == "fp8_dynamic":
            KV_CACHE_SPEC = FP8_PER_TENSOR_SPEC_DYNAMIC
        elif kv_cache_dtype == "int8_per_tensor_static":
            KV_CACHE_SPEC = INT8_PER_TENSOR_SPEC
        elif kv_cache_dtype == "int8_per_tensor_dynamic":
            KV_CACHE_SPEC = INT8_PER_TENSOR_DYNAMIC_SPEC
        elif kv_cache_dtype == "int8_per_token":
            KV_CACHE_SPEC = INT8_PER_TOKEN_DYNAMIC_SPEC
        elif kv_cache_dtype == 'uint4':
            KV_CACHE_SPEC = UINT4_PER_CHANNEL_ASYM_DYNAMIC_SPEC
        elif kv_cache_dtype == 'mx_fp8':
            KV_CACHE_SPEC = MX_FP8_PER_GROUP_SYM_SPEC
        elif kv_cache_dtype == 'mx_fp6e2m3':
            KV_CACHE_SPEC = MX_FP6E2M3_PER_GROUP_SYM_SPEC
        elif kv_cache_dtype == 'mx_fp6e3m2':
            KV_CACHE_SPEC = MX_FP6E3M2_PER_GROUP_SYM_SPEC

        if model_type not in MODEL_NAME_KV_LAYERS_MAP.keys():
            raise ValueError(f"KV cache configuration of {model_type} could not be supported automaticly,"
                             "please add the KV layers in MODEL_NAME_KV_LAYERS_MAP")

        kv_layers_name = MODEL_NAME_KV_LAYERS_MAP[model_type]
        for layer_name in kv_layers_name:
            kv_cache_quant_config[layer_name] = QuantizationConfig(input_tensors=global_quant_config.input_tensors,
                                                                   weight=global_quant_config.weight,
                                                                   output_tensors=KV_CACHE_SPEC)
        layer_quant_config = kv_cache_quant_config.copy()

    group_size_per_layer = group_size_per_layer or []
    for layer, group_size in group_size_per_layer:
        try:
            group_size = int(group_size)
        except ValueError:
            raise ValueError(
                f"Invalid group size '{group_size}' for layer '{layer}'. " "Group size must be an integer."
            )
        layer_config = layer_quant_config.get(layer, copy.deepcopy(global_quant_config))
        layer_config.weight.group_size = group_size
        layer_quant_config[layer] = layer_config

    if fp8_attention_quant:
        attn_qspec = FP8_PER_TENSOR_SPEC

        if model_type not in MODEL_NAME_Q_LAYERS_MAP.keys():
            raise ValueError(f"Q_proj configuration of {model_type} could not be supported automaticly,"
                             "please add the q_proj layers in MODEL_NAME_Q_LAYERS_MAP")

        q_layers_name = MODEL_NAME_Q_LAYERS_MAP[model_type]
        layer_quant_config[q_layers_name] = QuantizationConfig(input_tensors=global_quant_config.input_tensors,
                                                               weight=global_quant_config.weight,
                                                               output_tensors=attn_qspec)
    else:
        attn_qspec = None

    # Set up `exclude`
    if "c4ai-command-r-08-2024" in model_dir.lower():  # no quantization for particular layer
        MODEL_NAME_EXCLUDE_LAYERS_MAP["cohere"].append("*2.down_proj")

    if exclude_layers is None:
        if model_type in MODEL_NAME_EXCLUDE_LAYERS_MAP:
            EXCLUDE_LAYERS = MODEL_NAME_EXCLUDE_LAYERS_MAP[model_type]
        else:
            EXCLUDE_LAYERS = ["lm_head"]
            import warnings
            warnings.warn(
                f'Exclude layers configuration for {model_type} could not be supported automatically.'
                'Using EXCLUDE_LAYERS = ["lm_head"]. Please customize the exclude layers in MODEL_NAME_EXCLUDE_LAYERS_MAP.',
                UserWarning)
    else:
        EXCLUDE_LAYERS = exclude_layers

    # Set up `pre_opt_config`
    pre_optimization_configs = []
    if "rotation" in pre_quantization_optimization:
        pre_optimization_configs.append(RotationConfig())
    if "quarot" in pre_quantization_optimization:
        pre_optimization_config_file_path = pre_optimization_config_file_path if pre_optimization_config_file_path else 'models/' + model_type + '/quarot_config.json'
        pre_quant_opt_config = load_pre_optimization_config_from_file(pre_optimization_config_file_path)
        pre_quant_opt_config.kv_cache_quant = kv_cache_dtype is not None
        pre_quant_opt_config.act_quant = global_quant_config.input_tensors is not None
        pre_optimization_configs.append(pre_quant_opt_config)
    if 'smoothquant' in pre_quantization_optimization:
        pre_optimization_config_file_path = pre_optimization_config_file_path if pre_optimization_config_file_path else 'models/' + model_type + '/smooth_config.json'
        pre_opt_config = load_pre_optimization_config_from_file(pre_optimization_config_file_path)
        pre_optimization_configs.append(pre_opt_config)

        # TODO: These warnings should be moved into quark.torch directly at some point.
        smoothquant_alpha = pre_opt_config.alpha
        if global_quant_config.input_tensors is None and smoothquant_alpha > 0:
            print(
                f"[WARNING] Weight-only quantization is used, but SmoothQuant alpha={smoothquant_alpha} is larger than 0.0. In this case, using alpha = 0.0 is recommended to shift all the quantization difficulty from the weights into from the activations."
            )

        if global_quant_config.weight is None and smoothquant_alpha < 1:
            print(
                f"[WARNING] Activation-only quantization is used, but SmoothQuant alpha={smoothquant_alpha} is smaller than 1.0. In this case, using alpha = 1.0 is recommended to shift all the quantization difficulty from the activations into the weights."
            )

        if global_quant_config.weight is not None and global_quant_config.input_tensors is not None and smoothquant_alpha in [
                0.0, 1.0
        ]:
            print(
                f"[WARNING] Both weights and activations are quantized, but SmoothQuant alpha={smoothquant_alpha} is used. alpha = 0.0 shifts all the quantization difficulty to activations, while alpha = 1.0 shifts all the quantization difficulty to the weights. If this is the desired behavior, this warning can be ignored."
            )

    # Set up `algo_config`
    algo_config = load_algo_config(quant_algo, quant_scheme, quant_algo_config_file_path, model_type) if quant_algo else None

    quant_config = Config(
        global_quant_config=global_quant_config,
        layer_quant_config=layer_quant_config,
        kv_cache_quant_config=kv_cache_quant_config,
        softmax_quant_spec=attn_qspec,
        exclude=EXCLUDE_LAYERS,
        pre_quant_opt_config=pre_optimization_configs,
        algo_config=algo_config
    )

    return quant_config


def load_algo_config(quant_algo, quant_scheme, quant_algo_config_file_path, model_type):
    if quant_algo == 'awq':
        default_algo_config_file = 'models/' + model_type + '/awq_config.json'
    elif quant_algo == 'autosmoothquant':
        default_algo_config_file = 'models/' + model_type + '/autosmoothquant_config.json'
    elif quant_algo == 'gptq':
        assert quant_scheme in ['w_uint4_per_group_asym', 'w_uint4_per_channel_asym']  # GPTQ is only tested with uint4_per_group and w_uint4_per_channel_asym quantization in Quark
        default_algo_config_file = 'models/' + model_type + '/gptq_config.json'
    quant_algo_config_file_path = quant_algo_config_file_path if quant_algo_config_file_path else default_algo_config_file
    if os.path.exists(quant_algo_config_file_path):
        algo_config = load_quant_algo_config_from_file(quant_algo_config_file_path)
    else:
        if quant_algo == 'awq':
            algo_config = AWQConfig()
        elif quant_algo == 'autosmoothquant':
            algo_config = AutoSmoothQuantConfig()
        else:
            raise ValueError("Missing quantization algorithm configuration")
    return algo_config


MERGE_REALQ_CONFIG = JsonExporterConfig(weight_merge_groups=[["*up_proj", "*gate_proj"],
                                                             ["*q_proj", "*k_proj", "*v_proj"]],
                                        weight_format="real_quantized",
                                        pack_method="reorder")

NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized", pack_method="reorder")


def get_export_config(args: argparse.Namespace, model_type: str) -> ExporterConfig:
    export_config = None
    if args.weight_matrix_merge is True:
        export_config = ExporterConfig(json_export_config=MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig())
    else:
        export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG,
                                       onnx_export_config=OnnxExporterConfig())

    if args.kv_cache_dtype is not None:
        if model_type not in MODEL_NAME_KV_LAYERS_MAP.keys():
            raise ValueError(f"KV cache configuration of {model_type} could not be supported automaticly,"
                             "please add the KV layers in MODEL_NAME_KV_LAYERS_MAP")
        export_config.json_export_config.kv_cache_group = MODEL_NAME_KV_LAYERS_MAP[model_type]

    if args.pack_method == "order":
        export_config.json_export_config.pack_method = "order"

    export_config.json_export_config.min_kv_scale = args.min_kv_scale

    return export_config
