#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import copy
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from customized_configuration import (
    DEPRECATED_QUANT_SCHEME,
    FP4_PER_GROUP_SYM_SPEC,
    FP6_E2M3_PER_GROUP_SYM_SPEC,
    FP6_E3M2_PER_GROUP_SYM_SPEC,
    INT8_PER_TENSOR_DYNAMIC_SPEC,
    INT8_PER_TENSOR_SPEC,
    INT8_PER_TOKEN_DYNAMIC_SPEC,
    OCP_MX_QUANT_SCHEME,
    OCP_MXFP4_SPEC,
    OCP_MXFP6_E2M3_SPEC,
    OCP_MXFP6_E3M2_SPEC,
    OCP_MXFP8_E4M3_SPEC,
    UINT4_PER_CHANNEL_ASYM_DYNAMIC_SPEC,
    get_global_config,
)
from llm_utils.model_preparation import (
    MODEL_NAME_EXCLUDE_LAYERS_MAP,
    MODEL_NAME_KV_LAYERS_MAP,
    MODEL_NAME_Q_LAYERS_MAP,
    MOE_MODEL_NAME_EXPERTS_LAYERS_MAP,
)

from quark.shares.utils.log import ScreenLogger
from quark.torch.export import ExporterConfig, JsonExporterConfig, OnnxExporterConfig
from quark.torch.quantization import (
    AutoSmoothQuantConfig,
    AWQConfig,
    Config,
    FP8E4M3PerTensorSpec,
    Int4PerChannelSpec,
    ProgressiveSpec,
    QuantizationConfig,
    load_quant_algo_config_from_file,
)

logger = ScreenLogger(__name__)

"""
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
"""

# Step 1: Configure `DataTypeSpec` for torch.Tensors. Specify attributes such as dtype, observer_method, etc. More customer settings refer customized_configuration.py
FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
FP8_PER_TENSOR_SPEC_DYNAMIC = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=True).to_quantization_spec()
INT4_PER_CHANNEL_SPEC = Int4PerChannelSpec(
    symmetric=True, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
).to_quantization_spec()
MXFP8_PER_GROUP_SPEC = OCP_MXFP8_E4M3_SPEC(is_dynamic=True)


# Step 3: Set up the overall `Config` for the model.
def get_config(args: argparse.Namespace, model_type: str) -> Config:
    quant_scheme = args.quant_scheme
    model_dir = args.model_dir
    kv_cache_dtype = args.kv_cache_dtype
    attention_dtype = args.attention_dtype
    moe_experts_second_step_config = args.moe_experts_second_step_config
    exclude_layers = args.exclude_layers
    quant_algo = args.quant_algo
    quant_algo_config_file = args.quant_algo_config_file
    group_size_per_layer = args.group_size_per_layer
    scale_format = args.scale_format
    scale_calculation_mode = args.scale_calculation_mode
    min_kv_scale = args.min_kv_scale

    if quant_scheme in DEPRECATED_QUANT_SCHEME:
        print(
            f"[WARNING] The quantization scheme `'{quant_scheme}'` is deprecated and will be removed in the next AMD Quark release in favor of `'{DEPRECATED_QUANT_SCHEME[quant_scheme]}'`. Please use `--quant_scheme {DEPRECATED_QUANT_SCHEME[quant_scheme]}`."
        )

        quant_scheme = DEPRECATED_QUANT_SCHEME[quant_scheme]

    if quant_scheme in OCP_MX_QUANT_SCHEME and args.group_size != 32:
        logger.warning(
            f"OCP MX quantization requires a group_size of 32. The specified group_size {args.group_size} has been automatically replaced with 32."
        )
        args.group_size = 32
    group_size = args.group_size

    if quant_algo and quant_algo_config_file:
        if len(quant_algo) != len(quant_algo_config_file):
            raise ValueError(
                "Mismatch: The number of '--quant_algo' must match the number of '--quant_algo_config_file'."
            )

    supported_kv_cache_type = {
        "fp8": FP8_PER_TENSOR_SPEC,
        "fp8_dynamic": FP8_PER_TENSOR_SPEC_DYNAMIC,
        "int8_per_tensor_static": INT8_PER_TENSOR_SPEC,
        "int8_per_tensor_dynamic": INT8_PER_TENSOR_DYNAMIC_SPEC,
        "int8_per_token": INT8_PER_TOKEN_DYNAMIC_SPEC,
        "uint4": UINT4_PER_CHANNEL_ASYM_DYNAMIC_SPEC,
        "mxfp8": MXFP8_PER_GROUP_SPEC,
        "fp4_per_group": FP4_PER_GROUP_SYM_SPEC(group_size, scale_format, None, True),
        "mxfp4": OCP_MXFP4_SPEC(scale_calculation_mode, True),
        "fp6e2m3_per_group": FP6_E2M3_PER_GROUP_SYM_SPEC(group_size, scale_format, None, True),
        "mxfp6_e2m3": OCP_MXFP6_E2M3_SPEC(scale_calculation_mode, True),
        "fp6e3m2_per_group": FP6_E3M2_PER_GROUP_SYM_SPEC(group_size, scale_format, None, True),
        "mxfp6_e3m2": OCP_MXFP6_E3M2_SPEC(scale_calculation_mode, True),
    }

    if kv_cache_dtype is not None and kv_cache_dtype not in supported_kv_cache_type:
        raise ValueError(f"The kv_cache_dtype='{kv_cache_dtype}' is not supported, only {supported_kv_cache_type} are.")

    global_quant_config = get_global_config(quant_scheme, group_size, scale_format, scale_calculation_mode)

    # Set up `layer_quant_config` and `kv_cache_quant_config`
    layer_quant_config = {}
    kv_cache_quant_config = {}
    kv_cache_group = []
    if kv_cache_dtype is not None:
        kv_cache_spec = supported_kv_cache_type[kv_cache_dtype]

        if model_type not in MODEL_NAME_KV_LAYERS_MAP.keys():
            raise ValueError(
                f"KV cache configuration of {model_type} could not be supported automaticly,"
                "please add the KV layers in MODEL_NAME_KV_LAYERS_MAP"
            )

        kv_layers_name = MODEL_NAME_KV_LAYERS_MAP[model_type]
        kv_cache_group = kv_layers_name
        for layer_name in kv_layers_name:
            kv_cache_quant_config[layer_name] = QuantizationConfig(
                input_tensors=global_quant_config.input_tensors,
                weight=global_quant_config.weight,
                output_tensors=kv_cache_spec,
            )
        layer_quant_config = kv_cache_quant_config.copy()

    group_size_per_layer = group_size_per_layer or []
    for layer, group_size in group_size_per_layer:
        try:
            group_size = int(group_size)
        except ValueError:
            raise ValueError(f"Invalid group size '{group_size}' for layer '{layer}'. Group size must be an integer.")
        layer_config = layer_quant_config.get(layer, copy.deepcopy(global_quant_config))
        layer_config.weight.group_size = group_size
        layer_quant_config[layer] = layer_config

    if attention_dtype is not None:
        if attention_dtype == "fp8":
            attn_qspec = FP8_PER_TENSOR_SPEC
        else:
            raise ValueError(f"Attention dtype {attention_dtype} is not supported currently.")

        if model_type not in MODEL_NAME_Q_LAYERS_MAP.keys():
            raise ValueError(
                f"Q_proj configuration of {model_type} could not be supported automaticly,"
                "please add the q_proj layers in MODEL_NAME_Q_LAYERS_MAP"
            )

        q_layers_name = MODEL_NAME_Q_LAYERS_MAP[model_type]
        layer_quant_config[q_layers_name] = QuantizationConfig(
            input_tensors=global_quant_config.input_tensors,
            weight=global_quant_config.weight,
            output_tensors=attn_qspec,
        )
    else:
        attn_qspec = None

    #  Add MoE experts second step quantization configuration
    if moe_experts_second_step_config is not None:
        assert moe_experts_second_step_config in ["w_int4_per_channel_sym"], (
            "Currently, only w_int4_per_channel_sym is supported for MoE experts second step quantization, "
            "please add the MoE experts second step quantization configuration in quantize_quark.py"
        )

        assert model_type in MOE_MODEL_NAME_EXPERTS_LAYERS_MAP.keys(), (
            f"Currently, {model_type} is not supported for MoE experts second step quantization, "
            f"please add {model_type} model in MOE_MODEL_NAME_EXPERTS_LAYERS_MAP"
        )

        if moe_experts_second_step_config == "w_int4_per_channel_sym":
            weight_quant_spec = ProgressiveSpec(
                first_stage=global_quant_config.weight, second_stage=INT4_PER_CHANNEL_SPEC
            )

        experts_layers_name = MOE_MODEL_NAME_EXPERTS_LAYERS_MAP[model_type]
        for layer_name in experts_layers_name:
            layer_quant_config[layer_name] = QuantizationConfig(
                input_tensors=global_quant_config.input_tensors,
                weight=weight_quant_spec.to_quantization_spec(),
                output_tensors=global_quant_config.output_tensors,
            )

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
                f"Exclude layers configuration for {model_type} could not be supported automatically."
                'Using EXCLUDE_LAYERS = ["lm_head"]. Please customize the exclude layers in MODEL_NAME_EXCLUDE_LAYERS_MAP.',
                UserWarning,
            )
    else:
        EXCLUDE_LAYERS = exclude_layers

    # Set up `algo_config`
    algo_configs = []
    if quant_algo is not None:
        for i in range(len(quant_algo)):
            algo = quant_algo[i]
            algo_config_file = (
                quant_algo_config_file[i]
                if quant_algo_config_file is not None and quant_algo_config_file[i] != ""
                else None
            )
            algo_config = load_algo_config(algo, quant_scheme, algo_config_file, model_type)
            algo_configs.append(algo_config)

    quant_config = Config(
        global_quant_config=global_quant_config,
        layer_quant_config=layer_quant_config,
        kv_cache_quant_config=kv_cache_quant_config,
        kv_cache_group=kv_cache_group,
        min_kv_scale=min_kv_scale,
        softmax_quant_spec=attn_qspec,
        exclude=EXCLUDE_LAYERS,
        algo_config=algo_configs,
    )

    return quant_config


def load_algo_config(quant_algo, quant_scheme, quant_algo_config_file_path, model_type):
    if quant_algo == "rotation":
        default_algo_config_file = "models/" + model_type + "/rotation_config.json"
    elif quant_algo == "quarot":
        default_algo_config_file = "models/" + model_type + "/quarot_config.json"
    elif quant_algo == "smoothquant":
        default_algo_config_file = "models/" + model_type + "/smooth_config.json"
    elif quant_algo == "awq":
        default_algo_config_file = "models/" + model_type + "/awq_config.json"
    elif quant_algo == "autosmoothquant":
        default_algo_config_file = "models/" + model_type + "/autosmoothquant_config.json"
    elif quant_algo == "gptq":
        if quant_scheme not in ["w_uint4_per_group_asym", "w_uint4_per_channel_asym", "w_mxfp4_a_mxfp4"]:
            logger.warning(
                f"GPTQ is only tested with uint4_per_group, w_uint4_per_channel_asym, and w_mxfp4_a_mxfp4 quantization in Quark, be careful to apply GPTQ on {quant_scheme}."
            )
        default_algo_config_file = "models/" + model_type + "/gptq_config.json"
    elif quant_algo == "qronos":
        default_algo_config_file = Path("model", model_type, "qronos_config.json")

    quant_algo_config_file_path = (
        quant_algo_config_file_path if quant_algo_config_file_path else default_algo_config_file
    )

    if os.path.exists(quant_algo_config_file_path):
        algo_config = load_quant_algo_config_from_file(quant_algo_config_file_path)
    elif quant_algo_config_file_path:
        raise ValueError(
            f"The quantization configuration: {quant_algo_config_file_path}, does not exist."
            "Please ensure this is the correct path."
        )
    else:
        if quant_algo == "awq":
            algo_config = AWQConfig()
        elif quant_algo == "autosmoothquant":
            algo_config = AutoSmoothQuantConfig()
        else:
            raise ValueError("Missing quantization algorithm configuration")
    return algo_config


MERGE_REALQ_CONFIG = JsonExporterConfig(
    weight_merge_groups=[["*up_proj", "*gate_proj"], ["*q_proj", "*k_proj", "*v_proj"]],
    weight_format="real_quantized",
    pack_method="reorder",
)

NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized", pack_method="reorder")


def get_export_config(args: argparse.Namespace, model_type: str) -> ExporterConfig:
    export_config = None
    if args.weight_matrix_merge is True:
        export_config = ExporterConfig(json_export_config=MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig())
    else:
        export_config = ExporterConfig(
            json_export_config=NO_MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig()
        )

    if args.kv_cache_dtype is not None:
        if model_type not in MODEL_NAME_KV_LAYERS_MAP.keys():
            raise ValueError(
                f"KV cache configuration of {model_type} could not be supported automaticly,"
                "please add the KV layers in MODEL_NAME_KV_LAYERS_MAP"
            )
        export_config.json_export_config.kv_cache_group = MODEL_NAME_KV_LAYERS_MAP[model_type]

    if args.pack_method == "order":
        export_config.json_export_config.pack_method = "order"

    export_config.json_export_config.min_kv_scale = args.min_kv_scale

    return export_config
