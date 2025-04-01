#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.torch.quantization import QuantizationConfig, FP8E4M3PerTensorSpec, \
    Int4PerTensorSpec, Int4PerChannelSpec, Int8PerChannelSpec, Int4PerGroupSpec, Uint4PerGroupSpec, Int8PerTensorSpec, \
    Uint8PerTensorSpec, Int8PerGroupSpec, Uint8PerGroupSpec, MXSpec, BFP16Spec, MX6Spec, Float16Spec, Bfloat16Spec, \
    FP8E4M3PerChannelSpec, Uint4PerChannelSpec

FLOAT16_SPEC = Float16Spec().to_quantization_spec()

BFLOAT16_SPEC = Bfloat16Spec().to_quantization_spec()

FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max",
                                           is_dynamic=False).to_quantization_spec()

FP8_PER_TENSOR_SPEC_DYNAMIC = FP8E4M3PerTensorSpec(observer_method="min_max",
                                                   is_dynamic=True).to_quantization_spec()

INT4_PER_TENSOR_SPEC = Int4PerTensorSpec(observer_method="min_max",
                                         symmetric=True,
                                         scale_type="float",
                                         round_method="half_even",
                                         is_dynamic=False).to_quantization_spec()



INT8_PER_CHANNEL_SPEC = Int8PerChannelSpec(symmetric=True,
                                           scale_type="float",
                                           round_method="half_even",
                                           ch_axis=0,
                                           is_dynamic=False).to_quantization_spec()



UINT4_PER_GROUP_ASYM_SPEC = Uint4PerGroupSpec(symmetric=False,
                                              scale_type="float",
                                              round_method="half_even",
                                              ch_axis=1,
                                              is_dynamic=False,
                                              group_size=128).to_quantization_spec()

INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(observer_method="min_max",
                                         symmetric=True,
                                         scale_type="float",
                                         round_method="half_even",
                                         is_dynamic=False).to_quantization_spec()

UINT8_PER_TENSOR_ASPEC = Uint8PerTensorSpec(observer_method="min_max",
                                            symmetric=False,
                                            scale_type="float",
                                            round_method="half_even",
                                            is_dynamic=False).to_quantization_spec()

INT8_PER_TOKEN_DYNAMIC_SPEC = Int8PerChannelSpec(symmetric=True,
                                                 scale_type="float",
                                                 round_method="half_even",
                                                 ch_axis=1,
                                                 is_dynamic=True).to_quantization_spec()

INT8_PER_TENSOR_DYNAMIC_SPEC = Int8PerTensorSpec(observer_method="min_max",
                                                 symmetric=True,
                                                 scale_type="float",
                                                 round_method="half_even",
                                                 is_dynamic=True).to_quantization_spec()

INT8_PER_GROUP_SYM_SPEC = Int8PerGroupSpec(symmetric=True,
                                           scale_type="float",
                                           round_method="half_even",
                                           ch_axis=1,
                                           is_dynamic=False,
                                           group_size=128).to_quantization_spec()

UINT8_PER_GROUP_ASYM_SPEC = Uint8PerGroupSpec(symmetric=False,
                                              scale_type="float",
                                              round_method="half_even",
                                              ch_axis=1,
                                              is_dynamic=False,
                                              group_size=128).to_quantization_spec()

MX_FP8_PER_GROUP_SYM_STATIC_SPEC = MXSpec(mx_element_dtype="fp8_e4m3",
                                          ch_axis=-1,
                                          block_size=32,
                                          is_dynamic=False).to_quantization_spec()

MX_FP8_PER_GROUP_SYM_DYNAMIC_SPEC = MXSpec(mx_element_dtype="fp8_e4m3",
                                           ch_axis=-1,
                                           block_size=32,
                                           is_dynamic=True).to_quantization_spec()

W_BFP16_SPEC = BFP16Spec(is_dynamic=False, ch_axis=-1).to_quantization_spec()

A_BFP16_SPEC = BFP16Spec(is_dynamic=True, ch_axis=-1).to_quantization_spec()

W_MX6_SPEC = MX6Spec(ch_axis=-1, block_size=32, is_dynamic=False).to_quantization_spec()

A_MX6_SPEC = MX6Spec(ch_axis=-1, block_size=32, is_dynamic=True).to_quantization_spec()

# Data type spec for testing, not applied on the specific backend
UINT8_PER_TENSOR_SPEC = Uint8PerTensorSpec(observer_method="min_max",
                                           symmetric=True,
                                           scale_type="float",
                                           round_method="half_even",
                                           is_dynamic=False).to_quantization_spec()

FP8_PER_CHANNEL_SPEC = FP8E4M3PerChannelSpec(is_dynamic=False, ch_axis=0).to_quantization_spec()

INT4_PER_CHANNEL_ASYM_SPEC = Int4PerChannelSpec(symmetric=False,
                                                scale_type="float",
                                                round_method="half_even",
                                                ch_axis=0,
                                                is_dynamic=False).to_quantization_spec()

INT4_PER_GROUP_ASYM_SPEC = Int4PerGroupSpec(symmetric=False,
                                            scale_type="float",
                                            round_method="half_even",
                                            ch_axis=1,
                                            is_dynamic=False,
                                            group_size=128).to_quantization_spec()

UINT4_PER_GROUP_SYM_SPEC = Uint4PerGroupSpec(symmetric=True,
                                             scale_type="float",
                                             round_method="half_even",
                                             ch_axis=1,
                                             is_dynamic=False,
                                             group_size=128).to_quantization_spec()

UINT4_PER_CHANNEL_SYM_SPEC = Uint4PerChannelSpec(symmetric=True,
                                                 scale_type="float",
                                                 round_method="half_even",
                                                 ch_axis=0,
                                                 is_dynamic=False).to_quantization_spec()

UINT4_PER_CHANNEL_ASYM_SPEC = Uint4PerChannelSpec(symmetric=False,
                                                  scale_type="float",
                                                  round_method="half_even",
                                                  ch_axis=0,
                                                  is_dynamic=False).to_quantization_spec()

UINT4_PER_CHANNEL_ASYM_DYNAMIC_SPEC = Uint4PerChannelSpec(symmetric=False,
                                                          scale_type="float",
                                                          round_method="half_even",
                                                          ch_axis=0,
                                                          is_dynamic=True).to_quantization_spec()

INT8_PER_TENSOR_PERCENTILE_SPEC = Int8PerTensorSpec(observer_method="percentile",
                                                    symmetric=True,
                                                    scale_type="float",
                                                    round_method="half_even",
                                                    is_dynamic=False).to_quantization_spec()

INT8_PER_TENSOR_MSE_SPEC = Int8PerTensorSpec(observer_method="MSE",
                                             symmetric=True,
                                             scale_type="float",
                                             round_method="half_even",
                                             is_dynamic=False).to_quantization_spec()

UINT8_PER_TENSOR_PERCENTILE_SPEC = Uint8PerTensorSpec(observer_method="percentile",
                                                      symmetric=True,
                                                      scale_type="float",
                                                      round_method="half_even",
                                                      is_dynamic=False).to_quantization_spec()

UINT8_PER_TENSOR_MSE_SPEC = Uint8PerTensorSpec(observer_method="MSE",
                                               symmetric=True,
                                               scale_type="float",
                                               round_method="half_even",
                                               is_dynamic=False).to_quantization_spec()

MX_FP4_PER_GROUP_SYM_SPEC = MXSpec(mx_element_dtype="fp4",
                                   ch_axis=-1,
                                   block_size=32,
                                   is_dynamic=False).to_quantization_spec()

MX_FP4_PER_GROUP_SYM_DYN_SPEC = MXSpec(mx_element_dtype="fp4",
                                       ch_axis=-1,
                                       block_size=32,
                                       is_dynamic=True).to_quantization_spec()

MX_FP6_E2M3_PER_GROUP_SYM_SPEC = MXSpec(mx_element_dtype="fp6_e2m3",
                                        ch_axis=-1,
                                        block_size=32,
                                        is_dynamic=True).to_quantization_spec()

MX_FP6_E3M2_PER_GROUP_SYM_SPEC = MXSpec(mx_element_dtype="fp6_e3m2",
                                        ch_axis=-1,
                                        block_size=32,
                                        is_dynamic=True).to_quantization_spec()

MX_FP6_E2M3_PER_GROUP_SYM_STATIC_SPEC = MXSpec(mx_element_dtype="fp6_e2m3",
                                               ch_axis=-1,
                                               block_size=32,
                                               is_dynamic=False).to_quantization_spec()

MX_FP6_E3M2_PER_GROUP_SYM_STATIC_SPEC = MXSpec(mx_element_dtype="fp6_e3m2",
                                               ch_axis=-1,
                                               block_size=32,
                                               is_dynamic=False).to_quantization_spec()

MX_INT8_PER_GROUP_SYM_SPEC = MXSpec(mx_element_dtype="int8",
                                    ch_axis=-1,
                                    block_size=32,
                                    is_dynamic=True).to_quantization_spec()

BFP16_PER_GROUP_SYM_SPEC = BFP16Spec(is_dynamic=True, ch_axis=-1).to_quantization_spec()


# Float16 config
FLOAT16_CONFIG = QuantizationConfig(input_tensors=FLOAT16_SPEC, weight=FLOAT16_SPEC)



W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                        weight=FP8_PER_TENSOR_SPEC,
                                                        output_tensors=FP8_PER_TENSOR_SPEC)

# Int per tensor config
W_INT4_PER_TENSOR_CONFIG = QuantizationConfig(weight=INT4_PER_TENSOR_SPEC)

W_INT8_PER_TENSOR_CONFIG = QuantizationConfig(weight=INT8_PER_TENSOR_SPEC)

W_INT8_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC, weight=INT8_PER_TENSOR_SPEC)

W_UINT8_A_UINT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=UINT8_PER_TENSOR_ASPEC,
                                                       weight=UINT8_PER_TENSOR_ASPEC)

W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_DYNAMIC_SPEC,
                                                             weight=INT8_PER_TENSOR_DYNAMIC_SPEC)

# Int per Channel Config
W_INT8_PER_CHANNEL_CONFIG = QuantizationConfig(weight=INT8_PER_CHANNEL_SPEC)

W_INT8_PER_CHANNEL_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                                                 weight=INT8_PER_CHANNEL_SPEC)

W_INT8_PER_CHANNEL_A_INT8_PER_TENSOR_DYNAMIC_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_DYNAMIC_SPEC,
                                                                         weight=INT8_PER_CHANNEL_SPEC)

# Int per Group Config
W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG = QuantizationConfig(input_tensors=BFLOAT16_SPEC, weight=UINT4_PER_GROUP_ASYM_SPEC)

W_INT8_PER_GROUP_CONFIG = QuantizationConfig(weight=INT8_PER_GROUP_SYM_SPEC)

W_UINT8_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT8_PER_GROUP_ASYM_SPEC)

W_MX_FP8_CONFIG = QuantizationConfig(weight=MX_FP8_PER_GROUP_SYM_STATIC_SPEC)
W_MX_FP8_A_MX_FP8_CONFIG = QuantizationConfig(weight=MX_FP8_PER_GROUP_SYM_STATIC_SPEC, input_tensors=MX_FP8_PER_GROUP_SYM_DYNAMIC_SPEC)

W_INT8_A_INT8_PER_TOKEN_DYNAMIC_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TOKEN_DYNAMIC_SPEC,
                                                            weight=INT8_PER_CHANNEL_SPEC)

W_BFP16_CONFIG = QuantizationConfig(weight=W_BFP16_SPEC)
W_BFP16_A_BFP16_CONFIG = QuantizationConfig(input_tensors=A_BFP16_SPEC, weight=W_BFP16_SPEC)

W_MX6_CONFIG = QuantizationConfig(weight=W_MX6_SPEC)
W_MX6_A_MX6_CONFIG = QuantizationConfig(input_tensors=A_MX6_SPEC,
                                        weight=W_MX6_SPEC)

# quant_scheme for testing, not applied on the specific backend
W_UINT4_A_UINT4_PER_CHANNEL = QuantizationConfig(input_tensors=UINT4_PER_CHANNEL_ASYM_SPEC, weight=UINT4_PER_CHANNEL_ASYM_SPEC)
W_UINT4_PER_TOKEN_A_INT8_PER_CHANNEL = QuantizationConfig(input_tensors=INT8_PER_TOKEN_DYNAMIC_SPEC, weight=UINT4_PER_CHANNEL_ASYM_SPEC)
W_UINT4_PER_CHANNEL_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC, weight=UINT4_PER_CHANNEL_SYM_SPEC)
W_UINT4_PER_GROUP_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC, weight=UINT4_PER_GROUP_SYM_SPEC)
W_FP8_A_FP8_O_FP8_PER_CHANNEL_SYM_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_CHANNEL_SPEC)
W_INT8_A_INT8_PER_TENSOR_PERCENTILE_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC, weight=INT8_PER_TENSOR_PERCENTILE_SPEC)
W_INT8_A_INT8_PER_TENSOR_MSE_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC, weight=INT8_PER_TENSOR_MSE_SPEC)
W_UINT8_A_UINT8_PER_TENSOR_PERCENTILE_CONFIG = QuantizationConfig(input_tensors=UINT8_PER_TENSOR_SPEC, weight=UINT8_PER_TENSOR_PERCENTILE_SPEC)
W_UINT8_A_UINT8_PER_TENSOR_MSE_CONFIG = QuantizationConfig(input_tensors=UINT8_PER_TENSOR_SPEC, weight=UINT8_PER_TENSOR_MSE_SPEC)
W_UINT4_PER_CHANNEL_SYM_CONFIG = QuantizationConfig(weight=UINT4_PER_CHANNEL_SYM_SPEC)
W_UINT4_PER_CHANNEL_ASYM_CONFIG = QuantizationConfig(weight=UINT4_PER_CHANNEL_ASYM_SPEC)
W_UINT4_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)
W_INT4_PER_CHANNEL_ASYM_CONFIG = QuantizationConfig(weight=INT4_PER_CHANNEL_ASYM_SPEC)
W_INT4_PER_GROUP_ASYM_CONFIG = QuantizationConfig(weight=INT4_PER_GROUP_ASYM_SPEC)
W_MX_FP4_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=MX_FP4_PER_GROUP_SYM_SPEC)
W_MX_FP4_A_FP8_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=MX_FP4_PER_GROUP_SYM_SPEC, input_tensors=FP8_PER_TENSOR_SPEC)
W_MX_FP4_A_MX_FP4_DYN_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=MX_FP4_PER_GROUP_SYM_SPEC, input_tensors=MX_FP4_PER_GROUP_SYM_DYN_SPEC)
W_MX_FP6_E2M3_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=MX_FP6_E2M3_PER_GROUP_SYM_SPEC)
W_MX_FP6_E3M2_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=MX_FP6_E3M2_PER_GROUP_SYM_SPEC)
W_MX_INT8_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=MX_INT8_PER_GROUP_SYM_SPEC)
W_BFP16_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=BFP16_PER_GROUP_SYM_SPEC)
W_MX_FP4_A_MX_FP4_PER_GROUP_SYM_CONFIG = QuantizationConfig(input_tensors=MX_FP4_PER_GROUP_SYM_DYN_SPEC, weight=MX_FP4_PER_GROUP_SYM_SPEC)
W_MX_FP4_A_MX_FP8_PER_GROUP_SYM_CONFIG = QuantizationConfig(input_tensors=MX_FP8_PER_GROUP_SYM_DYNAMIC_SPEC, weight=MX_FP4_PER_GROUP_SYM_SPEC)
W_MX_FP4_A_MX_FP6E2M3_PER_GROUP_SYM_CONFIG = QuantizationConfig(input_tensors=MX_FP6_E2M3_PER_GROUP_SYM_SPEC, weight=MX_FP4_PER_GROUP_SYM_SPEC)
W_MX_FP4_A_MX_FP6E3M2_PER_GROUP_SYM_CONFIG = QuantizationConfig(input_tensors=MX_FP6_E3M2_PER_GROUP_SYM_SPEC, weight=MX_FP4_PER_GROUP_SYM_SPEC)
W_MX_FP6E2M3_A_MX_FP6E2M3_PER_GROUP_SYM_CONFIG = QuantizationConfig(input_tensors=MX_FP6_E2M3_PER_GROUP_SYM_SPEC, weight=MX_FP6_E2M3_PER_GROUP_SYM_STATIC_SPEC)
W_MX_FP6E3M2_A_MX_FP6E3M2_PER_GROUP_SYM_CONFIG = QuantizationConfig(input_tensors=MX_FP6_E3M2_PER_GROUP_SYM_SPEC, weight=MX_FP6_E3M2_PER_GROUP_SYM_STATIC_SPEC)
W_MX_FP4_PER_GROUP_A_FP8_PER_TENSOR_STATIC_SYM_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=MX_FP4_PER_GROUP_SYM_SPEC)
W_MX_FP4_PER_GROUP_A_FP8_PER_TENSOR_DYNAMIC_SYM_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC_DYNAMIC, weight=MX_FP4_PER_GROUP_SYM_SPEC)

def get_global_config(quant_scheme: str, group_size: int) -> QuantizationConfig:
    assert quant_scheme in ["w_uint4_a_bfloat16_per_group_asym", "w_int8_per_tensor_sym", "w_int8_per_group_sym", "w_uint8_per_group_asym", "w_int8_a_int8_per_tensor_sym",
                            "w_int8_a_int8_per_tensor_sym_dynamic", "w_uint8_a_uint8_per_tensor_asym", "w_fp8_a_fp8_o_fp8", "w_mx_fp8", "w_mx_fp8_a_mx_fp8", "w_int8_a_int8_per_token_dynamic",
                            "w_bfp16", "w_bfp16_a_bfp16", "w_mx6", "w_mx6_a_mx6", "w_int8_per_channel_a_int8_per_tensor_sym", "w_int8_per_channel_a_int8_per_tensor_dynamic",
                            'w_fp8_per_channel_sym', 'w_int4_per_channel_asym', 'w_int4_per_group_asym', 'w_uint4_per_group_sym', 'w_uint4_per_channel_sym',
                            'w_uint4_per_channel_asym', 'w_int8_per_tensor_percentile', 'w_int8_per_tensor_mse', 'w_uint8_per_tensor_percentile', 'w_uint8_per_tensor_mse',
                            "w_mx_fp4_per_group_sym", "w_mx_fp4_a_fp8_per_group_sym", "w_mx_fp4_a_mx_fp4_dyn_per_group_sym", "w_mx_fp6_e3m2_per_group_sym", "w_mx_fp6_e2m3_per_group_sym", "w_mx_int8_per_group_sym",
                            "w_uint4_per_channel_a_int8_per_tensor", "w_uint4_per_group_a_int8_per_tensor", "w_bfp16_per_group_sym",
                            "w_int8_per_channel_a_int8_per_tensor_sym", "w_int8_per_channel_a_int8_per_tensor_sym_dynamic", "w_uint4_per_token_a_int8_per_channel",
                            "w_uint4_a_uint4_per_channel", "w_mx_fp4_a_mx_fp4_per_group_sym", "w_mx_fp4_a_mx_fp8_per_group_sym",
                            "w_mx_fp4_per_group_a_fp8_per_tensor_static_sym", "w_mx_fp4_per_group_a_fp8_per_tensor_dynamic_sym", "w_mx_fp4_a_mx_fp6e2m3_per_group_sym",
                            "w_mx_fp4_a_mx_fp6e3m2_per_group_sym", "w_mx_fp6e2m3_a_mx_fp6e2m3_per_group_sym", "w_mx_fp6e3m2_a_mx_fp6e3m2_per_group_sym",
                            ]
    if quant_scheme == 'w_fp8_a_fp8_o_fp8':
        global_quant_config = W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG
    elif quant_scheme == 'w_int4_per_tensor':
        global_quant_config = W_INT4_PER_TENSOR_CONFIG
    elif quant_scheme == 'w_uint4_a_bfloat16_per_group_asym':
        global_quant_config = W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG
        global_quant_config.weight.set_group_size(group_size)
    elif quant_scheme == 'w_int8_per_tensor_sym':
        global_quant_config = W_INT8_PER_TENSOR_CONFIG
    elif quant_scheme == 'w_int8_per_group_sym':
        global_quant_config = W_INT8_PER_GROUP_CONFIG
    elif quant_scheme == 'w_uint8_per_group_asym':
        global_quant_config = W_UINT8_PER_GROUP_CONFIG
    elif quant_scheme == 'w_int8_per_channel_a_int8_per_tensor_sym':
        global_quant_config = W_INT8_PER_CHANNEL_A_INT8_PER_TENSOR_CONFIG
    elif quant_scheme == 'w_int8_per_channel_a_int8_per_tensor_sym_dynamic':
        global_quant_config = W_INT8_PER_CHANNEL_A_INT8_PER_TENSOR_DYNAMIC_CONFIG
    elif quant_scheme == 'w_int8_a_int8_per_tensor_sym':
        global_quant_config = W_INT8_A_INT8_PER_TENSOR_CONFIG
    elif quant_scheme == 'w_uint8_a_uint8_per_tensor_asym':
        global_quant_config = W_UINT8_A_UINT8_PER_TENSOR_CONFIG
    elif quant_scheme == 'w_int8_per_channel_a_int8_per_tensor_dynamic':
        global_quant_config = W_INT8_PER_CHANNEL_A_INT8_PER_TENSOR_DYNAMIC_CONFIG
    elif quant_scheme == 'w_int8_a_int8_per_tensor_sym_dynamic':
        global_quant_config = W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG
    elif quant_scheme == "w_mx_fp8":
        global_quant_config = W_MX_FP8_CONFIG
    elif quant_scheme == "w_mx_fp8_a_mx_fp8":
        global_quant_config = W_MX_FP8_A_MX_FP8_CONFIG
    elif quant_scheme == 'w_int8_a_int8_per_token_dynamic':  # only support batch_size=1 for activation per-token
        global_quant_config = W_INT8_A_INT8_PER_TOKEN_DYNAMIC_CONFIG
    elif quant_scheme == 'w_bfp16':
        global_quant_config = W_BFP16_CONFIG
    elif quant_scheme == 'w_bfp16_a_bfp16':
        global_quant_config = W_BFP16_A_BFP16_CONFIG
    elif quant_scheme == 'w_mx6':
        global_quant_config = W_MX6_CONFIG
    elif quant_scheme == 'w_mx6_a_mx6':
        global_quant_config = W_MX6_A_MX6_CONFIG
    elif quant_scheme == 'w_mx_fp4_a_mx_fp4_per_group_sym':
        global_quant_config = W_MX_FP4_A_MX_FP4_PER_GROUP_SYM_CONFIG
    elif quant_scheme == 'w_mx_fp4_a_mx_fp8_per_group_sym':
        global_quant_config = W_MX_FP4_A_MX_FP8_PER_GROUP_SYM_CONFIG
    elif quant_scheme == 'w_mx_fp4_per_group_a_fp8_per_tensor_static_sym':
        global_quant_config = W_MX_FP4_PER_GROUP_A_FP8_PER_TENSOR_STATIC_SYM_CONFIG
    elif quant_scheme == 'w_mx_fp4_per_group_a_fp8_per_tensor_dynamic_sym':
        global_quant_config = W_MX_FP4_PER_GROUP_A_FP8_PER_TENSOR_DYNAMIC_SYM_CONFIG
    elif quant_scheme == 'w_mx_fp4_a_mx_fp6e2m3_per_group_sym':
        global_quant_config = W_MX_FP4_A_MX_FP6E2M3_PER_GROUP_SYM_CONFIG
    elif quant_scheme == 'w_mx_fp4_a_mx_fp6e3m2_per_group_sym':
        global_quant_config = W_MX_FP4_A_MX_FP6E3M2_PER_GROUP_SYM_CONFIG
    elif quant_scheme == 'w_mx_fp6e2m3_a_mx_fp6e2m3_per_group_sym':
        global_quant_config = W_MX_FP6E2M3_A_MX_FP6E2M3_PER_GROUP_SYM_CONFIG
    elif quant_scheme == 'w_mx_fp6e3m2_a_mx_fp6e3m2_per_group_sym':
        global_quant_config = W_MX_FP6E3M2_A_MX_FP6E3M2_PER_GROUP_SYM_CONFIG
    # quant_scheme for release test.
    elif quant_scheme == 'w_fp8_per_channel_sym':
        global_quant_config = W_FP8_A_FP8_O_FP8_PER_CHANNEL_SYM_CONFIG
    elif quant_scheme == 'w_mx_fp6_e3m2_per_group_sym':
        global_quant_config = W_MX_FP6_E3M2_PER_GROUP_SYM_CONFIG
    elif quant_scheme == 'w_mx_fp6_e2m3_per_group_sym':
        global_quant_config = W_MX_FP6_E2M3_PER_GROUP_SYM_CONFIG
    elif quant_scheme == 'w_mx_fp4_per_group_sym':
        global_quant_config = W_MX_FP4_PER_GROUP_SYM_CONFIG
    elif quant_scheme == "w_mx_fp4_a_fp8_per_group_sym":
        global_quant_config = W_MX_FP4_A_FP8_PER_GROUP_SYM_CONFIG
    elif quant_scheme == 'w_mx_fp4_a_mx_fp4_dyn_per_group_sym':
        global_quant_config = W_MX_FP4_A_MX_FP4_DYN_PER_GROUP_SYM_CONFIG
    elif quant_scheme == 'w_mx_int8_per_group_sym':
        global_quant_config = W_MX_INT8_PER_GROUP_SYM_CONFIG
    elif quant_scheme == 'w_bfp16_per_group_sym':
        global_quant_config = W_BFP16_PER_GROUP_SYM_CONFIG
    elif quant_scheme == 'w_int4_per_channel_asym':
        global_quant_config = W_INT4_PER_CHANNEL_ASYM_CONFIG
    elif quant_scheme == 'w_int4_per_group_asym':
        global_quant_config = W_INT4_PER_GROUP_ASYM_CONFIG
        global_quant_config.weight.set_group_size(group_size)
    elif quant_scheme == 'w_uint4_per_group_sym':
        global_quant_config = W_UINT4_PER_GROUP_SYM_CONFIG
        global_quant_config.weight.set_group_size(group_size)
    elif quant_scheme == 'w_uint4_per_channel_sym':
        global_quant_config = W_UINT4_PER_CHANNEL_SYM_CONFIG
    elif quant_scheme == 'w_uint4_per_channel_asym':
        global_quant_config = W_UINT4_PER_CHANNEL_ASYM_CONFIG
    elif quant_scheme == 'w_int8_per_tensor_percentile':
        global_quant_config = W_INT8_A_INT8_PER_TENSOR_PERCENTILE_CONFIG
    elif quant_scheme == 'w_int8_per_tensor_mse':
        global_quant_config = W_INT8_A_INT8_PER_TENSOR_MSE_CONFIG
    elif quant_scheme == 'w_uint8_per_tensor_percentile':
        global_quant_config = W_UINT8_A_UINT8_PER_TENSOR_PERCENTILE_CONFIG
    elif quant_scheme == 'w_uint8_per_tensor_mse':
        global_quant_config = W_UINT8_A_UINT8_PER_TENSOR_MSE_CONFIG
    elif quant_scheme == 'w_uint4_per_group_a_int8_per_tensor':
        global_quant_config = W_UINT4_PER_GROUP_A_INT8_PER_TENSOR_CONFIG
    elif quant_scheme == 'w_uint4_per_channel_a_int8_per_tensor':
        global_quant_config = W_UINT4_PER_CHANNEL_A_INT8_PER_TENSOR_CONFIG
    elif quant_scheme == 'w_uint4_per_token_a_int8_per_channel':
        global_quant_config = W_UINT4_PER_TOKEN_A_INT8_PER_CHANNEL
    elif quant_scheme == 'w_uint4_a_uint4_per_channel':
        global_quant_config = W_UINT4_A_UINT4_PER_CHANNEL
    else:
        raise ValueError(f'please set global quant config for {quant_scheme} at customized_configuration.py')
    return global_quant_config
