#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Mapping of parameters to pre-defined Brevitas quantizers."""

import quark.torch.quantization.config.type as quark_config_type
import quark.torch.extensions.brevitas.config as brevitas_config
import brevitas.quant.scaled_int as brevitas_scaled_int  # type: ignore[import-not-found]
import brevitas.quant.shifted_scaled_int as brevitas_shifted_scaled_int  # type: ignore[import-not-found]
import brevitas.quant.fixed_point as brevitas_fixed_point  # type: ignore[import-not-found]
import brevitas.quant.experimental.float as brevitas_float  # type: ignore[import-not-found]

WEIGHT_QUANT_MAP = {
    brevitas_config.QuantType.int_quant: {
        quark_config_type.ScaleType.float: {
            brevitas_config.ParamType.stats: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_scaled_int.Int8WeightPerTensorFloat,
                    'asym': brevitas_shifted_scaled_int.ShiftedUint8WeightPerTensorFloat
                },
                quark_config_type.QSchemeType.per_channel: {
                    'sym': brevitas_scaled_int.Int8WeightPerChannelFloat,
                    'asym': brevitas_shifted_scaled_int.ShiftedUint8WeightPerChannelFloat
                }
            },
            brevitas_config.ParamType.mse: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_scaled_int.Int8WeightPerTensorFloatMSE,
                    'asym': brevitas_shifted_scaled_int.ShiftedUint8WeightPerTensorFloatMSE
                },
                quark_config_type.QSchemeType.per_channel: {
                    'sym': brevitas_scaled_int.Int8WeightPerChannelFloatMSE,
                    'asym': brevitas_shifted_scaled_int.ShiftedUint8WeightPerChannelFloatMSE
                },
            },
        },
        quark_config_type.ScaleType.pof2: {
            brevitas_config.ParamType.stats: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_fixed_point.Int8WeightPerTensorFixedPoint
                },
                quark_config_type.QSchemeType.per_channel: {
                    'sym': brevitas_fixed_point.Int8WeightPerChannelFixedPoint
                },
            },
            brevitas_config.ParamType.mse: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_fixed_point.Int8WeightPerTensorFixedPointMSE
                },
                quark_config_type.QSchemeType.per_channel: {
                    'sym': brevitas_fixed_point.Int8WeightPerChannelFixedPointMSE
                }
            },
        }
    },
    brevitas_config.QuantType.float_quant: {
        quark_config_type.ScaleType.float: {
            brevitas_config.ParamType.stats: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_float.Fp8e4m3WeightPerTensorFloat
                },
                quark_config_type.QSchemeType.per_channel: {
                    'sym': brevitas_float.Fp8e4m3WeightPerChannelFloat
                }
            },
            brevitas_config.ParamType.mse: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_float.Fp8e4m3WeightPerTensorFloatMSE
                },
                quark_config_type.QSchemeType.per_channel: {
                    'sym': brevitas_float.Fp8e4m3WeightPerChannelFloatMSE
                }
            }
        }
    }
}

INPUT_QUANT_MAP = {
    brevitas_config.QuantType.int_quant: {
        quark_config_type.ScaleType.float: {
            brevitas_config.ParamType.stats: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_scaled_int.Int8ActPerTensorFloat,
                    'asym': brevitas_shifted_scaled_int.ShiftedUint8ActPerTensorFloat
                }
            },
            brevitas_config.ParamType.mse: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_scaled_int.Int8ActPerTensorFloatMSE,
                    'asym': brevitas_shifted_scaled_int.ShiftedUint8ActPerTensorFloatMSE
                }
            }
        },
        quark_config_type.ScaleType.pof2: {
            brevitas_config.ParamType.stats: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_fixed_point.Int8ActPerTensorFixedPoint,
                    'asym': brevitas_shifted_scaled_int.ShiftedUint8ActPerTensorFixedPoint
                },
            },
            brevitas_config.ParamType.mse: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_fixed_point.Int8ActPerTensorFixedPointMSE
                }
            },
        }
    },
    brevitas_config.QuantType.float_quant: {
        quark_config_type.ScaleType.float: {
            brevitas_config.ParamType.stats: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_float.Fp8e4m3ActPerTensorFloat
                }
            },
            brevitas_config.ParamType.mse: {
                quark_config_type.QSchemeType.per_tensor: {
                    'sym': brevitas_float.Fp8e4m3ActPerTensorFloat
                },
            }
        }
    }
}

BIAS_QUANT_MAP = {
    8: brevitas_scaled_int.Int8Bias,
    16: brevitas_scaled_int.Int16Bias,
    24: brevitas_scaled_int.Int24Bias,
    32: brevitas_scaled_int.Int32Bias
}
