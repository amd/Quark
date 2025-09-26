#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from enum import Enum, auto
from typing import Dict

import torch

from quark.shares.data_type import (
    BaseBFloat16,
    BaseBFP16,
    BaseDataType,
    BaseFloat16,
    BaseFP4,
    BaseFP6_E2M3,
    BaseFP6_E3M2,
    BaseFP8_E4M3,
    BaseFP8_E5M2,
    BaseInt2,
    BaseInt3,
    BaseInt4,
    BaseInt8,
    BaseInt16,
    BaseInt32,
    BaseMX,
    BaseMX6,
    BaseMX9,
    BaseUInt4,
    BaseUInt8,
    BaseUInt16,
    BaseUInt32,
)


class QSchemeType(Enum):
    """
    The quantization schemes applicable to tensors within a model.

    - `per_tensor`: Quantization is applied uniformly across the entire tensor.
    - `per_channel`: Quantization parameters differ across channels of the tensor.
    - `per_group`: Quantization parameters differ across defined groups of weight tensor elements.

    """

    per_tensor = "per_tensor"
    per_channel = "per_channel"
    per_group = "per_group"


class ZeroPointType(Enum):
    """
    The zero point Dtype used for zero point.

    - 'int32': int zero point
    - 'float32': float zero point
    """

    int32 = "int32"
    float32 = "float32"


class Int4(BaseInt4):
    """Signed 4-bit integer quantization data type."""

    torch_packed_dtype = torch.int32


class UInt4(BaseUInt4):
    """Unsigned 4-bit integer quantization data type."""

    torch_packed_dtype = torch.int32


class Int8(BaseInt8):
    """Signed 8-bit integer quantization data type."""

    torch_packed_dtype = torch.int8


class UInt8(BaseUInt8):
    """Unsigned 8-bit integer quantization data type."""

    torch_packed_dtype = torch.uint8


class Int16(BaseInt16):
    """Signed 16-bit integer quantization data type."""

    torch_packed_dtype = torch.int16


class UInt16(BaseUInt16):
    """Unsigned 16-bit integer quantization data type."""

    torch_packed_dtype = torch.int16


class Int32(BaseInt32):
    """Signed 32-bit integer quantization data type."""

    torch_packed_dtype = torch.int32


class UInt32(BaseUInt32):
    """Unsigned 32-bit integer quantization data type."""

    torch_packed_dtype = torch.uint32


class Float16(BaseFloat16):
    """16-bit floating point quantization data type."""

    torch_packed_dtype = torch.float16


class BFloat16(BaseBFloat16):
    """16-bit Brain Floating Point quantization data type."""

    torch_packed_dtype = torch.bfloat16


class BFP16(BaseBFP16):
    """Block Floating Point data type."""

    pass


class MX(BaseMX):
    """Microscaling data type."""

    pass


class Int2(BaseInt2):
    """Signed 2-bit integer quantization data type."""

    torch_packed_dtype = torch.int32


class Int3(BaseInt3):
    """Signed 3-bit integer quantization data type."""

    torch_packed_dtype = torch.uint8


class FP8_E5M2(BaseFP8_E5M2):
    """8-bit floating point with E5M2 format."""

    torch_packed_dtype = torch.float8_e5m2


class FP8_E4M3(BaseFP8_E4M3):
    """8-bit floating point with E4M3 format."""

    torch_packed_dtype = torch.float8_e4m3fn


class FP6_E3M2(BaseFP6_E3M2):
    """6-bit floating point with E3M2 format."""

    torch_packed_dtype = torch.uint8


class FP6_E2M3(BaseFP6_E2M3):
    """6-bit floating point with E2M3 format."""

    torch_packed_dtype = torch.uint8


class FP4(BaseFP4):
    """4-bit floating point quantization data type."""

    torch_packed_dtype = torch.uint8


class MX6(BaseMX6):
    """6-bit microscaling data type."""

    pass


class MX9(BaseMX9):
    """9-bit microscaling data type."""

    pass


SUPPORT_DATA_TYPE = [
    Int8,
    UInt8,
    UInt16,
    Int16,
    Int32,
    UInt32,
    Int4,
    UInt4,
    Int3,
    Int2,
    BFloat16,
    Float16,
    FP8_E5M2,
    FP8_E4M3,
    FP6_E3M2,
    FP6_E2M3,
    FP4,
    MX,
    MX6,
    MX9,
    BFP16,
]


class Dtype(Enum):
    """
    The data types used for quantization of tensors.

    - `int8`: Signed 8-bit integer, range from -128 to 127.
    - `uint8`: Unsigned 8-bit integer, range from 0 to 255.
    - `int16`: Signed 16-bit integer, range from -2**15(-32768) to 2**15 - 1(32767).
    - `uint16`: USigned 16-bit integer, range from 0 to 65535.
    - `int32`: Signed 32-bit integer, range from -2**31 to 2**31 - 1.
    - `int4`: Signed 4-bit integer, range from -8 to 7.
    - `uint4`: Unsigned 4-bit integer, range from 0 to 15.
    - `int3`: Signed 3-bit integer, range from -4 to 3.
    - `int2`: Signed 2-bit integer, range from -2 to 1.
    - `bfloat16`: Bfloat16 format.
    - `float16`: Standard 16-bit floating point format.
    - `fp8_e4m3`: FP8 format with 4 exponent bits and 3 bits of mantissa.
    - `fp8_e5m2`: FP8 format with 5 exponent bits and 2 bits of mantissa.
    - `fp6_e3m2`: FP6 format with 3 exponent bits and 2 bits of mantissa.
    - `fp6_e2m3`: FP6 format with 2 exponent bits and 3 bits of mantissa.
    - `fp4`: FP4 format.
    - `mx`: MX format 8 bit shared exponent with specific element data types.
    - `mx6`, `mx9`: Block data representation with multi-level ultra-fine scaling factors.

    """

    int8 = Int8.__name__.lower()
    uint8 = UInt8.__name__.lower()
    uint16 = UInt16.__name__.lower()
    int16 = Int16.__name__.lower()
    int32 = Int32.__name__.lower()
    int4 = Int4.__name__.lower()
    uint4 = UInt4.__name__.lower()
    int3 = Int3.__name__.lower()
    int2 = Int2.__name__.lower()
    bfloat16 = BFloat16.__name__.lower()
    float16 = Float16.__name__.lower()
    fp8_e5m2 = FP8_E5M2.__name__.lower()
    fp8_e4m3 = FP8_E4M3.__name__.lower()
    fp6_e3m2 = FP6_E3M2.__name__.lower()
    fp6_e2m3 = FP6_E2M3.__name__.lower()
    fp4 = FP4.__name__.lower()
    mx = MX.__name__.lower()
    mx6 = MX6.__name__.lower()
    mx9 = MX9.__name__.lower()
    bfp16 = BFP16.__name__.lower()

    @staticmethod
    def from_torch_dtype(torch_dtype: torch.dtype) -> "Dtype":
        if torch_dtype in TORCH_TO_DTYPE_MAP:
            return TORCH_TO_DTYPE_MAP[torch_dtype]
        else:
            raise ValueError(f"The torch dtype {torch_dtype} does not correspond to a dtype in quark.")

    @staticmethod
    def from_str(s: str) -> "Dtype":
        assert s is not None, "String dtype is None"
        s = s.lower()
        if hasattr(Dtype, s):
            return getattr(Dtype, s)  # type: ignore
        else:
            raise ValueError("Undefined dtype", s)

    def to_bitwidth(self) -> int:  # pragma: no cover
        try:
            for k in SUPPORT_DATA_TYPE:
                if self.value == k.__name__.lower():
                    return k.bitwidth
            raise ValueError(f"Unknown bitwidth for dtype: {self.value}")
        except KeyError:
            raise ValueError(f"Unknown bitwidth for dtype: {self.value}")

    def to_torch_packed_dtype(self) -> torch.dtype:  # pragma: no cover
        if self.value in UNSUPPORTED_TYPES:
            raise NotImplementedError(
                f"Serialization of {self.value} models is not yet supported in Quark. Please open an issue."
            )

        for k in SUPPORT_DATA_TYPE:
            try:
                if self.value == k.__name__.lower():
                    if hasattr(k, "torch_packed_dtype"):
                        return k.torch_packed_dtype
                    else:
                        f"Unknown Dtype: {self.value}, missing attribute torch_packed_dtype in '{k}'"
            except KeyError:
                raise ValueError(f"Unknown Dtype: {self.value}")


TORCH_TO_DTYPE_MAP: dict[torch.dtype, Dtype] = {
    torch.int32: Dtype.int32,
    torch.int16: Dtype.int16,
    torch.uint16: Dtype.uint16,
    torch.int8: Dtype.int8,
    torch.uint8: Dtype.uint8,
    torch.bfloat16: Dtype.bfloat16,
    torch.float16: Dtype.float16,
    torch.float8_e4m3fn: Dtype.fp8_e4m3,
    torch.float8_e5m2: Dtype.fp8_e5m2,
}

UNSUPPORTED_TYPES = frozenset(["mx", "mx6", "mx9", "bfp16"])

ALL_DATA_TYPES = list(Dtype.__members__.values())


class ScaleType(Enum):
    """
    The types of scales used in quantization.

    - `float`: Scale values are floating-point numbers. They use the same floating point dtype as the original model dtype.
    - `pof2`: Scale values are powers of two.
    - `float32`: Scale values are float32 numbers.
    - `float16`: Scale values are float16 numbers.
    - `bfloat16`: Scale values are bfloat16 numbers.
    """

    float = "float"
    pof2 = "pof2"
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    def to_torch_dtype(self) -> torch.dtype:
        if self.value == "float16":
            return torch.float16
        elif self.value == "bfloat16":
            return torch.bfloat16
        elif self.value == "float32":
            return torch.float32
        else:
            raise ValueError(
                "ScaleType.float and ScaleType.pof2 could be implemented with various torch dtype. The method `ScaleType.to_torch_dtype` should not be called with these values."
            )


class RoundType(Enum):
    """
    The rounding methods used during quantization.

    - `round`: Rounds.
    - `floor`: Floors towards the nearest even number.
    - `half_even`: Rounds towards the nearest even number.

    """

    round = 2
    floor = 3
    half_even = 8


class DeviceType(Enum):
    """
    The target devices for model deployment and optimization.

    - `CPU`: CPU.
    - `IPU`: IPU.
    """

    CPU = "cpu"
    IPU = "ipu"


class QuantizationMode(Enum):
    """
    Different quantization modes.

    - `eager_mode`: The eager mode based on PyTorch in-place operator replacement.
    - `fx_graph_mode`: The graph mode based on torch.fx.
    """

    eager_mode = auto()
    fx_graph_mode = auto()


class TQTThresholdInitMeth(Enum):
    """
    The method of threshold initialization of TQT algorithm in QAT. See Table 2 in https://arxiv.org/pdf/1903.08066.pdf

    - `_3SD`: The method of threshold initialization with std and 3 as hyperparameters.
    - `_LL_J`: The method of threshold initialization in the Algorithm 1 of paper "Quantizing Convolutional Neural Networks for Low-Power High-Throughput Inference Engines" - Sean Settle et al. https://arxiv.org/pdf/1805.07941.pdf
    """

    _3SD = "_3sd"
    _KL_J = "_kl_j"
