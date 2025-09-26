#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization Base Data Type Classes"""

from typing import Union


class BaseDataType:
    """
    Base class for representing a quantization data type.

    Attributes:
        bitwidth (int): Number of bits used to represent the value.
        min_value (Union[float, int]): Minimum representable value.
        max_value (Union[float, int]): Maximum representable value.
    """

    bitwidth: int
    min_value: Union[float, int]
    max_value: Union[float, int]


class BaseInt4(BaseDataType):
    """Signed 4-bit integer quantization data type."""

    bitwidth = 4
    min_value = -8
    max_value = 7


class BaseUInt4(BaseDataType):
    """Unsigned 4-bit integer quantization data type."""

    bitwidth = 4
    min_value = 0
    max_value = 15


class BaseInt8(BaseDataType):
    """Signed 8-bit integer quantization data type."""

    bitwidth = 8
    min_value = -128
    max_value = 127


class BaseUInt8(BaseDataType):
    """Unsigned 8-bit integer quantization data type."""

    bitwidth = 8
    min_value = 0
    max_value = 255


class BaseInt16(BaseDataType):
    """Signed 16-bit integer quantization data type."""

    bitwidth = 16
    min_value = -32768
    max_value = 32767


class BaseUInt16(BaseDataType):
    """Unsigned 16-bit integer quantization data type."""

    bitwidth = 16
    min_value = 0
    max_value = 65535


class BaseInt32(BaseDataType):
    """Signed 32-bit integer quantization data type."""

    bitwidth = 32
    min_value = -(2**31)
    max_value = 2**31 - 1


class BaseUInt32(BaseDataType):
    """Unsigned 32-bit integer quantization data type."""

    bitwidth = 32
    min_value = 0
    max_value = 2**32 - 1


class BaseFloat16(BaseDataType):
    """16-bit floating point quantization data type."""

    bitwidth = 16


class BaseBFloat16(BaseDataType):
    """16-bit Brain Floating Point quantization data type."""

    bitwidth = 16


class BaseBFP16(BaseDataType):
    """Block Floating Point data type."""

    bitwidth = 16


class BaseInt2(BaseDataType):
    """Signed 2-bit integer quantization data type."""

    bitwidth = 2
    min_value = -2
    max_value = 1


class BaseMX(BaseDataType):
    """Microscaling data type."""

    pass


class BaseInt3(BaseDataType):
    """Signed 3-bit integer quantization data type."""

    bitwidth = 3
    min_value = -4
    max_value = 3


class BaseFP8_E5M2(BaseDataType):
    """8-bit floating point with E5M2 format."""

    bitwidth = 8
    min_value = -57344.0
    max_value = 57344.0


class BaseFP8_E4M3(BaseDataType):
    """8-bit floating point with E4M3 format."""

    bitwidth = 8
    min_value = -448.0
    max_value = 448.0


class BaseFP6_E3M2(BaseDataType):
    """6-bit floating point with E3M2 format."""

    bitwidth = 6
    min_value = -28.0
    max_value = 28.0


class BaseFP6_E2M3(BaseDataType):
    """6-bit floating point with E2M3 format."""

    bitwidth = 6
    min_value = -7.5
    max_value = 7.5


class BaseFP4(BaseDataType):
    """4-bit floating point quantization data type."""

    bitwidth = 4
    min_value = -6.0
    max_value = 6.0


class BaseMX6(BaseDataType):
    """6-bit microscaling data type."""

    bitwidth = 6


class BaseMX9(BaseDataType):
    """9-bit microscaling data type."""

    bitwidth = 9


# TODO: Add Quark ONNX MX data type classes.
