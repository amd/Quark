#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from enum import Enum


class Int16Method(Enum):
    MinMax = 0


class PowerOfTwoMethod(Enum):
    NonOverflow = 0
    MinMSE = 1


class LayerWiseMethod(Enum):
    LayerWisePercentile = 0


class ExtendedCalibrationMethod(Enum):
    MinMSE = 0
