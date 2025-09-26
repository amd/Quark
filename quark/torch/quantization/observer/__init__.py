#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from .lsq_observer import LSQObserver
from .observer import PlaceholderObserver, PerChannelMinMaxObserver, PerChannelPowOf2MinMaxObserver, PerChannelPowOf2MinMSEObserver, PerTensorMinMaxObserver, PerTensorPowOf2MinMaxObserver, PerTensorPowOf2MinMSEObserver, PerTensorHistogramObserver, PerTensorHistogramObserverPro, PerTensorPercentileObserver, PerTensorMSEObserver, PerBlockMXObserver, PerBlockMXDiffsObserver, PerBlockBFPObserver, PerGroupMinMaxObserver, UniformScalingObserver, ObserverBase
from .tqt_observer import TQTObserver

__all__ = [
    "LSQObserver", "PlaceholderObserver", "PerChannelMinMaxObserver", "PerChannelPowOf2MinMaxObserver",
    "PerChannelPowOf2MinMSEObserver", "PerTensorMinMaxObserver", "PerTensorPowOf2MinMaxObserver",
    "PerTensorPowOf2MinMSEObserver", "PerTensorHistogramObserver", "PerTensorHistogramObserverPro",
    "PerTensorPercentileObserver", "PerTensorMSEObserver", "PerBlockMXObserver", "PerBlockMXDiffsObserver",
    "PerBlockBFPObserver", "PerGroupMinMaxObserver", "UniformScalingObserver", "ObserverBase", "TQTObserver"
]

PLACEHOLDER_OBSERVERS = {PlaceholderObserver}
PER_CHANNEL_OBSERVERS = {
    PerChannelMinMaxObserver,
    PerChannelPowOf2MinMaxObserver,
    PerChannelPowOf2MinMSEObserver,
    LSQObserver,
}
PER_TENSOR_OBSERVERS = {
    PerTensorMinMaxObserver, PerTensorPowOf2MinMaxObserver, PerTensorPowOf2MinMSEObserver, PerTensorHistogramObserver,
    PerTensorHistogramObserverPro, PerTensorPercentileObserver, PerTensorMSEObserver, TQTObserver, LSQObserver
}

PER_GROUP_OBSERVERS = {PerBlockMXObserver, PerBlockMXDiffsObserver, PerBlockBFPObserver, PerGroupMinMaxObserver}

OBSERVER_CLASSES = PLACEHOLDER_OBSERVERS | PER_CHANNEL_OBSERVERS | PER_TENSOR_OBSERVERS | PER_GROUP_OBSERVERS | {
    UniformScalingObserver, ObserverBase
}

OBSERVER_MAP = {observer_cls.__name__: observer_cls for observer_cls in OBSERVER_CLASSES}
