# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import enum


# this should be kept in sync with the C++ version in onnx_custom_ops/dynamic_dispatch/external_buffers.hpp
class ModelType(enum.IntEnum):
    UNET = 0
    DECODER = enum.auto()
    UNET_BFP = enum.auto()
    DECODER_BFP = enum.auto()
    SD15_UNET = enum.auto()
    SD15_DECODER = enum.auto()
    SD30_MMDIT = enum.auto()
    SD30_VAE = enum.auto()
    LLM_PREFILL = enum.auto()
    LLM_TOKEN = enum.auto()
    UNKNOWN = enum.auto()
