#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""ctypes wrapper around llama.cpp libggml quantizers."""

from __future__ import annotations

import ctypes
from math import prod
from pathlib import Path

import numpy as np

from quark.common.utils.import_utils import is_gguf_available_and_minimum_version

if is_gguf_available_and_minimum_version():
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import quant_shape_to_byte_shape
else:
    GGMLQuantizationType = None  # type: ignore[misc, assignment]
    quant_shape_to_byte_shape = None  # type: ignore[assignment]

c_float_p = ctypes.POINTER(ctypes.c_float)


class _ggml_init_params(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


class GgmlQuantizer:
    """Quantize float32 arrays with llama.cpp's native ggml quantizers."""

    def __init__(self, libggml: Path | str) -> None:
        if GGMLQuantizationType is None:
            raise ImportError("gguf>=0.10.0 is required for GgmlQuantizer")

        lib_path = Path(libggml)
        if not lib_path.exists():
            raise FileNotFoundError(f"Missing libggml shared library: {lib_path}")

        self.libggml = ctypes.CDLL(str(lib_path))
        self.libggml.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.libggml.ggml_quantize_chunk.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        )
        self.libggml.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        self.libggml.ggml_quantize_requires_imatrix.argtypes = (ctypes.c_int,)
        self.libggml.ggml_init.argtypes = (_ggml_init_params,)
        self.libggml.ggml_init(_ggml_init_params(1 * 1024 * 1024, None, False))

        for t in (
            "q4_0",
            "q4_1",
            "q5_0",
            "q5_1",
            "q8_0",
            "q2_K",
            "q3_K",
            "q4_K",
            "q5_K",
            "q6_K",
            "tq1_0",
            "tq2_0",
            "mxfp4",
            "nvfp4",
            "iq2_xxs",
            "iq2_xs",
            "iq2_s",
            "iq3_xxs",
            "iq3_s",
            "iq1_s",
            "iq1_m",
            "iq4_nl",
            "iq4_xs",
        ):
            dequant = getattr(self.libggml, "dequantize_row_" + t)
            dequant.restype = None
            dequant.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int64,
            )

    def quantize(self, data: np.ndarray, qtype: "GGMLQuantizationType") -> np.ndarray:
        assert quant_shape_to_byte_shape is not None
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        if data.ndim < 1:
            raise ValueError(f"Cannot quantize scalar array with shape {data.shape}")

        result = np.zeros(
            quant_shape_to_byte_shape(data.shape, qtype),
            dtype=np.uint8,
            order="C",
        )
        if self.libggml.ggml_quantize_requires_imatrix(qtype.value):
            qw = np.sum(
                (data * data).reshape((-1, data.shape[-1])),
                axis=0,
            ).ctypes.data_as(c_float_p)
        else:
            qw = ctypes.cast(0, c_float_p)

        result_size = self.libggml.ggml_quantize_chunk(
            qtype.value,
            data.ctypes.data_as(c_float_p),
            result.ctypes.data_as(ctypes.c_void_p),
            0,
            prod(data.shape[:-1]) if data.ndim > 1 else 1,
            data.shape[-1],
            qw,
        )
        if result.size != result_size:
            raise RuntimeError(
                f"ggml_quantize_chunk size mismatch for {qtype.name}: "
                f"expected {result.size}, got {result_size}"
            )
        return result

    def dequantize(self, data: np.ndarray, qtype: "GGMLQuantizationType") -> np.ndarray:
        from gguf.quants import quant_shape_from_byte_shape

        result = np.zeros(
            quant_shape_from_byte_shape(data.shape, qtype),
            dtype=np.float32,
            order="C",
        )
        lw_qname = qtype.name.lower()
        if lw_qname.endswith("k"):
            lw_qname = lw_qname[:-1] + "K"
        dequant = getattr(self.libggml, "dequantize_row_" + lw_qname)
        dequant(
            data.ctypes.data_as(ctypes.c_void_p),
            result.ctypes.data_as(c_float_p),
            result.size,
        )
        return result
