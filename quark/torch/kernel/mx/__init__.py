#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch

from quark.torch.kernel.mx.hip import dq_mxfp4_hip, qdq_mxfp4_hip
from quark.torch.quantization import debug
from quark.shares.utils.import_utils import is_triton_available

if is_triton_available():  # pragma: no cover
    from quark.torch.kernel.mx.triton import dq_mxfp4_triton, qdq_mxfp4_triton  # type: ignore[attr-defined]
else:

    def _raise_import_error_when_used(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError("Failed to import MX-FP4 quantization kernels for Triton: "
                          "Please ensure triton is installed correctly.")

    dq_mxfp4_triton = _raise_import_error_when_used
    qdq_mxfp4_triton = _raise_import_error_when_used

__all__ = ["dq_mxfp4", "qdq_mxfp4"]

if debug.QUARK_MXFP4_IMPL == "hip":
    dq_mxfp4 = dq_mxfp4_hip
    qdq_mxfp4 = qdq_mxfp4_hip
elif debug.QUARK_MXFP4_IMPL == "triton":  # pragma: no cover
    dq_mxfp4 = dq_mxfp4_triton
    qdq_mxfp4 = qdq_mxfp4_triton
else:
    raise ValueError(f"Unsupported QUARK_MXFP4_IMPL='{debug.QUARK_MXFP4_IMPL}'.")
