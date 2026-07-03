#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch  # noqa  # TODO: can we remove this?

from quark.common.utils.import_utils import is_triton_available
from quark.torch.kernel.mx.hip import dq_mxfp4_hip, qdq_mxfp4_hip
from quark.torch.utils import QUARK_MXFP4_IMPL

if is_triton_available():  # pragma: no cover
    from quark.torch.kernel.mx.triton import dq_mxfp4_triton, qdq_mxfp4_triton  # type: ignore[attr-defined]
else:

    def _raise_import_error_when_used(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError(
            "Failed to import MX-FP4 quantization kernels for Triton: Please ensure triton is installed correctly."
        )

    dq_mxfp4_triton = _raise_import_error_when_used
    qdq_mxfp4_triton = _raise_import_error_when_used

__all__ = ["dq_mxfp4", "qdq_mxfp4"]

if QUARK_MXFP4_IMPL == "hip":
    dq_mxfp4 = dq_mxfp4_hip
    qdq_mxfp4 = qdq_mxfp4_hip
elif QUARK_MXFP4_IMPL == "triton":  # pragma: no cover
    dq_mxfp4 = dq_mxfp4_triton
    qdq_mxfp4 = qdq_mxfp4_triton
else:
    raise ValueError(f"Unsupported QUARK_MXFP4_IMPL='{QUARK_MXFP4_IMPL}'.")
