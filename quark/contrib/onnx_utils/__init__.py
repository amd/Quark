# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This is a temporary wrapper around onnx_utils to allow it to be used as a
standalone package as well as being imported from Quark. Once the passes and
infrastructure is migrated to Quark proper, this file can be removed.
"""

try:
    import ryzenai_onnx_utils
except ImportError:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    import ryzenai_onnx_utils

__all__ = ["ryzenai_onnx_utils"]
