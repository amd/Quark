#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
""" **AMD Quark** is a comprehensive cross-platform toolkit designed to simplify and
enhance the quantization of deep learning models. Supporting both PyTorch and
ONNX models, Quark empowers developers to optimize their models for deployment
on a wide range of hardware backends, achieving significant performance gains
without compromising accuracy.

For further details on the features and capabilities of Quark, please refer to the

* [Documentation](https://quark.docs.amd.com>)
* [Pytorch examples](https://quark.docs.amd.com/latest/pytorch/pytorch_examples.html>)
* [ONNX examples](https://quark.docs.amd.com/latest/onnx/onnx_examples.html>).

"""

try:
    from .version import __version__  # type: ignore[unused-ignore, import-not-found]
except ImportError:
    __version__ = 'unknown'
