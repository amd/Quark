#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import onnx
import numpy as np
import torch

from .create_model_ops import (convert_ops_to_modules, set_modules_original_weight, get_modules_optimized_weight,
                               set_modules_original_bias, get_modules_optimized_bias)
from numpy.typing import NDArray
from typing import Any


class TorchModel(torch.nn.Module):
    """
    A torch model converted from a onnx model.
    """

    def __init__(self, onnx_model: onnx.ModelProto) -> None:
        """
        Supports basic structure of "compute_op + act_op" (e.g. comv + relu),
        there may be an additional "pad" module for ONNX's asymmetric pads conversion.
        """
        super().__init__()

        self._onnx_model = onnx_model

        self._module, self._module_pad, self._module_act, self._output_qdq = convert_ops_to_modules(self._onnx_model)

        self._input_name = self._onnx_model.graph.input[0].name
        self._output_name = self._onnx_model.graph.output[0].name

    def forward(self, inputs: torch.Tensor) -> Any:
        """ Support the models with single input and single output """
        if self._module_pad is not None:
            tensor = self._module_pad(inputs)
        else:
            tensor = inputs

        assert (self._module is not None), "self _module is None"
        outputs = self._module(tensor)

        if self._module_act is not None:
            outputs = self._module_act(outputs)

        if self._output_qdq is not None:
            outputs = self._output_qdq(outputs)

        return outputs

    def set_weight(self, weight: NDArray[np.float32]) -> None:
        """ Set the original float weight for the compute module """
        assert (self._module is not None), "self._module is None"
        set_modules_original_weight(self._module, weight)

    def get_weight(self) -> Any:
        """ Get the optimized quantized weight of the compute module """
        assert (self._module is not None), "self._module is None"
        return get_modules_optimized_weight(self._module)

    def set_bias(self, bias: NDArray[np.float32]) -> None:
        """ Set the original float bias for the compute module """
        assert (self._module is not None), "self._module is None"
        set_modules_original_bias(self._module, bias)

    def get_bias(self) -> Any:
        """ Get the optimized quantized bias of the compute module """
        assert (self._module is not None), "self._module is None"
        return get_modules_optimized_bias(self._module)
