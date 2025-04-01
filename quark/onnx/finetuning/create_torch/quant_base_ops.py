#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import numpy
from onnx import onnx_pb as onnx_proto
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, Any, List

from .base_qdq_quantizers import INTQuantizer, FPQuantizer
from .base_fn_quantizers import BFPQuantizer, MXQuantizer

map_onnx_dtype_to_torch = {
    onnx_proto.TensorProto.UINT4: torch.uint8,  # Note: torch does not have a uint4 type
    onnx_proto.TensorProto.INT4: torch.int8,  # Note: torch does not have a int4 type
    onnx_proto.TensorProto.UINT8: torch.uint8,
    onnx_proto.TensorProto.INT8: torch.int8,
    onnx_proto.TensorProto.UINT16: torch.int16,  # Note: torch does not have a uint16 type
    onnx_proto.TensorProto.INT16: torch.int16,
    onnx_proto.TensorProto.UINT32: torch.int32,  # Note: torch does not have a uint32 type
    onnx_proto.TensorProto.INT32: torch.int32,
    onnx_proto.TensorProto.FLOAT16: torch.float16,
    onnx_proto.TensorProto.BFLOAT16: torch.bfloat16,
    onnx_proto.TensorProto.FLOAT: torch.float32,
}


def create_qdq_quantizer(scale: torch.Tensor,
                         zero_point: torch.Tensor,
                         min_q: torch.Tensor,
                         max_q: torch.Tensor,
                         ch_axis: int = 0,
                         q_folded: bool = False,
                         quant_type: torch.dtype = torch.bfloat16) -> Union[INTQuantizer, FPQuantizer]:
    if quant_type in [torch.float16, torch.bfloat16]:
        return FPQuantizer(scale, zero_point, min_q, max_q, ch_axis, q_folded, quant_type)
    else:
        return INTQuantizer(scale, zero_point, min_q, max_q, ch_axis, q_folded)


def create_fn_quantizer(quant_info: Dict[str, Any]) -> Union[BFPQuantizer, MXQuantizer]:
    if 'MX' in quant_info['op_type']:
        return MXQuantizer(quant_info['op_attrs'])
    else:
        return BFPQuantizer(quant_info['op_attrs'])


class QuantizationModule(torch.nn.Module):
    """ A pytorch module that behaves as ONNX quantization nodes """

    def __init__(
        self, quant_info: Union[Tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[Any],
                                      numpy.typing.NDArray[Any], numpy.typing.NDArray[Any], int, bool,
                                      onnx_proto.TensorProto], Dict[str, Any], None]
    ) -> None:
        super().__init__()

        self.quantizer: Optional[Union[INTQuantizer, FPQuantizer, BFPQuantizer, MXQuantizer]] = None

        if isinstance(quant_info, dict) and len(quant_info) >= 2:
            self.quantizer = create_fn_quantizer(quant_info)
        elif isinstance(quant_info, tuple) and len(quant_info) >= 7:
            self.quantizer = create_qdq_quantizer(torch.from_numpy(quant_info[0]), torch.from_numpy(quant_info[1]),
                                                  torch.from_numpy(quant_info[2]), torch.from_numpy(quant_info[3]),
                                                  quant_info[4], quant_info[5], map_onnx_dtype_to_torch[quant_info[6]])
        else:
            raise ValueError("Invalid quantization info: {}".format(quant_info))

    def forward(self, tensor: torch.Tensor) -> Any:
        if self.quantizer is None:
            return tensor
        else:
            return self.quantizer(tensor)


class QuantizeWrapper(torch.nn.Module, ABC):
    """ A wrapper for torch layer's input/weight/bias quantization """

    def __init__(self, w_alpha: float = 1.0, b_beta: float = 1.0, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)

        # These parameters are used to implement ONNX Gemm op's formula,
        # and compatiable with other ops since their default values are 1
        self.w_alpha: float = w_alpha
        self.b_beta: float = b_beta

        self.input_quantizer: Optional[Union[INTQuantizer, FPQuantizer, BFPQuantizer, MXQuantizer]] = None
        self.weight_quantizer: Optional[Union[INTQuantizer, FPQuantizer, BFPQuantizer, MXQuantizer]] = None
        self.bias_quantizer: Optional[Union[INTQuantizer, FPQuantizer, BFPQuantizer, MXQuantizer]] = None

        # This quantize wrapper is for the modules that should have weight
        assert hasattr(self, 'weight') and f'{type(self)} does not contain weight'

        # This is a flag to indicate if we have a accuracy improvement after optimization.
        # If false, do not to get the optimized weight or bias from the wrapperred module.
        self.opt_gained: Optional[bool] = None

    def create_input_quantizer(
        self, quant_info: Union[Tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[Any],
                                      numpy.typing.NDArray[Any], numpy.typing.NDArray[Any], int, bool,
                                      onnx_proto.TensorProto], Dict[str, Any], None]
    ) -> None:
        if isinstance(quant_info, dict) and len(quant_info) >= 2:
            self.input_quantizer = create_fn_quantizer(quant_info)
        elif isinstance(quant_info, tuple) and len(quant_info) >= 7:
            self.input_quantizer = create_qdq_quantizer(torch.from_numpy(quant_info[0]),
                                                        torch.from_numpy(quant_info[1]),
                                                        torch.from_numpy(quant_info[2]),
                                                        torch.from_numpy(quant_info[3]), quant_info[4], quant_info[5],
                                                        map_onnx_dtype_to_torch[quant_info[6]])
        else:
            raise ValueError("Invalid quantization info: {}".format(quant_info))

    def create_weight_quantizer(
        self, quant_info: Union[Tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[Any],
                                      numpy.typing.NDArray[Any], numpy.typing.NDArray[Any], int, bool,
                                      onnx_proto.TensorProto], Dict[str, Any], None]
    ) -> None:
        if isinstance(quant_info, dict) and len(quant_info) >= 2:
            self.weight_quantizer = create_fn_quantizer(quant_info)
        elif isinstance(quant_info, tuple) and len(quant_info) >= 7:
            self.weight_quantizer = create_qdq_quantizer(torch.from_numpy(quant_info[0]),
                                                         torch.from_numpy(quant_info[1]),
                                                         torch.from_numpy(quant_info[2]),
                                                         torch.from_numpy(quant_info[3]), quant_info[4], quant_info[5],
                                                         map_onnx_dtype_to_torch[quant_info[6]])
        else:
            raise ValueError("Invalid quantization info: {}".format(quant_info))

    def create_bias_quantizer(
        self, quant_info: Union[Tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[Any],
                                      numpy.typing.NDArray[Any], numpy.typing.NDArray[Any], int, bool,
                                      onnx_proto.TensorProto], Dict[str, Any], None]
    ) -> None:
        if isinstance(quant_info, dict) and len(quant_info) >= 2:
            self.bias_quantizer = create_fn_quantizer(quant_info)
        elif isinstance(quant_info, tuple) and len(quant_info) >= 7:
            self.bias_quantizer = create_qdq_quantizer(torch.from_numpy(quant_info[0]), torch.from_numpy(quant_info[1]),
                                                       torch.from_numpy(quant_info[2]), torch.from_numpy(quant_info[3]),
                                                       quant_info[4], quant_info[5],
                                                       map_onnx_dtype_to_torch[quant_info[6]])
        else:
            raise ValueError("Invalid quantization info: {}".format(quant_info))

    @abstractmethod
    def forward_impl(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, input: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        if self.input_quantizer is not None:
            tensor = self.input_quantizer(input)
        else:
            tensor = input

        if self.weight_quantizer is not None:
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight.data

        weight = weight * self.w_alpha

        output = self.forward_impl(tensor, weight)

        if self.bias is not None:
            if self.bias_quantizer is not None:
                bias = self.bias_quantizer(self.bias)
            else:
                bias = self.bias.data

            bias = bias * self.b_beta

            # The bias is always 1D tensor (vector). If the output is not 1D,
            # should expand the dims of bias for broadcasting
            if output.dim() >= 2 and output.shape[1] == bias.shape[0]:
                bias = bias.view(1, -1, *([1] * (output.dim() - 2)))

            output = output + bias

        return output
