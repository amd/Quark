#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import onnx
import numpy as np

from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase


class QDQSoftmax(QDQOperatorBase):  # type: ignore

    def quantize(self) -> None:
        super().quantize()
        output_name = self.node.output[0]

        if self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.UINT8:
            if self.quantizer.is_activation_symmetric:
                out_scale = np.array(1 / 128.0, dtype=np.float32)
                out_zero_point = np.array(128, dtype=np.uint8)
            else:
                out_scale = np.array(1 / 256.0, dtype=np.float32)
                out_zero_point = np.array(0, dtype=np.uint8)
        elif self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.INT8:
            if self.quantizer.is_activation_symmetric:
                out_scale = np.array(1 / 128.0, dtype=np.float32)
                out_zero_point = np.array(0, dtype=np.int8)
            else:
                out_scale = np.array(1 / 256.0, dtype=np.float32)
                out_zero_point = np.array(-128, dtype=np.int8)
        elif self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.UINT16:
            if self.quantizer.is_activation_symmetric:
                out_scale = np.array(1 / 32768.0, dtype=np.float32)
                out_zero_point = np.array(32768, dtype=np.uint16)
            else:
                out_scale = np.array(1 / 65536.0, dtype=np.float32)
                out_zero_point = np.array(0, dtype=np.uint16)
        elif self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.INT16:
            if self.quantizer.is_activation_symmetric:
                out_scale = np.array(1 / 32768.0, dtype=np.float32)
                out_zero_point = np.array(0, dtype=np.int16)
            else:
                out_scale = np.array(1 / 65536.0, dtype=np.float32)
                out_zero_point = np.array(-32768, dtype=np.int16)
        elif self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.UINT32:
            if self.quantizer.is_activation_symmetric:
                out_scale = np.array(1. / 2**31, dtype=np.float32)
                out_zero_point = np.array(2**31, dtype=np.uint32)
            else:
                out_scale = np.array(1. / 2**32, dtype=np.float32)
                out_zero_point = np.array(0, dtype=np.uint32)
        elif self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.INT32:
            if self.quantizer.is_activation_symmetric:
                out_scale = np.array(1. / 2**31, dtype=np.float32)
                out_zero_point = np.array(0, dtype=np.int32)
            else:
                out_scale = np.array(1. / 2**32, dtype=np.float32)
                out_zero_point = np.array(-2**31, dtype=np.int32)
        else:
            out_scale = np.array(1, dtype=np.float32)
            out_zero_point = np.array(0, dtype=np.float32)
        self.quantizer.set_quant_scale_zp(output_name, (out_scale, out_zero_point))
