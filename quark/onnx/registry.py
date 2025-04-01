#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from onnx import NodeProto
from onnxruntime.quantization.operators.activation import (QDQRemovableActivation, QLinearActivation)
from onnxruntime.quantization.operators.argmax import QArgMax
from onnxruntime.quantization.operators.attention import AttentionQuant
from onnxruntime.quantization.operators.base_operator import QuantOperatorBase
from onnxruntime.quantization.operators.binary_op import QLinearBinaryOp
from onnxruntime.quantization.operators.concat import QLinearConcat
from onnxruntime.quantization.operators.conv import (ConvInteger, QDQConv, QLinearConv)
from onnxruntime.quantization.operators.direct_q8 import (Direct8BitOp, QDQDirect8BitOp)
from onnxruntime.quantization.operators.embed_layernorm import EmbedLayerNormalizationQuant
from onnxruntime.quantization.operators.gather import GatherQuant, QDQGather
from onnxruntime.quantization.operators.gavgpool import QGlobalAveragePool
from onnxruntime.quantization.operators.gemm import QDQGemm, QLinearGemm
from onnxruntime.quantization.operators.lstm import LSTMQuant
from onnxruntime.quantization.operators.matmul import (MatMulInteger, QDQMatMul, QLinearMatMul)
from onnxruntime.quantization.operators.maxpool import QDQMaxPool, QMaxPool
from onnxruntime.quantization.operators.norm import QDQNormalization
from onnxruntime.quantization.operators.pad import QPad
from onnxruntime.quantization.operators.pooling import QLinearPool
from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
from onnxruntime.quantization.operators.resize import QDQResize, QResize
from onnxruntime.quantization.operators.softmax import QLinearSoftmax
from onnxruntime.quantization.operators.split import QDQSplit, QSplit
from onnxruntime.quantization.operators.where import QDQWhere, QLinearWhere
from onnxruntime.quantization.quant_utils import QuantizationMode
from typing import Union, Any

from .operators.vai_ops.hardsigmoid import QDQHardSigmoid
from .operators.vai_ops.layernorm import QDQLayerNorm
from .operators.vai_ops.prelu import QDQPRelu
from .quant_utils import is_ort_version_below

CommonOpsRegistry = {
    "Gather": GatherQuant,
    "Transpose": Direct8BitOp,
    "EmbedLayerNormalization": EmbedLayerNormalizationQuant,
}

IntegerOpsRegistry = {
    "Conv": ConvInteger,
    "MatMul": MatMulInteger,
    "Attention": AttentionQuant,
    "LSTM": LSTMQuant,
}
IntegerOpsRegistry.update(CommonOpsRegistry)

QLinearOpsRegistry = {
    "ArgMax": QArgMax,
    "Conv": QLinearConv,
    "Gemm": QLinearGemm,
    "MatMul": QLinearMatMul,
    "Add": QLinearBinaryOp,
    "Mul": QLinearBinaryOp,
    "Relu": QLinearActivation,
    "Clip": QLinearActivation,
    "LeakyRelu": QLinearActivation,
    "Sigmoid": QLinearActivation,
    "MaxPool": QMaxPool,
    "GlobalAveragePool": QGlobalAveragePool,
    "Split": QSplit,
    "Pad": QPad,
    "Reshape": Direct8BitOp,
    "Squeeze": Direct8BitOp,
    "Unsqueeze": Direct8BitOp,
    "Resize": QResize,
    "AveragePool": QLinearPool,
    "Concat": QLinearConcat,
    "Softmax": QLinearSoftmax,
    "Where": QLinearWhere,
}
QLinearOpsRegistry.update(CommonOpsRegistry)

QDQRegistry = {
    "Conv": QDQConv,
    "ConvTranspose": QDQConv,
    "Gemm": QDQGemm,
    "Clip": QDQRemovableActivation,
    "Relu": QDQRemovableActivation,
    "Reshape": QDQDirect8BitOp,
    "Transpose": QDQDirect8BitOp,
    "Squeeze": QDQDirect8BitOp,
    "Unsqueeze": QDQDirect8BitOp,
    "Resize": QDQResize,
    "MaxPool": QDQMaxPool,
    "AveragePool": QDQDirect8BitOp,
    "MatMul": QDQMatMul,
    "Split": QDQSplit,
    "Gather": QDQGather,
    "Where": QDQWhere,
    "Tanh": QDQOperatorBase,
    "Gelu": QDQOperatorBase,
}

QDQRegistry["InstanceNormalization"] = QDQNormalization
QDQRegistry["LayerNormalization"] = QDQLayerNorm

if is_ort_version_below("1.18.0"):
    from .operators.vai_ops.softmax import QDQSoftmax
    QDQRegistry["Softmax"] = QDQSoftmax

NPUCnnRegistry = {
    "HardSigmoid": QDQHardSigmoid,
    "LayerNormalization": QDQLayerNorm,
    "Div": QDQOperatorBase,
    "Erf": QDQOperatorBase,
    "Tanh": QDQOperatorBase,
    "Sub": QDQOperatorBase,
    "Min": QDQOperatorBase,
    "Max": QDQOperatorBase,
    "ReduceMean": QDQOperatorBase,
    "DepthToSpace": QDQOperatorBase,
    "SpaceToDepth": QDQOperatorBase,
    "Slice": QDQOperatorBase,
    "PRelu": QDQPRelu,
    "LpNormalization": QDQOperatorBase,
    "AveragePool": QDQOperatorBase,
}

NPUTransformerRegistry = {
    "Gemm": QDQGemm,
    "MatMul": QDQMatMul,
}


def CreateDefaultOpQuantizer(onnx_quantizer: Any, node: NodeProto) -> QuantOperatorBase:
    return QuantOperatorBase(onnx_quantizer, node)


def CreateOpQuantizer(onnx_quantizer: Any, node: NodeProto) -> Union[QuantOperatorBase, Any]:
    registry = IntegerOpsRegistry if onnx_quantizer.mode == QuantizationMode.IntegerOps else QLinearOpsRegistry
    if node.op_type in registry:
        op_quantizer = registry[node.op_type](onnx_quantizer, node)
        if op_quantizer.should_quantize():
            return op_quantizer
    return QuantOperatorBase(onnx_quantizer, node)


def CreateNPUTransformerQDQQuantizer(onnx_quantizer: Any, node: NodeProto) -> Union[QDQOperatorBase, Any]:
    if node.op_type in NPUTransformerRegistry:
        return NPUTransformerRegistry[node.op_type](onnx_quantizer, node)
    else:
        return QDQOperatorBase(onnx_quantizer, node)


def CreateNPUCnnQDQQuantizer(onnx_quantizer: Any, node: NodeProto) -> Union[QDQOperatorBase, Any]:
    if node.op_type in NPUCnnRegistry:
        return NPUCnnRegistry[node.op_type](onnx_quantizer, node)

    elif node.op_type in QDQRegistry:
        return QDQRegistry[node.op_type](onnx_quantizer, node)
    return QDQOperatorBase(onnx_quantizer, node)


def CreateQDQQuantizer(onnx_quantizer: Any, node: NodeProto) -> Union[QDQOperatorBase, Any]:
    if node.op_type in QDQRegistry:
        return QDQRegistry[node.op_type](onnx_quantizer, node)
    return QDQOperatorBase(onnx_quantizer, node)
