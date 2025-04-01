#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import numpy as np
import onnx
import onnxruntime

import torch
from torch import nn
from quark.onnx.quant_utils import PowerOfTwoMethod
from quark.onnx.finetuning.torch_utils import convert_onnx_to_torch, train_torch_module_api
from quark.onnx.quantize import quantize_static
from typing import List, Any

data_shape = [1, 3, 56, 56]
input_data = np.random.randint(0, high=256, size=data_shape).astype(np.float32)


def create_onnx_model(model_path: str) -> None:
    from onnx.onnx_ml_pb2 import TensorProto
    from onnx import helper

    def _make_initializer_tensor(name: str, dims: List[Any]) -> TensorProto:
        value = np.random.random(dims).astype(np.float32)
        tensor = helper.make_tensor(name=name,
                                    data_type=TensorProto.FLOAT,
                                    dims=list(value.shape),
                                    vals=value.tobytes(),
                                    raw=True)
        return tensor

    feat_channels = 16

    conv_input = helper.make_tensor_value_info('x', TensorProto.FLOAT, data_shape)

    weight = _make_initializer_tensor("weight", [feat_channels, 3, 3, 3])
    bias = _make_initializer_tensor("bias", [feat_channels])

    conv = helper.make_node(
        "Conv",
        name='conv',
        inputs=["x", "weight", 'bias'],
        outputs=["conv_output"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        dilations=[1, 1],
        pads=[1, 1, 1, 1],
    )

    conv_output = helper.make_tensor_value_info('conv_output', TensorProto.FLOAT, [1, feat_channels] + data_shape[2:])

    relu = helper.make_node("Relu", name='relu', inputs=["conv_output"], outputs=["y"])

    relu_output = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, feat_channels] + data_shape[2:])

    graph_def = onnx.helper.make_graph(nodes=[conv, relu],
                                       name="conv_relu",
                                       inputs=[conv_input],
                                       outputs=[relu_output],
                                       initializer=[weight, bias],
                                       value_info=[conv_output, relu_output])

    produce_opset_version = 11  # you could set any opset here
    opset_imports = [onnx.helper.make_operatorsetid("", produce_opset_version)]
    model_def = onnx.helper.make_model(graph_def, producer_name="", ir_version=8, opset_imports=opset_imports)

    onnx.save(model_def, model_path)
    print(f"{model_path} save succeed!")


def run_onnx_model(model_path: str) -> Any:
    so = onnxruntime.SessionOptions()
    sess = onnxruntime.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])

    input_name = sess.get_inputs()[0].name
    # input_data = np.random.randint(
    #    0, high=256, size=sess.get_inputs()[0].shape).astype(np.float32)
    outputs = sess.run(None, {input_name: input_data})[0]

    print(f"{model_path} inference succeed!", outputs[0, 0, 0, :5])
    return outputs


def onnx_to_torch(onnx_model_path: str) -> nn.Module:
    onnx_model = onnx.load(onnx_model_path)

    weight = np.random.random([16, 3, 3, 3]).astype(np.float32)
    torch_model = convert_onnx_to_torch(onnx_model, weight)

    # Use torch.jit.script or torch.jit.trace to save torch model
    # from quark.onnx.finetuning.torch_utils import save_torch_model
    # torch_model_path = onnx_model_path.rstrip(".onnx") + ".pth"
    # save_torch_model(torch_model, torch_model_path)
    # save_torch_model(torch_model, torch_model_path, input_data)

    # Load the saved torch model
    # torch_model = torch.load(torch_model_path)

    # Export onnx model
    # from quark.onnx.finetuning.torch_utils import convert_torch_to_onnx
    # convert_torch_to_onnx(torch_model, input_data)

    print("torch model convert succeed!")
    return torch_model


def run_torch_model(torch_model: nn.Module) -> Any:
    device = torch.device("cpu")
    torch_model = torch_model.to(device)

    inputs = torch.from_numpy(input_data)
    outputs = torch_model(inputs).detach().numpy()

    print("torch model inference succeed!", outputs[0, 0, 0, :5])
    return outputs


def quantize_onnx_model(float_model_path: str) -> str:
    quant_model_path = "quant-" + float_model_path.split('/')[0]

    op_types_to_quantize = ['Conv', 'ConvTranspose', 'MatMul', 'Gemm']

    quantize_static(
        float_model_path,
        quant_model_path,
        None,
        calibrate_method=PowerOfTwoMethod.MinMSE,

        # 8bit quantization
        quant_format=onnxruntime.quantization.quant_utils.QuantFormat.QDQ,
        # activation_type=onnxruntime.quantization.quant_utils.QuantType.QUInt8,
        # weight_type=onnxruntime.quantization.quant_utils.QuantType.QUInt8,
        activation_type=onnxruntime.quantization.quant_utils.QuantType.QInt8,
        weight_type=onnxruntime.quantization.quant_utils.QuantType.QInt8,

        # 16bit quantization
        # quant_format=VitisQuantFormat.QDQ,
        # activation_type=VitisQuantType.QUInt16,
        # weight_type=VitisQuantType.QUInt16,
        # activation_type=VitisQuantType.QInt16,
        # weight_type=VitisQuantType.QInt16,
        # activation_type=VitisQuantType.QFloat16,
        # weight_type=VitisQuantType.QFloat16,
        # activation_type=VitisQuantType.QBFloat16,
        # weight_type=VitisQuantType.QBFloat16,

        # Quantize compute op only
        op_types_to_quantize=op_types_to_quantize,
        extra_options={
            'AddQDQPairToWeight': False,
            'QuantizeBias': True,
            'OpTypesToExcludeOutputQuantization': op_types_to_quantize,
        },
    )

    print(f"{quant_model_path} quantized succeed!")
    return quant_model_path


def train_torch_model(quant_torch_model: nn.Module, out_data_float: Any) -> nn.Module:
    inp_data_quant = input_data
    inp_data_float = input_data

    weight = quant_torch_model.get_weight()
    print("weight before optimize: ", weight[0, 0, 0, :])

    extra_options = {
        'FastFinetune': {
            'BatchSize': 1,
            'NumBatches': 1,
            'NumIterations': 500,
            'LearningRate': 0.001,
            'OptimAlgorithm': 'adaround',
            'OptimDevice': 'cpu',
            'LRAdjust': (),
            'EarlyStop': False,
            'DropRatio': 0.5,
            'RegParam': 0.01,
            'BetaRange': (20, 2),
            'WarmStart': 0.2,
            'LogPeriod': 100,
        }
    }

    weight, _ = train_torch_module_api(quant_torch_model, inp_data_quant, inp_data_float, out_data_float, extra_options)

    print("weight after optimize: ", weight[0, 0, 0, :])
    return quant_torch_model


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=False, type=str, help="model path", default='block_model.onnx')
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    print("=======================float model=======================")

    create_onnx_model(args.model_path)
    onnx_out_data_float = run_onnx_model(args.model_path)

    torch_model = onnx_to_torch(args.model_path)
    torch_out_data_float = run_torch_model(torch_model)

    print("-----------------------quant model-----------------------")

    quant_model_path = quantize_onnx_model(args.model_path)
    onnx_out_data_quant = run_onnx_model(quant_model_path)

    quant_torch_model = onnx_to_torch(quant_model_path)
    torch_out_data_quant = run_torch_model(quant_torch_model)

    print("-----------------------fintuning-----------------------")

    optimized_quant_torch_model = train_torch_model(quant_torch_model, onnx_out_data_float)
    optimized_torch_out_data_quant = run_torch_model(optimized_quant_torch_model)


if __name__ == "__main__":
    main()
