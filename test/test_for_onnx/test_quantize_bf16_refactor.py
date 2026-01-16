#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from onnx import TensorProto, helper
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization import CalibrationDataReader

from quark.onnx import BFloat16Spec, ModelQuantizer, QConfig, QLayerConfig
from quark.onnx.quantization.quant_utils import convert_to_bf16
from quark.shares.utils.testing_utils import use_temporary_directory

input_tensor = np.array(
    [
        [
            [
                [0.26921557, 0.79500909, 0.6102178, 0.04375664],
                [0.06221361, 0.98258356, 0.38635129, 0.06492238],
                [0.49631707, 0.35442799, 0.51719146, 0.52100111],
                [0.04145599, 0.88960236, 0.50627326, 0.57204613],
            ],
            [
                [0.99185097, 0.93582153, 0.13174529, 0.42896287],
                [0.14552133, 0.02538564, 0.0732355, 0.25725371],
                [0.09856916, 0.43015628, 0.55679755, 0.66560074],
                [0.9439425, 0.45701841, 0.86791293, 0.64728276],
            ],
            [
                [0.29159685, 0.79021383, 0.3117182, 0.11342342],
                [0.16660495, 0.46426165, 0.31348552, 0.143383],
                [0.96454802, 0.63258874, 0.30295267, 0.96720039],
                [0.29879457, 0.79916527, 0.02905061, 0.20115725],
            ],
        ]
    ]
).astype(np.float32)

output_golden = np.array([[-0.5097736]], dtype=np.float32)


class DataReader(CalibrationDataReader):
    def __init__(self, input_tensor):
        self.data = [input_tensor]
        self.input_name = "input"
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            input_dict = {self.input_name: self.data[self.index]}
            self.index += 1
            return input_dict
        else:
            return None

    def rewind(self):
        self.index = 0


class DoubleConvModel(nn.Module):
    def __init__(self):
        super(DoubleConvModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1, 1)

        with torch.no_grad():
            self.conv2.weight *= 100.0

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def prepare_add_model(tmp_path):
    initializer_value = np.full((1, 3, 4, 4), 3.39e38, dtype=np.float32)
    initializer_tensor = helper.make_tensor(
        name="initializer_tensor",
        data_type=TensorProto.FLOAT,
        dims=initializer_value.shape,
        vals=initializer_value.flatten().tolist(),
    )

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])

    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])

    node = helper.make_node("Add", inputs=["input", "initializer_tensor"], outputs=["output"])

    graph = helper.make_graph(
        nodes=[node],
        name="example_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[initializer_tensor],
    )

    opset_version = 13
    model = helper.make_model(
        graph, producer_name="example_model", opset_imports=[helper.make_opsetid("", opset_version)], ir_version=10
    )

    onnx_model_path = Path(tmp_path, "add_model.onnx").as_posix()
    onnx_quantized_model_path = Path(tmp_path, "add_bf16_model.onnx").as_posix()
    onnx.save(model, onnx_model_path)
    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_quantized_model_path


def prepare_conv_model(tmp_path):
    torch.manual_seed(42)
    model = DoubleConvModel()

    dummy_input = torch.randn(1, 3, 4, 4)
    onnx_model_path = Path(tmp_path, "double_conv_model.onnx").as_posix()
    onnx_quantized_model_path = Path(tmp_path, "bf16.onnx").as_posix()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_quantized_model_path


def prepare_constant_model(tmp_path):
    value = np.array([1, 2, 3], dtype=np.float32)
    const_tensor = onnx.numpy_helper.from_array(value, name="const_value")

    constant_node = helper.make_node("Constant", inputs=[], outputs=["const_output"], value=const_tensor)

    graph = helper.make_graph(
        nodes=[constant_node],
        name="ConstantGraph",
        inputs=[],
        outputs=[helper.make_tensor_value_info("const_output", onnx.TensorProto.FLOAT, [3])],
        initializer=[],
    )

    model = helper.make_model(graph, producer_name="constant-model", ir_version=10)
    fp32_model_path = Path(tmp_path, "constant_model.onnx").as_posix()
    bf16_model_path = Path(tmp_path, "bf16.onnx").as_posix()
    onnx.save(model, fp32_model_path)
    return fp32_model_path, bf16_model_path


def prepare_cast_model(tmp_path):
    input_tensor = helper.make_tensor_value_info("input", TensorProto.INT32, [1, 3, 5, 5])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 3, 3])

    cast_node = helper.make_node(
        op_type="Cast", inputs=["input"], outputs=["cast_output"], to=TensorProto.FLOAT, name="CastToFloat32"
    )

    weight_tensor = helper.make_tensor(
        name="conv_weight",
        data_type=TensorProto.FLOAT,
        dims=[1, 3, 3, 3],
        vals=np.random.randn(1, 3, 3, 3).astype(np.float32).flatten().tolist(),
    )

    conv_node = helper.make_node(
        op_type="Conv", inputs=["cast_output", "conv_weight"], outputs=["output"], name="ConvNode"
    )

    graph = helper.make_graph(
        nodes=[cast_node, conv_node],
        name="CastAndConvGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_tensor],
    )

    model = helper.make_model(graph, producer_name="onnx-cast-and-conv-example", ir_version=10)
    fp32_model_path = Path(tmp_path, "cast_model.onnx").as_posix()
    bf16_model_path = Path(tmp_path, "bf16.onnx").as_posix()
    onnx.save(model, fp32_model_path)
    return fp32_model_path, bf16_model_path


def prepare_2output_model(tmp_path):
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 5, 5])
    conv1_output_tensor = helper.make_tensor_value_info("conv1_output", TensorProto.FLOAT, [1, 4, 3, 3])
    final_output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 1, 1])

    conv1_weight = helper.make_tensor(
        name="conv1_weight",
        data_type=TensorProto.FLOAT,
        dims=[4, 3, 3, 3],
        vals=np.random.randn(4, 3, 3, 3).astype(np.float32).flatten().tolist(),
    )

    conv2_weight = helper.make_tensor(
        name="conv2_weight",
        data_type=TensorProto.FLOAT,
        dims=[2, 4, 3, 3],
        vals=np.random.randn(2, 4, 3, 3).astype(np.float32).flatten().tolist(),
    )

    conv1_node = helper.make_node(
        op_type="Conv", inputs=["input", "conv1_weight"], outputs=["conv1_output"], name="Conv1"
    )

    conv2_node = helper.make_node(
        op_type="Conv", inputs=["conv1_output", "conv2_weight"], outputs=["output"], name="Conv2"
    )

    graph = helper.make_graph(
        nodes=[conv1_node, conv2_node],
        name="TwoConvGraph",
        inputs=[input_tensor],
        outputs=[conv1_output_tensor, final_output_tensor],
        initializer=[conv1_weight, conv2_weight],
    )

    model = helper.make_model(graph, producer_name="onnx-two-conv-example", ir_version=10)
    fp32_model_path = Path(tmp_path, "2output_model.onnx").as_posix()
    bf16_model_path = Path(tmp_path, "bf16.onnx").as_posix()
    onnx.save(model, fp32_model_path)
    return fp32_model_path, bf16_model_path


def prepare_config():
    quant_config = QConfig(QLayerConfig(activation=BFloat16Spec(), weight=BFloat16Spec()), EnableVaimlBF16=True)
    return quant_config


def prepare_bf16_with_cast_config():
    quant_config = QConfig(QLayerConfig(activation=BFloat16Spec(), weight=BFloat16Spec()), BF16QDQToCast=True)
    return quant_config


def prepare_data():
    data_reader = DataReader(input_tensor)
    return data_reader


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(quantizer, input_model_path, output_model_path, data_reader):
    quantizer.quantize_model(input_model_path, output_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", output_model_path)
    return output_model_path


def infer_quantized_model(quantized_model_path):
    sess = onnxruntime.InferenceSession(quantized_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize_bf16_with_cast(tmp_path):
    input_model_path, output_model_path = prepare_add_model(tmp_path)
    data_reader = prepare_data()
    quant_config = prepare_bf16_with_cast_config()
    quantizer = prepare_quantizer(quant_config)
    _ = quantize_static(quantizer, input_model_path, output_model_path, data_reader)


def tensor_quantize(tmp_path):
    input_model_path, output_model_path = prepare_conv_model(tmp_path)
    data_reader = prepare_data()
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def test_conv_model_convert_fp32_to_bf16(tmp_path):
    input_model_path, output_model_path = prepare_conv_model(tmp_path)
    fp32_model = onnx.load(input_model_path)
    qType = onnx_proto.TensorProto.BFLOAT16
    bf16_model = convert_to_bf16(fp32_model, qType)
    fp32_flag = 0
    bf16_flag = 0
    for init in bf16_model.graph.initializer:
        if init.data_type == 1:
            fp32_flag += 1
        elif init.data_type == 16:
            bf16_flag += 1
    assert fp32_flag == 0 and bf16_flag > 0, "Failed. The conv model can not convert from fp32 to bf16."


def test_constant_model_convert_fp32_to_bf16(tmp_path):
    input_model_path, output_model_path = prepare_constant_model(tmp_path)
    fp32_model = onnx.load(input_model_path)
    qType = onnx_proto.TensorProto.BFLOAT16
    bf16_model = convert_to_bf16(fp32_model, qType)
    fp32_flag = 0
    for init in bf16_model.graph.initializer:
        if init.data_type == 1:
            fp32_flag += 1
    assert fp32_flag == 0, "Failed. The constant model can not convert from fp32 to bf16."


def test_cast_model_convert_fp32_to_bf16(tmp_path):
    input_model_path, output_model_path = prepare_cast_model(tmp_path)
    fp32_model = onnx.load(input_model_path)
    qType = onnx_proto.TensorProto.BFLOAT16
    bf16_model = convert_to_bf16(fp32_model, qType)
    fp32_flag = 0
    bf16_flag = 0
    for init in bf16_model.graph.initializer:
        if init.data_type == 1:
            fp32_flag += 1
        elif init.data_type == 16:
            bf16_flag += 1
    assert fp32_flag == 0 and bf16_flag > 0, "Failed. The cast model can not convert from fp32 to bf16."


def test_2output_model_convert_fp32_to_bf16(tmp_path):
    input_model_path, output_model_path = prepare_2output_model(tmp_path)
    fp32_model = onnx.load(input_model_path)
    qType = onnx_proto.TensorProto.BFLOAT16
    bf16_model = convert_to_bf16(fp32_model, qType)
    fp32_flag = 0
    bf16_flag = 0
    for init in bf16_model.graph.initializer:
        if init.data_type == 1:
            fp32_flag += 1
        elif init.data_type == 16:
            bf16_flag += 1
    assert fp32_flag == 0 and bf16_flag > 0, "Failed. The 2output model can not convert from fp32 to bf16."


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_bf16_with_cast(self, tmpdir: str):
        tensor_quantize_bf16_with_cast(tmpdir)

    @use_temporary_directory
    def test_quantize_remove_bf16_cast(self, tmpdir: str):
        output = tensor_quantize(tmpdir)
        comp_equal = np.allclose(output, output_golden, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_conv_model_convert_fp32_to_bf16(self, tmpdir: str):
        fp32_flag = test_conv_model_convert_fp32_to_bf16(tmpdir)
        self.assertEqual(fp32_flag, None)

    @use_temporary_directory
    def test_quantize_constant_model_convert_fp32_to_bf16(self, tmpdir: str):
        fp32_flag = test_constant_model_convert_fp32_to_bf16(tmpdir)
        self.assertEqual(fp32_flag, None)

    @use_temporary_directory
    def test_quantize_cast_model_convert_fp32_to_bf16(self, tmpdir: str):
        fp32_flag = test_cast_model_convert_fp32_to_bf16(tmpdir)
        self.assertEqual(fp32_flag, None)

    @use_temporary_directory
    def test_quantize_2output_model_convert_fp32_to_bf16(self, tmpdir: str):
        fp32_flag = test_2output_model_convert_fp32_to_bf16(tmpdir)
        self.assertEqual(fp32_flag, None)


if __name__ == "__main__":
    unittest.main()
