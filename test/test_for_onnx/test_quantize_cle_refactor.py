#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import CLEConfig, Int16Spec, ModelQuantizer, QConfig, QLayerConfig, XInt8Spec


def make_input_tensor():
    return np.array(
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


output_golden = np.array([[-0.46404064]], dtype=np.float32)
output_golden_with_gemm_gemm = np.array(
    [
        [
            -27.396326,
            -73.73458,
            -31.340204,
            5.5969443,
            -44.355606,
            -63.986744,
            -51.95005,
            -11.384575,
            -66.99817,
            -57.421787,
        ]
    ],
    dtype=np.float32,
)
output_golden_with_gemm_gemm_identity = np.array(
    [
        [
            9.4531221e00,
            3.8214649e01,
            -1.9299913e01,
            2.5200323e01,
            5.3104904e01,
            3.0621435e01,
            -5.7787322e-02,
            6.6381897e01,
            1.8221943e01,
            2.4687962e01,
        ]
    ],
    dtype=np.float32,
)
output_golden_with_conv_depthwiseconv_conv = np.array(
    [
        [
            -0.09978256,
            0.03257076,
            0.05322019,
            0.0328774,
            -0.0582874,
            -0.04417268,
            -0.03517536,
            -0.01375716,
            -0.05499446,
            -0.10006751,
        ]
    ],
    dtype=np.float32,
)


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
        super().__init__()
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
        x = torch.clip(x, 0, 6)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = DoubleConvModel()
    dummy_input = torch.randn(1, 3, 4, 4)
    onnx_model_path = Path(output_dir, f"double_conv_model_{np.random.randint(0, 10000)}.onnx").as_posix()
    onnx_quantized_model_path = Path(
        output_dir, f"double_conv_model_quantized_{np.random.randint(0, 10000)}.onnx"
    ).as_posix()
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


def prepare_model_with_gemm_gemm(output_dir):
    np.random.seed(123)
    N, C, H, W = 1, 3, 4, 4
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [N, C, H, W])
    conv_out_channels = 4
    kernel_h, kernel_w = 3, 3
    W_conv = np.random.randn(conv_out_channels, C, kernel_h, kernel_w).astype(np.float32)
    B_conv = np.random.randn(conv_out_channels).astype(np.float32)
    W_conv_init = numpy_helper.from_array(W_conv, name="W_conv")
    B_conv_init = numpy_helper.from_array(B_conv, name="B_conv")
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "W_conv", "B_conv"],
        outputs=["conv_out"],
        kernel_shape=[kernel_h, kernel_w],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        name="Conv1",
    )
    H_out = H - kernel_h + 1
    W_out = W - kernel_w + 1
    flat_dim = conv_out_channels * H_out * W_out
    flatten_node = helper.make_node("Flatten", inputs=["conv_out"], outputs=["flat_out"], axis=1, name="Flatten1")
    K1 = 16
    B1 = np.random.randn(flat_dim, K1).astype(np.float32)
    C1 = np.random.randn(K1).astype(np.float32)
    B1_init = numpy_helper.from_array(B1, name="B1")
    C1_init = numpy_helper.from_array(C1, name="C1")
    gemm1 = helper.make_node(
        "Gemm",
        inputs=["flat_out", "B1", "C1"],
        outputs=["gemm1_out"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0,
        name="Gemm1",
    )
    K2 = 10
    B2 = np.random.randn(K1, K2).astype(np.float32)
    C2 = np.random.randn(K2).astype(np.float32)
    B2_init = numpy_helper.from_array(B2, name="B2")
    C2_init = numpy_helper.from_array(C2, name="C2")
    gemm2 = helper.make_node(
        "Gemm", inputs=["gemm1_out", "B2", "C2"], outputs=["Y"], alpha=1.0, beta=1.0, transA=0, transB=0, name="Gemm2"
    )
    graph = helper.make_graph(
        nodes=[conv_node, flatten_node, gemm1, gemm2],
        name="ConvFlattenGemmGemmGraph",
        inputs=[input],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N, K2])],
        initializer=[W_conv_init, B_conv_init, B1_init, C1_init, B2_init, C2_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model)
    onnx_model_path = Path(output_dir, f"conv_flat_gemm_gemm_{np.random.randint(0, 10000)}.onnx").as_posix()
    quant_onnx_model_path = Path(
        output_dir, f"conv_flat_gemm_gemm_quantized_{np.random.randint(0, 10000)}.onnx"
    ).as_posix()
    model.ir_version = 10
    onnx.save(model, str(onnx_model_path))
    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, quant_onnx_model_path


def prepare_model_with_gemm_gemm_identity(output_dir):
    np.random.seed(123)
    N, C, H, W = 1, 3, 4, 4
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [N, C, H, W])
    conv_out_channels = 4
    kernel_h, kernel_w = 3, 3
    W_conv = np.random.randn(conv_out_channels, C, kernel_h, kernel_w).astype(np.float32)
    B_conv = np.random.randn(conv_out_channels).astype(np.float32)
    W_conv_init = numpy_helper.from_array(W_conv, name="W_conv")
    B_conv_init = numpy_helper.from_array(B_conv, name="B_conv")
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "W_conv", "B_conv"],
        outputs=["conv_out"],
        kernel_shape=[kernel_h, kernel_w],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        name="Conv1",
    )
    H_out = H - kernel_h + 1
    W_out = W - kernel_w + 1
    flat_dim = conv_out_channels * H_out * W_out
    flatten_node = helper.make_node("Flatten", inputs=["conv_out"], outputs=["flat_out"], axis=1, name="Flatten1")
    K1 = 16
    B1 = np.random.randn(K1, flat_dim).astype(np.float32)
    C1 = np.random.randn(K1).astype(np.float32)
    B1_init = numpy_helper.from_array(B1, name="B1")
    C1_init = numpy_helper.from_array(C1, name="C1")
    gemm1 = helper.make_node(
        "Gemm",
        inputs=["flat_out", "B1", "C1"],
        outputs=["gemm1_out"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
        name="Gemm1",
    )
    K2 = 10
    B2 = np.random.randn(K2, K1).astype(np.float32)
    C2 = np.random.randn(K2).astype(np.float32)
    B2_init = numpy_helper.from_array(B2, name="B2")
    C2_init = numpy_helper.from_array(C2, name="C2")
    gemm2 = helper.make_node(
        "Gemm", inputs=["gemm1_out", "B2", "C2"], outputs=["Y"], alpha=1.0, beta=1.0, transA=0, transB=1, name="Gemm2"
    )
    weird_node = helper.make_node("Identity", inputs=["Y"], outputs=["valid_output"], name="WeirdIdentity")
    graph = helper.make_graph(
        nodes=[conv_node, flatten_node, gemm1, gemm2, weird_node],
        name="ConvFlattenGemmGemmGraph",
        inputs=[input],
        outputs=[helper.make_tensor_value_info("valid_output", TensorProto.FLOAT, [N, K2])],
        initializer=[W_conv_init, B_conv_init, B1_init, C1_init, B2_init, C2_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model)
    onnx_model_path = Path(output_dir, f"conv_flat_gemm_gemm_identity_{np.random.randint(0, 10000)}.onnx").as_posix()
    quant_onnx_model_path = Path(
        output_dir, f"conv_flat_gemm_gemm_identity_quantized_{np.random.randint(0, 10000)}.onnx"
    ).as_posix()
    model.ir_version = 10
    onnx.save(model, str(onnx_model_path))
    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, quant_onnx_model_path


class ConvDepthWiseConvConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=2, groups=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def prepare_model_with_conv_depthwiseconv_conv(output_dir):
    torch.manual_seed(42)
    model = ConvDepthWiseConvConvModel()
    dummy_input = torch.randn(1, 3, 4, 4)
    onnx_model_path = Path(output_dir, f"conv_depthwiseconv_conv_model_{np.random.randint(0, 10000)}.onnx").as_posix()
    onnx_quantized_model_path = Path(
        output_dir, f"conv_depthwiseconv_conv_model_quantized_{np.random.randint(0, 10000)}.onnx"
    ).as_posix()
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


def prepare_config():
    cle_algo = CLEConfig(cle_steps=2)
    quant_config = QConfig(
        global_config=QLayerConfig(input_tensors=XInt8Spec(), weight=XInt8Spec()),
        specific_layer_config={
            QLayerConfig(input_tensors=Int16Spec(), weight=Int16Spec()): ["/conv1/Conv", "/conv2/Conv"]
        },
        layer_type_config={None: ["Gemm"]},
        algo_config=[cle_algo],
        extra_options={"SimplifyModel": False, "Int32Bias": False},
    )
    return quant_config


def prepare_data():
    data_reader = DataReader(make_input_tensor())
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
    input_data = make_input_tensor()
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_with_gemm_gemm(output_dir: str):
    input_model_path, output_model_path = prepare_model_with_gemm_gemm(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_with_gemm_gemm_identity(output_dir: str):
    input_model_path, output_model_path = prepare_model_with_gemm_gemm_identity(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_with_conv_depthwiseconv_conv(output_dir: str):
    input_model_path, output_model_path = prepare_model_with_conv_depthwiseconv_conv(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_cle(self, tmpdir: str):
        output = tensor_quantize(tmpdir)
        comp_equal = np.allclose(output, output_golden, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_with_gemm_gemm(self, tmpdir: str):
        output = tensor_quantize_with_gemm_gemm(tmpdir)
        comp_equal = np.allclose(output, output_golden_with_gemm_gemm, atol=1e-1)
        self.assertTrue(comp_equal)

    @use_temporary_directory
    def test_quantize_with_gemm_gemm_identity(self, tmpdir: str):
        output = tensor_quantize_with_gemm_gemm_identity(tmpdir)
        comp_equal = np.allclose(output, output_golden_with_gemm_gemm_identity, atol=1e-1)
        self.assertTrue(comp_equal)

    @use_temporary_directory
    def test_quantize_with_conv_depthwiseconv_conv(self, tmpdir: str):
        output = tensor_quantize_with_conv_depthwiseconv_conv(tmpdir)
        comp_equal = np.allclose(output, output_golden_with_conv_depthwiseconv_conv, atol=1e-1)
        self.assertTrue(comp_equal)


if __name__ == "__main__":
    unittest.main()
