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
from onnx import TensorProto, numpy_helper
from onnx_testing_utils import run_onnx_op_variants
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import (
    assert_outputs_equivalent,
    use_temporary_directory,
)
from quark.onnx import (
    AutoMixprecisionConfig,
    BFloat16Spec,
    CLEConfig,
    ExtendedQuantType,
    Int8,
    ModelQuantizer,
    QConfig,
    QLayerConfig,
    get_library_path,
)
from quark.onnx.quantization.config.spec import BFP16Spec, Int16Spec, MXInt8Spec, XInt8Spec


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


auto_mix_precision_output_golden = np.array([[-0.4639878]], dtype=np.float32)

mix_precision_output_golden = np.array([[-0.46404064]], dtype=np.float32)

mix_precision_with_dual_quant_nodes_output_golden = np.array([[-0.46484375]], dtype=np.float32)


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
        x_1 = x[:, :8, :, :]
        x_2 = x[:, 8:, :, :]
        x = torch.cat([x_1, x_2], dim=1)
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


def prepare_auto_mix_precision_config():
    auto_mixprecision_algo = AutoMixprecisionConfig(
        act_target_quant_type=Int8, weight_target_quant_type=Int8, l2_target=10000, output_index=0
    )
    quant_config = QConfig(
        QLayerConfig(input_tensors=Int16Spec(), weight=Int16Spec()),
        algo_config=[auto_mixprecision_algo],
        extra_options={"Percentile": 99.9999, "Int32Bias": False, "Int16Bias": False},
    )
    return quant_config


def prepare_mix_precision_config():
    cle_algo = CLEConfig(cle_steps=2)
    quant_config = QConfig(
        global_config=QLayerConfig(input_tensors=XInt8Spec(), weight=XInt8Spec()),
        specific_layer_config={QLayerConfig(weight=Int16Spec(), bias=Int16Spec()): ["/conv1/Conv", "/conv2/Conv"]},
        layer_type_config={
            QLayerConfig(input_tensors=Int16Spec(), weight=Int16Spec(), bias=Int16Spec()): ["Flatten"],
            None: ["Gemm"],
        },
        algo_config=[cle_algo],
        extra_options={
            "SimplifyModel": False,
            "Int32Bias": False,
            "NodesWithMixedPrecision": [],
            "MixedPrecisionTensor": {},
            "SpecificTensorPrecision": True,
        },
    )
    return quant_config


def prepare_mix_precision_with_dual_quant_nodes_config():
    cle_algo = CLEConfig(cle_steps=2)
    quant_config = QConfig(
        global_config=QLayerConfig(input_tensors=XInt8Spec(), weight=XInt8Spec()),
        specific_layer_config={
            QLayerConfig(input_tensors=Int16Spec()): ["/conv1/Conv", "/relu/Relu"],
            QLayerConfig(input_tensors=BFP16Spec()): ["/conv2/Conv"],
        },
        layer_type_config={
            QLayerConfig(input_tensors=Int16Spec(), weight=Int16Spec(), bias=Int16Spec()): ["Flatten"],
            QLayerConfig(input_tensors=MXInt8Spec(), weight=MXInt8Spec(), bias=MXInt8Spec()): ["Gemm"],
        },
        algo_config=[cle_algo],
        extra_options={
            "SimplifyModel": False,
            "Int32Bias": False,
            "TensorQuantOverrides": {
                "/Slice_output_0": [{"quant_type": ExtendedQuantType.QBFP}],
                "/Slice_1_output_0": [{"quant_type": ExtendedQuantType.QBFP}],
            },
            "EnableDualQuantNodePairs": True,
        },
    )
    return quant_config


def prepare_mix_precision_bf16_xint8_dual_boundary_config():
    """XInt8 globally with BFloat16 on conv2 and dual Q/DQ at precision boundaries (QUARK-510)."""
    quant_config = QConfig(
        global_config=QLayerConfig(input_tensors=XInt8Spec(), weight=XInt8Spec()),
        specific_layer_config={
            QLayerConfig(
                input_tensors=BFloat16Spec(),
                weight=BFloat16Spec(),
                bias=BFloat16Spec(),
            ): ["/conv2/Conv"],
        },
        layer_type_config={
            QLayerConfig(input_tensors=XInt8Spec(), weight=XInt8Spec(), bias=XInt8Spec()): ["Flatten"],
            QLayerConfig(input_tensors=MXInt8Spec(), weight=MXInt8Spec(), bias=MXInt8Spec()): ["Gemm"],
        },
        extra_options={
            "SimplifyModel": False,
            "Int32Bias": False,
            "EnableDualQuantNodePairs": True,
        },
    )
    return quant_config


def assert_bf16_extended_qdq_scales_are_unity(quantized_model_path: str) -> None:
    """BF16 Extended Q/DQ scale initializer must be exactly 1."""
    model = onnx.load(quantized_model_path)
    init_map = {init.name: init for init in model.graph.initializer}
    bf16_qdq_count = 0
    for node in model.graph.node:
        if node.op_type not in ("ExtendedQuantizeLinear", "ExtendedDequantizeLinear"):
            continue
        if len(node.input) < 3 or not node.input[2]:
            continue
        zp_init = init_map.get(node.input[2])
        if zp_init is None or zp_init.data_type != TensorProto.BFLOAT16:
            continue
        scale_init = init_map.get(node.input[1])
        assert scale_init is not None, f"Missing scale initializer for {node.name}"
        scale = numpy_helper.to_array(scale_init)
        bf16_qdq_count += 1
        scale_arr = np.asarray(scale)
        expected = np.ones_like(scale_arr)
        np.testing.assert_array_equal(
            scale_arr,
            expected,
            err_msg=f"{node.name}: BF16 Q/DQ scale must equal 1 exactly (boundary scale fix), got {scale!r}",
        )
    assert bf16_qdq_count > 0, "Expected at least one Extended Q/DQ with BF16 zero-point in the graph."


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
    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    sess = onnxruntime.InferenceSession(quantized_model_path, so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = make_input_tensor()
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(output_dir, quant_config):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def quantize_model_to_path(output_dir: str, quant_config) -> str:
    """Quantize the model to a path.
    :param output_dir: The directory to save the quantized model.
    :param quant_config: The quantization configuration.
    :return: The path to the quantized model.
    """
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quantizer = prepare_quantizer(quant_config)
    quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    return output_model_path


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_auto_mix_precision(self, tmpdir: str):
        quant_config = prepare_auto_mix_precision_config()
        out = run_onnx_op_variants(lambda: tensor_quantize(tmpdir, quant_config))
        assert_outputs_equivalent(out[0], auto_mix_precision_output_golden, atol=1e-1)

    @use_temporary_directory
    def test_quantize_mix_precision(self, tmpdir: str):
        quant_config = prepare_mix_precision_config()
        output = tensor_quantize(tmpdir, quant_config)
        comp_equal = np.allclose(output, mix_precision_output_golden, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_mix_precision_with_dual_quant_nodes(self, tmpdir: str):
        quant_config = prepare_mix_precision_with_dual_quant_nodes_config()
        with self.assertLogs("quark.onnx.quantization.quant_utils_screen", level="INFO") as cm:
            output = tensor_quantize(tmpdir, quant_config)
        self.assertTrue(
            any(
                "Inserted 10 quant nodes at the boundary tensors of two different precisions" in message
                for message in cm.output
            )
        )
        comp_equal = np.allclose(output, mix_precision_with_dual_quant_nodes_output_golden, atol=1e-3)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_dual_quant_bf16_xint8_boundary_bf16_scales_are_unity(self, tmpdir: str):
        """QUARK-510: BF16 boundary Q/DQ scales must use compute_scale_zp_fp, scale should be 1.0."""
        quant_config = prepare_mix_precision_bf16_xint8_dual_boundary_config()
        quantized_path = quantize_model_to_path(tmpdir, quant_config)
        assert_bf16_extended_qdq_scales_are_unity(quantized_path)


if __name__ == "__main__":
    unittest.main()
