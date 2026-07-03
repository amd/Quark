#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import unittest

import numpy as np
import onnxruntime
from onnx_testing_utils import prepare_model_vit
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import Config, ModelQuantizer
from quark.onnx.quantization.config.custom_config import MATMUL_NBITS_CONFIG
from quark.onnx.quantization.quant_utils import is_version_below

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

output_tensor = np.array(
    [
        [
            0.13596348,
            0.7069947,
            -0.745268,
            -0.8128995,
            0.25301066,
            0.43890983,
            0.10404871,
            0.88007313,
            -0.11417447,
            -0.7310455,
        ]
    ]
).astype(np.float32)
output_tensor_gptq = np.array(
    [
        [
            0.14073174,
            0.72223854,
            -0.7341187,
            -0.8268701,
            0.22473133,
            0.44346523,
            0.1087392,
            0.8950456,
            -0.13907433,
            -0.74229425,
        ]
    ]
).astype(np.float32)
output_tensor_hqq = np.array(
    [
        [
            0.13278924,
            0.7027687,
            -0.7434849,
            -0.8260411,
            0.26645112,
            0.4434718,
            0.10804553,
            0.89408153,
            -0.1228957,
            -0.7243466,
        ]
    ]
).astype(np.float32)


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


def prepare_config():
    config_copy = copy.deepcopy(MATMUL_NBITS_CONFIG)
    config_copy.extra_options["MatMulNBitsParams"]["Symmetric"] = False
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_config_gptq():
    config_copy = copy.deepcopy(MATMUL_NBITS_CONFIG)
    config_copy.extra_options["MatMulNBitsParams"]["Symmetric"] = False
    config_copy.extra_options["MatMulNBitsParams"]["Algorithm"] = "GPTQ"
    config_copy.extra_options["GPTQParams"] = {"MSE": False, "GroupSize": 32, "ActOrder": True, "PerChannel": True}
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_config_gptq_asym():
    # Asymmetric GPTQ (WeightSymmetric=False) emits a zero_points initializer for each
    # MatMulNBits node. Regression guard for the zero-point packing in
    # GptqProcessor.prepare_matmul4bits_node, which previously produced a {N*k_blocks, 1}
    # tensor that ORT rejected instead of the expected {N, ceil(k_blocks/2)} layout.
    config_copy = copy.deepcopy(MATMUL_NBITS_CONFIG)
    config_copy.extra_options["MatMulNBitsParams"]["Symmetric"] = False
    config_copy.extra_options["MatMulNBitsParams"]["Algorithm"] = "GPTQ"
    config_copy.extra_options["GPTQParams"] = {
        "MSE": False,
        "GroupSize": 32,
        "ActOrder": True,
        "PerChannel": True,
        "WeightSymmetric": False,
    }
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_config_hqq():
    config_copy = copy.deepcopy(MATMUL_NBITS_CONFIG)
    config_copy.extra_options["MatMulNBitsParams"]["Symmetric"] = False
    config_copy.extra_options["MatMulNBitsParams"]["Algorithm"] = "HQQ"
    quant_config = Config(global_quant_config=config_copy)
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


def tensor_quantize_matmul_4bits(output_dir):
    input_model_path, output_model_path = prepare_model_vit(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_matmul_4bits_none_calibration_data_reader(output_dir):
    input_model_path, output_model_path = prepare_model_vit(output_dir)

    quant_config = prepare_config()
    quant_config.global_quant_config.extra_options["UseRandomData"] = True

    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, None)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_matmul_4bits_gptq(output_dir):
    input_model_path, output_model_path = prepare_model_vit(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config_gptq()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_matmul_4bits_gptq_asym(output_dir):
    import math

    import onnx

    input_model_path, output_model_path = prepare_model_vit(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config_gptq_asym()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)

    # Every asymmetric MatMulNBits node must carry a zero_points input shaped
    # {N, ceil(k_blocks/2)} (matching its scales {N, k_blocks}), not the buggy {N*k_blocks, 1}.
    model = onnx.load(quantized_model_path)
    initializers = {init.name: init for init in model.graph.initializer}
    checked = 0
    for node in model.graph.node:
        if node.op_type != "MatMulNBits" or len(node.input) < 4:
            continue
        scales_dims = list(initializers[node.input[2]].dims)
        zp_dims = list(initializers[node.input[3]].dims)
        n, k_blocks = scales_dims
        assert zp_dims == [n, math.ceil(k_blocks / 2)], (
            f"Unexpected zero_points shape {zp_dims} for scales {scales_dims}"
        )
        checked += 1
    assert checked > 0, "No asymmetric MatMulNBits node was produced"

    # Inference must succeed; the malformed zero_points shape made the kernel raise.
    output = infer_quantized_model(quantized_model_path)
    return output


def tensor_quantize_matmul_4bits_gptq_asym_odd_kblocks(output_dir):
    # Same asymmetric GPTQ path, but with a K that makes k_blocks odd
    # (K=384, GroupSize=128 -> k_blocks=3) to exercise the zero-point padding branch.
    import math
    import os

    import onnx

    rng = np.random.default_rng(0)
    model = onnx.parser.parse_model(
        """
        < ir_version: 10, opset_import: ["" : 21] >
        test_model (float[N, 384] input) => (float [N, ?] output)
        <float[384, 64] W>
        { output = MatMul(input, W) }
        """
    )
    W = onnx.numpy_helper.from_array(rng.normal(size=(384, 64)).astype(np.float32), name="W")
    model.graph.initializer.extend([W])
    input_model_path = os.path.join(output_dir, "matmul_odd.onnx")
    output_model_path = os.path.join(output_dir, "matmul_odd_quantized.onnx")
    onnx.save(model, input_model_path)

    odd_input = rng.random((4, 384)).astype(np.float32)
    data_reader = DataReader(odd_input)
    # GroupSize=128 over K=384 gives k_blocks=3 (odd), unlike the GroupSize=32 default.
    config_copy = copy.deepcopy(MATMUL_NBITS_CONFIG)
    config_copy.extra_options["MatMulNBitsParams"]["Symmetric"] = False
    config_copy.extra_options["MatMulNBitsParams"]["Algorithm"] = "GPTQ"
    config_copy.extra_options["GPTQParams"] = {
        "MSE": False,
        "GroupSize": 128,
        "ActOrder": True,
        "PerChannel": True,
        "WeightSymmetric": False,
    }
    quantizer = prepare_quantizer(Config(global_quant_config=config_copy))
    quantize_static(quantizer, input_model_path, output_model_path, data_reader)

    quantized = onnx.load(output_model_path)
    initializers = {init.name: init for init in quantized.graph.initializer}
    checked = 0
    for node in quantized.graph.node:
        if node.op_type != "MatMulNBits" or len(node.input) < 4:
            continue
        n, k_blocks = list(initializers[node.input[2]].dims)
        assert k_blocks % 2 != 0, f"Expected odd k_blocks to exercise padding, got {k_blocks}"
        assert list(initializers[node.input[3]].dims) == [n, math.ceil(k_blocks / 2)]
        checked += 1
    assert checked > 0, "No asymmetric MatMulNBits node was produced"

    sess = onnxruntime.InferenceSession(output_model_path)
    return sess.run(None, {"input": odd_input})


def tensor_quantize_matmul_4bits_hqq(output_dir):
    input_model_path, output_model_path = prepare_model_vit(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config_hqq()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize(self, tmpdir: str):
        output = tensor_quantize_matmul_4bits(tmpdir)
        comp_equal = np.allclose(output, output_tensor, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_none_calibration_data_reader(self, tmpdir: str):
        output = tensor_quantize_matmul_4bits_none_calibration_data_reader(tmpdir)
        comp_equal = np.allclose(output, output_tensor, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_gptq(self, tmpdir: str):
        output_gptq = tensor_quantize_matmul_4bits_gptq(tmpdir)
        comp_equal = np.allclose(output_gptq, output_tensor_gptq, atol=1e-1)
        self.assertEqual(comp_equal, True)

    @use_temporary_directory
    def test_quantize_gptq_asym(self, tmpdir: str):
        # Regression test for the asymmetric GPTQ zero_points packing bug
        # (MatMulNBits kernel rejected the {N*k_blocks, 1} zero_points tensor).
        # The shape assertions live in the helper; reaching here means inference passed.
        tensor_quantize_matmul_4bits_gptq_asym(tmpdir)

    @use_temporary_directory
    def test_quantize_gptq_asym_odd_kblocks(self, tmpdir: str):
        # Covers the odd-k_blocks zero-point padding branch in prepare_matmul4bits_node.
        tensor_quantize_matmul_4bits_gptq_asym_odd_kblocks(tmpdir)

    @use_temporary_directory
    def test_quantize_hqq(self, tmpdir: str):
        if not is_version_below(onnxruntime, "1.18.0"):
            output_hqq = tensor_quantize_matmul_4bits_hqq(tmpdir)
            comp_equal = np.allclose(output_hqq, output_tensor_hqq, atol=1e-1)
            self.assertEqual(comp_equal, True)


if __name__ == "__main__":
    unittest.main()
