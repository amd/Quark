#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from onnx_testing_utils import SimpleConvModel

from quark.common.utils.testing_utils import delete_directory_content, use_temporary_directory
from quark.onnx import Config, ModelQuantizer
from quark.onnx.calibration import RandomDataReader
from quark.onnx.quantization.config.custom_config import U8S8_AAWS_CONFIG

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

calib_data = np.array(
    [
        [
            [
                [0.9296161, 0.31637555, 0.18391882, 0.20456028],
                [0.567725, 0.5955447, 0.9645145, 0.6531771],
                [0.7489066, 0.6535699, 0.7477148, 0.96130675],
                [0.0083883, 0.10644437, 0.2987037, 0.6564112],
            ],
            [
                [0.80981255, 0.87217593, 0.9646476, 0.7236853],
                [0.6424753, 0.7174536, 0.467599, 0.32558468],
                [0.4396446, 0.72968906, 0.99401456, 0.6768737],
                [0.7908225, 0.17091426, 0.02684928, 0.8003702],
            ],
            [
                [0.9037225, 0.02467621, 0.49174732, 0.5262552],
                [0.596366, 0.05195754, 0.8950895, 0.7282662],
                [0.81835, 0.50022274, 0.8101894, 0.09596852],
                [0.21895005, 0.25871906, 0.46810576, 0.4593732],
            ],
        ]
    ]
).astype(np.float32)


golden_output = np.array(
    [
        [
            [
                [-0.52279645, -0.39127532, -0.3189387, -0.25646618],
                [-0.04274436, -0.28934646, -0.26633024, -0.09535281],
                [-0.20714575, -0.13809717, -0.11179294, -0.28934646],
                [-0.12165703, -0.23016195, -0.19399364, -0.07233661],
            ]
        ]
    ]
).astype(np.float32)


def prepare_model(output_dir):
    torch.manual_seed(12345)  # Not the same seed as in testing_utils.py...
    model = SimpleConvModel()

    dummy_input = torch.randn(1, 3, 4, 4)

    onnx_model_path = Path(output_dir, "simple_conv_model.onnx").as_posix()
    quant_onnx_model_path = Path(output_dir, "simple_conv_model_quantized.onnx").as_posix()

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
    return onnx_model_path, quant_onnx_model_path


def prepare_config():
    quant_config = Config(global_quant_config=U8S8_AAWS_CONFIG)
    return quant_config


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(
    quantizer, input_model_path, output_model_path, calibration_data_reader=None, calibration_data_path=None
):
    quantizer.quantize_model(
        input_model_path,
        output_model_path,
        calibration_data_reader=calibration_data_reader,
        calibration_data_path=calibration_data_path,
    )
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


def prepare_calib_data(calib_data_dir="output"):
    if not os.path.exists(calib_data_dir):
        os.makedirs(calib_data_dir)
    file_path = os.path.join(calib_data_dir, "data.npy")
    np.save(file_path, calib_data)


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_random_data_reader_with_shape(self, tmpdir: str):
        input_model_path, output_model_path = prepare_model(tmpdir)
        data_reader = RandomDataReader(input_model_path, {"input": [1, 3, 4, 4]})
        quant_config = prepare_config()
        quantizer = prepare_quantizer(quant_config)
        quantized_model_path = quantize_static(
            quantizer, input_model_path, output_model_path, calibration_data_reader=data_reader
        )
        output = infer_quantized_model(quantized_model_path)
        comp_equal = np.allclose(output, golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_random_data_reader_no_shape(self, tmpdir: str):
        input_model_path, output_model_path = prepare_model(tmpdir)
        data_reader = RandomDataReader(input_model_path)
        quant_config = prepare_config()
        quantizer = prepare_quantizer(quant_config)
        quantized_model_path = quantize_static(
            quantizer, input_model_path, output_model_path, calibration_data_reader=data_reader
        )
        output = infer_quantized_model(quantized_model_path)
        comp_equal = np.allclose(output, golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_CalibrationDataPath(self, tmpdir: str):
        calib_data_dir = "calib_data"
        prepare_calib_data(calib_data_dir)
        input_model_path, output_model_path = prepare_model(tmpdir)
        quant_config = prepare_config()
        quantizer = prepare_quantizer(quant_config)
        quantized_model_path = quantize_static(
            quantizer,
            input_model_path,
            output_model_path,
            calibration_data_reader=None,
            calibration_data_path=calib_data_dir,
        )
        output = infer_quantized_model(quantized_model_path)
        comp_equal = np.allclose(output, golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)
        shutil.rmtree(calib_data_dir)

    @use_temporary_directory
    def test_data_reader_none(self, tmpdir: str):
        input_model_path, output_model_path = prepare_model(tmpdir)
        quant_config = prepare_config()
        quantizer = prepare_quantizer(quant_config)

        # Default: raise error on static quantization when a calibration reader is not provided
        with self.assertRaises(Exception) as context:
            quantized_model_path = quantize_static(
                quantizer, input_model_path, output_model_path, calibration_data_reader=None
            )

        self.assertIn("A calibration data reader is required for quantization", str(context.exception))
        delete_directory_content(tmpdir)

        # Enable random data reader.
        input_model_path, output_model_path = prepare_model(tmpdir)
        quant_config = prepare_config()
        quant_config.global_quant_config.extra_options["UseRandomData"] = True

        quantizer = prepare_quantizer(quant_config)

        quantized_model_path = quantize_static(
            quantizer, input_model_path, output_model_path, calibration_data_reader=None
        )

        output = infer_quantized_model(quantized_model_path)
        comp_equal = np.allclose(output, golden_output, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)


if __name__ == "__main__":
    unittest.main()
