#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest

import numpy as np
import onnxruntime
from onnx_testing_utils import prepare_model
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import (
    CalibMethod,
    Int8Spec,
    ModelQuantizer,
    QConfig,
    QLayerConfig,
    get_library_path,
)

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

golden_output_0 = np.array(
    [
        [
            [
                [0.16587953, 0.15924434, 0.01990554, -0.05308145],
                [0.09289254, -0.08625735, 0.12938604, 0.0331759],
                [0.10284531, 0.26540723, 0.3549822, 0.10948049],
                [0.12938604, 0.13602121, 0.22891375, 0.2521369],
            ]
        ]
    ]
).astype(np.float32)


golden_output_1 = np.array(
    [
        [
            [
                [0.19242026, 0.15924434, 0.02322313, -0.04644627],
                [0.1061629, -0.07298699, 0.13602121, 0.0364935],
                [0.11943326, 0.27204242, 0.37157014, 0.11279808],
                [0.1426564, 0.13602121, 0.23554893, 0.26872483],
            ]
        ]
    ]
).astype(np.float32)

golden_output_2 = np.array(
    [
        [
            [
                [0.37488773, 0.421334, 0.421334, 0.16919711],
                [0.421334, 0.421334, 0.421334, 0.36493495],
                [0.421334, 0.421334, 0.421334, 0.421334],
                [0.421334, 0.421334, 0.421334, 0.3616174],
            ]
        ]
    ]
).astype(np.float32)

golden_output_3 = np.array(
    [
        [
            [
                [0.33839422, 0.421334, 0.421334, 0.16587953],
                [0.421334, 0.421334, 0.421334, 0.421334],
                [0.421334, 0.421334, 0.421334, 0.421334],
                [0.33507666, 0.421334, 0.421334, 0.421334],
            ]
        ]
    ]
).astype(np.float32)

golden_output_4 = np.array(
    [
        [
            [
                [0.32512388, 0.421334, 0.421334, 0.16256194],
                [0.421334, 0.421334, 0.421334, 0.421334],
                [0.421334, 0.421334, 0.421334, 0.421334],
                [0.33175907, 0.421334, 0.421334, 0.421334],
            ]
        ]
    ]
).astype(np.float32)

golden_output_5 = np.array(
    [
        [
            [
                [0.22227857, 0.15924434, 0.02322313, -0.04976386],
                [0.11943326, -0.05639904, 0.13602121, 0.0364935],
                [0.12275085, 0.26872483, 0.39147568, 0.10948049],
                [0.14929157, 0.13270362, 0.23223133, 0.2853128],
            ]
        ]
    ]
).astype(np.float32)

golden_output_6 = np.array(
    [
        [
            [
                [0.02985832, -0.00995277, -0.04976386, -0.08293977],
                [0.05308145, -0.01990554, 0.0696694, -0.02985832],
                [0.15924434, 0.08625735, 0.05639904, -0.07298699],
                [0.0, -0.13602121, -0.21896097, -0.26540723],
            ]
        ]
    ]
).astype(np.float32)

golden_output_7 = np.array(
    [
        [
            [
                [0.20569062, 0.15924434, 0.02322313, -0.04976386],
                [0.11279808, -0.05639904, 0.1393388, 0.03981109],
                [0.12275085, 0.27204242, 0.38815808, 0.10948049],
                [0.14929157, 0.1393388, 0.23223133, 0.27536002],
            ]
        ]
    ]
).astype(np.float32)

golden_output_8 = np.array(
    [
        [
            -0.14297572,
            -0.38560116,
            -0.04332598,
            0.00866519,
            -0.4852509,
            0.08231935,
            0.11698013,
            0.55023986,
            -0.42459455,
            -0.38126856,
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
    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())
    sess = onnxruntime.InferenceSession(quantized_model_path, sess_options)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(quant_config, output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_1(self, tmpdir: str):
        config = QConfig(
            global_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec(calibration_method=CalibMethod.MinMax)),
            AlignSlice=False,
            FoldRelu=True,
            AlignConcat=True,
        )
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, golden_output_1, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_2(self, tmpdir: str):
        config = QConfig(
            global_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec(calibration_method=CalibMethod.MinMSE)),
            AlignSlice=False,
            FoldRelu=True,
            AlignConcat=True,
        )
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, golden_output_1, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_3(self, tmpdir: str):
        config = QConfig(
            global_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec(calibration_method=CalibMethod.MinMSE)),
            AlignSlice=False,
            FoldRelu=True,
            AlignConcat=True,
            MinMSEModeFloatScale="HistCenter",
        )
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, golden_output_0, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_4(self, tmpdir: str):
        config = QConfig(
            global_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec(calibration_method=CalibMethod.MinMSE)),
            AlignSlice=False,
            FoldRelu=True,
            AlignConcat=True,
            MinMSEModeFloatScale="All",
        )
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, golden_output_0, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_5(self, tmpdir: str):
        config = QConfig(
            global_config=QLayerConfig(
                activation=Int8Spec(), weight=Int8Spec(calibration_method=CalibMethod.MinMax, symmetric=False)
            ),
            AlignSlice=False,
            FoldRelu=True,
            AlignConcat=True,
        )
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, golden_output_1, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_6(self, tmpdir: str):
        config = QConfig(
            global_config=QLayerConfig(
                activation=Int8Spec(), weight=Int8Spec(calibration_method=CalibMethod.MinMSE, symmetric=False)
            ),
            AlignSlice=False,
            FoldRelu=True,
            AlignConcat=True,
        )
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, golden_output_5, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_7(self, tmpdir: str):
        config = QConfig(
            global_config=QLayerConfig(
                activation=Int8Spec(), weight=Int8Spec(calibration_method=CalibMethod.MinMSE, symmetric=False)
            ),
            AlignSlice=False,
            FoldRelu=True,
            AlignConcat=True,
            MinMSEModeFloatScale="HistCenter",
        )
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, golden_output_7, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)

    @use_temporary_directory
    def test_quantize_8(self, tmpdir: str):
        config = QConfig(
            global_config=QLayerConfig(
                activation=Int8Spec(), weight=Int8Spec(calibration_method=CalibMethod.MinMSE, symmetric=False)
            ),
            AlignSlice=False,
            FoldRelu=True,
            AlignConcat=True,
            MinMSEModeFloatScale="All",
        )
        output = tensor_quantize(config, tmpdir)
        comp_equal = np.allclose(output, golden_output_7, atol=1e-1)
        self.assertEqual(np.all(comp_equal), True)


if __name__ == "__main__":
    unittest.main()
