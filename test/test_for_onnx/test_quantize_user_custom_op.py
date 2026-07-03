#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import helper
from onnx.onnx_ml_pb2 import TensorProto
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime_extensions import PyCustomOpDef, onnx_op
from onnxruntime_extensions import get_library_path as ext_lib_path

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import Config, ModelQuantizer
from quark.onnx.operators.custom_ops import _COP_DOMAIN, _COP_IN_OP_NAME, get_library_path
from quark.onnx.quantization.config.custom_config import S16S16_MIXED_S8S8_CONFIG

op_type = "MyCustomOp"
op_domain = "ai.onnx.contrib"

input_tensor = np.array(
    [
        [
            [
                [0.39450988, 0.34032542, 0.91402656, 0.40040675, 0.5765305],
                [0.40323386, 0.07389439, 0.38661167, 0.8645387, 0.55553377],
                [0.21639073, 0.10061733, 0.19072777, 0.32449463, 0.79694337],
                [0.66819394, 0.03191912, 0.397995, 0.01690937, 0.63425934],
                [0.37730116, 0.80095553, 0.77266306, 0.54853624, 0.27609143],
            ],
            [
                [0.8222164, 0.5256697, 0.2953402, 0.47371042, 0.40800324],
                [0.6019997, 0.7506883, 0.5605579, 0.7274801, 0.19008774],
                [0.76555413, 0.6223917, 0.27387974, 0.85017425, 0.70976704],
                [0.868642, 0.18798842, 0.26945123, 0.8975411, 0.1434885],
                [0.30794197, 0.13901855, 0.8121448, 0.8238567, 0.33238393],
            ],
            [
                [0.57792556, 0.98300576, 0.8607786, 0.6592352, 0.22613065],
                [0.7223881, 0.46592003, 0.3890724, 0.868129, 0.691695],
                [0.4210194, 0.5127264, 0.6360194, 0.30745587, 0.1583932],
                [0.67081225, 0.16967775, 0.6681447, 0.71011454, 0.3408417],
                [0.83913565, 0.3341194, 0.8299601, 0.9870858, 0.35757536],
            ],
        ]
    ],
    dtype=np.float32,
)
output_golden = np.array(
    [
        [
            [
                [-0.1796875, -0.390625, 1.828125, -0.15625, 0.53125],
                [-0.1484375, -1.421875, -0.2109375, 1.625, 0.4375],
                [-0.8671875, -1.3125, -0.96875, -0.453125, 1.375],
                [0.875, -1.578125, -0.1640625, -1.640625, 0.734375],
                [-0.25, 1.390625, 1.28125, 0.40625, -0.640625],
            ],
            [
                [1.125, -0.03125, -0.9375, -0.234375, -0.4921875],
                [0.265625, 0.84375, 0.109375, 0.75, -1.34375],
                [0.90625, 0.34375, -1.0234375, 1.25, 0.6875],
                [1.3125, -1.359375, -1.0390625, 1.4375, -1.53125],
                [-0.8828125, -1.546875, 1.09375, 1.140625, -0.7890625],
            ],
            [
                [0.015625, 1.6875, 1.171875, 0.359375, -1.421875],
                [0.609375, -0.4296875, -0.75, 1.203125, 0.484375],
                [-0.6171875, -0.25, 0.265625, -1.0859375, -1.6953125],
                [0.40625, -1.6484375, 0.390625, 0.5625, -0.9453125],
                [1.09375, -0.9765625, 1.046875, 1.6875, -0.8828125],
            ],
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


@onnx_op(op_type=op_type, domain=op_domain, inputs=[PyCustomOpDef.dt_float], outputs=[PyCustomOpDef.dt_float])
def my_custom_op(x: np.ndarray[np.dtype[np.float32]]):
    return x * 2


def prepare_model(float_model_path):
    np.random.seed(123)
    data_type = TensorProto.FLOAT
    data_shape = (1, 3, 5, 5)

    gamma = np.ones(data_shape).astype(np.float32)
    beta = np.zeros(data_shape).astype(np.float32)

    in_param_nodes = [
        helper.make_node(
            "Constant", [], ["gamma"], value=onnx.helper.make_tensor("y_scale", data_type, data_shape, gamma)
        ),
        helper.make_node(
            "Constant", [], ["beta"], value=onnx.helper.make_tensor("y_zero_point", data_type, data_shape, beta)
        ),
    ]

    graph_def = helper.make_graph(
        nodes=[
            helper.make_node(op_type, ["input"], ["input_out"], domain=op_domain),
            helper.make_node(
                _COP_IN_OP_NAME,
                ["input_out", "gamma", "beta"],
                ["y"],
                domain=_COP_DOMAIN,
            ),
        ]
        + in_param_nodes,
        name="test-in",
        inputs=[helper.make_tensor_value_info("input", data_type, shape=None)],
        outputs=[helper.make_tensor_value_info("y", data_type, shape=None)],
    )

    produce_opset_version = 19
    opset_imports = [onnx.helper.make_operatorsetid("", produce_opset_version)]
    model_def = helper.make_model(graph_def, producer_name="quark.onnx", ir_version=9, opset_imports=opset_imports)
    onnx.save(model_def, float_model_path)
    print(f"Model has been saved to {float_model_path}")


def quantize_model(input_model_path, quantized_model_path, data_reader):
    config_copy = copy.deepcopy(S16S16_MIXED_S8S8_CONFIG)
    config_copy.extra_options["UserCustomOpLibPath"] = [get_library_path(), ext_lib_path()]
    quant_config = Config(global_quant_config=config_copy)
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(input_model_path, quantized_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", quantized_model_path)


def infer_quantized_model(quantized_model_path):
    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(ext_lib_path())
    sess_options.register_custom_ops_library(get_library_path())
    sess = onnxruntime.InferenceSession(quantized_model_path, sess_options)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


class TestTensorQuantize(unittest.TestCase):
    @use_temporary_directory
    def test_quantize_custom_ops(self, tmpdir: str):
        float_model_path = Path(tmpdir, "user_custom_op_model.onnx").as_posix()
        quantized_model_path = Path(tmpdir, "user_custom_op_model_quantized.onnx").as_posix()
        prepare_model(float_model_path)
        data_reader = DataReader(input_tensor)
        quantize_model(float_model_path, quantized_model_path, data_reader)
        output = infer_quantized_model(quantized_model_path)
        comp_equal = np.allclose(output, output_golden, atol=1e-1)
        self.assertEqual(comp_equal, True)


if __name__ == "__main__":
    unittest.main()
