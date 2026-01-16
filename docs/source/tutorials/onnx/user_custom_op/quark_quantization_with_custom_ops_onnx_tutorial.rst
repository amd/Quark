Quantizing ONNX Models with Custom Operators Using Quark
========================================================

This tutorial demonstrates how to use Quark to quantize an ONNX model
containing Custom Operators. The example includes two types of custom
operator implementations: - Python Custom Operator (my_custom_op) - C++
Custom Operator (\_COP_IN_OP_NAME)

This tutorial will guide you through the following steps: - Building a
floating-point ONNX model with custom operators - Preparing calibration
data for quantization - Using Quark to quantize the model - Running
inference on the quantized model using ONNX Runtime and printing the
output

Preparing the Floating-Point Model
----------------------------------

Before quantization, we first build a floating-point model containing
two custom operators: - my_custom_op: A custom operator implemented in
Python. - \_COP_IN_OP_NAME: A custom operator implemented in C++.

The following code demonstrates how to incorporate these operators when
constructing the ONNX graph and generate a floating-point model ready
for quantization.

.. code:: ipython3

    import copy
    from pathlib import Path
    
    import numpy as np
    import onnx
    import onnxruntime
    from onnx import helper
    from onnx.onnx_ml_pb2 import TensorProto
    from onnxruntime.quantization import CalibrationDataReader
    from onnxruntime_extensions import PyCustomOpDef, onnx_op
    from onnxruntime_extensions import get_library_path as ext_lib_path
    
    from quark.onnx import Config, ModelQuantizer
    from quark.onnx.operators.custom_ops import _COP_DOMAIN, _COP_IN_OP_NAME, get_library_path
    from quark.onnx.quantization.config.custom_config import S16S16_MIXED_S8S8_CONFIG
    
    op_type = "MyCustomOp"
    op_domain = "ai.onnx.contrib"
    
    
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
    
    
    float_model_path = Path("user_custom_op_model.onnx").as_posix()
    prepare_model(float_model_path)

Preparing Calibration Data
--------------------------

To demonstrate the quantization process, a set of pseudo input data with
the shape ``(1, 3, 5, 5)`` is constructed as calibration samples. The
calibration data is provided through a custom DataReader class, which
serves as the standard input data interface for the Quark quantizer.

.. code:: ipython3

    input_tensor = np.array(
        [
            [
                [
                    [0.39250988, 0.34032542, 0.91402656, 0.40040675, 0.39050988],
                    [0.39150988, 0.07389439, 0.38661167, 0.8645387, 0.55553377],
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
    
    
    data_reader = DataReader(input_tensor)

Quantizing the Model
--------------------

In this step, we use the prepared floating-point model and calibration
data, and perform quantization by configuring and executing it with the
``quantize_model`` function. Quark will automatically handle custom
operators.

.. code:: ipython3

    def quantize_model(float_model_path, quantized_model_path, data_reader):
        config_copy = copy.deepcopy(S16S16_MIXED_S8S8_CONFIG)
        config_copy.extra_options["UserCustomOpLibPath"] = [get_library_path(), ext_lib_path()]
        quant_config = Config(global_quant_config=config_copy)
        quantizer = ModelQuantizer(quant_config)
        quantizer.quantize_model(float_model_path, quantized_model_path, data_reader)
        print("Quantized the ONNX model and saved it at:", quantized_model_path)
    
    
    quantized_model_path = Path("user_custom_op_model_quantized.onnx").as_posix()
    quantize_model(float_model_path, quantized_model_path, data_reader)

Inference with the Quantized Model
----------------------------------

After quantization, we use ONNX Runtime to load the quantized model and
perform inference. The model output is printed to verify that the
quantized model runs successfully and that custom operators are
correctly applied.

.. code:: ipython3

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
    
    
    infer_quantized_model(quantized_model_path)

Summary
-------

This guide demonstrated how to: - Build an ONNX model with custom
operators (Python & C++) - Prepare calibration data for quantization -
Quantize the model using Quark - Perform inference on the quantized
model using ONNX Runtime

With these steps, you can seamlessly apply Quark to any ONNX model
containing custom operators, making quantization more flexible and
controllable.
