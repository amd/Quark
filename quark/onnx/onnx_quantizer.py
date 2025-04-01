#
# Modifications copyright(c) 2023 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from quark.shares.utils.log import ScreenLogger
import numpy as np
import onnx
import onnx.numpy_helper
from onnx import ModelProto, NodeProto
from onnx import onnx_pb as onnx_proto
from typing import Any, List, Dict, Optional, Tuple

try:
    from onnx.reference.op_run import to_array_extended
except ImportError:
    # old version of onnx.
    to_array_extended = None  # type: ignore

from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer as OrtONNXQuantizer
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizationMode,
    QuantizedValue,
    QuantizedValueType,
    QuantType,
    find_by_name,
    model_has_infer_metadata,
    tensor_proto_to_array,
)
from onnxruntime.quantization.registry import CreateOpQuantizer

from .quant_utils import (__producer__, __version__, VitisQuantType, ONNX_WBIT_QTYPES_LIST, ONNX_FP_QTYPES_LIST,
                          ONNX_BFP_QTYPES_LIST, get_tensor_type_from_qType, get_qmin_qmax_for_qType, compute_scale_zp,
                          compute_scale_zp_fp, quantize_data_pof2s, ONNX_TYPE_TO_NP_TYPE, check_relu_like_node)

logger = ScreenLogger(__name__)


class ONNXQuantizer(OrtONNXQuantizer):  # type: ignore
    """
    A class to perform quantization on an ONNX model.

    Args:
        model (ModelProto): The ONNX model to be quantized.
        per_channel (bool): Whether to perform per-channel quantization.
        reduce_range (bool): Whether to reduce the quantization range.
        mode (QuantizationMode.QLinearOps): The quantization mode to be used.
        static (bool): Whether to use static quantization.
        weight_qType (Any): The quantization type for weights.
        activation_qType (Any): The quantization type for activations.
        tensors_range (Any): The range of tensors for quantization.
        nodes_to_quantize (List[str]): List of node names to be quantized.
        nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
        op_types_to_quantize (List[str]): List of operation types to be quantized.
        extra_options (Optional[Dict[str, Any]]): Additional options for quantization.

    Inherits from:
        OrtONNXQuantizer: Base class for ONNX quantization.
    """

    def __init__(
        self,
        model: ModelProto,
        per_channel: bool,
        reduce_range: bool,
        mode: QuantizationMode.QLinearOps,
        static: bool,
        weight_qType: Any,
        activation_qType: Any,
        tensors_range: Any,
        nodes_to_quantize: List[str],
        nodes_to_exclude: List[str],
        op_types_to_quantize: List[str],
        extra_options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model,
            per_channel=per_channel,
            reduce_range=reduce_range,
            mode=mode,
            static=static,
            weight_qType=weight_qType,
            activation_qType=activation_qType,
            tensors_range=tensors_range,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            op_types_to_quantize=op_types_to_quantize,
            extra_options=extra_options,
        )


class VitisONNXQuantizer(OrtONNXQuantizer):  # type: ignore
    """
    A class to perform quantization on an ONNX model specifically optimized for Vitis AI.

    Args:
        model (ModelProto): The ONNX model to be quantized.
        per_channel (bool): Whether to perform per-channel quantization.
        reduce_range (bool): Whether to reduce the quantization range.
        mode (QuantizationMode.QLinearOps): The quantization mode to be used.
        static (bool): Whether to use static quantization.
        weight_qType (Any): The quantization type for weights.
        activation_qType (Any): The quantization type for activations.
        tensors_range (Any): Dictionary specifying the min and max values for tensors.
        nodes_to_quantize (List[str]): List of node names to be quantized.
        nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
        op_types_to_quantize (List[str]): List of operation types to be quantized.
        calibrate_method (Any): The calibration method to be used.
        quantized_tensor_type (Dict[Any, Any], optional): Dictionary specifying the types for quantized tensors.
        extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.

    Inherits from:
        OrtONNXQuantizer: Base class for ONNX quantization.
    """

    def __init__(
        self,
        model: ModelProto,
        per_channel: bool,
        reduce_range: bool,
        mode: QuantizationMode.QLinearOps,
        static: bool,
        weight_qType: Any,
        activation_qType: Any,
        tensors_range: Any,
        nodes_to_quantize: List[str],
        nodes_to_exclude: List[str],
        op_types_to_quantize: List[str],
        calibrate_method: Any,
        quantized_tensor_type: Dict[Any, Any] = {},
        extra_options: Optional[Dict[str, Any]] = None,
    ):
        self.calibrate_method = calibrate_method
        self.quantized_tensor_type = quantized_tensor_type
        OrtONNXQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            activation_qType,
            None,  # base class no need to calculate quantization params
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options=None,
        )
        if not model_has_infer_metadata(model):
            from onnxruntime.quantization.quant_utils import save_and_reload_model_with_shape_infer
            model = save_and_reload_model_with_shape_infer(model)
        self.value_infos = {vi.name: vi for vi in model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in model.graph.output})
        self.value_infos.update({it.name: it for it in model.graph.input})

        self.model = ONNXModel(model)
        if not static:
            self.model.replace_gemm_with_matmul()

        self.per_channel = per_channel  # weight-pack per channel
        self.reduce_range = reduce_range
        self.mode = mode  # QuantizationMode.Value
        self.static = static  # use static quantization for inputs.
        self.fuse_dynamic_quant = False

        self.extra_options = extra_options if extra_options else {}
        self.enable_subgraph_quantization = ("EnableSubgraph" in self.extra_options
                                             and self.extra_options["EnableSubgraph"])
        self.force_quantize_no_input_check = ("ForceQuantizeNoInputCheck" in self.extra_options
                                              and self.extra_options["ForceQuantizeNoInputCheck"])
        self.q_matmul_const_b_only = "MatMulConstBOnly" in self.extra_options and self.extra_options["MatMulConstBOnly"]

        self.use_qdq_vitis_custom_ops = True
        if "UseQDQVitisCustomOps" in self.extra_options:
            self.use_qdq_vitis_custom_ops = self.extra_options["UseQDQVitisCustomOps"]
        self.use_power_of_2_scale = True
        if "UsePowerOf2Scale" in self.extra_options:
            self.use_power_of_2_scale = self.extra_options["UsePowerOf2Scale"]

        self.is_weight_symmetric = (weight_qType in (QuantType.QInt8, VitisQuantType.QInt16, VitisQuantType.QInt32,
                                                     VitisQuantType.QFloat16, VitisQuantType.QBFloat16,
                                                     VitisQuantType.QBFP, VitisQuantType.QMX) if "WeightSymmetric"
                                    not in self.extra_options else self.extra_options["WeightSymmetric"])
        self.is_activation_symmetric = (activation_qType in (VitisQuantType.QFloat16, VitisQuantType.QBFloat16,
                                                             VitisQuantType.QBFP,
                                                             VitisQuantType.QMX) if "ActivationSymmetric"
                                        not in self.extra_options else self.extra_options["ActivationSymmetric"])

        self.is_weight_scaled = True
        if weight_qType in (VitisQuantType.QFloat16, VitisQuantType.QBFloat16, VitisQuantType.QBFP, VitisQuantType.QMX):
            self.is_weight_scaled = False if "WeightScaled" not in self.extra_options else self.extra_options[
                "WeightScaled"]
        self.is_activation_scaled = True
        if activation_qType in (VitisQuantType.QFloat16, VitisQuantType.QBFloat16, VitisQuantType.QBFP,
                                VitisQuantType.QMX):
            self.is_activation_scaled = False if "ActivationScaled" not in self.extra_options else self.extra_options[
                "ActivationScaled"]

        self.use_unsigned_relu = (False if "UseUnsignedReLU" not in self.extra_options else
                                  self.extra_options["UseUnsignedReLU"])
        self.activation_qType = get_tensor_type_from_qType(activation_qType)
        self.weight_qType = get_tensor_type_from_qType(weight_qType)
        self.tensors_range = tensors_range

        #    Dictionary specifying the min and max values for tensors. It has following format:
        #        {
        #            "param_name": [min, max]
        #        }
        #    example:
        #        {
        #            'Conv_3:0': [np.float32(0), np.float32(0.5)],
        #            'Conv_4:0': [np.float32(1), np.float32(3.5)]
        #        }

        self.nodes_to_quantize = nodes_to_quantize  # specific nodes to quantize
        self.nodes_to_exclude = nodes_to_exclude  # specific nodes to exclude
        self.op_types_to_quantize = op_types_to_quantize
        self.new_nodes: List[NodeProto] = []
        self.parent = None
        self.graph_scope = "/"  # for human readable debug information
        self.tensor_names = {}  # in case the shape inference not totally working
        self.tensor_names.update({ot.name: 1 for ot in model.graph.output})
        self.tensor_names.update({it.name: 1 for it in model.graph.input})
        for node in self.model.model.graph.node:
            self.tensor_names.update({output_name: 1 for output_name in node.output})

        self.opset_version = self.check_opset_version()

        if self.mode not in QuantizationMode:
            raise ValueError("unsupported quantization mode {}".format(self.mode))

        self.quantization_params = self.calculate_quantization_params()

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_uint8_name = "fixed_quantization_range_uint8"
        self.fixed_qrange_int8_name = "fixed_quantization_range_int8"
        # For uint8 data-type, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"
        # For int8 data-type, zero point is always zero (respresented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"

        # Map of all original value names to quantized value names
        self.quantized_value_map: Dict[Any, Any] = {}
        # some output from nodes will be quantized, yet itself should be treat as existing so
        # no dequantized will be applied when needed later
        self.generated_value_names = self.model.get_non_initializer_inputs()
        # to store specified scale and zeropoint instead of calculated value, tensor_name->(scale, zeropoint)
        self.used_scale_zp_map: Dict[Any, Any] = {}

    def _get_quantization_params(self,
                                 param_name: str,
                                 use_scale: Any = None,
                                 use_zeropoint: Any = None,
                                 zero_point_type: Any = None) -> Tuple[bool, str, str, Any, Any]:
        """
        Create initializers and inputs in the graph for zero point and scale of output.
        Zero point and scale values are obtained from self.quantization_params if specified.
            parameter param_name: Name of the quantization parameter.
            return: result, scale_name, zero_point_name, scale_shape, zero_point_shape.
        """
        from onnxruntime.quantization.onnx_quantizer import QuantizationParams
        if zero_point_type in [VitisQuantType.QFloat16, VitisQuantType.QBFloat16]:
            zero_point_values = np.array([0], dtype=np.float32)
            scale_values = np.array([1], dtype=np.float32)
            zero_point_type = get_tensor_type_from_qType(zero_point_type)
        elif use_scale is None or use_zeropoint is None:
            if zero_point_type is None:
                zero_point_type = self.activation_qType
            else:
                zero_point_type = get_tensor_type_from_qType(zero_point_type)
            if self.quantization_params is None or param_name not in self.quantization_params:
                logger.info(f'Quantization parameters for tensor:"{param_name}" not specified')
                return False, "", "", "", ""

            params = self.quantization_params[param_name]
            if not isinstance(params, QuantizationParams):
                raise TypeError(f"Unexpected type {type(params)} for {param_name!r}.")
            if params is None or len(params) != 3:
                raise ValueError("Quantization parameters should contain zero point, scale, quant type. "
                                 f"Specified values for output {param_name}: {params}")

            zero_point_values = np.array([params["zero_point"]])
            if not hasattr(params["scale"], "dtype") or params["scale"].dtype not in (np.float32, np.float16):
                raise ValueError(f"Unexpected type {type(params['scale'])} and param_name={param_name!r}")
            scale_values = np.array([params["scale"]])
            assert scale_values.dtype != np.float64
        else:
            if zero_point_type is None:
                zero_point_type = self.activation_qType
            else:
                zero_point_type = get_tensor_type_from_qType(zero_point_type)
            zero_point_values = np.array([use_zeropoint])
            scale_values = np.array([use_scale])
            params = self.quantization_params[param_name]
            if "scale" in params:
                dtype = params["scale"].dtype
                scale_values = scale_values.astype(dtype)
            assert scale_values.dtype != np.float64

        zero_point_shape: List[Any] = []
        zero_point_name = param_name + "_zero_point"
        scale_shape: List[Any] = []
        scale_name = param_name + "_scale"

        # Add initializers
        init_zp = onnx.helper.make_tensor(zero_point_name, zero_point_type, zero_point_shape,
                                          zero_point_values.ravel().tolist())
        self.model.add_initializer(init_zp)
        if scale_values.dtype == np.float32:
            scale_type = onnx_proto.TensorProto.FLOAT
        elif scale_values.dtype == np.float16:
            scale_type = onnx_proto.TensorProto.FLOAT16
        else:
            raise ValueError(f"Unexpected dtype={scale_values.dtype} for param_name={param_name!r}")
        init_scale = onnx.helper.make_tensor(scale_name, scale_type, scale_shape, scale_values.reshape((-1, )).tolist())
        self.model.add_initializer(init_scale)

        return True, scale_name, zero_point_name, scale_shape, zero_point_shape

    def find_quant_scale_zp(self, input_name: str) -> Any:
        """
        Finds the quantization scale and zero-point for a given input.

        This method looks up the quantization scale and zero-point values for the specified input name.
        It first checks the current instance's `used_scale_zp_map`. If not found, it recursively checks
        the parent instance if one exists.

        :param input_name: The name of the input for which to find the quantization scale and zero-point.
        :type input_name: str
        :return: A tuple containing the quantization scale and zero-point if found, otherwise (None, None).
        :rtype: Any
        """
        if input_name in self.used_scale_zp_map:
            return self.used_scale_zp_map[input_name]
        if self.parent is not None:
            return self.parent.find_quantized_value(input_name)
        return (None, None)

    def find_quantized_value(self, input_name: str) -> Any:
        """
        Finds the quantized value for a given input.

        This method looks up the quantized value for the specified input name.
        It first checks the current instance's `quantized_value_map`. If not found, it recursively checks
        the parent instance if one exists.

        :param input_name: The name of the input for which to find the quantized value.
        :type input_name: str
        :return: The quantized value if found, otherwise None.
        :rtype: Any
        """
        if input_name in self.quantized_value_map:
            return self.quantized_value_map[input_name]
        if self.parent is not None:
            return self.parent.find_quantized_value(input_name)
        return None

    def quantize_model(self) -> Any:
        if self.has_QDQ_nodes():
            logger.warning(
                "Please check if the model is already quantized."
                "Note you don't need to quantize a QAT model. OnnxRuntime support to run QAT model directly.")

        for node in self.model.nodes():
            # quantize subgraphes if have
            if self.enable_subgraph_quantization:
                node = self.quantize_node_with_sub_graph(node)

            number_of_existing_new_nodes = len(self.new_nodes)
            op_quantizer = CreateOpQuantizer(self, node)
            op_quantizer.quantize()
            for i in range(number_of_existing_new_nodes, len(self.new_nodes)):
                for output_name in self.new_nodes[i].output:
                    self.generated_value_names.add(output_name)

        self._dequantize_outputs()

        # extend is used to append to the list for a protobuf fields
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated?csw=1#fields
        self.model.graph().ClearField("node")
        self.model.graph().node.extend(self.new_nodes)

        # Remove ununsed initializers from graph, starting from the top level graph.
        if self.parent is None:
            _, initializers_not_found = self.model.clean_initializers()
            if len(initializers_not_found) > 0:
                raise RuntimeError("Invalid model with unknown initializers/tensors." + str(initializers_not_found))

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def quantize_bias_static(self, bias_name: str, input_name: str, weight_name: str, beta: float = 1.0) -> Any:
        """
        Quantizes the bias using static quantization. Zero Point == 0 and Scale == Input_Scale * Weight_Scale.

        This method performs the following steps:
        1. Validates the weight quantization type.
        2. Retrieves the scale for the weight.
        3. Retrieves the bias data and its scale.
        4. Retrieves the scale for the input.
        5. Calculates the scale for the bias.
        6. Quantizes the bias data.
        7. Updates the bias, scale, and zero-point initializers in the model.
        8. Updates the quantized value map with the new quantized bias information.

        :param bias_name: The name of the bias to be quantized.
        :type bias_name: str
        :param input_name: The name of the input associated with the bias.
        :type input_name: str
        :param weight_name: The name of the weight associated with the bias.
        :type weight_name: str
        :param beta: A scaling factor applied during quantization. Default is 1.0.
        :type beta: float
        :return: The name of the quantized bias.
        :rtype: Any

        :raises ValueError: If the weight quantization type is not supported or if the input name is not found in the quantized value map.
        """
        # Handle case where bias already in quantization map
        if bias_name in self.quantized_value_map:
            return self.quantized_value_map[bias_name].q_name

        # get scale for weight
        weight_scale_name = self.quantized_value_map[weight_name].scale_name
        weight_initializer = find_by_name(weight_scale_name, self.model.initializer())
        weight_scale = tensor_proto_to_array(weight_initializer)

        # get bias
        bias_initializer = find_by_name(bias_name, self.model.initializer())
        bias_data = tensor_proto_to_array(bias_initializer)
        quantized_bias_name = bias_name + TENSOR_NAME_QUANT_SUFFIX

        # get scale for input
        if input_name in self.quantized_value_map:
            input_scale_name = self.quantized_value_map[input_name].scale_name
        elif input_name in self.quantization_params:
            _, input_scale_name, _, _, _ = self._get_quantization_params(input_name)
        else:
            raise ValueError("Expected {} to be in quantized value map for static quantization".format(input_name))

        inputscale_initializer = find_by_name(input_scale_name, self.model.initializer())
        input_scale = tensor_proto_to_array(inputscale_initializer)

        # calcuate scale for bias
        bias_scale = input_scale * weight_scale * beta

        # quantize bias
        quantized_data = (np.asarray(bias_data) / bias_scale).round()
        quantized_data = np.clip(quantized_data, -2**31, 2**31 - 1)  # Clip for Int32 datatype
        quantized_data = quantized_data.astype(np.int32)

        # update bias initializer
        bias_np_data = np.asarray(quantized_data, dtype=np.int32).reshape(bias_initializer.dims)
        packed_bias_initializer = onnx.numpy_helper.from_array(bias_np_data, quantized_bias_name)
        self.model.initializer().extend([packed_bias_initializer])

        # update scale initializer
        quantized_bias_scale_name = quantized_bias_name + "_scale"
        bias_scale_data = np.asarray(bias_scale, dtype=np.float32).reshape(-1)
        if self.is_per_channel():
            packed_bias_scale_initializer = onnx.numpy_helper.from_array(bias_scale_data, quantized_bias_scale_name)
        else:
            packed_bias_scale_initializer = onnx.helper.make_tensor(quantized_bias_scale_name,
                                                                    onnx_proto.TensorProto.FLOAT, [], bias_scale_data)
        self.model.initializer().extend([packed_bias_scale_initializer])

        # update zero initializer
        quantized_bias_zp_name = quantized_bias_name + "_zero_point"
        bias_zp_data = np.zeros(bias_scale.shape, dtype=np.int32).reshape(-1)
        if self.is_per_channel():
            packed_bias_zp_initializer = onnx.numpy_helper.from_array(bias_zp_data, quantized_bias_zp_name)
        else:
            packed_bias_zp_initializer = onnx.helper.make_tensor(quantized_bias_zp_name, onnx_proto.TensorProto.INT32,
                                                                 [], bias_zp_data)
        self.model.initializer().extend([packed_bias_zp_initializer])

        assert bias_name not in self.quantized_value_map
        quantized_value = QuantizedValue(
            bias_name,
            quantized_bias_name,
            quantized_bias_scale_name,
            quantized_bias_zp_name,
            QuantizedValueType.Initializer,
            0 if bias_scale_data.size > 1 else None,
        )
        self.quantized_value_map[bias_name] = quantized_value

        return quantized_bias_name

    # In some circumstances a weight is not an initializer, for example of MatMul, if both A and B are not
    # initializer, B can still be considered as Weight
    def quantize_weight(
        self,
        node: NodeProto,
        indices: Any,
        reduce_range: bool = False,
        op_level_per_channel: bool = False,
        axis: int = -1,
        from_subgraph: bool = False,
    ) -> Any:
        """
        Quantizes the weights of a given node.

        In some circumstances, a weight is not an initializer. For example, in MatMul, if both A and B are not initializers,
        B can still be considered as a weight.

        This method calls `__quantize_inputs` to perform the weight quantization.

        :param node: The node containing the weights to be quantized.
        :type node: NodeProto
        :param indices: The indices of the inputs to be quantized.
        :type indices: Any
        :param reduce_range: Flag to indicate whether to reduce the quantization range. Default is False.
        :type reduce_range: bool, optional
        :param op_level_per_channel: Flag to indicate whether to use per-channel quantization at the operator level. Default is False.
        :type op_level_per_channel: bool, optional
        :param axis: The axis for per-channel quantization. Default is -1.
        :type axis: int, optional
        :param from_subgraph: Flag to indicate whether the node is from a subgraph. Default is False.
        :type from_subgraph: bool, optional
        :return: The result of the weight quantization process.
        :rtype: Any
        """
        return self.__quantize_inputs(
            node=node,
            indices=indices,
            initializer_use_weight_qType=True,
            reduce_range=reduce_range,
            op_level_per_channel=op_level_per_channel,
            axis=axis,
            from_subgraph=from_subgraph,
        )

    def __quantize_inputs(
        self,
        node: NodeProto,
        indices: Any,
        initializer_use_weight_qType: bool = True,
        reduce_range: bool = False,
        op_level_per_channel: bool = False,
        axis: int = -1,
        from_subgraph: bool = False,
    ) -> Any:
        """
        Given a node, this function quantizes the inputs as follows:
            - If input is an initializer, quantize the initializer data, replace old initializer
              with new initializer
            - Else, add QuantizeLinear nodes to perform quantization
            parameter node: node being quantized in NodeProto format.
            parameter indices: input indices to quantize.
            return: (List of quantized input names,
                     List of zero point names used for input quantization,
                     List of scale names used for input quantization,
                     List of new QuantizeLinear nodes created)
        """

        scale_names = []
        zero_point_names = []
        quantized_input_names = []
        nodes = []

        for input_index in indices:
            node_input = node.input[input_index]

            # Find if this input is already quantized
            if node_input in self.quantized_value_map:
                quantized_value = self.quantized_value_map[node_input]
                scale_names.append(quantized_value.scale_name)
                zero_point_names.append(quantized_value.zp_name)
                quantized_input_names.append(quantized_value.q_name)
                continue

            # Quantize the input
            initializer = find_by_name(node_input, self.model.initializer())
            if initializer is not None:
                if self.per_channel and op_level_per_channel:
                    (
                        q_weight_name,
                        zp_name,
                        scale_name,
                    ) = self.quantize_weight_per_channel(
                        initializer.name,
                        self.weight_qType if initializer_use_weight_qType else self.activation_qType,
                        axis,
                        self.calibrate_method,
                        reduce_range,
                    )
                else:
                    q_weight_name, zp_name, scale_name = self.quantize_initializer(
                        initializer,
                        self.weight_qType if initializer_use_weight_qType else self.activation_qType,
                        self.calibrate_method,
                        reduce_range,
                    )

                quantized_input_names.append(q_weight_name)
                zero_point_names.append(zp_name)
                scale_names.append(scale_name)
            elif self.contains_tensor(node_input):
                # Add QuantizeLinear node.
                qlinear_node = self.model.find_node_by_name(node_input + "_QuantizeLinear", self.new_nodes,
                                                            self.model.graph())
                if qlinear_node is None:
                    quantize_input_nodes = self._get_quantize_input_nodes(node, input_index, self.activation_qType)
                    if quantize_input_nodes is None:
                        return (None, None, None, None)
                    if from_subgraph:
                        self.add_new_nodes(quantize_input_nodes)
                    else:
                        nodes.extend(quantize_input_nodes)
                    qlinear_node = quantize_input_nodes[-1]

                if qlinear_node.op_type == "QuantizeLinear":
                    quantized_input_names.extend(qlinear_node.output)
                    scale_names.append(qlinear_node.input[1])
                    zero_point_names.append(qlinear_node.input[2])
                else:
                    quantized_input_names.append(qlinear_node.output[0])
                    scale_names.append(qlinear_node.output[1])
                    zero_point_names.append(qlinear_node.output[2])
            elif self.parent is not None:
                (
                    parent_quantized_input_names,
                    parent_zero_point_names,
                    parent_scale_names,
                    _,
                ) = self.parent.__quantize_inputs(
                    node,
                    [input_index],
                    initializer_use_weight_qType=initializer_use_weight_qType,
                    reduce_range=reduce_range,
                    op_level_per_channel=op_level_per_channel,
                    axis=axis,
                    from_subgraph=True,
                )
                quantized_input_names.append(parent_quantized_input_names[0])
                scale_names.append(parent_scale_names[0])
                zero_point_names.append(parent_zero_point_names[0])
                # node should not be add this child level here
            else:
                raise ValueError("Invalid tensor name to quantize: {} @graph scope{}".format(
                    node_input, self.graph_scope))

        return quantized_input_names, zero_point_names, scale_names, nodes

    def quantize_initializer(self,
                             weight: Any,
                             qType: Any,
                             method: Any,
                             reduce_range: bool = False,
                             keep_float_weight: bool = False) -> Tuple[str, str, str]:
        """
        :param weight: TensorProto initializer
        :param qType: type to quantize to. Note that it may be different with weight_qType because of mixed precision
        :param keep_float_weight: Whether to quantize the weight. In some cases, we only want to qunatize scale and zero point.
                                  If keep_float_weight is False, quantize the weight, or don't quantize the weight.
        :return: quantized weight name, zero point name, scale name
        """
        # Find if this input is already quantized
        if weight.name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight.name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name = weight.name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight.name + "_zero_point"
        scale_name = weight.name + "_scale"

        # Update packed weight, zero point, and scale initializers
        weight_data = tensor_proto_to_array(weight)

        _, _, zero_point, scale, q_weight_data = quantize_data_pof2s(
            weight_data.flatten(),
            qType,
            self.is_weight_symmetric,
            self.reduce_range and reduce_range,
            method=method,
            use_pof2s=self.use_power_of_2_scale,
            use_scaling=self.is_weight_scaled,
        )
        scale_dtype = weight.data_type
        scale_initializer = onnx.helper.make_tensor(scale_name, scale_dtype, [], scale.reshape((-1, )).tolist())
        if qType in ONNX_BFP_QTYPES_LIST:
            # BFP data types do not need zero point, but we need to consider the case of reusing zero point of
            # weight for activation, such as Gather aligns its output with input.
            if self.activation_qType in ONNX_BFP_QTYPES_LIST:
                zero_initializer = onnx.helper.make_tensor(zp_name, onnx_proto.TensorProto.FLOAT, [],
                                                           zero_point.reshape((-1, )).tolist())
            else:
                if self.activation_qType not in ONNX_FP_QTYPES_LIST:
                    zero_point = zero_point.astype(ONNX_TYPE_TO_NP_TYPE[self.activation_qType])
                zero_initializer = onnx.helper.make_tensor(zp_name, self.activation_qType, [],
                                                           zero_point.reshape((-1, )).tolist())
        else:
            zero_initializer = onnx.helper.make_tensor(zp_name, qType, [], zero_point.reshape((-1, )).tolist())
        self.model.initializer().extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            if qType in ONNX_FP_QTYPES_LIST:
                q_weight_initializer = onnx.TensorProto()
                q_weight_initializer.data_type = qType
                q_weight_initializer.dims.extend(weight.dims)
                q_weight_initializer.name = q_weight_name
                # Do not remove .flatten().copy() numpy is not clear about data persistence.
                q_weight_initializer.raw_data = q_weight_data.flatten().copy().tobytes()
                # Commented out this check since this QType is just for experimental usage.
                '''
                if to_array_extended is not None:
                    # This test should not be needed but it helped catch some issues
                    # with data persistence and tobytes.
                    check = to_array_extended(q_weight_initializer)
                    if check.shape != weight_data.shape or check.tobytes(
                    ) != q_weight_data.tobytes():
                        raise RuntimeError(
                            f"The initializer of shape {weight_data.shape} could not be created, expecting "
                            f"{q_weight_data.tobytes()[:10]}, got {check.tobytes()[:10]} and shape={weight.shape}"
                            f"\nraw={str(q_weight_initializer)[:200]}.")
                '''
            elif qType in ONNX_BFP_QTYPES_LIST:
                # We just use original values for BFP data types, because the quantized weight is not actually used
                q_weight_initializer = onnx.TensorProto()
                q_weight_initializer.CopyFrom(weight)
                q_weight_initializer.name = q_weight_name
            else:
                q_weight_data = np.asarray(q_weight_data,
                                           dtype=onnx.helper.tensor_dtype_to_np_dtype(qType)).reshape(weight.dims)
                q_weight_initializer = onnx.numpy_helper.from_array(q_weight_data, q_weight_name)

            self.model.initializer().extend([q_weight_initializer])

        # Log entry for this quantized weight
        quantized_value = QuantizedValue(
            weight.name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight.name] = quantized_value

        return q_weight_name, zp_name, scale_name

    def quantize_weight_per_channel(
        self,
        weight_name: str,
        weight_qType: Any,
        channel_axis: Any,
        method: Any,
        reduce_range: bool = True,
        keep_float_weight: bool = False,
    ) -> Tuple[str, str, str]:
        """
        Quantizes the given weight tensor per channel.

        This method quantizes the weights per channel, creating separate quantization parameters (scale and zero-point) for each channel.

        :param weight_name: The name of the weight tensor to be quantized.
        :type weight_name: str
        :param weight_qType: The data type to use for quantization.
        :type weight_qType: Any
        :param channel_axis: The axis representing the channel dimension in the weight tensor.
        :type channel_axis: Any
        :param method: The quantization method to use.
        :type method: Any
        :param reduce_range: Whether to reduce the quantization range. Default is True.
        :type reduce_range: bool, optional
        :param keep_float_weight: Whether to keep the original floating-point weights. Default is False.
        :type keep_float_weight: bool, optional
        :return: A tuple containing the names of the quantized weight tensor, zero-point tensor, and scale tensor.
        :rtype: Tuple[str, str, str]

        :raises ValueError: If the specified weight is not an initializer.
        """
        if weight_name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight_name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        initializer = find_by_name(weight_name, self.model.initializer())
        if initializer is None:
            raise ValueError("{} is not an initializer", weight_name)

        weights = tensor_proto_to_array(initializer)
        channel_count = weights.shape[channel_axis]
        rmin_list = []
        rmax_list = []
        zero_point_list = []
        scale_list = []
        quantized_per_channel_data_list = []
        for i in range(channel_count):
            per_channel_data = weights.take(i, channel_axis)
            rmin, rmax, zero_point, scale, quantized_per_channel_data = quantize_data_pof2s(
                np.array(per_channel_data.flatten().tolist()),
                weight_qType,
                self.is_weight_symmetric or weight_qType
                in (onnx_proto.TensorProto.INT8, onnx_proto.TensorProto.INT16, onnx_proto.TensorProto.INT32,
                    onnx_proto.TensorProto.FLOAT16, onnx_proto.TensorProto.BFLOAT16),
                self.reduce_range and reduce_range,
                method=method,
                use_pof2s=self.use_power_of_2_scale,
            )
            rmin_list.append(rmin)
            rmax_list.append(rmax)
            zero_point_list.append(zero_point.item())
            scale_list.append(scale.item())
            quantized_per_channel_data_list.append(quantized_per_channel_data)

        # combine per_channel_data into one
        reshape_dims = list(weights.shape)  # deep copy
        reshape_dims[channel_axis] = 1  # only one per channel for reshape
        quantized_weights = np.asarray(quantized_per_channel_data_list[0]).reshape(reshape_dims)
        for i in range(1, len(quantized_per_channel_data_list)):
            channel_weights = np.asarray(quantized_per_channel_data_list[i]).reshape(reshape_dims)
            quantized_weights = np.concatenate((quantized_weights, channel_weights), channel_axis)

        q_weight_name = weight_name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight_name + "_zero_point"
        scale_name = weight_name + "_scale"

        quantized_value = QuantizedValue(
            weight_name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight_name] = quantized_value

        # Update packed weight, zero point, and scale initializers
        zero_scale_shape = [initializer.dims[channel_axis]]
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape,
                                                    scale_list)
        zero_initializer = onnx.helper.make_tensor(zp_name, weight_qType, zero_scale_shape, zero_point_list)

        self.model.initializer().extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            quantized_weights = np.asarray(
                quantized_weights,
                dtype=onnx.helper.tensor_dtype_to_np_dtype(weight_qType),
            ).reshape(initializer.dims)
            q_weight_initializer = onnx.numpy_helper.from_array(quantized_weights, q_weight_name)
            self.model.initializer().extend([q_weight_initializer])

        return q_weight_name, zp_name, scale_name

    def calculate_quantization_params(self) -> Any:
        """
        Calculates the quantization parameters for each tensor in the model.

        This method computes the quantization parameters (scale and zero-point) for each tensor in the model
        based on its range (rmin and rmax). It adjusts the tensor ranges for the inputs of Clip and Relu nodes
        and ensures the correct quantization parameters are used for each tensor type.

        :return: A dictionary containing the quantization parameters for each tensor.
        :rtype: Any

        :raises ValueError: If a weight is not an initializer.

        Notes:
            - If `self.tensors_range` is None, the method returns immediately.
            - Adjusts tensor ranges for Clip and Relu nodes.
            - For versions of ONNX Runtime below 1.16.0, specific quantization parameters are computed.
            - For versions of ONNX Runtime 1.16.0 and above, the `QuantizationParams` class is used.
            - Forces asymmetric quantization for ReLU-like output tensors if `self.use_unsigned_relu` is True.
        """
        if self.tensors_range is None:
            return

        # adjust tensor_ranges for input of Clip and Relu node
        relu_like_output_tensors: Any = []
        for node in self.model.nodes():
            if node.op_type not in ["Clip", "Relu"]:
                continue
            elif self.should_quantize_node(node) and check_relu_like_node(self.model.model, node):
                relu_like_output_tensors.append(node.output[0])

            if self.is_activation_symmetric:
                continue
            if self.activation_qType in ONNX_WBIT_QTYPES_LIST or self.use_qdq_vitis_custom_ops:
                continue  # TODO : check what's the differences
            if not self.should_quantize_node(node):
                continue
            if len(self.model.input_name_to_nodes()[node.input[0]]) != 1:
                continue
            if node.input[0] not in self.tensors_range or node.output[0] not in self.tensors_range:
                continue
            self.tensors_range[node.input[0]] = self.tensors_range[node.output[0]]

        quantization_params = {}

        from onnxruntime.quantization.onnx_quantizer import QuantizationParams

        for tensor_name in self.tensors_range:
            td = self.tensors_range[tensor_name]
            rmin, rmax = td.range_value
            zp_type = self.activation_qType
            if tensor_name in self.quantized_tensor_type:
                zp_type = get_tensor_type_from_qType(self.quantized_tensor_type[tensor_name])
                logger.info(
                    f"The type of tensor {tensor_name} is {self.quantized_tensor_type[tensor_name]}: using specific tensor precision"
                )

            if zp_type in ONNX_FP_QTYPES_LIST:
                reduce_range = self.is_activation_scaled  # If scale the activation, it will use a reduced range
                qmin, qmax = get_qmin_qmax_for_qType(zp_type, reduce_range=reduce_range)
                zero, scale = compute_scale_zp_fp(rmin, rmax, qmin, qmax, zp_type, self.calibrate_method,
                                                  self.is_activation_symmetric, self.is_activation_scaled)
                quantization_params[tensor_name] = QuantizationParams(zero_point=zero, scale=scale, quant_type=zp_type)
            else:
                symmetric = self.is_activation_symmetric

                if tensor_name in relu_like_output_tensors and (self.is_activation_symmetric
                                                                and self.use_unsigned_relu):
                    symmetric = False  # Force it to be asymmetric to utilize representation range fully
                    rmin = 0  # Reduce the tensor range to the positive half axis

                    node = self.model.output_name_to_node()[tensor_name]
                    logger.info("Node {} output tensor {} is forced to be asymmetric.".format(node.name, tensor_name))

                qmin, qmax = get_qmin_qmax_for_qType(zp_type, symmetric=symmetric)

                zero, scale = compute_scale_zp(rmin, rmax, qmin, qmax, zp_type, self.calibrate_method, symmetric,
                                               self.use_power_of_2_scale)

                quantization_params[tensor_name] = QuantizationParams(zero_point=zero, scale=scale, quant_type=zp_type)

        return quantization_params
