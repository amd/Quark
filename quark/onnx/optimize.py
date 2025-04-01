#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from quark.shares.utils.log import ScreenLogger
import numpy as np
from math import sqrt
import copy
import onnx
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.transformers.onnx_model import OnnxModel
from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization
from onnxruntime.transformers.fusion_gelu import FusionGelu
from .quant_utils import QUANT_OP_TYPES, DEQUANT_OP_TYPES, get_clip_min_max, get_opset_version
from typing import Tuple, List, Optional, Union
from numpy.typing import NDArray
from onnx import ModelProto, NodeProto

logger = ScreenLogger(__name__)


class Optimize(object):
    """
    A class for optimizations to be applied to onnx model before quantization.

    Args:
        model (onnx.ModelProto): The ONNX model to be optimized.
        op_types_to_quantize (list): A list of operation types to be quantized.
        nodes_to_quantize (list): A list of node names to be quantized.
        nodes_to_exclude (list): A list of node names to be excluded from quantization.

    """

    def __init__(self, model: ModelProto, op_types_to_quantize: List[str], nodes_to_quantize: Optional[List[str]],
                 nodes_to_exclude: Optional[List[str]]) -> None:
        self.model = model
        self.op_types_to_quantize = op_types_to_quantize
        self.nodes_to_quantize = nodes_to_quantize
        self.nodes_to_exclude = nodes_to_exclude

    def should_quantize_node(self, node: NodeProto) -> bool:
        if (self.nodes_to_quantize is not None and len(self.nodes_to_quantize) != 0
                and node.name not in self.nodes_to_quantize):
            return False

        if node.op_type not in self.op_types_to_quantize:
            return False

        if self.nodes_to_exclude is not None and node.name in self.nodes_to_exclude:
            return False

        return True

    def replace_node_with(self, node: NodeProto, replaced_type: str) -> NodeProto:
        new_node = onnx.helper.make_node(replaced_type, inputs=node.input, outputs=node.output, name=node.name)

        self.model.graph.node.append(new_node)
        return new_node

    def convert_bn_to_conv(self) -> None:
        """Convert BatchNormalization to Conv.
        """

        def _get_folded_conv_weights(bn_gamma: NDArray[np.float32], bn_beta: NDArray[np.float32],
                                     bn_mm: NDArray[np.float32], bn_mv: NDArray[np.float32],
                                     bn_epsilon: float) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
            if bn_gamma is not None:
                multiplier = bn_gamma / np.sqrt(bn_mv + bn_epsilon)
            else:
                multiplier = 1 / np.sqrt(bn_mv + bn_epsilon)

            folded_conv_kernel = multiplier
            folded_conv_bias = bn_beta + (-bn_mm) * multiplier
            return folded_conv_kernel, folded_conv_bias

        self.op_types_to_quantize.append("BatchNormalization")
        nodes_to_remove: List[NodeProto] = []
        init_to_remove: List[str] = []
        onnx_model = ONNXModel(self.model)
        for node in onnx_model.model.graph.node:

            if node.op_type == 'BatchNormalization' and self.should_quantize_node(node):
                input_name = node.input[0]
                input_shape: List[str] = []
                for input_info in onnx_model.model.graph.value_info:
                    if input_info.name == input_name:
                        input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
                if len(node.input) == 5 and len(input_shape) == 4:
                    bn_epsilon = next((attr.f for attr in node.attribute if attr.name == 'epsilon'), 1e-10)
                    for init in onnx_model.model.graph.initializer:
                        if init.name == node.input[1]:
                            bn_gamma = onnx.numpy_helper.to_array(init)
                        elif init.name == node.input[2]:
                            bn_beta = onnx.numpy_helper.to_array(init)
                        elif init.name == node.input[3]:
                            bn_mm = onnx.numpy_helper.to_array(init)
                        elif init.name == node.input[4]:
                            bn_mv = onnx.numpy_helper.to_array(init)
                    try:
                        weights, bias = _get_folded_conv_weights(bn_gamma, bn_beta, bn_mm, bn_mv, bn_epsilon)
                        num_channel = bn_mm.shape[0]
                        weights = weights.reshape([num_channel, 1, 1, 1])
                        weights_tensor = onnx.numpy_helper.from_array(weights, name=node.output[0] + "weights")
                        bias_tensor = onnx.numpy_helper.from_array(bias, name=node.output[0] + "bias")
                        onnx_model.model.graph.initializer.extend([weights_tensor, bias_tensor])
                        new_node = onnx.helper.make_node(
                            "Conv",
                            inputs=[node.input[0], node.output[0] + "weights", node.output[0] + "bias"],
                            outputs=[node.output[0]],
                            group=num_channel,
                            kernel_shape=[1, 1],
                            strides=[1, 1],
                            name=node.name)

                        nodes_to_remove.append(node)
                        init_to_remove.extend([node.input[1], node.input[2], node.input[3], node.input[4]])
                        onnx_model.model.graph.node.append(new_node)
                        logger.info(f"Found BatchNormalization node {node.name}. "
                                    f"Replacing with Conv.")
                    except Exception as e:
                        logger.warning(
                            f"Fail to generate conv's weights and bias beacuse of {e}, skip converting bn to conv")
                else:
                    logger.warning(
                        f"Fail to convert bn {node.name} to conv beacuse BatchNormalization's input or shape does not meet the requirements"
                    )
        onnx_model.remove_nodes(nodes_to_remove)
        onnx_model.remove_initializers(init_to_remove)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        self.model = onnx_model.model

    def convert_reduce_mean_to_global_avg_pool(self) -> None:
        """Convert ReduceMean to GlobalAveragePool.
        """

        from .quant_utils import check_reduce_mean_condition
        nodes_to_remove = []
        onnx_model = ONNXModel(self.model)
        for node in onnx_model.model.graph.node:

            if node.op_type == 'ReduceMean' and check_reduce_mean_condition(onnx_model.model,
                                                                            node) and self.should_quantize_node(node):
                if len(node.input) == 1:
                    new_node = self.replace_node_with(node, 'GlobalAveragePool')
                    nodes_to_remove.append(node)
                    logger.info(f"Found ReduceMean node {node.name} with axes=[2, 3]. "
                                f"Replacing with GlobalAveragePool.")
                # Handling opset >= 18 for Reduce Mean
                elif len(node.input) == 2:
                    new_node = onnx.helper.make_node('GlobalAveragePool',
                                                     inputs=[node.input[0]],
                                                     outputs=node.output,
                                                     name=node.name)

                    nodes_to_remove.append(node)
                    onnx_model.model.graph.node.append(new_node)
                    logger.info(f"Found ReduceMean node {node.name} with axes=[2, 3]. "
                                f"Replacing with GlobalAveragePool.")
        onnx_model.remove_nodes(nodes_to_remove)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        self.model = onnx_model.model

    def split_large_kernel_pool(self) -> None:
        """
        For pooling with an excessively large kernel size in the onnx model,
        split it into multiple smaller poolings.
        """

        def _get_factors(num: int) -> Tuple[int, int]:
            factor_1 = int(sqrt(num))
            while (factor_1 > 1):
                if (num % (factor_1) == 0):
                    factor_2 = num / factor_1
                    return int(factor_1), int(factor_2)
                factor_1 = factor_1 - 1
            factor_2 = num
            return int(factor_1), int(factor_2)

        onnx_model = ONNXModel(self.model)
        for node in onnx_model.model.graph.node:
            if node.op_type == "GlobalAveragePool" and self.should_quantize_node(node):
                input_name = node.input[0]
                kw = None
                kh = None
                for input_info in onnx_model.model.graph.value_info:
                    if input_info.name == input_name:
                        input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
                        if len(input_shape) == 4:
                            shape_to_check = True
                            kh = input_shape[2]
                            kw = input_shape[3]
                        break
                if not kw or not kh:
                    logger.warning('Failed to get the input shape, skip optimizing for GlobalAveragePool {}.'.format(
                        node.name))
                    continue
                # Only one split is supported.
                # TODO: Support multiple split operations
                elif kw * kh > 512:
                    kh1, kh2 = _get_factors(kh)
                    kw1, kw2 = _get_factors(kw)
                    if kh1 * kw1 > 512 or kh2 * kw2 > 512:
                        logger.warning("After split, the kernel size is still too large."
                                       "Currently, only one split is supported. Skip optimization.")
                    else:
                        split_tensor = node.input[0] + "_Split"
                        pool_node = onnx.helper.make_node("AveragePool",
                                                          inputs=[node.input[0]],
                                                          outputs=[split_tensor],
                                                          kernel_shape=[kh1, kw1],
                                                          strides=[kh1, kw1],
                                                          name=split_tensor)
                        if not node.name:
                            node.name = node.output[0]
                        node.input[0] = split_tensor
                        onnx_model.model.graph.node.extend([pool_node])
                        logger.info(f"Found GlobalAveragePool node {node.name} with large kernel size. "
                                    f"Split it into multiple AveragePools.")
        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        self.model = onnx_model.model

    def convert_split_to_slice(self) -> None:
        """Convert Split to Slice.
        """
        nodes_to_remove: List[NodeProto] = []
        init_to_remove: List[str] = []
        onnx_model = ONNXModel(self.model)
        for node in onnx_model.model.graph.node:
            if node.op_type == 'Split' and self.should_quantize_node(node):
                num_input = len(node.input)
                axis_attr = next((attr for attr in node.attribute if attr.name == 'axis'), None)
                assert (axis_attr is not None), "No axis attribute founded in Split node"
                axis = axis_attr.i  # if axis_attr is not None else 0
                input_name = node.input[0]
                output_names = node.output
                if num_input == 2:
                    splits = None
                    for init in onnx_model.model.graph.initializer:
                        if init.name == node.input[1]:
                            splits = onnx.numpy_helper.to_array(init).tolist()
                    if splits is None:
                        logger.warning(f"No split detected of {node.name}, "
                                       "failed to convert split to slice, please check the input model.")
                        break
                elif num_input == 1:
                    split_attr = next((attr for attr in node.attribute if attr.name == 'split'), None)
                    if split_attr is None:
                        logger.warning(f"No split detected of {node.name}, "
                                       "failed to convert split to slice, please check the input model.")
                        break
                    splits = split_attr.ints
                else:
                    logger.warning(f"Failed to convert split of {node.name} to slice, "
                                   "the number of input nodes is not supported.")
                    break
                starts = [sum(splits[:i]) for i in range(len(splits))]
                ends = [sum(splits[:i + 1]) for i in range(len(splits))]
                for i in range(len(output_names)):
                    starts_node = onnx.helper.make_node('Constant',
                                                        inputs=[],
                                                        outputs=[output_names[i] + '_starts_' + str(i)],
                                                        value=onnx.helper.make_tensor(name=output_names[i] +
                                                                                      '_starts_' + str(i),
                                                                                      data_type=onnx.TensorProto.INT64,
                                                                                      dims=[1],
                                                                                      vals=[starts[i]]))
                    ends_node = onnx.helper.make_node('Constant',
                                                      inputs=[],
                                                      outputs=[output_names[i] + '_ends_' + str(i)],
                                                      value=onnx.helper.make_tensor(name=output_names[i] + '_ends_' +
                                                                                    str(i),
                                                                                    data_type=onnx.TensorProto.INT64,
                                                                                    dims=[1],
                                                                                    vals=[ends[i]]))
                    axes_node = onnx.helper.make_node('Constant',
                                                      inputs=[],
                                                      outputs=[output_names[i] + '_axes_' + str(i)],
                                                      value=onnx.helper.make_tensor(name=output_names[i] + '_axes_' +
                                                                                    str(i),
                                                                                    data_type=onnx.TensorProto.INT64,
                                                                                    dims=[1],
                                                                                    vals=[axis]))
                    steps_node = onnx.helper.make_node('Constant',
                                                       inputs=[],
                                                       outputs=[output_names[i] + '_steps_' + str(i)],
                                                       value=onnx.helper.make_tensor(name=output_names[i] + '_steps_' +
                                                                                     str(i),
                                                                                     data_type=onnx.TensorProto.INT64,
                                                                                     dims=[1],
                                                                                     vals=[1]))
                    slice_node = onnx.helper.make_node("Slice",
                                                       inputs=[
                                                           input_name, output_names[i] + '_starts_' + str(i),
                                                           output_names[i] + '_ends_' + str(i),
                                                           output_names[i] + '_axes_' + str(i),
                                                           output_names[i] + '_steps_' + str(i)
                                                       ],
                                                       outputs=[output_names[i]],
                                                       name=output_names[i] + '_' + str(i))
                    onnx_model.model.graph.node.extend([slice_node, starts_node, ends_node, axes_node, steps_node])
                nodes_to_remove.append(node)
                if len(node.input) > 1:
                    init_to_remove.append(node.input[1])
                logger.info(f"Found Split node {node.name}. "
                            f"Replacing with Slice.")
        onnx_model.remove_nodes(nodes_to_remove)
        onnx_model.remove_initializers(init_to_remove)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        self.model = onnx_model.model

    def fuse_instance_norm(self) -> None:
        '''
        The split instance norm operation will be fused to InstanceNorm operation
        '''
        onnx_model = ONNXModel(self.model)
        tensor_to_producer_dict = {}
        remove_nodes: List[NodeProto] = []
        remove_inits: List[onnx.TensorProto] = []
        for node in onnx_model.model.graph.node:
            for output in node.output:
                tensor_to_producer_dict[output] = node
        for init in onnx_model.model.graph.initializer:
            tensor_to_producer_dict[init.name] = init
        for node in onnx_model.model.graph.node:
            if node.op_type == "Add":
                try:
                    add0_i0 = node.input[0]
                    add0_i1 = node.input[1]
                    add0_i0_node = tensor_to_producer_dict[add0_i0]
                    add0_i1_node = tensor_to_producer_dict[add0_i1]
                    # TODO: Use different dictionaries to distinguish between node and init.
                    if add0_i0_node.op_type == "Mul" and add0_i1_node.op_type == "Sub":
                        sub0_node = add0_i1_node
                        sub0_i0 = sub0_node.input[0]
                        sub0_i1 = sub0_node.input[1]
                        sub0_i1_node = tensor_to_producer_dict[sub0_i1]
                        if sub0_i1_node.op_type == "Mul":
                            mul0_node = sub0_i1_node
                            mul0_i0 = mul0_node.input[0]
                            mul0_i1 = mul0_node.input[1]
                            mul0_i0_node = tensor_to_producer_dict[mul0_i0]
                            mul0_i1_node = tensor_to_producer_dict[mul0_i1]
                            if mul0_i0_node.op_type == "GlobalAveragePool" and mul0_i1_node.op_type == "Mul":
                                mul1_node = mul0_i1_node
                                mul1_i0 = mul1_node.input[0]
                                mul1_i1 = mul1_node.input[1]
                                mul1_i0_node = tensor_to_producer_dict[mul1_i0]
                                mul1_i1_node = tensor_to_producer_dict[mul1_i1]
                                if mul1_i0_node.op_type == "Reciprocal":
                                    rec0_node = mul1_i0_node
                                    rec0_i0 = rec0_node.input[0]
                                    rec0_i0_node = tensor_to_producer_dict[rec0_i0]
                                    if rec0_i0_node.op_type == "Sqrt":
                                        sqr0_node = rec0_i0_node
                                        sqr0_i0 = sqr0_node.input[0]
                                        sqr0_i0_node = tensor_to_producer_dict[sqr0_i0]
                                        if sqr0_i0_node.op_type == "Add":
                                            add1_node = sqr0_i0_node
                                            add1_i0 = add1_node.input[0]
                                            add1_i1 = add1_node.input[1]
                                            add1_i0_node = tensor_to_producer_dict[add1_i0]
                                            if add1_i0_node.op_type == "GlobalAveragePool":
                                                gap0_node = add1_i0_node
                                                gap0_i0 = gap0_node.input[0]
                                                gap0_i0_node = tensor_to_producer_dict[gap0_i0]
                                                if gap0_i0_node.op_type == "Mul":
                                                    mul2_node = gap0_i0_node
                                                    mul2_i0 = mul2_node.input[0]
                                                    mul2_i0_node = tensor_to_producer_dict[mul2_i0]
                                                    if mul2_i0_node.op_type == "Sub":
                                                        sub1_node = mul2_i0_node
                                                        sub1_i0 = sub1_node.input[0]
                                                        sub1_i1 = sub1_node.input[1]
                                                        sub1_i0_node = tensor_to_producer_dict[sub1_i0]
                                                        sub1_i1_node = tensor_to_producer_dict[sub1_i1]
                                                        if sub1_i1_node.op_type == "GlobalAveragePool":

                                                            # Remove nodes
                                                            remove_node_list = [
                                                                node,
                                                                add0_i0_node,
                                                                add0_i1_node,
                                                                sub0_i1_node,
                                                                mul0_i0_node,
                                                                mul0_i1_node,
                                                                mul1_i0_node,
                                                                rec0_i0_node,
                                                                sqr0_i0_node,
                                                                add1_i0_node,
                                                                gap0_i0_node,
                                                                mul2_i0_node,
                                                            ]

                                                            # Add InstanceNormalization
                                                            bias_init = onnx_model.get_initializer(sub0_i0)
                                                            bias_init.dims[:] = [bias_init.dims[1]]
                                                            weight_init = onnx_model.get_initializer(mul1_i1)
                                                            weight_init.dims[:] = [weight_init.dims[1]]
                                                            eps_init = onnx_model.get_initializer(add1_i1)

                                                            instance_norm_node = onnx.helper.make_node(
                                                                "InstanceNormalization", [sub1_i0, mul1_i1, sub0_i0],
                                                                node.output,
                                                                node.name,
                                                                epsilon=onnx.numpy_helper.to_array(eps_init).item())
                                                            logger.info(
                                                                f"Matched Instance Normalization, fuse it into InstanceNormalization {node.name}"
                                                            )
                                                            onnx_model.add_node(instance_norm_node)

                                                            remove_nodes.extend(remove_node_list)
                                                            remove_inits.append(eps_init)
                except Exception as e:
                    logger.debug(
                        f"FuseInstanceNorm is enabled, but {node.name} does not meet the matching rules:{e}, skipping this node"
                    )
        onnx_model.remove_nodes(remove_nodes)
        onnx_model.remove_initializers(remove_inits)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        self.model = onnx_model.model

    def fuse_l2_norm(self) -> None:
        """
        convert L2norm ops to LpNormalization
        """
        onnx_model = ONNXModel(self.model)
        tensor_to_producer_dict = {}
        remove_nodes: List[NodeProto] = []
        remove_inits: List[onnx.TensorProto] = []
        for node in onnx_model.model.graph.node:
            for output in node.output:
                tensor_to_producer_dict[output] = node
        for init in onnx_model.model.graph.initializer:
            tensor_to_producer_dict[init.name] = init
        for node in onnx_model.model.graph.node:
            if node.op_type == "Mul":
                try:
                    inp_0 = node.input[0]
                    inp_1 = node.input[1]
                    inp_0_node = tensor_to_producer_dict[inp_0]
                    inp_1_node = tensor_to_producer_dict[inp_1]
                    if inp_0_node.op_type == "Unsqueeze" and inp_1_node.op_type == "Reciprocal":
                        rec_node = inp_1_node
                        rec_inp_0 = rec_node.input[0]
                        rec_inp_0_node = tensor_to_producer_dict[rec_inp_0]
                        if rec_inp_0_node.op_type == "Sqrt":
                            sqrt_node = rec_inp_0_node
                            sqrt_inp_0 = sqrt_node.input[0]
                            sqrt_inp_0_node = tensor_to_producer_dict[sqrt_inp_0]
                            if sqrt_inp_0_node.op_type == "Max":
                                max_node = sqrt_inp_0_node
                                max_inp_0 = max_node.input[0]
                                max_inp_1 = max_node.input[1]
                                max_inp_0_node = tensor_to_producer_dict[max_inp_0]
                                if max_inp_0_node.op_type == "ReduceSum":
                                    red_node = max_inp_0_node
                                    red_inp_0 = red_node.input[0]
                                    red_inp_0_node = tensor_to_producer_dict[red_inp_0]
                                if red_inp_0_node.op_type == "Mul":
                                    mul_node = red_inp_0_node
                                    mul_inp_0 = mul_node.input[0]
                                    mul_inp_0_node = tensor_to_producer_dict[mul_inp_0]
                                    if mul_inp_0_node.op_type == "Unsqueeze":
                                        uns_node = mul_inp_0_node
                                        # Remove nodes
                                        logger.info(f"Found L2norm ops from {node.name}.")
                                        nodes_to_remove_list = [
                                            node,
                                            rec_node,
                                            sqrt_node,
                                            max_node,
                                            red_node,
                                            mul_node,
                                        ]
                                        remove_nodes.extend(nodes_to_remove_list)
                                        eps_init = onnx_model.get_initializer(max_inp_1)
                                        remove_inits.append(eps_init)
                                        # Add LpNormalization
                                        inp = uns_node.output[0]
                                        out = node.output[0]
                                        l2norm_node = onnx.helper.make_node("LpNormalization", [inp], [out],
                                                                            node.name,
                                                                            p=2)
                                        onnx_model.add_node(l2norm_node)
                                        logger.info("Converted L2norm ops from {node.name} to LpNormalization.")
                except Exception as e:
                    logger.debug(
                        f"FuseL2Norm is enabled, but {node.name} does not meet the matching rules:{e}, skipping this node"
                    )
        onnx_model.remove_nodes(remove_nodes)
        onnx_model.remove_initializers(remove_inits)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        self.model = onnx_model.model

    def fold_batch_norm(self) -> None:
        """
        fold BatchNormalization to target operations
        """

        def _get_folded_weight_bias(target_type: str, target_weight: NDArray[np.float32],
                                    target_bias: Union[NDArray[np.float32], NDArray[np.float64]],
                                    bn_gamma: Optional[NDArray[np.float32]], bn_beta: Optional[NDArray[np.float32]],
                                    bn_mean: NDArray[np.float32], bn_var: NDArray[np.float32],
                                    bn_epsilon: float) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
            if bn_gamma is not None:
                multiplier = bn_gamma / np.sqrt(bn_var + bn_epsilon)
            else:
                multiplier = 1 / np.sqrt(bn_var + bn_epsilon)

            if target_type == "Gemm":
                bn_weight = np.diag(multiplier)
            elif target_type == "ConvTranspose":
                bn_weight = multiplier.reshape(1, len(multiplier), 1, 1)

            if bn_beta is not None:
                bn_bias = bn_beta + (-bn_mean) * multiplier
            else:
                bn_bias = (-bn_mean) * multiplier

            if target_type == "Gemm":
                folded_weight = np.dot(bn_weight, target_weight)
                folded_bias = np.dot(bn_weight, target_bias) + bn_bias
            elif target_type == "ConvTranspose":
                folded_weight = bn_weight * target_weight
                folded_bias = bn_weight.reshape(1, -1) * target_bias + bn_bias
                folded_bias = folded_bias.reshape(-1)

            return folded_weight, folded_bias

        onnx_model = ONNXModel(self.model)

        TARGET_OPS = ('ConvTranspose', 'Gemm')

        remove_nodes = []

        for node in onnx_model.model.graph.node:
            if node.op_type != 'BatchNormalization' or self.should_quantize_node(node):
                continue

            if len(node.input) != 5:
                logger.warning(f"BatchNorm {node.name} with {len(node.input)} inputs cannot be folded.")
                continue

            target_node = onnx_model.get_parent(node, 0)
            if target_node is None:
                logger.warning(f"BatchNorm {node.name} that is isolated node cannot be folded.")
                continue

            if target_node.op_type not in TARGET_OPS:
                logger.debug(f"BatchNorm {node.name} after node {target_node.name} cannot be folded.")
                continue

            bn_gamma_init = onnx_model.get_initializer(node.input[1])
            bn_gamma = None if bn_gamma_init is None else onnx.numpy_helper.to_array(bn_gamma_init)
            bn_beta_init = onnx_model.get_initializer(node.input[2])
            bn_beta = None if bn_beta_init is None else onnx.numpy_helper.to_array(bn_beta_init)
            bn_mean_init = onnx_model.get_initializer(node.input[3])
            bn_mean = None if bn_mean_init is None else onnx.numpy_helper.to_array(bn_mean_init)
            bn_var_init = onnx_model.get_initializer(node.input[4])
            bn_var = None if bn_var_init is None else onnx.numpy_helper.to_array(bn_var_init)
            bn_epsilon = next((attr.f for attr in node.attribute if attr.name == 'epsilon'), 1e-10)

            if bn_mean is None or bn_var is None:
                logger.warning(f"BatchNorm {node.name} that is missing mean or variance cannot be folded.")
                continue

            target_weight_init = onnx_model.get_initializer(target_node.input[1])
            target_weight = None if target_weight_init is None else onnx.numpy_helper.to_array(target_weight_init)

            target_bias_init = onnx_model.get_initializer(target_node.input[2]) if len(target_node.input) > 2 else None
            target_bias = None if target_bias_init is None else onnx.numpy_helper.to_array(target_bias_init)

            if target_weight is None:
                logger.warning(f"BatchNorm {node.name}'s target node f{target_node.name} is not foldable.")
                continue

            target_type = target_node.op_type
            if target_type == "Gemm":
                transB = next((attr.i for attr in target_node.attribute if attr.name == 'transB'), 0)
                # TODO: Support transB is 0
                if transB == 0:
                    logger.debug(f"Target node f{target_node.name}'s transB=0 is not supported.")
                    continue
            if target_type == "ConvTranspose":
                group = next((attr.i for attr in target_node.attribute if attr.name == 'group'), 1)
                # TODO: Support ConvTranspose group != 1
                if group != 1:
                    logger.debug(f"Target node f{target_node.name}'s group !=1 is not supported.")
                    continue

            if target_bias is None:
                if target_type == "Gemm":
                    target_bias = np.zeros(target_weight.shape[0])
                else:  # target_type == "ConvTranspose":
                    target_bias = np.zeros(target_weight.shape[1])

                target_bias_name = target_node.name + "_bias_4bn"
                target_bias_init = onnx.numpy_helper.from_array(target_bias.astype(np.float32), name=target_bias_name)
                onnx_model.add_initializer(target_bias_init)
                target_node.input.append(target_bias_name)

            # Calculate the weight and bias after folded
            folded_weight, folded_bias = _get_folded_weight_bias(target_type, target_weight, target_bias, bn_gamma,
                                                                 bn_beta, bn_mean, bn_var, bn_epsilon)

            # Update target node's weight and bias
            folded_weight_init = onnx.numpy_helper.from_array(folded_weight.astype(np.float32),
                                                              name=target_weight_init.name)
            target_weight_init.CopyFrom(folded_weight_init)
            folded_bias_init = onnx.numpy_helper.from_array(folded_bias.astype(np.float32), name=target_bias_init.name)
            target_bias_init = onnx_model.get_initializer(target_node.input[2])
            target_bias_init.CopyFrom(folded_bias_init)

            # Deal with the tensor name
            children = onnx_model.get_children(target_node)

            for child in children:
                if child is node:  # this node will be removed
                    continue

                for input_index, input_name in enumerate(child.input):
                    if input_name == target_node.output[0]:
                        child.input[input_index] = node.output[0]

            target_node.output[0] = node.output[0]

            # TODO: has shared initializers?
            remove_nodes.append(node)

            logger.info(f"Folded {node.op_type} {node.name} to {target_node.op_type} {target_node.name}.")

        onnx_model.remove_nodes(remove_nodes)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        self.model = onnx_model.model

    def convert_clip_to_relu(self) -> None:
        '''
        Convert Clip to Relu.
        '''
        nodes_to_remove = []
        init_to_remove = []
        onnx_model = ONNXModel(self.model)

        for node in onnx_model.model.graph.node:
            if node.op_type == 'Clip' and self.should_quantize_node(node):
                min_value, max_value, para_type = get_clip_min_max(onnx_model.model, node)

                if min_value is None or min_value < 0:
                    continue  # could not be replaced with Relu

                if para_type == 1:
                    # This Clip node's min and max come from initializers
                    for init in onnx_model.model.graph.initializer:
                        if len(node.input) > 1 and init.name == node.input[1]:
                            init_to_remove.append(init)
                        if len(node.input) > 2 and init.name == node.input[2]:
                            init_to_remove.append(init)

                elif para_type == 2:
                    # This Clip node's min and max come from other nodes
                    for nd in onnx_model.model.graph.node:
                        if ((len(node.input) > 1 and node.input[1] in nd.output)
                                or (len(node.input) > 2 and node.input[2] in nd.output)) is False:
                            continue

                        if nd.op_type == 'Identity':
                            for init in onnx_model.model.graph.initializer:
                                if len(nd.input) > 1 and init.name == nd.input[1]:
                                    init_to_remove.append(init)
                                if len(nd.input) > 2 and init.name == nd.input[2]:
                                    init_to_remove.append(init)
                            nodes_to_remove.append(nd)

                        elif nd.op_type == 'Constant':
                            nodes_to_remove.append(nd)

                logger.info(f"Convert Clip node {node.name} to Relu, "
                            f"its min is {min_value}, max is {max_value} and type is {para_type}")
                relu_node = onnx.helper.make_node("Relu", [node.input[0]], node.output, node.name)
                onnx_model.model.graph.node.extend([relu_node])  # insert a Relu node
                nodes_to_remove.append(node)  # to remove this Clip node

        onnx_model.remove_nodes(nodes_to_remove)
        onnx_model.remove_initializers(init_to_remove)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        self.model = onnx_model.model

    def fold_batch_norm_after_concat(self) -> None:
        """
        fold BatchNormalization (after concat) to target operations
        """

        def _get_folded_weight_bias(target_type: str, target_weight: NDArray[np.float32],
                                    target_bias: Union[NDArray[np.float32],
                                                       NDArray[np.float64]], bn_gamma: Union[NDArray[np.float32], None],
                                    bn_beta: Union[NDArray[np.float32], None], bn_mean: NDArray[np.float32],
                                    bn_var: NDArray[np.float32], bn_epsilon: float, start: int,
                                    end: int) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
            if bn_gamma is not None:
                multiplier = bn_gamma[start:end] / np.sqrt(bn_var[start:end] + bn_epsilon)
            else:
                multiplier = 1 / np.sqrt(bn_var[start:end] + bn_epsilon)

            if target_type == "Gemm":
                bn_weight = np.diag(multiplier)
            elif target_type == "ConvTranspose":
                bn_weight = multiplier.reshape(1, len(multiplier), 1, 1)
            elif target_type == "Conv":
                bn_weight = multiplier.reshape(len(multiplier), 1, 1, 1)

            if bn_beta is not None:
                bn_bias = bn_beta[start:end] + (-bn_mean[start:end]) * multiplier
            else:
                bn_bias = (-bn_mean[start:end]) * multiplier

            if target_type == "Gemm":
                folded_weight = np.dot(bn_weight, target_weight)
                folded_bias = np.dot(bn_weight, target_bias) + bn_bias
            elif target_type == "ConvTranspose":
                folded_weight = bn_weight * target_weight
                folded_bias = bn_weight.reshape(1, -1) * target_bias + bn_bias
                folded_bias = folded_bias.reshape(-1)
            elif target_type == "Conv":
                folded_weight = bn_weight * target_weight
                folded_bias = bn_weight.reshape(-1) * target_bias + bn_bias

            return folded_weight, folded_bias

        onnx_model = ONNXModel(self.model)

        TARGET_OPS = ('ConvTranspose', 'Gemm', 'Conv')

        remove_nodes = []

        for node in onnx_model.model.graph.node:

            if node.op_type != 'BatchNormalization' or self.should_quantize_node(node):
                continue

            if len(node.input) != 5:
                logger.warning(f"BatchNorm {node.name} with {len(node.input)} inputs cannot be folded.")
                continue

            # find potential target nodes
            parent_node = onnx_model.get_parent(node, 0)
            if parent_node is None:
                logger.warning(f"BatchNorm {node.name} that is isolated node cannot be folded.")
                continue

            if parent_node.op_type == 'Concat':
                grandparent_nodes = onnx_model.get_parents(parent_node)
            else:
                continue

            # check if all target nodes satisfy the requirements to be folded
            is_foldable = True
            for target_node in grandparent_nodes:

                target_type = target_node.op_type
                if target_type not in TARGET_OPS:
                    logger.debug(
                        f"Not all parent nodes of Concat are in ['ConvTranspose', 'Gemm', 'Conv'], so BatchNorm {node.name} after Concat node cannot be folded."
                    )
                    is_foldable = False
                    break
                if target_type == "Gemm":
                    transB = next((attr.i for attr in target_node.attribute if attr.name == 'transB'), 0)
                    # TODO: Support transB is 0
                    if transB == 0:
                        logger.debug(f"Target node f{target_node.name}'s transB=0 is not supported.")
                        is_foldable = False
                        break
                if target_type == "ConvTranspose":
                    group = next((attr.i for attr in target_node.attribute if attr.name == 'group'), 1)
                    # TODO: Support ConvTranspose group != 1
                    if group != 1:
                        logger.debug(f"Target node f{target_node.name}'s group !=1 is not supported.")
                        is_foldable = False
                        break

                target_weight_init = onnx_model.get_initializer(target_node.input[1])
                target_weight = None if target_weight_init is None else onnx.numpy_helper.to_array(target_weight_init)
                if target_weight is None:
                    logger.warning(f"BatchNorm {node.name}'s target node f{target_node.name} is not foldable.")
                    is_foldable = False
                    break

            if is_foldable is False:
                continue

            bn_gamma_init = onnx_model.get_initializer(node.input[1])
            bn_gamma = None if bn_gamma_init is None else onnx.numpy_helper.to_array(bn_gamma_init)
            bn_beta_init = onnx_model.get_initializer(node.input[2])
            bn_beta = None if bn_beta_init is None else onnx.numpy_helper.to_array(bn_beta_init)
            bn_mean_init = onnx_model.get_initializer(node.input[3])
            bn_mean = None if bn_mean_init is None else onnx.numpy_helper.to_array(bn_mean_init)
            bn_var_init = onnx_model.get_initializer(node.input[4])
            bn_var = None if bn_var_init is None else onnx.numpy_helper.to_array(bn_var_init)
            bn_epsilon = next((attr.f for attr in node.attribute if attr.name == 'epsilon'), 1e-10)

            if bn_mean is None or bn_var is None:
                logger.warning(f"BatchNorm {node.name} that is missing mean or variance cannot be folded.")
                continue

            # fold batchnorm to target nodes
            start_idx, end_idx = 0, 0
            for i in range(len(grandparent_nodes)):
                target_node = grandparent_nodes[i]

                target_weight_init = onnx_model.get_initializer(target_node.input[1])
                target_weight = onnx.numpy_helper.to_array(target_weight_init)

                target_bias_init = onnx_model.get_initializer(target_node.input[2]) if len(
                    target_node.input) > 2 else None
                target_bias = None if target_bias_init is None else onnx.numpy_helper.to_array(target_bias_init)

                if target_bias is None:
                    if target_type == "Conv":
                        target_bias = np.zeros(target_weight.shape[0])
                    elif target_type == "Gemm":
                        target_bias = np.zeros(target_weight.shape[0])
                    else:  # if target_type == "ConvTranspose":
                        target_bias = np.zeros(target_weight.shape[1])

                    target_bias_name = target_node.name + "_bias_4bn"
                    target_bias_init = onnx.numpy_helper.from_array(target_bias.astype(np.float32),
                                                                    name=target_bias_name)
                    onnx_model.add_initializer(target_bias_init)
                    target_node.input.append(target_bias_name)

                end_idx += target_bias.shape[0]

                # Calculate the weight and bias after folded
                folded_weight, folded_bias = _get_folded_weight_bias(target_type, target_weight, target_bias, bn_gamma,
                                                                     bn_beta, bn_mean, bn_var, bn_epsilon, start_idx,
                                                                     end_idx)

                start_idx += target_bias.shape[0]

                # Update target node's weight and bias
                folded_weight_init = onnx.numpy_helper.from_array(folded_weight.astype(np.float32),
                                                                  name=target_weight_init.name)
                target_weight_init.CopyFrom(folded_weight_init)
                folded_bias_init = onnx.numpy_helper.from_array(folded_bias.astype(np.float32),
                                                                name=target_bias_init.name)
                target_bias_init = onnx_model.get_initializer(target_node.input[2])
                target_bias_init.CopyFrom(folded_bias_init)

            # Deal with the tensor name
            children = onnx_model.get_children(parent_node)

            for child in children:
                if child is node:  # this node will be removed
                    continue

                for input_index, input_name in enumerate(child.input):
                    if input_name == parent_node.output[0]:
                        child.input[input_index] = node.output[0]

            parent_node.output[0] = node.output[0]

            # TODO: has shared initializers?
            remove_nodes.append(node)

            logger.info(f"Folded {node.op_type} {node.name} to {target_node.op_type} {target_node.name}.")

        onnx_model.remove_nodes(remove_nodes)
        onnx_model.clean_initializers()
        onnx_model.topological_sort()
        self.model = onnx_model.model

    def dedicate_dq_node(self) -> None:
        onnx_model = ONNXModel(self.model)
        output_name_to_node = onnx_model.output_name_to_node()
        input_name_to_nodes = onnx_model.input_name_to_nodes()

        nodes_to_add = []

        for node in onnx_model.model.graph.node:
            if node.op_type not in DEQUANT_OP_TYPES:
                continue

            # Deal with the implicit condition of multiple consumers
            consumers = []
            if onnx_model.is_graph_output(node.output[0]):
                consumers = [node]  # Just to occupy the position

            if node.output[0] not in input_name_to_nodes:
                continue
            children = input_name_to_nodes[node.output[0]]
            if len(children) + len(consumers) < 2:
                continue

            consumers = consumers + children

            if node.input[0] not in output_name_to_node:
                continue
            parent = output_name_to_node[node.input[0]]
            if parent.op_type not in QUANT_OP_TYPES:
                continue

            # If this is a shared weight, copy Q as well
            if onnx_model.get_initializer(parent.input[0]) is not None:
                copy_q = True
            else:
                copy_q = False

            for index, consumer in enumerate(consumers):
                if index == 0:
                    continue

                postfix = f"_{index}"

                if copy_q:  # Copy a new QuantizedLinear node
                    parent_new = copy.deepcopy(parent)
                    parent_new.name = parent_new.name + postfix
                    parent_new.output[0] = parent_new.output[0] + postfix
                    nodes_to_add.append(parent_new)

                    output_info = next(
                        (info for info in onnx_model.model.graph.value_info if info.name == parent.output[0]), None)
                    output_info_new = next(
                        (info for info in onnx_model.model.graph.value_info if info.name == parent_new.output[0]), None)
                    if output_info is not None and output_info_new is None:
                        output_info_new = copy.deepcopy(output_info)
                        output_info_new.name = parent_new.output[0]
                        onnx_model.model.graph.value_info.extend([output_info_new])

                # Copy a new DequantizeLinear node
                node_new = copy.deepcopy(node)
                node_new.name = node.name + postfix
                node_new.output[0] = node.output[0] + postfix
                if copy_q:  # Should Connect with the new q
                    node_new.input[0] = parent_new.output[0]
                nodes_to_add.append(node_new)

                # Copy shape info
                output_info = next((info for info in onnx_model.model.graph.value_info if info.name == node.output[0]),
                                   None)
                output_info_new = next(
                    (info for info in onnx_model.model.graph.value_info if info.name == node_new.output[0]), None)
                if output_info is not None and output_info_new is None:
                    output_info_new = copy.deepcopy(output_info)
                    output_info_new.name = node_new.output[0]
                    onnx_model.model.graph.value_info.extend([output_info_new])

                onnx_model.replace_node_input(consumer, node.output[0], node_new.output[0])

        if len(nodes_to_add):
            logger.info(f"Dedicate {len(nodes_to_add)} DQs in post-processing.")
            onnx_model.add_nodes(nodes_to_add)
            onnx_model.topological_sort()
            self.model = onnx_model.model


def optimize(model: ModelProto,
             op_types_to_quantize: List[str],
             nodes_to_quantize: Optional[List[str]],
             nodes_to_exclude: Optional[List[str]],
             convert_bn_to_conv: bool = True,
             convert_reduce_mean_to_global_avg_pool: bool = True,
             split_large_kernel_pool: bool = True,
             convert_split_to_slice: bool = True,
             fuse_instance_norm: bool = True,
             fuse_l2_norm: bool = True,
             fuse_gelu: bool = True,
             fuse_layer_norm: bool = True,
             fold_batch_norm: bool = True,
             convert_clip_to_relu: bool = True,
             fold_batch_norm_after_concat: bool = True,
             dedicate_dq_node: bool = False) -> ModelProto:
    """
    Optimize an ONNX model to meet specific constraints and requirements for deployment on an CPU/NPU.

    This function applies various optimization techniques to the provided ONNX model based on the specified parameters. The optimizations include fusing operations, converting specific layers, and folding batch normalization layers, among others.

    :param model: The ONNX model to be optimized.
    :type model: ModelProto
    :param op_types_to_quantize: List of operation types to be quantized.
    :type op_types_to_quantize: List[str]
    :param nodes_to_quantize: List of node names to explicitly quantize. If `None`, quantization is applied based on the operation types.
    :type nodes_to_quantize: Optional[List[str]]
    :param nodes_to_exclude: List of node names to exclude from quantization.
    :type nodes_to_exclude: Optional[List[str]]
    :param convert_bn_to_conv: Flag indicating whether to convert BatchNorm layers to Conv layers.
    :type convert_bn_to_conv: bool
    :param convert_reduce_mean_to_global_avg_pool: Flag indicating whether to convert ReduceMean layers to GlobalAveragePool layers.
    :type convert_reduce_mean_to_global_avg_pool: bool
    :param split_large_kernel_pool: Flag indicating whether to split large kernel pooling operations.
    :type split_large_kernel_pool: bool
    :param convert_split_to_slice: Flag indicating whether to convert Split layers to Slice layers.
    :type convert_split_to_slice: bool
    :param fuse_instance_norm: Flag indicating whether to fuse InstanceNorm layers.
    :type fuse_instance_norm: bool
    :param fuse_l2_norm: Flag indicating whether to fuse L2Norm layers.
    :type fuse_l2_norm: bool
    :param fuse_gelu: Flag indicating whether to fuse Gelu layers.
    :type fuse_gelu: bool
    :param fuse_layer_norm: Flag indicating whether to fuse LayerNorm layers.
    :type fuse_layer_norm: bool
    :param fold_batch_norm: Flag indicating whether to fold BatchNorm layers into preceding Conv layers.
    :type fold_batch_norm: bool
    :param convert_clip_to_relu: Flag indicating whether to convert Clip layers to ReLU layers.
    :type convert_clip_to_relu: bool
    :param fold_batch_norm_after_concat: Flag indicating whether to fold BatchNorm layers after concatenation operations.
    :type fold_batch_norm_after_concat: bool

    :return: The optimized ONNX model.
    :rtype: ModelProto

    Notes:
        - The `Optimize` class is used to apply the optimizations based on the provided flags.
        - The function returns the optimized model with the applied transformations.
    """
    onnx_model = OnnxModel(model)
    opset_version = get_opset_version(onnx_model.model)

    optimizer = Optimize(
        model,
        op_types_to_quantize,
        nodes_to_quantize,
        nodes_to_exclude,
    )

    if fuse_instance_norm:
        optimizer.fuse_instance_norm()

    if convert_reduce_mean_to_global_avg_pool:
        optimizer.convert_reduce_mean_to_global_avg_pool()

    if split_large_kernel_pool:
        optimizer.split_large_kernel_pool()

    if convert_split_to_slice:
        optimizer.convert_split_to_slice()

    if fuse_l2_norm:
        optimizer.fuse_l2_norm()

    if fuse_layer_norm:
        if opset_version < 17:
            logger.warning(f"The opset version is {opset_version} < 17. Skipping fusing layer normalization.")
        else:
            fusion_layernorm = FusionLayerNormalization(onnx_model)
            fusion_layernorm.apply()

    if fuse_gelu:
        if opset_version < 20:
            logger.warning(f"The opset version is {opset_version} < 20. Skipping fusing Gelu.")
        else:
            fusion_gelu = FusionGelu(onnx_model)
            fusion_gelu.apply()

    if fold_batch_norm:
        optimizer.fold_batch_norm()

    if convert_clip_to_relu:
        optimizer.convert_clip_to_relu()

    if fold_batch_norm_after_concat:
        optimizer.fold_batch_norm_after_concat()

    if convert_bn_to_conv:
        optimizer.convert_bn_to_conv()

    # Only for quantization post-processing
    if dedicate_dq_node:
        optimizer.dedicate_dq_node()

    return optimizer.model
