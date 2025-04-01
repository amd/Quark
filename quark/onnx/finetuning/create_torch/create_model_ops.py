#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import numpy as np
import torch
from torch import nn
import onnx

from .create_model_utils import (ComputeOperations, NormalizationOperations, ActivationMapping, QuantizeLinearOps,
                                 DequantizeLinearOps, FixNeuronOps, ONNXModelParser, extract_attr_values)
from .quant_base_ops import QuantizationModule, QuantizeWrapper
from .quant_conv_ops import (QConv1d, QConv2d, QConv3d, QConvTranspose1d, QConvTranspose2d, QConvTranspose3d)
from .quant_gemm_ops import QGemm
from .quant_matmul_ops import QMatMul
from .quant_norm_ops import QInstanceNorm2d, QLayerNorm
from typing import Any, Tuple, Dict, List, Union, Optional
from numpy.typing import NDArray
from quark.shares.utils.log import ScreenLogger, log_errors

logger = ScreenLogger(__name__)


def param_is_symmetric(params: List[Any]) -> bool:
    """
    Check if parameters are symmetric, all values [2,2,2,2].
    Then we can use only [2,2].
    """
    assert len(params) // 2 == len(params) / 2, "Non even number of parameters."
    idx = len(params) // 2
    for i in range(0, idx):
        if params[i] != params[idx + i]:
            return False
    return True


def extract_padding_params(params: List[Any]) -> Any:
    """Extract padding parameters for Pad layers."""
    pad_dim = len(params) // 2
    if pad_dim == 0:
        return []

    # Let it to be a (2 rows * pad_dim cols) array
    pads = np.array(params).reshape(-1, pad_dim)
    # Flip last two cols (swap H and W axis)
    if pad_dim > 1:
        pads[:, [-2, -1]] = pads[:, [-1, -2]]
    # Transpose and flatten
    pads = pads.T.flatten()

    # Some padding modes do not support padding in batch and channel dimension.
    # If batch and channel dimension have no padding, discard.
    if pad_dim > 2 and (pads[:4] == 0).all():
        pads = pads[4:]

    pads = pads.tolist()

    return pads


def extract_padding_params_for_conv(params: List[Any]) -> Any:
    """
    Padding params in onnx are different than in pytorch. That is why we need to
    check if they are symmetric and cut half or return a padding layer.
    """
    if param_is_symmetric(params):
        return params[:len(params) // 2]
    else:
        pad_dim = len(params) // 2
        pad_layer = getattr(torch.nn, "ConstantPad{}d".format(pad_dim))
        pads = extract_padding_params(params)
        return pad_layer(pads, value=0)


def extract_weight_and_bias(params: List[Any]) -> Tuple[NDArray[Any], Union[NDArray[Any], None]]:
    """Extract weights and biases."""
    param_length = len(params)
    if param_length == 1:
        weight = params[0]
        bias = None
    elif param_length == 2:
        weight = params[0]
        bias = params[1]
    else:
        raise ValueError("Unexpected number of parameters: {}".format(param_length))
    return weight, bias


def load_weight_and_bias(layer: nn.Module, weight: NDArray[Any], bias: Union[NDArray[Any], None]) -> None:
    """Load weight and bias to a given layer from onnx format."""
    # Even if the weight and bias was quantized to integer, has to be kept in float32
    # in order to be updated by the optimizer
    layer.weight.data = torch.tensor(weight, dtype=torch.float)
    if bias is not None:
        layer.bias.data = torch.tensor(bias, dtype=torch.float)


def convert_conv(node: onnx.NodeProto, layer_params: List[Any],
                 layer_qinfos: List[Any]) -> Tuple[QuantizeWrapper, Union[QuantizeWrapper, None]]:
    """Use to convert Conv ONNX node to Torch module (or called layer).
       This function supports onnx's Conv and ConvTranspose from 1 to 11.

    :param node : ONNX node.
    :param layer_params : Layer weight and bias parameters.
    :param layer_qinfos : Layer quantization information.
    :return: Converted conv layer, perhaps it has a pad layer.
    """

    def _extract_attributes(node: onnx.NodeProto) -> Dict[str, Any]:
        kwargs = {}

        for attr in node.attribute:
            if attr.name == "dilations":
                kwargs["dilation"] = extract_attr_values(attr)
            elif attr.name == "group":
                kwargs["groups"] = extract_attr_values(attr)
            elif attr.name == "kernel_shape":
                kwargs["kernel_size"] = extract_attr_values(attr)
            elif attr.name == "pads":
                kwargs["padding"] = extract_padding_params_for_conv(extract_attr_values(attr))
            elif attr.name == "strides":
                kwargs["stride"] = extract_attr_values(attr)
            elif attr.name == "auto_pad":
                value = extract_attr_values(attr)
                if value == "NOTSET":
                    pass
                else:
                    # This feature is not implemented yet
                    raise NotImplementedError("auto_pad={} functionality not implemented.".format(value))

            # This two attributes are for ConvTranspose
            elif attr.name == "output_shape" and node.op_type == "ConvTranspose":
                raise NotImplementedError("ConvTranspose with output shape not implemented.")
            elif attr.name == "output_padding" and node.op_type == "ConvTranspose":
                raise NotImplementedError("ConvTranspose with dynamic padding not implemented.")

        return kwargs

    assert node.op_type in [
        "Conv",
        "ConvTranspose",
    ], "Incorrect layer type: {}".format(node.op_type)

    kwargs = _extract_attributes(node)
    kernel_size_length = len(kwargs["kernel_size"])
    layer: Union[QuantizeWrapper, type[QuantizeWrapper]] = QConv2d
    if kernel_size_length == 1:
        layer = QConv1d if node.op_type == 'Conv' else QConvTranspose1d
    elif kernel_size_length == 2:
        layer = QConv2d if node.op_type == 'Conv' else QConvTranspose2d
    elif kernel_size_length == 3:
        layer = QConv3d if node.op_type == 'Conv' else QConvTranspose3d
    else:
        raise ValueError("Unexpected length of kernel_size dimension: {}".format(kernel_size_length))

    weight, bias = extract_weight_and_bias(layer_params)
    kwargs["bias"] = bias is not None
    kwargs["in_channels"] = weight.shape[1] * kwargs.get("groups", 1)
    kwargs["out_channels"] = weight.shape[0]

    if node.op_type == "ConvTranspose":
        kwargs["in_channels"], kwargs["out_channels"] = (
            kwargs["out_channels"],
            kwargs["in_channels"],
        )

    # If padding is a layer, remove from kwargs and prepend later
    pad_layer = None
    if "padding" in kwargs and isinstance(kwargs["padding"], nn.Module):
        pad_layer = kwargs.pop("padding")

    # Initialize layer and load weights
    layer = layer(**kwargs)

    # Create layer's quantizer
    input_quant_info = layer_qinfos[0]
    if input_quant_info is not None:
        layer.create_input_quantizer(input_quant_info)

    weight_quant_info = layer_qinfos[1]
    if weight_quant_info is not None:
        layer.create_weight_quantizer(weight_quant_info)
        assert (layer.weight_quantizer is not None), "Layer Weight Quantizer is None"
        if layer.weight_quantizer.q_folded:  # Restore weight
            weight = layer.weight_quantizer.dequantize(torch.tensor(weight)).numpy()

    if len(layer_qinfos) > 2:
        bias_quant_info = layer_qinfos[2]
        if bias_quant_info is not None:
            layer.create_bias_quantizer(bias_quant_info)
            assert (layer.bias_quantizer is not None), "Layer Bias Quantizer is None"
            if layer.bias_quantizer.q_folded:  # Restore bias
                bias = layer.bias_quantizer.dequantize(torch.tensor(bias)).numpy()

    load_weight_and_bias(layer, weight, bias)

    return layer, pad_layer


def convert_matmul(node: onnx.NodeProto, layer_params: List[Any], layer_qinfos: List[Any]) -> Tuple[QMatMul, None]:
    """Use to convert MatMul ONNX node to Torch module.

       This function supports onnx's MatMul from 6.
       :param node : ONNX node.
       :param layer_params : Layer weight parameters.
       :param layer_qinfos : Layer quantization informations.
       :return: Converted MatMul layer.
    """

    def _extract_attributes(node: onnx.NodeProto) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        return kwargs

    kwargs = _extract_attributes(node)

    layer: Union[QMatMul, type[QMatMul]] = QMatMul

    weight, bias = extract_weight_and_bias(layer_params)

    # The arguments for torch.nn.Linear
    kwargs["in_features"] = weight.shape[1]
    kwargs["out_features"] = weight.shape[0] if bias is None else bias.shape[0]

    # Initialize layer and load weights
    layer = layer(**kwargs)

    # Create layer's quantizer
    input_quant_info = layer_qinfos[0]
    if input_quant_info is not None:
        layer.create_input_quantizer(input_quant_info)

    weight_quant_info = layer_qinfos[1]
    if weight_quant_info is not None:
        layer.create_weight_quantizer(weight_quant_info)
        assert (layer.weight_quantizer is not None), "Layer Weight Quantizer is None"
        if layer.weight_quantizer.q_folded:  # Restore weight
            weight = layer.weight_quantizer.dequantize(torch.tensor(weight)).numpy()
    bias = None
    load_weight_and_bias(layer, weight, bias)

    return layer, None


def convert_gemm(node: onnx.NodeProto, layer_params: List[Any], layer_qinfos: List[Any]) -> Tuple[QGemm, None]:
    """Use to convert Gemm ONNX node to Torch module.
       This function supports onnx's Instance Norm from 6.

    :param node : ONNX node.
    :param layer_params : Layer weight and bias parameters.
    :param layer_qinfos : Layer quantization information.
    :return: Converted Gemm layer.
    """

    def _extract_attributes(node: onnx.NodeProto) -> Dict[str, Any]:
        kwargs = {}

        for attr in node.attribute:
            if attr.name == "alpha":
                kwargs["w_alpha"] = extract_attr_values(attr)
            elif attr.name == "beta":
                kwargs["b_beta"] = extract_attr_values(attr)
            elif attr.name == "transA":
                kwargs["transA"] = extract_attr_values(attr)
            elif attr.name == "transB":
                kwargs["transB"] = extract_attr_values(attr)

        return kwargs

    kwargs = _extract_attributes(node)

    layer: Union[QGemm, type[QGemm]] = QGemm

    weight, bias = extract_weight_and_bias(layer_params)

    # The arguments for torch.nn.Linear
    if "transB" not in kwargs or kwargs["transB"] == 0:
        kwargs["in_features"] = weight.shape[0]
        kwargs["out_features"] = weight.shape[1] if bias is None else bias.shape[0]
    else:
        kwargs["in_features"] = weight.shape[1]
        kwargs["out_features"] = weight.shape[0] if bias is None else bias.shape[0]

    # Initialize layer and load weights
    layer = layer(**kwargs)

    # Create layer's quantizer
    input_quant_info = layer_qinfos[0]
    if input_quant_info is not None:
        layer.create_input_quantizer(input_quant_info)

    weight_quant_info = layer_qinfos[1]
    if weight_quant_info is not None:
        layer.create_weight_quantizer(weight_quant_info)
        assert (layer.weight_quantizer is not None), "Layer Weight Quantizer is None"
        if layer.weight_quantizer.q_folded:  # Restore weight
            weight = layer.weight_quantizer.dequantize(torch.tensor(weight)).numpy()

    if len(layer_qinfos) > 2:
        bias_quant_info = layer_qinfos[2]
        if bias_quant_info is not None:
            layer.create_bias_quantizer(bias_quant_info)
            assert (layer.bias_quantizer is not None), "Layer Bias Quantizer is None"
            if layer.bias_quantizer.q_folded:  # Restore bias
                bias = layer.bias_quantizer.dequantize(torch.tensor(bias)).numpy()

    load_weight_and_bias(layer, weight, bias)

    return layer, None


def convert_norm(node: onnx.NodeProto, layer_params: List[Any],
                 layer_qinfos: List[Any]) -> Tuple[Union[QInstanceNorm2d, QLayerNorm], None]:
    """Use to convert norm (Instance/Layer Norm) ONNX node to Torch module.
       This function supports onnx's Instance Norm from 6.

    :param node : ONNX node.
    :param layer_params : Layer weight and bias parameters.
    :param layer_qinfos : Layer quantization information.
    :return: Converted norm (Instance/Layer Norm) layer.
    """

    def _extract_attributes(node: onnx.NodeProto) -> Dict[str, Any]:
        kwargs = {}

        for attr in node.attribute:
            if attr.name == "epsilon":
                kwargs["eps"] = extract_attr_values(attr)
            elif attr.name == "axis":
                kwargs["axis"] = extract_attr_values(attr)

        return kwargs

    kwargs = _extract_attributes(node)

    layer: Union[
        QInstanceNorm2d,
        type[QInstanceNorm2d],
        QLayerNorm,
        type[QLayerNorm],
    ] = QInstanceNorm2d  # QInstanceNorm2d should be compatiable with 1d and 3d

    weight, bias = extract_weight_and_bias(layer_params)

    if "InstanceNormalization" in node.op_type:
        # The affine argument enables the initialization of IN's weight and bias
        kwargs["num_features"] = weight.shape[0] if bias is None else bias.shape[0]
        kwargs["affine"] = True

        layer = QInstanceNorm2d
    elif node.op_type == "LayerNormalization":
        # Support normalization at the last dimension
        if "axis" in kwargs and kwargs.pop("axis") != -1:
            raise NotImplementedError("Unsupported LayerNorm {} whose axis is not -1.".format(node.name))
        kwargs["normalized_shape"] = weight.shape[0] if bias is None else bias.shape[0]
        kwargs["elementwise_affine"] = True

        layer = QLayerNorm
    else:
        raise NotImplementedError("Unsupported op type {}.".format(node.op_type))

    # Initialize layer and load weights
    layer = layer(**kwargs)

    # Create layer's quantizer
    input_quant_info = layer_qinfos[0]
    if input_quant_info is not None:
        layer.create_input_quantizer(input_quant_info)

    weight_quant_info = layer_qinfos[1]
    if weight_quant_info is not None:
        layer.create_weight_quantizer(weight_quant_info)
        assert (layer.weight_quantizer is not None), "Layer Weight Quantizer is None"
        if layer.weight_quantizer.q_folded:  # Restore weight
            weight = layer.weight_quantizer.dequantize(torch.tensor(weight)).numpy()

    if len(layer_qinfos) > 2:
        bias_quant_info = layer_qinfos[2]
        if bias_quant_info is not None:
            layer.create_bias_quantizer(bias_quant_info)
            assert (layer.bias_quantizer is not None), "Layer Bias Quantizer is None"
            if layer.bias_quantizer.q_folded:  # Restore bias
                bias = layer.bias_quantizer.dequantize(torch.tensor(bias)).numpy()

    load_weight_and_bias(layer, weight, bias)

    return layer, None


class Clip(nn.Module):

    def __init__(self, min: Optional[float] = None, max: Optional[float] = None) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.min is None or self.max is None:
            return tensor  # TODO: support one of them is None
        else:
            return torch.clamp(tensor, self.min, self.max)


def convert_act(node: onnx.NodeProto) -> Union[nn.Module, None]:
    """Use to convert Activation ONNX node to Torch module (or called layer).

    :param node : ONNX node.
    :return: Converted act layer.
    """

    def _extract_attributes(node: onnx.NodeProto) -> Dict[str, Any]:
        """This function supports LeakyRelu from 1 to 16 and Softmax from 1 to 13"""
        kwargs = {}

        for attr in node.attribute:
            if attr.name == "alpha":
                if node.op_type == "LeakyRelu":
                    kwargs["negative_slope"] = extract_attr_values(attr)
                else:
                    raise NotImplementedError("node {}'s alpha is not implemented.".format(node.name))
            elif attr.name == "axis":
                if node.op_type == "Softmax":
                    kwargs["dim"] = extract_attr_values(attr)
                else:
                    raise NotImplementedError("node {}'s axis is not implemented.".format(node.name))
            elif attr.name == "approximate":
                if node.op_type == "Gelu":
                    kwargs["approximate"] = extract_attr_values(attr)
                else:
                    raise NotImplementedError("node {}'s approximate is not implemented.".format(node.name))
            elif attr.name == "min":
                if node.op_type == "Clip":
                    kwargs["min"] = extract_attr_values(attr)
                else:
                    raise NotImplementedError("node {}'s min is not implemented.".format(node.name))
            elif attr.name == "max":
                if node.op_type == "Clip":
                    kwargs["max"] = extract_attr_values(attr)
                else:
                    raise NotImplementedError("node {}'s min is not implemented.".format(node.name))

        return kwargs

    if node is None:
        logger.debug("No activation for this module")
        return None
    elif node.op_type not in ActivationMapping.keys():
        logger.warning(f"Not supported activation node {node.name} for conversion")
        return None
    else:
        if node.op_type == 'Clip' and len(node.input) == 1:
            return Clip(**_extract_attributes(node))
        elif node.op_type == 'LeakyRelu':
            return nn.LeakyReLU(**_extract_attributes(node), inplace=True)
        elif node.op_type == 'Softmax':
            kwargs = dict(dim=-1)
            kwargs.update(_extract_attributes(node))
            return nn.Softmax(**kwargs)
        else:
            return ActivationMapping[node.op_type]


def convert_output_nodes_to_module(onnx_parser: ONNXModelParser, node: onnx.NodeProto) -> Optional[QuantizationModule]:
    if node is None:
        logger.warning(f"Could not convert output QDQ for {node}")
        return None

    qinfos = onnx_parser.get_outputs_qinfo(node)

    for quant_info in qinfos:
        if quant_info is not None:
            return QuantizationModule(quant_info)

    return None


@log_errors
def convert_ops_to_modules(
    onnx_model: onnx.ModelProto
) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[nn.Module], Optional[QuantizationModule]]:
    """Convert ONNX operations to Torch modules."""

    opset_version = onnx_model.opset_import[0].version
    onnx_parser = ONNXModelParser(onnx_model)
    target_ops = ComputeOperations + NormalizationOperations
    module: Optional[nn.Module] = None
    module_pad: Optional[nn.Module] = None
    for node in onnx_model.graph.node:
        if node.op_type not in target_ops:
            continue

        qinfos = onnx_parser.get_inputs_qinfo(node)
        params = onnx_parser.get_inputs_param(node)
        if node.op_type in NormalizationOperations:
            module, module_pad = convert_norm(node, params, qinfos)
        elif node.op_type == "Gemm":
            module, module_pad = convert_gemm(node, params, qinfos)
        elif node.op_type == "MatMul":
            module, module_pad = convert_matmul(node, params, qinfos)
        else:
            module, module_pad = convert_conv(node, params, qinfos)

        act_node = onnx_parser.get_output_node(node)
        if act_node is None:
            return module, module_pad, None, None

        # Special circumstances 1: Has no act but has output quantization ops
        if act_node.op_type in QuantizeLinearOps + DequantizeLinearOps + FixNeuronOps:
            return module, module_pad, None, convert_output_nodes_to_module(onnx_parser, node)

        module_act = None

        # Special circumstances 2: Min and Max of Clip is inputs not attributes
        if act_node.op_type == "Clip" and len(act_node.input) == 3:
            min_value = None
            max_value = None

            for initializer in onnx_model.graph.initializer:
                if initializer.name == act_node.input[1] and min_value is None:
                    min_value = onnx.numpy_helper.to_array(initializer).item()
                if initializer.name == act_node.input[2] and max_value is None:
                    max_value = onnx.numpy_helper.to_array(initializer).item()

                if min_value is not None and max_value is not None:
                    break

            if min_value is not None and max_value is not None:
                act_node_new = onnx.helper.make_node(act_node.op_type,
                                                     name=act_node.name,
                                                     min=min_value,
                                                     max=max_value,
                                                     inputs=[act_node.input[0]],
                                                     outputs=act_node.output)

                module_act = convert_act(act_node_new)

        if module_act is None:
            module_act = convert_act(act_node)

        return module, module_pad, module_act, convert_output_nodes_to_module(onnx_parser, act_node)

    raise RuntimeError("Not found any compute operations in the onnx model")


def set_modules_original_weight(module: nn.Module, weight: NDArray[Any]) -> None:
    """ For setting original float weight """
    if hasattr(module, 'weight') and module.weight is not None:
        module.weight.data = torch.tensor(weight).to(device=module.weight.device)


def get_modules_optimized_weight(module: nn.Module) -> Any:
    """ For getting optimized quantized weight """
    if hasattr(module, "opt_gained") and module.opt_gained is False:
        return None

    quantizer = module.weight_quantizer
    if quantizer is None:
        logger.warning("Not found quantizer for weight")
        return None

    if hasattr(quantizer, 'alpha'):
        # This is for adaround
        # It requires folding the QuantizeLinear, quantize the weight by adaround
        # and simulate the QuantizeLinear behaviour
        quantizer.use_soft_rounding = False
        return quantizer.quantize(module.weight.cpu().detach()).numpy()
    else:
        # This is for adaquant
        if quantizer.q_folded:
            # If QuantizeLinear was folded, quantize the weight
            return quantizer.quantize(module.weight.cpu().detach()).numpy()
        else:
            # We just export the optimized weight tensor (not recommended)
            return module.weight.cpu().detach().numpy()


def set_modules_original_bias(module: nn.Module, bias: NDArray[Any]) -> None:
    """ For setting original float bias """
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data = torch.tensor(bias).to(device=module.bias.device)


def get_modules_optimized_bias(module: nn.Module) -> Any:
    """ For getting optimized quantized bias """
    if hasattr(module, "opt_gained") and module.opt_gained is False:
        return None

    quantizer = module.bias_quantizer
    if quantizer is None:
        logger.debug("Not found quantizer for bias")
        return None

    # This is for adaquant only
    if quantizer.q_folded:
        # If QuantizeLinear was folded, quantize the bias
        return quantizer.quantize(module.bias.cpu().detach()).numpy()
    else:
        # We just export the optimized bias tensor (not recommended)
        return module.bias.cpu().detach().numpy()
