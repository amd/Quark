#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from torch.fx import GraphModule, Node
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.graph.optimization.utils import replace_ops_module_name_suffix
from quark.torch.quantization.config.config import QuantizationConfig
from quark.torch.quantization.graph.torch_utils import is_linear_node
from quark.torch.quantization.graph.optimization.utils import is_all_nodes_save_parameters
from quark.torch.quantization.graph.processor.processor_utils import _is_skip_quant_node
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def replace_linear_qtlinear(m: GraphModule) -> GraphModule:
    '''
    replace [ops.aten.linear] to QuantLinear
    ops.aten.linear:
        args: (Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
        required: [input, weight]
        optional: [bias=None]
    '''
    count_replace_num = 0  # used for debug and trace
    recognized_but_not_optimized = 0
    quant_module_id_2_name: dict[str, str] = {}
    device = [module for module in m.parameters()][0].device  # cpu/gpu
    need_to_delete_node = []
    for n in m.graph.nodes:
        if not is_linear_node(n):
            continue
        linear_node = n

        weight_node = linear_node.args[1]
        bias_node = linear_node.args[2] if len(linear_node.args) > 2 else None

        skip_quant = True if _is_skip_quant_node(linear_node) else False

        # pre check if linear's weight/bias is not parameters, we skip replace
        need_check_node = [weight_node] if bias_node is None else [weight_node, bias_node]
        if (not all(isinstance(item, Node)
                    for item in need_check_node)) or (not is_all_nodes_save_parameters(m, need_check_node)):
            logger.warning(
                "Not all Nodes: {} save Parameters, skip replace to QuantLinear model".format(need_check_node))
            recognized_but_not_optimized += 1
            continue

        linear_weight = _get_tensor_constant_from_node(weight_node, m)  # type: ignore [no-untyped-call]
        linear_bias = _get_tensor_constant_from_node(
            bias_node, m) if bias_node is not None else None  # type: ignore [no-untyped-call]

        used_param_id = weight_node.target + "_" + bias_node.target if bias_node is not None else weight_node.target
        input_activation_node = linear_node.args[0]
        to_delete_node = [linear_node]
        # Process node need to be deleted
        if used_param_id in quant_module_id_2_name:
            # exist share param
            quant_linear_name = quant_module_id_2_name[used_param_id]
        else:  # instance a QuantLinear
            to_delete_node.append(weight_node)
            if bias_node is not None:
                to_delete_node.append(bias_node)
            in_features = linear_weight.shape[1]
            out_features = linear_weight.shape[0]
            bias = True if bias_node is not None else False
            empty_config = QuantizationConfig()  # Note Set to empty config

            # init convbn
            quantized_linear = QuantLinear(in_features, out_features, device, bias, empty_config).to(device=device)
            quantized_linear.weight.data = linear_weight.data.clone()
            if bias_node is not None:
                assert linear_bias is not None
                quantized_linear.bias.data = linear_bias.data.clone()

            quant_linear_name = linear_node.name + replace_ops_module_name_suffix[QuantLinear]
            setattr(m, quant_linear_name, quantized_linear)
            quant_module_id_2_name[used_param_id] = quant_linear_name
            count_replace_num += 1
        need_to_delete_node += to_delete_node
        with m.graph.inserting_after(input_activation_node):
            quant_linear_node = m.graph.create_node('call_module', quant_linear_name, (input_activation_node, ), {})
            # TODO try to use deepcopy inthe future
            quant_linear_node.meta = linear_node.meta
            quant_linear_node.meta["skip_quant"] = skip_quant
            linear_node.replace_all_uses_with(quant_linear_node)

    logger.info("Totally replace ops.aten.linear to {} count:\t{}, found but skip: {}".format(
        QuantLinear.__name__, count_replace_num, recognized_but_not_optimized))
    [m.graph.erase_node(node) for node in need_to_delete_node]
    m.graph.eliminate_dead_code()
    m.recompile()
    return m
