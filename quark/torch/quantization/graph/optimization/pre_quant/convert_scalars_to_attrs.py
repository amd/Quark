#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from quark.torch.quantization.graph.torch_utils import is_math_arithmetic_node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    '''
    Convert constant number to tensor
    e.g.

    before:
        torch.ops.aten.mul.Tensor(unsqueeze, 1.0)
    after:
        _tensor_constant0 = self._tensor_constant0 # self._tensor_constant0 is a tensor
        torch.ops.aten.mul.Tensor(unsqueeze, _tensor_constant0)
    NOTE:
        In some case, like model samvit_base_patch16_224(TIMM)(VisionTransformerSAM)
        e.g The model in GPU, but some operations/Tensors in CPU
        In this case, we will skip convert if one operation's Tensor device diff with model.
        ref: torch/ao/quantization/quantizer/xnnpack_quantizer_utils.py: _convert_scalars_to_attrs
    '''
    model_device = [module for module in model.parameters()][0].device  # cpu/gpu
    for n in model.graph.nodes:
        if not is_math_arithmetic_node(n):
            continue

        args = list(n.args)

        # NOTE in some case
        # model in GPU, but some operations/Tensor in CPU
        nodes = list(filter(lambda n: isinstance(n, torch.fx.Node) and ('val' in n.meta), args))
        tensor_device = [n.meta['val'].device for n in nodes]
        if len(set(tensor_device)) >= 2 or (len(tensor_device) >= 1 and tensor_device[0] != model_device):
            logger.warning(
                "In Node: {}'s args, contaion multi/diff (with model) devices:{}, skip convert to attrs".format(
                    n.name, tensor_device))
            continue

        new_args = []
        for i in range(len(args)):
            if isinstance(args[i], torch.fx.Node):
                new_args.append(args[i])
                continue
            prefix = "_tensor_constant_"
            get_new_attr_name = get_new_attr_name_with_prefix(prefix)
            tensor_constant_name = get_new_attr_name(model)
            attr_tensor = torch.tensor(args[i]).to(model_device)
            model.register_buffer(tensor_constant_name, attr_tensor)
            fake_mode = n.meta["val"].fake_mode
            with model.graph.inserting_before(n):
                get_attr_node = model.graph.create_node("get_attr", tensor_constant_name, (), {})
                get_attr_node.meta["val"] = fake_mode.from_tensor(attr_tensor, static_shapes=True)
                # NOTE note the skip info for the node, Default set to not to skip quant
                get_attr_node.meta['skip_quant'] = n.meta['skip_quant'] if 'skip_quant' in n.meta else False
                new_args.append(get_attr_node)
            logger.info("Node: {}'s {}_th args, convert scalar: {} to Tensor (type: {}) and save in attr Node".format(
                n.name, i, args[i], attr_tensor.dtype))
        n.args = tuple(new_args)
    model.graph.eliminate_dead_code()
    model.recompile()
    return model
