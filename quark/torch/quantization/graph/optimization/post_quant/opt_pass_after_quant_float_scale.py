#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any, Iterable, Tuple, Union

from torch.fx import GraphModule, Node

from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.graph.optimization.opt_pass_manager import OptPassBase
from quark.torch.quantization.graph.optimization.utils import is_quantizer_node
from quark.torch.quantization.graph.torch_utils import (
    ADAPTIVE_AVG_POOL_OP,
    AVG_POOL_2D_OP,
    CAT_OPS,
    MAX_POOL_2D_OP,
    PAD_OP,
    PERMUTE_OP,
    RESHAPE_OP,
    SLICE_OP,
)
from quark.torch.quantization.nn.modules import QuantAdaptiveAvgPool2d, QuantAvgPool2d

logger = ScreenLogger(__name__)
__all__ = [
    "AlignConcatQOPass",
    "AlignPoolQOPass",
    "AlignPadQOPass",
    "AlignSliceQOPass",
    "AlignTransposeQOPass",
    "AlignReshapeQOPass",
]


def _is_has_one_user_and_followed_quantizer(m: GraphModule, n: Node) -> bool:
    if len(n.users) != 1 or (not is_quantizer_node(m, next(iter(n.users)))):
        logger.warning(
            f"node: {n.name} user num != 1 or not followed by quantizer node, Please check whether meet the quant demand."
        )
        return False
    return True


def _make_iterable(obj: Union[Iterable[Any], Any]) -> Iterable[Any]:
    if isinstance(obj, Iterable):
        return obj
    else:
        return [obj]


class AliginScaleOutputToInputBase(OptPassBase):
    def __init__(self) -> None:
        super().__init__()
        self.target_op: Union[list[Any], tuple[Any, ...]] = []

    def get_target_node(self, g: GraphModule, n: Node) -> bool:
        if n.op == "call_function" and (n.target in self.target_op) and _is_has_one_user_and_followed_quantizer(g, n):
            return True
        else:
            return False

    def call(self, m: GraphModule) -> GraphModule:
        for n in m.graph.nodes:
            if not self.get_target_node(m, n):
                continue

            cat_node = n
            # Session 1: get the scale from inputs & outputs quantizer
            output_node = next(iter(cat_node.users))
            output_quantizer = getattr(m, output_node.target)
            if output_quantizer.scale.numel() != 1:
                logger.warning(
                    f"In quantizer Node {output_node.name} scale is not a scale num, Skip apply this strategy"
                )
                continue

            scale_list = [output_quantizer.scale.item()]

            skip_optimize_this_concat = False
            for input_node in _make_iterable(cat_node.args[0]):
                if not is_quantizer_node(m, input_node):
                    skip_optimize_this_concat = True
                    continue

                input_quantizer = getattr(m, input_node.target)
                if input_quantizer.scale.numel() != 1:
                    skip_optimize_this_concat = True
                    continue
                scale_list.append(input_quantizer.scale.item())

            if skip_optimize_this_concat:
                logger.warning(f"Skip apply this strategy for cat node: {cat_node.name}")
                continue

            if len(set(scale_list)) == 1:
                continue

            # Session2: if need change, change quantizer's scale
            for input_node in _make_iterable(cat_node.args[0]):
                input_quantizer = getattr(m, input_node.target)
                if input_quantizer.scale.item() != output_quantizer.scale.item():
                    logger.info(
                        f"Node: {cat_node.name} input quantizer: {input_node.name} scale change from {input_quantizer.scale.item()} to {output_quantizer.scale.item()}"
                    )
                    input_quantizer.scale.fill_(output_quantizer.scale.item())
        return m


class AliginScaleInputToOutputBase(OptPassBase):
    def __init__(self) -> None:
        super().__init__()
        self.target_op: Union[list[Any], tuple[Any, ...]] = []
        self.target_module: tuple[Any, ...] = ()

    def get_target_node(self, g: GraphModule, n: Node) -> bool:
        is_target_op = n.op == "call_function" and n.target in self.target_op
        is_target_module = (
            n.op == "call_module"
            and hasattr(self, "target_module")
            and isinstance(getattr(g, n.target), self.target_module)
        )  # type: ignore[arg-type]
        if (is_target_op or is_target_module) and _is_has_one_user_and_followed_quantizer(g, n):
            return True
        else:
            return False

    def call(self, m: GraphModule) -> GraphModule:
        for n in m.graph.nodes:
            if not self.get_target_node(m, n):
                continue

            cat_node = n
            skip_optimize_this_concat = False
            scale_list = []
            # Session 1: get the scale from inputs & outputs quantizer
            for input_node in _make_iterable(cat_node.args[0]):
                if not is_quantizer_node(m, input_node):
                    skip_optimize_this_concat = True
                    continue
                input_quantizer = getattr(m, input_node.target)
                if input_quantizer.scale.numel() != 1:
                    logger.warning(
                        f"In quantizer Node {input_node.name} scale is not a scale num, Skip apply this strategy"
                    )
                    continue
                scale_list.append(input_quantizer.scale.item())

            output_node = next(iter(cat_node.users))
            if not is_quantizer_node(m, output_node):
                skip_optimize_this_concat = True
                continue
            output_quantizer = getattr(m, output_node.target)
            if output_quantizer.scale.numel() != 1:
                logger.warning(
                    f"Out quantizer Node {output_node.name} scale is not a scale num, Skip apply this strategy"
                )
                continue

            scale_list.append(output_quantizer.scale.item())

            if skip_optimize_this_concat:
                logger.warning(f"Skip apply this strategy for cat node: {cat_node.name}")
                continue

            if len(set(scale_list)) == 1:
                continue

            # Session2: if need change, change quantizer's scale
            input_node = cat_node.args[0]
            input_quantizer = getattr(m, input_node.target)
            if input_quantizer.scale.item() != output_quantizer.scale.item():
                logger.info(
                    f"Node: {cat_node.name} input quantizer: {input_node.name} scale change from {input_quantizer.scale.item()} to {output_quantizer.scale.item()}"
                )
                output_quantizer.scale.fill_(input_quantizer.scale.item())
        return m


class AlignConcatQOPass(AliginScaleOutputToInputBase):
    """
    NOTE: adjust with quark/onnx/refine.py::QuantInfoManager::align_concat
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_op = CAT_OPS


class AlignPoolQOPass(AliginScaleInputToOutputBase):
    """
    NOTE: adjust with quark/onnx/refine.py::QuantInfoManager::align_pool
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_op = [MAX_POOL_2D_OP, AVG_POOL_2D_OP, ADAPTIVE_AVG_POOL_OP]
        self.target_module = (QuantAvgPool2d, QuantAdaptiveAvgPool2d)


class AlignPadQOPass(AliginScaleOutputToInputBase):
    """
    NOTE: adjust with quark/onnx/refine.py::QuantInfoManager::align_pad
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_op = [PAD_OP]


class AlignSliceQOPass(AliginScaleInputToOutputBase):
    """
    NOTE: adjust with quark/onnx/refine.py::QuantInfoManager::align_slice
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_op = [SLICE_OP]


class AlignTransposeQOPass(AliginScaleOutputToInputBase):
    """
    NOTE: adjust with quark/onnx/refine.py::QuantInfoManager::align_transpose
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_op = [PERMUTE_OP]


class AlignReshapeQOPass(AliginScaleOutputToInputBase):
    """
    NOTE: adjust with quark/onnx/refine.py::QuantInfoManager::align_reshape
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_op = [RESHAPE_OP]
