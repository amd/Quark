#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import onnx
import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs

from quark.onnx_adapter.onnx_adapter_pass import PatternConfig, PatternPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class RemoveQDQ(PatternPass):
    """
    This pass removes redundant back-to-back DequantizeLinear and QuantizeLinear
    operators from the graph to simplify it.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        return {}

    def _pattern(self) -> list[PatternConfig]:
        return [PatternConfig("DQ -> Q", ["DequantizeLinear([?, ?, ?], a0)", "QuantizeLinear([a0, ?, ?], ?)"])]

    def _run_for_pattern(
        self, extractor: onnx.utils.Extractor, pass_id: str, subgraph: list[onnx.NodeProto], config: dict[str, Any]
    ) -> PassOutputArgs:
        dequant = subgraph[0]
        quant = subgraph[1]

        multiple_successors = ryzenai_onnx_utils.matcher.has_multiple_successors(dequant.output[0], extractor.graph)
        if not multiple_successors:
            logger.debug(f"Removing simple back-to-back quantization ops {dequant.name} and {quant.name}")
            return [], [], None
        # in this case, we need to keep the first dequant around because its output
        # is going to multiple places but there is also a second redundant quant
        # to some nodes. Find the nodes where this second quant output is going
        # and rewrite it to the first dequant's input
        quant_successors = ryzenai_onnx_utils.matcher.find_nodes_by_input(quant.output[0], extractor.graph)
        for node in quant_successors:
            for index, input_name in enumerate(node.input):
                if input_name == quant.output[0]:
                    node.input[index] = dequant.input[0]
        logger.debug(f"Removing quant with multiple successors from {dequant.name}")
        return [dequant], [], None
