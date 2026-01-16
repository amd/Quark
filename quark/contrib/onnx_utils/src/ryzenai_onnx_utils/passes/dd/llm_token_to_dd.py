# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
import ryzenai_onnx_utils.partitioner
import ryzenai_onnx_utils.pattern
import ryzenai_onnx_utils.pattern_generator as pg
from ryzenai_onnx_utils.transform.dd import build_dd_node, split_dd_node
from ryzenai_onnx_utils.typing import PassOutputArgs

from . import ModelType


def generate_pattern(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    params: ryzenai_onnx_utils.ReplaceParams,
) -> list[list[str]]:
    fuse_qk_mha = params.get_bool_attr("fuse_qk_mha", True)
    lora = params.get_bool_attr("lora", False)
    hints_key = {"MladfMatMul", "FlatMLP", "FlatRMSAdd", "FLATMHA", "MLADFRMSNORM"}
    if lora:
        hints_key = {"MladfMatMul", "FlatRMSAdd", "FLATMHA", "SILU", "ELWMUL", "MLADFRMSNORM"}

    # topological sort before pattern generation to support splitting by layers
    # accurately
    ryzenai_onnx_utils.matcher.graph_topological_sort(extractor, True)

    hints = ryzenai_onnx_utils.partitioner.make_hints(extractor, params, hints_key)
    engine = pg.PartitionGraph(extractor, hints, hints_key)
    engine.partition()
    pattern_ = pg.PatternGenerator(engine)
    patterns = pattern_.get_patterns()
    assert len(patterns) == 1
    # for LLMs, we have a "floating" v_matmul on the front that doesn't match
    # otherwise so insert a parent cast so we can match correctly
    patterns[0].insert(0, "CastAvx(?, b0)")

    matmul_0 = ryzenai_onnx_utils.pattern.Pattern(patterns[0][1])
    matmul_0.inputs[0] = "b0"
    if "MLADFRMSNORM" in patterns[0][1]:
        patterns[0][1] = str(matmul_0)
        return patterns
    patterns[0][1] = str(matmul_0)
    matmul_1 = ryzenai_onnx_utils.pattern.Pattern(patterns[0][2])
    matmul_1.inputs[0] = "b0"
    patterns[0][2] = str(matmul_1)
    # extra k matmul if qk not fused
    if not fuse_qk_mha:
        idx = 3 if not params.get_bool_attr("fuse_qk_mha", True) or params.get_bool_attr("lora", False) else 4
        matmul_2 = ryzenai_onnx_utils.pattern.Pattern(patterns[0][idx])
        matmul_2.inputs[0] = "b0"
        patterns[0][idx] = str(matmul_2)

    return patterns


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    new_nodes = [subgraph[0]]

    split_count = int(params.attributes.get("split_dd_fusion", 1))
    for split, extra_outputs in split_dd_node(subgraph[1:], split_count, extractor):
        dd_node = build_dd_node(
            extractor,
            split,
            params,
            extra_outputs=extra_outputs,
        )

        # must match with ModelType enum for Llm_Token in dynamic_dispatch.hpp
        ryzenai_onnx_utils.matcher.add_attribute(dd_node, "model_type", int(ModelType.LLM_TOKEN))
        if "model_hash" in params.attributes:
            ryzenai_onnx_utils.matcher.add_attribute(dd_node, "model_hash", params.attributes["model_hash"])
        lora = params.get_bool_attr("lora", False)
        if lora:
            ryzenai_onnx_utils.matcher.add_attribute(dd_node, "input_num", len(dd_node.input))
        new_nodes.append(dd_node)

    return new_nodes, [], None


PATTERN = generate_pattern
REPLACEMENT = replacement
