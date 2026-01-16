# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass fuses Cast + Mul operations into a single CastMulAvx custom operator.

Pattern matched:
    Cast -> Mul  or  Mul with Cast on one input

The CastMulAvx operator performs cast and multiplication in a single fused operation,
leveraging AVX-512 instructions for optimal performance.

Supported conversions:
    - Float32 ↔ BFloat16
    - Float32 ↔ Float16
    - Float16 ↔ BFloat16
    - Float32 → Float32 (multiplication only)
"""

import logging

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs

_logger = logging.getLogger(__name__)


def get_cast_to_attribute(cast_node: onnx.NodeProto) -> int:
    """Extract the 'to' attribute from a Cast node."""
    for attr in cast_node.attribute:
        if attr.name == "to":
            return attr.i
    raise ValueError(f"Cast node {cast_node.name} missing 'to' attribute")


def is_supported_cast_type(from_type: int, to_type: int) -> bool:
    """Check if the cast conversion is supported by CastMulAvx."""
    FLOAT = onnx.TensorProto.FLOAT
    FLOAT16 = onnx.TensorProto.FLOAT16
    BFLOAT16 = onnx.TensorProto.BFLOAT16

    supported_pairs = {
        (FLOAT, BFLOAT16),
        (FLOAT, FLOAT16),
        (BFLOAT16, FLOAT),
        (BFLOAT16, FLOAT16),
        (FLOAT16, FLOAT),
        (FLOAT16, BFLOAT16),
    }

    return (from_type, to_type) in supported_pairs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    """
    Replace Cast -> Mul sequence with CastMulAvx.

    Pattern: Cast -> Mul
    """
    domain = params.get_domain("CastMulAvx")
    cast_node = subgraph[0]
    mul_node = subgraph[1]

    # Get Cast attributes
    cast_to_type = ryzenai_onnx_utils.matcher.get_attribute(cast_node, "to")
    cast_input = cast_node.input[0]
    cast_output = cast_node.output[0]

    # Get input data type
    input_dtype = ryzenai_onnx_utils.matcher.get_dtype(cast_input, extractor)

    # Check if this cast is supported
    if not is_supported_cast_type(input_dtype, cast_to_type):
        _logger.debug(f"Skipping Cast+Mul fusion: unsupported cast from {input_dtype} to {cast_to_type}")
        return subgraph, [], None

    # Check if cast output is only used by mul (no other consumers)
    if ryzenai_onnx_utils.matcher.has_multiple_successors(cast_output, extractor.graph):
        _logger.debug(f"Skipping Cast+Mul fusion: cast output {cast_output} has multiple consumers")
        return subgraph, [], None

    # Determine which input of Mul comes from Cast
    if mul_node.input[0] == cast_output:
        # Cast is first input: CastMulAvx(cast_input, mul_input[1])
        castmul_inputs = [cast_input, mul_node.input[1]]
    # elif mul_node.input[1] == cast_output:
    #     # Cast is second input: CastMulAvx(cast_input, mul_input[0])
    #     castmul_inputs = [cast_input, mul_node.input[0]]
    else:
        _logger.warning(f"Cast output {cast_output} not found in Mul inputs: {mul_node.input}")
        return subgraph, [], None

    # Create CastMulAvx node
    castmul_node = onnx.helper.make_node(
        "CastMulAvx",
        inputs=castmul_inputs,
        outputs=mul_node.output,
        domain=domain,
        name=f"{mul_node.name}_castmul_avx",
        to=cast_to_type,
    )

    _logger.debug(f"Fused Cast({input_dtype}->{cast_to_type}) + Mul into CastMulAvx: {castmul_node.name}")

    return [castmul_node], [], None


# Pattern 1: Cast followed by Mul
PATTERN = ["CastAvx([?],c0)", "Mul([c0,?],?)"]


REPLACEMENT = replacement
