#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

#!/usr/bin/env python3
"""Step 1: load an ONNX model and print a structural node listing.

For quantized models (detected via Q/DQ op presence) the Q/DQ wrapper nodes
are suppressed in the printed listing so architectural landmarks remain
visible; Q/DQ counts are reported separately.

Usage:
    python3 inspect_model.py <model.onnx>
"""

import sys

import onnx

QDQ_OPS = {
    "QuantizeLinear",
    "DequantizeLinear",
    "ExtendedQuantizeLinear",
    "ExtendedDequantizeLinear",
    "BFPQuantizeDequantize",
    "MXQuantizeDequantize",
}
SKIP_OPS = QDQ_OPS | {"Constant", "ConstantOfShape"}


def inspect(model_path: str) -> None:
    model = onnx.load(model_path)
    g = model.graph
    vi_map = {vi.name: vi for vi in list(g.value_info) + list(g.input) + list(g.output)}
    weight_map = {init.name: list(init.dims) for init in g.initializer}

    qdq_count = sum(1 for n in g.node if n.op_type in QDQ_OPS)
    is_quant = qdq_count > 0

    def shape(name: str):
        if name in vi_map:
            t = vi_map[name].type.tensor_type
            if t.HasField("shape"):
                return [d.dim_value for d in t.shape.dim]
        return None

    print(f"Nodes total : {len(g.node)}  (Q/DQ wrappers: {qdq_count})")
    print(f"Quantized   : {is_quant}")
    print(f"Graph inputs:  {[i.name for i in g.input]}")
    print(f"Graph outputs: {[o.name for o in g.output]}")
    print()
    for i, n in enumerate(g.node):
        if is_quant and n.op_type in SKIP_OPS:
            continue  # hide wrappers; they are captured by BFS automatically
        out = n.output[0] if n.output else "?"
        w = weight_map.get(n.input[1]) if n.op_type in ("Conv", "Gemm", "MatMul") and len(n.input) > 1 else None
        print(f"{i:5d}  {n.op_type:28s}  {n.name:70s}  out={shape(out)}  w={w}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model.onnx>", file=sys.stderr)
        sys.exit(1)
    inspect(sys.argv[1])
