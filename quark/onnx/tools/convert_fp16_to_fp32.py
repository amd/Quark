#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Convert tensor float16 type in the ONNX ModelProto input to tensor float.

:param input: Input ONNX model file path.
:param output: Output ONNX model file path.
:param save_as_external_data: Whether to save the model as external data.
:param subgraphs_to_include: List of (start_node_names, end_node_names) tuples defining subgraphs to convert.
                             Each tuple contains a list of start node names and a list of end node names.
                             Nodes on paths between start and end nodes are included. If start_node_names
                             is empty, all upstream nodes leading to end nodes are included. If end_node_names
                             is empty, all downstream nodes from start nodes are included. Multiple tuples
                             result in the union of all specified subgraphs being converted.
:return: converted ONNX ModelProto object

Examples:

::

Use the convert_float16_to_float.py to convert a float16 model to a float32 model, both floating-point models
and quantized models are supported.

```
python -m quark.onnx.tools.convert_fp16_to_fp32 --input $FLOAT_16_ONNX_MODEL_PATH --output $FLOAT_32_ONNX_MODEL_PATH
```

To convert only specific subgraphs (with boundary Cast nodes inserted automatically):

```
python -m quark.onnx.tools.convert_fp16_to_fp32 --input model.onnx --output model_fp32.onnx \
    --subgraphs_to_include "nodeA,nodeB:nodeZ" ":nodeX" "nodeC:"
```

Each ``--subgraphs_to_include`` argument is a ``start_nodes:end_nodes`` pair where node
names are comma-separated. Either side may be empty for open-ended ranges.
"""

from argparse import ArgumentParser, Namespace

import onnx
from onnxslim import slim

from . import float16


def parse_args() -> Namespace:
    parser = ArgumentParser("float16Converter")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save_as_external_data", action="store_true")
    parser.add_argument(
        "--subgraphs_to_include",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Subgraphs to convert, specified as start:end pairs of comma-separated "
            "node names. Use empty string for open-ended ranges. "
            "E.g. --subgraphs_to_include nodeA,nodeB:nodeZ  :nodeX,nodeY  nodeC:"
        ),
    )
    args, _ = parser.parse_known_args()
    return args


def _parse_subgraphs(raw: list[str]) -> list[tuple[list[str], list[str]]]:
    result: list[tuple[list[str], list[str]]] = []
    for item in raw:
        if ":" not in item:
            raise ValueError(
                f"Invalid subgraph spec '{item}'. Expected 'start_nodes:end_nodes' "
                "(comma-separated names, either side may be empty)."
            )
        start_str, end_str = item.split(":", 1)
        start_nodes = [n.strip() for n in start_str.split(",") if n.strip()]
        end_nodes = [n.strip() for n in end_str.split(",") if n.strip()]
        result.append((start_nodes, end_nodes))
    return result


def convert(args: Namespace) -> None:
    model = onnx.load(args.input)

    subgraphs_to_include = None
    if args.subgraphs_to_include:
        subgraphs_to_include = _parse_subgraphs(args.subgraphs_to_include)

    model_fp32 = float16.convert_float16_to_float(model, subgraphs_to_include=subgraphs_to_include)
    try:
        model_simp = slim(model_fp32)
    except Exception as e:
        print(f"Fail to Simplify ONNX model because of {e}.")
        model_simp = model_fp32

    onnx.save(model_simp, args.output, save_as_external_data=args.save_as_external_data)
    print(f"Convert the float16 model {args.input} to the float32 model {args.output}.")


if __name__ == "__main__":
    args = parse_args()
    convert(args)
