#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
Convert ONNX ModelProto with shared initializer to be unique initializer.

:param model: ONNX ModelProto object
:param op_types: The op_type need to copy the shared initializer
:return: converted ONNX ModelProto object

Examples:

::

    Example 1: Convert ONNX ModelProto object:
    from quark.onnx.tools import convert_shared_initializer_to_unique
    new_onnx_model = convert_shared_initializer_to_unique.convert(onnx_model)


Use the convert_shared_initializer_to_unique.py to duplicate reused initializer in onnx model,
so that there do not exist nodes to share initializer

```
python convert_shared_initializer_to_unique.py --input $ONNX_MODEL_PATH_WITH_INIT_SHARED --output $ONNX_MODEL_PATH_WITHOUT_INIT_SHARED --op_types ["Cnv", "Gemm"]
```

If need to duplicate all op_types in the given onnx model, the op_types could include all op_types or keep None.

```
python convert_shared_initializer_to_unique.py --input $ONNX_MODEL_PATH_WITH_INIT_SHARED --output $ONNX_MODEL_PATH_WITHOUT_INIT_SHARED
```


The conversion from reused initializer to that one without initializer shared
for given node op_types e.g. ["Conv", "Gemm"]. Empty list [] will include all
op_types in the given onnx model, default is [].
It is recommended to do conversion to satisfy the compilation need and model
quantization FastFinetune need.
'''

import onnx
import copy
from argparse import ArgumentParser, Namespace
from typing import Dict, Any


def parse_args() -> Namespace:
    parser = ArgumentParser("SharedInitializerConverter")
    parser.add_argument("--input", type=str)
    parser.add_argument("--op_types", type=list[str], default=[])
    parser.add_argument("--output", type=str)
    args, _ = parser.parse_known_args()
    return args


def convert(onnx_model: onnx.ModelProto,
            support_op_types: list[str] = [],
            prefix: str = "duplicated") -> onnx.ModelProto:

    if support_op_types == []:
        support_op_types = []
        for node_idx in range(len(onnx_model.graph.node)):
            node_op_type = onnx_model.graph.node[node_idx].op_type
            if node_op_type not in support_op_types:
                support_op_types.append(node_op_type)

    all_initializer_names = [item.name for item in onnx_model.graph.initializer]
    all_initializer_dict = {}
    for ini_item in onnx_model.graph.initializer:
        all_initializer_dict[ini_item.name] = ini_item

    ini_used_static: Dict[str, Any] = {}

    for i in range(len(onnx_model.graph.node)):
        if onnx_model.graph.node[i].op_type in support_op_types:
            # get all input for one node
            inputs_name = onnx_model.graph.node[i].input
            for idx, input_name in enumerate(inputs_name):
                # get the initializer from the input
                if input_name in all_initializer_names:
                    # copy initializer for shared initializer or pass
                    if input_name in list(ini_used_static.keys()):
                        ini_used_static[input_name] += 1
                        new_ini = copy.deepcopy(all_initializer_dict[input_name])
                        new_ini.name = prefix + new_ini.name + str(ini_used_static[input_name])
                        onnx_model.graph.node[i].input[idx] = new_ini.name
                        onnx_model.graph.initializer.append(new_ini)
                    else:
                        ini_used_static[input_name] = 1
    return onnx_model


if __name__ == '__main__':
    args = parse_args()
    onnx_model = onnx.load(args.input)
    if args.op_types is None:
        converted_model = convert(onnx_model)
    else:
        converted_model = convert(onnx_model, args.op_types)
    onnx.save(converted_model, args.output)
