#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
If the model has tensors without a shape, this tool will assign shapes to them.

Use the fix_shapes.py to assign shapes for a model:

```
python fix_shapes.py --input_model_path $INPUT_MODEL_PATH --output_model_path $OUTPUT_MODEL_PATH
```

"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import ModelProto, helper

from quark.common.utils.log import ScreenLogger
from quark.onnx.utils.model_utils import collect_tensor_shapes_from_feed, create_infer_session_for_onnx_model
from quark.onnx.utils.system_utils import create_tmp_dir

logger = ScreenLogger(__name__)


def parse_input_and_output_shapes(fix_shapes: str) -> dict[str, list[int]]:
    shapes_dict = {}
    name_shape_list = [item.strip() for item in fix_shapes.split(";")]
    names = []
    shapes = []
    for name_shape_item in name_shape_list:
        name_shape = [item.strip() for item in name_shape_item.rsplit(":", 1)]
        names.append(name_shape[0])
        if name_shape[1].startswith("[") and name_shape[1].endswith("]"):
            shapes.append([int(dim) for dim in name_shape[1][1:-1].split(",")])
        else:
            logger.info(
                "Has Error: Plase Check the input shape format. like: 'input_1:[1,224,224,3];input_2:[1,96,96,3];output_1:[1, 1000];output_2:[1,10]'"
            )
            exit(-1)
    assert len(names) == len(shapes)
    for i in range(len(names)):
        shapes_dict[names[i]] = shapes[i]
    return shapes_dict


def fix_input_and_output_shapes(model_input: str | Path | ModelProto, fix_shapes: str) -> ModelProto:
    model = model_input if isinstance(model_input, ModelProto) else onnx.load(model_input)
    shapes_dict = parse_input_and_output_shapes(fix_shapes)
    for i in range(len(model.graph.input)):
        name = model.graph.input[i].name
        shapes = shapes_dict[name]
        for j in range(len(shapes)):
            dim_value = shapes[j]
            model.graph.input[i].type.tensor_type.shape.dim[j].dim_value = dim_value
    for i in range(len(model.graph.output)):
        name = model.graph.output[i].name
        shapes = shapes_dict[name]
        for j in range(len(shapes)):
            dim_value = shapes[j]
            model.graph.output[i].type.tensor_type.shape.dim[j].dim_value = dim_value
    return model


def generate_random_data(model_input: str | Path | ModelProto) -> dict[str, np.ndarray[Any, Any]]:
    np.random.seed(42)
    sess = create_infer_session_for_onnx_model(model_input)
    input_info = sess.get_inputs()
    input_data = {}

    for inp in input_info:
        input_name = inp.name
        input_shape = inp.shape
        input_dtype = inp.type

        if input_dtype == "tensor(int8)":
            dtype = np.int8
        elif input_dtype == "tensor(uint8)":
            dtype = np.uint8  # type: ignore
        elif input_dtype == "tensor(int16)":
            dtype = np.int16  # type: ignore
        elif input_dtype == "tensor(uint16)":
            dtype = np.uint16  # type: ignore
        elif input_dtype == "tensor(int32)":
            dtype = np.int32  # type: ignore
        elif input_dtype == "tensor(uint32)":
            dtype = np.uint32  # type: ignore
        elif input_dtype == "tensor(int64)":
            dtype = np.int64  # type: ignore
        elif input_dtype == "tensor(uint64)":
            dtype = np.uint64  # type: ignore
        elif input_dtype == "tensor(float16)":
            dtype = np.float16  # type: ignore
        elif input_dtype == "tensor(float)":
            dtype = np.float32  # type: ignore
        elif input_dtype == "tensor(double)":
            dtype = np.float64  # type: ignore
        elif input_dtype == "tensor(bool)":
            dtype = np.bool_  # type: ignore
        else:
            raise ValueError(f"Unsupported dtype: {input_dtype}")

        random_input_data = np.random.random(input_shape).astype(dtype)
        input_data[input_name] = random_input_data

    return input_data


def infer_all_tensors_shape(
    model_input: str | Path | ModelProto, save_as_external_data: bool = False
) -> dict[str, tuple[int]]:
    input_data = generate_random_data(model_input)
    model = model_input if isinstance(model_input, ModelProto) else onnx.load(model_input)
    shape_map = collect_tensor_shapes_from_feed(model, input_data)
    return {name: info[1] for name, info in shape_map.items()}


def save_all_tensors_shape(
    model_input: str | Path | ModelProto, tensor_name_shape_dict: dict[str, tuple[int]]
) -> ModelProto:
    model = model_input if isinstance(model_input, ModelProto) else onnx.load(model_input)
    for tensor_name, new_shape in tensor_name_shape_dict.items():
        if len(new_shape) > 0:
            for value_info in model.graph.value_info:
                if value_info.name == tensor_name:
                    new_dtype = value_info.type.tensor_type.elem_type
                    updated_value_info = helper.make_tensor_value_info(tensor_name, new_dtype, new_shape)
                    value_info.CopyFrom(updated_value_info)
    return model


def find_nms(model: ModelProto) -> list[str]:
    nms_node_names = []
    for node in model.graph.node:
        if node.op_type == "NonMaxSuppression":
            nms_node_names.append(node.name)
    return nms_node_names


def parse_args() -> Namespace:
    parser = ArgumentParser("FixShapes")
    parser.add_argument("--input_model_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument(
        "--fix_shapes",
        type=str,
        required=False,
        help="Model input/output name & input/output shape to replace shape of. Provide fix_shapes if name specified. like: 'input_1:[1,224,224,3];input_2:[1.96.96.3];output_1:[1, 1000];output_2:[1,10]'",
    )
    parser.add_argument("--save_as_external_data", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def fix_shapes(args: Namespace) -> None:
    try:
        tmp_path = create_tmp_dir(prefix="quark_onnx.tools.")
        tmp_model_path = Path(tmp_path.name).joinpath("fixed_shapes.onnx").as_posix()
        if args.fix_shapes:
            temp_model = fix_input_and_output_shapes(args.input_model_path, args.fix_shapes)
        else:
            temp_model = onnx.load(args.input_model_path)
        onnx.save(temp_model, tmp_model_path, save_as_external_data=args.save_as_external_data)

        tensor_name_shape_dict = infer_all_tensors_shape(tmp_model_path, args.save_as_external_data)
        if not tensor_name_shape_dict:
            # Distinguish a model with no intermediate tensors (legitimate empty result)
            # from an ORT failure that silently returned no shapes.
            tmp_model = onnx.load(tmp_model_path)
            boundary = (
                {inp.name for inp in tmp_model.graph.input}
                | {out.name for out in tmp_model.graph.output}
                | {init.name for init in tmp_model.graph.initializer}
            )
            has_intermediate = any(
                name for node in tmp_model.graph.node for name in node.output if name and name not in boundary
            )
            if has_intermediate:
                logger.warning(
                    f"Shape inference returned no results for '{args.input_model_path}'. "
                    "ORT inference may have failed; intermediate tensor shapes were not fixed."
                )
                return None
        model = save_all_tensors_shape(tmp_model_path, tensor_name_shape_dict)
        nms_node_names = find_nms(model)
        onnx.save(model, args.output_model_path, save_as_external_data=args.save_as_external_data)
        logger.info(f"Shapes fixed model is saved at {args.output_model_path}")
        for nms_node_name in nms_node_names:
            logger.warning(
                f"The shapes of the nodes following NMS {nms_node_name} should remain dynamic and should not be fixed."
            )
    except Exception as e:
        logger.warning(f"Fail to fix shapes of the input model {args.input_model_path} beacuse {e}, ")


if __name__ == "__main__":
    args = parse_args()
    fix_shapes(args)
