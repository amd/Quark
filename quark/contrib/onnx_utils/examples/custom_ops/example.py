# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as rt


def create_random_data(
    shape: Sequence[int], dtype: npt.DTypeLike, min_value: int, max_value: int, seed: int | None
) -> npt.NDArray[Any]:
    np_type = np.dtype(dtype)
    np.random.seed(seed)
    return ((max_value - min_value) * np.random.sample(shape) + min_value).astype(np_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dll-path", type=str, default=None, help="Path to custom op DLL")
    parser.add_argument("model", type=Path, help="Path to the model")
    args = parser.parse_args()

    model = onnx.load(args.model)
    input_dict = {}
    output_names = []
    for i in model.graph.input:
        shape_arr = []
        shape_obj = i.type.tensor_type.shape
        for dim in shape_obj.dim:
            if dim.HasField("dim_param"):
                shape_arr.append(1)
            if dim.HasField("dim_value"):
                shape_arr.append(dim.dim_value)
        dtype = i.type.tensor_type.elem_type
        d = create_random_data(shape_arr, onnx.helper.tensor_dtype_to_np_dtype(dtype), 0, 1, None)
        input_dict[i.name] = d

    for o in model.graph.output:
        output_names.append(o.name)

    session_options = rt.SessionOptions()
    if args.dll_path is not None:
        session_options.register_custom_ops_library(args.dll_path)
        session_options.optimized_model_filepath = str(args.model.parent / f"{args.model.stem}_optimized.onnx")

    session = rt.InferenceSession(args.model, sess_options=session_options)
    res = session.run(output_names, input_dict)
    print("Ran successfully")
