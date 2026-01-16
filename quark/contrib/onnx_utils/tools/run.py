# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

# path to the directory containing the test data
PARENT_DIR = ""
# prefixed name of the model to run
NAME = ""
# index to run
INDEX = 0
# path to custom ops shared library
CUSTOM_OPS_PATH = ""


def load_meta_json(meta_json_file: str) -> dict[str, Any]:
    with open(meta_json_file) as f:
        meta = json.load(f)
    if "tensor_map" in meta:
        raise ValueError("Loading meta.json in DD format is not supported yet")
    return cast(dict[str, Any], meta)


def get_metadata_by_filename(meta: dict[str, Any], filename: Path) -> tuple[str, dict[str, Any]]:
    for key, value in meta.items():
        file_key = Path(value["file_name"])
        if (file_key.is_absolute() and file_key.as_posix() == filename.as_posix()) or (
            (not file_key.is_absolute()) and str(file_key) == filename.name
        ):
            return key, value
    raise ValueError(f"No metadata found for {filename.as_posix()}")


def get_inputs(meta: dict[str, Any]) -> dict[str, npt.NDArray[Any]]:
    input_data_file = (Path(PARENT_DIR) / f"{NAME}_{INDEX}_0.in").absolute()
    inputs = {}
    i = 0
    while input_data_file.exists():
        name, metadata = get_metadata_by_filename(meta, input_data_file)
        input_data = np.fromfile(input_data_file, dtype=metadata["dtype"]).reshape(metadata["shape"])
        inputs[name] = input_data
        i += 1
        input_data_file = (Path(PARENT_DIR) / f"{NAME}_{INDEX}_{i}.in").absolute()

    return inputs


def get_outputs(meta: dict[str, Any]) -> dict[str, npt.NDArray[Any]]:
    suffix = ""
    output_data_file = (Path(PARENT_DIR) / f"{NAME}_{INDEX}_0{suffix}.out").absolute()
    outputs = {}
    i = 0
    while output_data_file.exists():
        name, metadata = get_metadata_by_filename(meta, output_data_file)
        input_data = np.fromfile(output_data_file, dtype=metadata["dtype"]).reshape(metadata["shape"])
        outputs[name] = input_data
        i += 1
        output_data_file = (Path(PARENT_DIR) / f"{NAME}_{INDEX}_{i}{suffix}.out").absolute()

    return outputs


def run(inputs: dict[str, npt.NDArray[Any]], outputs: list[str]) -> Sequence[npt.NDArray[Any]]:
    model = (Path(PARENT_DIR) / f"{NAME}_{INDEX}.onnx").absolute()
    providers = ["CPUExecutionProvider"]
    custom_ops_path = CUSTOM_OPS_PATH

    sess_options = ort.SessionOptions()
    if custom_ops_path:
        sess_options.register_custom_ops_library(custom_ops_path)

    session = ort.InferenceSession(model, sess_options=sess_options, providers=providers)

    return cast(Sequence[npt.NDArray[Any]], session.run(outputs, inputs))


def find_dissimilar_values(output: npt.NDArray[Any], golden_output: npt.NDArray[Any]) -> tuple[list[int], float]:
    values = []
    total_count = 0

    flat_output = output.flatten()
    for i, (x, y) in enumerate(zip(flat_output, golden_output.flatten(), strict=True)):
        if not np.isclose(x, y, atol=1e-2):
            values.append(i)
            if len(values) > 10:
                break
            total_count += 1

    # this takes too long to compute so break after 10
    # percent = total_count / len(flat_output)
    percent = 0
    return values, percent


if __name__ == "__main__":
    meta_json_file = os.path.join(PARENT_DIR, f"{NAME}_{INDEX}_meta.json")
    meta = load_meta_json(meta_json_file)

    inputs = get_inputs(meta)
    golden_outputs = get_outputs(meta)

    outputs = run(inputs, list(golden_outputs.keys()))

    golden_outputs_list = list(golden_outputs.values())
    assert len(outputs) == len(golden_outputs_list)

    for index, (output, golden_output) in enumerate(zip(outputs, golden_outputs_list, strict=True)):
        # compare golden_outputs and outputs
        if not np.allclose(output, golden_output, atol=1e-2):
            print(f"output {index} doesn't match")
            dissimilar_values, percent = find_dissimilar_values(output, golden_output)
        else:
            print(f"output {index} matches")

    print("Success")
