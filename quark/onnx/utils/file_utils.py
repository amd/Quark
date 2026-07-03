#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import csv
import json
import os
from typing import Any

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationMethod, TensorsData

from quark.common.utils.log import ScreenLogger

from .model_utils import save_onnx_model_with_external_data

logger = ScreenLogger(__name__)

CRYPTO_MODE: bool = False


def update_crypto_mode(crypto_mode: bool) -> None:
    global CRYPTO_MODE
    CRYPTO_MODE = crypto_mode


def save_quantized_info(input_rows: list[Any], write_mode: str = "a") -> None:
    quantized_info_path = "quantized_info.csv"

    if not CRYPTO_MODE:
        with open(quantized_info_path, write_mode, newline="") as file:
            writer = csv.writer(file)
            writer.writerows(input_rows)


def save_and_restore_func(
    save_and_restore: None | str,
    command_type: str,
    save_content: Any = None,
    stage_name: str = "tensors_range",
) -> Any:
    """
    Save or restore intermediate data (e.g., tensor ranges or finetuning metadata) to/from a JSON file.

    This function supports multiple "stages" of data persistence, controlled by `stage_name`, and can either write data to disk or load it back into memory.

    :param Union[str, None] save_and_restore: Path to the JSON file used for saving or restoring data. If None, the function does nothing and logs a warning.
    :param str command_type: Operation mode. Must be either:
        - "save": serialize data and write it to the JSON file
        - "restore": load data from the JSON file and return it
    :param Any save_content: Operation mode. Could be:
        - Any tensors_range: Tensor range data used when `stage_name == "tensors_range"`.
        - Union[str, None] model_to_finetune:  Model identifier or path to be saved/restored when `stage_name == "model_to_finetune"`.
        - list[int | None] layers_to_finetune: List of layer indices to be saved/restored when `stage_name == "layers_to_finetune"`.
        - Any: quark version.
    :param str stage_name: Specifies which type of data to save or restore. Supported values: ["tensors_range", "model_to_finetune","layers_to_finetune"].
    :return Any:  When `save_or_restore == "restore"`, returns the restored object depending on `stage_name`:
        - "tensors_range": a `TensorsData` object
        - "model_to_finetune": str or None
        - "layers_to_finetune": list[int | None]
        Returns None for save operations or unsupported stages.
    """
    assert command_type in ["save", "restore"], 'Please select "save" or "restore" for parameter "save_or_restore" '
    if save_and_restore is not None:
        if os.path.exists(save_and_restore):
            with open(save_and_restore) as json_file:
                loaded_dict = json.load(json_file)

        if command_type == "save":
            save_and_restore_dict = {} if not os.path.exists(save_and_restore) else loaded_dict
            if stage_name in list(save_and_restore_dict.keys()):
                save_and_restore_dict.pop(stage_name)
                logger.warning(f"New {stage_name} content will be written, the old one will be removed!")
            if stage_name == "tensors_range":
                tensors_range_dict = {}
                for key in save_content.data:
                    temp_value = save_content.data[key].range_value
                    tensors_range_dict[key] = (temp_value[0].tolist(), temp_value[1].tolist())
                save_and_restore_dict["tensors_range"] = tensors_range_dict
            else:
                save_and_restore_dict[stage_name] = save_content

            with open(save_and_restore, "w") as json_file:
                json.dump(save_and_restore_dict, json_file, indent=2)

        elif command_type == "restore":
            assert os.path.exists(save_and_restore), "The SaveAndRestore does not exist."
            if stage_name == "tensors_range":
                tensors_range_dict = {}
                saved_tensors_range_dict = loaded_dict.get("tensors_range")
                assert saved_tensors_range_dict is not None, 'Please set "tensors_range" in "SaveAndRestore" .'
                for key in saved_tensors_range_dict:
                    temp_value = saved_tensors_range_dict[key]
                    tensors_range_dict[key] = (
                        np.array(temp_value[0], dtype=np.float32),
                        np.array(temp_value[1], dtype=np.float32),
                    )
                tensors_range = TensorsData(CalibrationMethod.MinMax, tensors_range_dict)
                return tensors_range
            else:
                return loaded_dict.get(stage_name, None)

    else:
        logger.warning('Please set "SaveAndRestore" for using this function!')


def load_model_layers_to_finetune(save_and_restore: str) -> tuple[Any, Any]:
    model_to_finetune = save_and_restore_func(
        save_and_restore=save_and_restore, command_type="restore", stage_name="model_to_finetune"
    )
    layers_to_finetune = save_and_restore_func(
        save_and_restore=save_and_restore, command_type="restore", stage_name="layers_to_finetune"
    )
    restored_model = None
    restored_layers = None
    if model_to_finetune is not None:
        restored_model = onnx.load(model_to_finetune)
    if layers_to_finetune is not None:
        restored_layers = layers_to_finetune
    return restored_model, restored_layers


def save_model_layers_to_finetune(save_and_restore: str, idx: int, sg: Any, use_external_data_format: bool) -> None:
    temp_model_output = "model_to_finetune.onnx"
    if save_and_restore.endswith(".json"):
        temp_model_output = save_and_restore.replace(".json", ".onnx")
    temp_layers_index = list(range(idx, len(sg.subgraph_qmodel_list)))
    save_and_restore_func(save_and_restore, "save", save_content=temp_model_output, stage_name="model_to_finetune")
    save_and_restore_func(save_and_restore, "save", save_content=temp_layers_index, stage_name="layers_to_finetune")
    temp_quant_model = sg.convert_qmodel_batch_size() if sg.dynamic_batch else sg.qmodel
    save_onnx_model_with_external_data(
        temp_quant_model, temp_model_output, save_as_external_data=use_external_data_format
    )
    return None
