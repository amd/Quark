#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
from typing import Any

import yaml


class LoadConfigFromFileOrDict:
    """
    A utility class for loading configuration data.
    It supports three kinds of input:
        1. A Python dictionary
        2. A JSON string
        3. A file path pointing to a JSON or YAML config file

    Example YAML config:
    --------------------------------
    input_model_path: "mnist-7.onnx"            # Path to the input ONNX model
    passes:                                     # Pass configurations for optimization/transformation
      onnx_convert_opset_version:               # Pass name: convert ONNX opset version
        target_opset_version: 12                # Parameter of this pass: target opset version
    output_model_path: "mnist-7_optimized.onnx" # Path to the output optimized model
    --------------------------------

    After loading, the internal data will look like:
    {
        "input_model_path": "mnist-7.onnx",
        "passes": {
            "onnx_convert_opset_version": {
                "target_opset_version": 12
            }
        },
        "output_model_path": "mnist-7_optimized.onnx"
    }
    """

    def __init__(self, input_data: dict[str, Any] | str):
        """
        Initialize the config loader.

        Args:
            input_data (Union[Dict[str, Any], str]):
                - dict: a pre-parsed configuration dictionary
                - str: either a JSON string or a path to a JSON/YAML file

        Raises:
            ValueError: if the input is neither a dict nor a str
        """
        if isinstance(input_data, dict):
            # Input is already a Python dictionary
            self.data = input_data
        elif isinstance(input_data, str):
            # Input is a string: try parsing as JSON string first,
            # if that fails, treat it as a file path
            self.data = self._load_config_file_from_path(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(self.data)}. Must be yaml, json or dict.")

    def _load_config_file_from_path(self, input_str: str) -> dict[str, Any]:
        """
        Internal method to load configuration from a string
        that may be either a JSON string or a file path.

        Args:
            input_str (str):
                - JSON string, or
                - path to a JSON/YAML file

        Returns:
            dict: parsed configuration dictionary

        Raises:
            ValueError: if the input is neither valid JSON string
                        nor a valid YAML/JSON file
        """
        # First, try parsing the input as a JSON string
        try:
            return json.loads(input_str)
        except json.JSONDecodeError:
            # If it's not valid JSON, treat it as a file path
            try:
                with open(input_str, encoding="utf-8") as f:
                    # Try loading as YAML (works for both YAML and JSON files)
                    return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Unsupported input type: {type(self.data)}. Must be yaml, json or dict.\n\nDetails:\n{e}"
                )
