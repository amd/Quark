#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
A tool for replace `inf` and `-inf` values in ONNX model weights with specified replacement values.'

    Example : python -m quark.onnx.tools.replace_inf_weights --input_model [INPUT_MODEL_PATH] --output_model [OUTPUT_MODEL_PATH] --replace_inf_value [REPLACE_INF_VALUE]

'''

import argparse
import onnx
import numpy as np
from onnx import numpy_helper
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def replace_inf_in_onnx_weights(input_model_path: str,
                                output_model_path: str,
                                replace_inf_value: float = 10000.0) -> None:
    """
    Replaces `inf` and `-inf` values in the weights of an ONNX model with specified default values.

    Parameters:
        input_model_path (str): Path to the input ONNX model file.
        output_model_path (str): Path to save the modified ONNX model file.
        replace_inf_value (float): The base value used to replace `inf` and `-inf`.
                              - Positive `inf` values are replaced with `replace_inf_value`.
                              - Negative `inf` values are replaced with `-replace_inf_value`.

    Returns:
        None: The function directly modifies the model and saves it to the output path.
    """

    model = onnx.load(input_model_path)

    for weight in model.graph.initializer:
        weight_array = numpy_helper.to_array(weight)
        is_weight_updated = False

        if np.isinf(weight_array).any():
            updated_weight_array = np.copy(weight_array)
            updated_weight_array[weight_array == np.inf] = replace_inf_value
            updated_weight = numpy_helper.from_array(updated_weight_array, name=weight.name)
            logger.info(f"Weight '{weight.name}': Updated 'inf' values to {replace_inf_value}.")
            is_weight_updated = True

        if np.any(weight_array == -np.inf):
            updated_weight_array = np.copy(weight_array)
            updated_weight_array[weight_array == -np.inf] = -replace_inf_value
            updated_weight = numpy_helper.from_array(updated_weight_array, name=weight.name)
            logger.info(f"Weight '{weight.name}': Updated '-inf' values to {-replace_inf_value}.")
            is_weight_updated = True

        if is_weight_updated:
            model.graph.initializer.remove(weight)
            model.graph.initializer.append(updated_weight)

    onnx.save(model, output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Replace `inf` and `-inf` values in ONNX model weights with specified replacement values.')
    parser.add_argument('--input_model', type=str, required=True, help='Path to the input ONNX model file')
    parser.add_argument('--output_model', type=str, required=True, help='Path to save the modified ONNX model file')
    parser.add_argument('--replace_inf_value',
                        type=float,
                        default=10000.0,
                        help='Value used to replace `inf` and `-inf`: '
                        '`inf` is replaced with this value, and `-inf` is replaced with its negative.')
    args = parser.parse_args()
    replace_inf_in_onnx_weights(args.input_model, args.output_model, args.replace_inf_value)
