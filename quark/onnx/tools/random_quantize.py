#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from quark.onnx import quantize_static, PowerOfTwoMethod  # type: ignore
import argparse
import os
from quark.shares.utils.log import ScreenLogger
from typing import Optional

logger = ScreenLogger(__name__)


def is_valid_path(path: Optional[str]) -> bool:
    if not path:
        logger.warning("path is null")
        return False

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        logger.warning(f"path is not exist: {directory}")
        return False

    if not os.path.isdir(directory):
        logger.warning(f"path is not directory: {directory}")
        return False

    if os.path.exists(path) and not os.access(path, os.W_OK):
        logger.warning(f"the file is read-only: {path}")
        return False

    return True


def onnx_random_quantize() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, default="", help="input onnx model file path.")
    parser.add_argument("--quant_model", type=str, default="", help="output quant model file path.")
    FLAGS, uparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.input_model):
        logger.warning("Input model file '{}' does not exist!".format(FLAGS.input_model))
        logger.warning("Usage: python -m quark.onnx.tools.random_quantize "
                       "--input_model INPUT_MODEL_PATH --quant_model QUANT_MODEL_PATH.")
        exit()

    if not is_valid_path(FLAGS.quant_model):
        logger.warning("Usage: python -m quark.onnx.tools.random_quantize "
                       "--input_model INPUT_MODEL_PATH --quant_model QUANT_MODEL_PATH.")
        exit()

    # `input_model_path` is the path to the original, unquantized ONNX model.
    model_input = FLAGS.input_model

    # `quant_model_path` is the path where the quantized model will be saved.
    model_output = FLAGS.quant_model

    calibration_data_reader = None
    quantize_static(model_input,
                    model_output,
                    calibration_data_reader,
                    calibrate_method=PowerOfTwoMethod.NonOverflow,
                    enable_dpu=True,
                    extra_options={
                        'ActivationSymmetric': True,
                        "UseRandomData": True
                    })

    logger.info(f'Calibrated and quantized model saved at: {model_output}')


if __name__ == '__main__':
    onnx_random_quantize()
