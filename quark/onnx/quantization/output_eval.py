#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization.calibrate import CalibrationDataReader

from quark.onnx.operators.custom_ops import get_library_path
from quark.onnx.utils.model_utils import create_infer_session_for_onnx_model
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def calculate_cos(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    arr1 = x.astype(np.float32).flatten()
    arr2 = y.astype(np.float32).flatten()
    dot_product = np.dot(arr1, arr2)
    norm_arr1 = np.linalg.norm(arr1)
    norm_arr2 = np.linalg.norm(arr2)
    cos_sim = dot_product / (norm_arr1 * norm_arr2)
    cos_sim = np.array(cos_sim)
    assert isinstance(cos_sim, np.ndarray)
    return cos_sim


def calculate_l2_distance(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    l2_distance = np.linalg.norm(x.astype(np.float32) - y.astype(np.float32))
    l2_distance = np.array(l2_distance)
    assert isinstance(l2_distance, np.ndarray)
    return l2_distance


def calculate_l1_distance(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Calculate the L1 (Manhattan) distance between two NumPy arrays.

    The L1 distance is defined as the sum of the absolute differences
    between corresponding elements of the two arrays:
        D_L1(x, y) = Σ |x_i - y_i|

    Parameters
    ----------
    x : np.ndarray
        First input array.
    y : np.ndarray
        Second input array (must be broadcastable to x).

    Returns
    -------
    np.ndarray
        A NumPy array containing the L1 distance.
    """
    # Convert both arrays to float32 for consistency
    l1_distance = np.sum(np.abs(x.astype(np.float32) - y.astype(np.float32)))

    # Wrap scalar in NumPy array for type consistency
    l1_distance = np.array(l1_distance)

    # Ensure output is indeed a NumPy array
    assert isinstance(l1_distance, np.ndarray)

    return l1_distance


def calculate_ssim(base_input: np.ndarray[Any, Any], ref_input: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Calculate the Structural Similarity Index (SSIM) between two NumPy arrays.

    The SSIM metric measures perceptual similarity between two images or signals.
    It combines luminance, contrast, and structure comparisons into a single value
    between 0 and 1, where 1 indicates perfect similarity.

    Parameters
    ----------
    base_input : np.ndarray
        Baseline input (predicted image or signal) as a NumPy array of float32.
    ref_input : np.ndarray
        Reference input (ground truth) as a NumPy array of float32.

    Returns
    -------
    np.ndarray
        A NumPy array containing the SSIM value (float32).

    Notes
    -----
    - Only `np.ndarray` inputs are accepted.
    - The dynamic range (R) is assumed to be 255.
    """
    # Validate input types
    assert isinstance(base_input, np.ndarray) and isinstance(ref_input, np.ndarray), "Inputs must be NumPy arrays."

    # Cast to float32 for numerical consistency
    y_pred = base_input.astype(np.float32)
    y_true = ref_input.astype(np.float32)

    # Mean values (luminance)
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)

    # Variances and standard deviations (contrast)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)

    # Constants for stability (based on pixel range)
    R = 255.0
    c1 = (0.01 * R) ** 2
    c2 = (0.03 * R) ** 2

    # SSIM calculation
    numerator = (2 * u_true * u_pred + c1) * (2 * std_true * std_pred + c2)
    denominator = (u_true**2 + u_pred**2 + c1) * (var_true + var_pred + c2)
    ssim_value = numerator / denominator

    # Wrap scalar in NumPy array for consistent output type
    ssim_value = np.array(ssim_value, dtype=np.float32)

    # Type check for safety
    assert isinstance(ssim_value, np.ndarray)

    return ssim_value


def calculate_psnr(reference_image: np.ndarray[Any, Any], noisy_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    reference_image = reference_image.astype(np.float32)
    mse = np.mean((reference_image - noisy_image) ** 2)
    if mse == np.array(0):
        mse = mse + np.array(1e-10)
    max_pixel_value: np.ndarray[Any, Any] = np.max(reference_image)  # type: ignore
    if max_pixel_value <= np.array(0):
        max_pixel_value = np.array(1e-10)
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    psnr = np.array(psnr)
    assert isinstance(psnr, np.ndarray)
    return psnr


def eval_metrics(
    float_model: str | Path | onnx.ModelProto,
    quant_model: str | Path | onnx.ModelProto,
    eval_data_reader: CalibrationDataReader,
    execution_providers: list[str] | None = ["CPUExecutionProvider"],
    use_external_data_format: bool = False,
) -> None:
    metric_values_l2 = []
    metric_values_cos = []

    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())
    float_sess = create_infer_session_for_onnx_model(
        float_model, sess_options, providers=execution_providers, use_external_data_format=use_external_data_format
    )
    quant_sess = create_infer_session_for_onnx_model(
        quant_model, sess_options, providers=execution_providers, use_external_data_format=use_external_data_format
    )

    while True:
        inputs = eval_data_reader.get_next()
        if not inputs:
            break
        float_output = float_sess.run([], inputs)[0]
        quant_output = quant_sess.run([], inputs)[0]
        metric_values_cos.append(calculate_cos(float_output, quant_output))
        metric_values_l2.append(calculate_l2_distance(float_output, quant_output))

    logger.info("Quantization Metrics (float vs quantized):")
    logger.info(f"Mean Cosine Similarity: {np.mean(metric_values_cos)}")
    logger.info(f"Min Cosine Similarity: {np.min(metric_values_cos)}")
    logger.info(f"Mean L2 Distance: {np.mean(metric_values_l2)}")
    logger.info(f"Max L2 Distance: {np.max(metric_values_l2)}")

    return None
