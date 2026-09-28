#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

_EPS = 1e-10

BUILTIN_METRICS: dict[str, MetricFn] = {}


class MetricFn(Protocol):
    """Callable measuring quality loss. Lower return value = quantized model closer to float (better)."""

    def __call__(
        self,
        float_out: list[list[np.ndarray[Any, Any]]],
        quant_out: list[list[np.ndarray[Any, Any]]],
    ) -> float: ...


def l2_metric(
    float_out: list[list[np.ndarray[Any, Any]]],
    quant_out: list[list[np.ndarray[Any, Any]]],
) -> float:
    """Mean L2 norm (np.linalg.norm) of element-wise differences, averaged over all
    (sample, output) pairs.  Matches the ``average_L2`` formula used in
    ``quark.onnx.algorithm.finetuning.onnx_evaluate``:
    ``np.linalg.norm(float_arr.astype(float32) - quant_arr.astype(float32))``
    averaged across all samples and outputs.

    Returns 0.0 when both lists are empty.
    Raises ValueError when the two lists have different lengths.
    """
    if len(float_out) != len(quant_out):
        raise ValueError(
            f"float_out and quant_out must have the same number of samples, got {len(float_out)} vs {len(quant_out)}."
        )
    total = 0.0
    count = 0
    for f_sample, q_sample in zip(float_out, quant_out, strict=False):
        for f_arr, q_arr in zip(f_sample, q_sample, strict=False):
            f = np.array(f_arr, dtype=np.float32)
            q = np.array(q_arr, dtype=np.float32)
            total += float(np.linalg.norm(f - q))
            count += 1
    return total / count if count > 0 else 0.0


def kl_divergence_metric(
    float_out: list[list[np.ndarray[Any, Any]]],
    quant_out: list[list[np.ndarray[Any, Any]]],
) -> float:
    """Mean KL divergence KL(P_float ‖ P_quant) averaged over all (sample, output) pairs.

    Each output array is treated as an unnormalized distribution: values are
    clipped to [0, ∞), shifted so the minimum is 0, then normalized to a
    probability simplex.  A small epsilon (1e-10) is added before the log to
    avoid log(0).

    Returns 0.0 when both lists are empty.
    Raises ValueError when the two lists have different lengths.
    """
    if len(float_out) != len(quant_out):
        raise ValueError(
            f"float_out and quant_out must have the same number of samples, got {len(float_out)} vs {len(quant_out)}."
        )
    total = 0.0
    count = 0
    for f_sample, q_sample in zip(float_out, quant_out, strict=False):
        for f_arr, q_arr in zip(f_sample, q_sample, strict=False):
            f = np.array(f_arr, dtype=np.float64).ravel()
            q = np.array(q_arr, dtype=np.float64).ravel()

            # Shift negatives to 0 before normalizing so all values are non-negative.
            f = f - min(f.min(), 0.0)
            q = q - min(q.min(), 0.0)

            f_sum = f.sum()
            q_sum = q.sum()
            p = f / f_sum if f_sum > 0 else np.ones_like(f) / len(f)
            r = q / q_sum if q_sum > 0 else np.ones_like(q) / len(q)

            # KL(P ‖ R) = sum(P * log(P / R))
            kl = float(np.sum(p * np.log((p + _EPS) / (r + _EPS))))
            total += kl
            count += 1
    return total / count if count > 0 else 0.0


def cosine_metric(
    float_out: list[list[np.ndarray[Any, Any]]],
    quant_out: list[list[np.ndarray[Any, Any]]],
) -> float:
    """Mean cosine distance (1 - cosine_similarity) averaged over all (sample, output) pairs.

    Both arrays are flattened to 1-D float32 vectors before computing the dot
    product and norms.  When either norm is zero the cosine distance for that
    pair is treated as 0.0 (identical).

    Returns 0.0 when both lists are empty or the arrays are identical.
    Raises ValueError when the two lists have different lengths.
    """
    if len(float_out) != len(quant_out):
        raise ValueError(
            f"float_out and quant_out must have the same number of samples, got {len(float_out)} vs {len(quant_out)}."
        )
    total = 0.0
    count = 0
    for f_sample, q_sample in zip(float_out, quant_out, strict=False):
        for f_arr, q_arr in zip(f_sample, q_sample, strict=False):
            f = np.array(f_arr, dtype=np.float32).flatten()
            q = np.array(q_arr, dtype=np.float32).flatten()
            norm_f = float(np.linalg.norm(f))
            norm_q = float(np.linalg.norm(q))
            if norm_f < _EPS or norm_q < _EPS:
                cos_sim = 1.0  # treat zero-norm as identical → distance 0
            else:
                cos_sim = float(np.dot(f, q) / (norm_f * norm_q))
            total += 1.0 - cos_sim
            count += 1
    return total / count if count > 0 else 0.0


def sqnr_metric(
    float_out: list[list[np.ndarray[Any, Any]]],
    quant_out: list[list[np.ndarray[Any, Any]]],
) -> float:
    """Mean negative SQNR (−dB) averaged over all (sample, output) pairs.

    SQNR (Signal-to-Quantization-Noise Ratio) measures how much noise
    quantization introduces relative to the original signal:

        SQNR = 10 * log10(E[||signal||²] / E[||error||²])

    where signal = float_out and error = float_out − quant_out.  Higher SQNR
    means less distortion.  This function negates SQNR to satisfy the
    lower-is-better convention used by all MetricFn implementations.

    When setting ``metric_threshold`` use a negative value, e.g.
    ``metric_threshold=-30`` means "stop mixing when SQNR drops below 30 dB".

    Returns 0.0 when both lists are empty.
    Raises ValueError when the two lists have different lengths.
    """
    if len(float_out) != len(quant_out):
        raise ValueError(
            f"float_out and quant_out must have the same number of samples, got {len(float_out)} vs {len(quant_out)}."
        )
    total = 0.0
    count = 0
    for f_sample, q_sample in zip(float_out, quant_out, strict=False):
        for f_arr, q_arr in zip(f_sample, q_sample, strict=False):
            f = np.array(f_arr, dtype=np.float32)
            q = np.array(q_arr, dtype=np.float32)
            signal_power = float(np.mean(f**2))
            noise_power = float(np.mean((f - q) ** 2))
            if signal_power < _EPS:
                signal_power = _EPS
            if noise_power < _EPS:
                noise_power = _EPS
            sqnr = 10.0 * np.log10(signal_power / noise_power)
            total += -sqnr  # negate: lower metric = higher SQNR = better
            count += 1
    return total / count if count > 0 else 0.0


def psnr_metric(
    float_out: list[list[np.ndarray[Any, Any]]],
    quant_out: list[list[np.ndarray[Any, Any]]],
) -> float:
    """Mean negative PSNR (−dB) averaged over all (sample, output) pairs.

    PSNR is higher-is-better, so this metric negates it to satisfy the
    lower-is-better convention.  A perfectly reconstructed pair yields
    PSNR → ∞ and thus a metric value of −∞; a poor reconstruction yields
    a small positive PSNR and thus a metric value close to 0.

    When setting ``metric_threshold`` with this metric use a negative value,
    e.g. ``metric_threshold=-20`` means "stop mixing when PSNR drops below
    20 dB".

    Returns 0.0 when both lists are empty.
    Raises ValueError when the two lists have different lengths.
    """
    if len(float_out) != len(quant_out):
        raise ValueError(
            f"float_out and quant_out must have the same number of samples, got {len(float_out)} vs {len(quant_out)}."
        )
    total = 0.0
    count = 0
    for f_sample, q_sample in zip(float_out, quant_out, strict=False):
        for f_arr, q_arr in zip(f_sample, q_sample, strict=False):
            f = np.array(f_arr, dtype=np.float32)
            q = np.array(q_arr, dtype=np.float32)
            mse = float(np.mean((f - q) ** 2))
            if mse == 0.0:
                mse = _EPS  # avoid log(0) / INF
            max_val = float(np.max(np.abs(f)))
            if max_val <= 0.0:
                max_val = _EPS
            psnr = 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)
            total += -psnr  # negate: lower metric = higher PSNR = better
            count += 1
    return total / count if count > 0 else 0.0


BUILTIN_METRICS["l2"] = l2_metric
BUILTIN_METRICS["kl"] = kl_divergence_metric
BUILTIN_METRICS["cosine"] = cosine_metric
BUILTIN_METRICS["sqnr"] = sqnr_metric
BUILTIN_METRICS["psnr"] = psnr_metric


def evaluate_fn_adapter(evaluate_fn: Callable[..., float]) -> MetricFn:
    """
    Wraps a higher-is-better evaluate_fn(model_out) -> float into a MetricFn.
    Returns evaluate(float_out) - evaluate(quant_out); lower = better convention preserved.
    """

    def _adapted(
        float_out: list[list[np.ndarray[Any, Any]]],
        quant_out: list[list[np.ndarray[Any, Any]]],
    ) -> float:
        return evaluate_fn(float_out) - evaluate_fn(quant_out)

    return _adapted


def resolve_metric_fn(
    metric_distance_fn: MetricFn | None,
    metric_evaluate_fn: Callable[..., float] | None,
    metric_default: str = "l2",
) -> MetricFn:
    """Select the active metric function from config.

    Resolution order:
    1. ``metric_distance_fn`` — user-supplied callable (highest priority).
    2. ``metric_evaluate_fn`` — wrapped with :func:`evaluate_fn_adapter`.
    3. ``metric_default`` — one of the built-in named metrics (``"l2"``, ``"kl"``, ``"cosine"``, ``"psnr"``).

    ``metric_distance_fn`` and ``metric_evaluate_fn`` are mutually exclusive.
    Raises ValueError if both callables are provided or if *metric_default* is unknown.
    """
    if metric_distance_fn is not None and metric_evaluate_fn is not None:
        raise ValueError("metric_distance_fn and metric_evaluate_fn are mutually exclusive; provide at most one.")
    if metric_evaluate_fn is not None:
        logger.info(
            f"AMP metric: using custom evaluate function '{getattr(metric_evaluate_fn, '__name__', repr(metric_evaluate_fn))}'."
        )
        return evaluate_fn_adapter(metric_evaluate_fn)
    if metric_distance_fn is not None:
        logger.info(
            f"AMP metric: using custom distance function '{getattr(metric_distance_fn, '__name__', repr(metric_distance_fn))}'."
        )
        return metric_distance_fn
    if metric_default not in BUILTIN_METRICS:
        raise ValueError(f"Unknown metric_default '{metric_default}'. Valid options are: {sorted(BUILTIN_METRICS)}.")
    logger.info(f"AMP metric: using built-in metric '{metric_default}'.")
    return BUILTIN_METRICS[metric_default]
