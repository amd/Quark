#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
from onnxruntime.quantization.calibrate import TensorsData

from quark.common.utils.log import ScreenLogger
from quark.onnx.algorithm.finetuning.onnx_evaluate import inference_model

from .metric_funcs import resolve_metric_fn
from .mixing_executor import MixingExecutor
from .mixing_strategy import MixingStrategy
from .mprecision_config import AutoMixprecisionConfig
from .sensitivity_analyzer import (
    SensitivityAnalyzer,
    compute_cache_key,
    load_sensitivity_results,
    save_sensitivity_results,
)
from .subgraph_parser import parse_subgraph_json

logger = ScreenLogger(__name__)


def auto_mixprecision(
    float_model: str | Path | onnx.ModelProto,
    quant_model: str | Path | onnx.ModelProto,
    use_external_data_format: bool,
    data_reader: Any,
    tensors_range: TensorsData,
    extra_options: dict[str, Any],
) -> onnx.ModelProto:
    """Full AMP pipeline: sensitivity analysis followed by greedy precision promotion.

    :param float_model: The original float-precision ONNX model
        (path or in-memory proto).
    :param quant_model: The quantized ONNX model to modify
        (path or in-memory proto).
    :param use_external_data_format: Whether the model uses external
        data format (for models > 2 GB).
    :param data_reader: Calibration data reader used for inference
        during sensitivity analysis and the promotion loop.
    :param tensors_range: Data range for all quantizing tensors,
        which is used for re-compute the quantization parameters.
    :param extra_options: Extra options for the algorithm.
    :returns: The modified quantized model with promoted candidates.
    """
    float_proto = float_model if isinstance(float_model, onnx.ModelProto) else onnx.load(float_model)
    quant_proto = quant_model if isinstance(quant_model, onnx.ModelProto) else onnx.load(quant_model)

    if data_reader is None:
        logger.warning("No data reader provided. Returning original quantized model.")
        return quant_proto

    config = AutoMixprecisionConfig._from_extra_options(extra_options)
    if config.target_layer_config is None:
        logger.warning("No target_layer_config in the config. Returning original quantized model.")
        return quant_proto

    metric_fn = resolve_metric_fn(config.metric_distance_fn, config.metric_evaluate_fn, config.metric_default)

    float_out = inference_model(float_proto, data_reader, config.data_size, config.metric_output_index)
    quant_out = inference_model(quant_proto, data_reader, config.data_size, config.metric_output_index)
    baseline_score = metric_fn(float_out, quant_out)
    if config.metric_threshold is None:
        logger.info(f"The baseline score is {baseline_score:.4f}. The threshold is None — sensitivity analysis only.")
    elif config.metric_threshold == 0:
        logger.info(f"The baseline score is {baseline_score:.4f}. No threshold set — all candidates will be mixed.")
    else:
        logger.info(f"The baseline score is {baseline_score:.4f} and the threshold is {config.metric_threshold}.")
        if config.metric_optimize_object == "speed":
            if baseline_score > config.metric_threshold:
                logger.warning(
                    f"The baseline (score {baseline_score:.4f}) has already exceeded the threshold ({config.metric_threshold}), "
                    f"no optimization room for optimizing '{config.metric_optimize_object}'. Returning original quantized model."
                )
                return quant_proto
        else:
            if baseline_score <= config.metric_threshold:
                logger.warning(
                    f"The baseline (score {baseline_score:.4f}) has better quality than the threshold ({config.metric_threshold}), "
                    f"no optimization needed for the object '{config.metric_optimize_object}'. Returning original quantized model."
                )
                return quant_proto

    subgraph_specs = None
    if config.subgraph_json is not None and Path(config.subgraph_json).exists():
        logger.info(f"Parsing subgraphs from the provided {config.subgraph_json}")
        subgraph_specs = parse_subgraph_json(config.subgraph_json, float_proto, quant_proto)

    # Sensitivity analysis path
    strategy = MixingStrategy(
        config.target_layer_config, shared_param_mode=config.shared_param_mode, extra_options=extra_options
    )

    ranked: list[Any] | None = None

    current_key = compute_cache_key(quant_proto, config)
    cache_path = config.sensitivity_cache_file
    if cache_path and Path(cache_path).exists():
        ranked = load_sensitivity_results(cache_path, current_key)

    if ranked is None:
        analyzer = SensitivityAnalyzer(config, metric_fn, float_out, strategy)
        ranked = analyzer.analyze(quant_proto, data_reader, tensors_range, subgraph_specs)
        if cache_path:
            save_sensitivity_results(ranked, cache_path, current_key)

    SensitivityAnalyzer.print_sensitivity_table(ranked)

    if not ranked:
        logger.warning("No candidate nodes found. Returning original quantized model.")
        return quant_proto

    if config.metric_threshold is None:
        logger.info("Because the threshold is None, skipping the mixing step.")
        return quant_proto

    # Mixing precision path
    executor = MixingExecutor(config, metric_fn, float_out, strategy)
    result_proto = executor.execute(quant_proto, data_reader, tensors_range, ranked)
    return executor.insert_dual_quant_nodes(result_proto, tensors_range, extra_options)
