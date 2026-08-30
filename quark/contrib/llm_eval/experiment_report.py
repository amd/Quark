#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone

UTC = timezone.utc
from typing import Any

import numpy as np

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ExperimentReport:
    VERSION = 1

    def __init__(self, experiment: dict[str, Any], metrics: list[Any], metadata: dict[str, Any]):
        self.experiment = experiment
        self.metrics = metrics
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "data": [{"experiment": self.experiment, "metrics": self.metrics, "metadata": self.metadata}],
        }

    def to_json(self, indent: int = 2) -> str:
        def convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32 | np.float64):
                return float(obj)
            elif isinstance(obj, np.int32 | np.int64):
                return int(obj)
            return obj

        return json.dumps(self.to_dict(), indent=indent, default=convert)


def generate_metadata() -> dict[str, Any]:
    """
    Generate environment metadata.

    If environment collection fails, this function will log the issue and return
    an empty metadata dictionary instead of raising an error.
    """

    metadata: dict[str, Any] = {}

    try:
        from quark.common.utils.collect_env import collect_environment

        env_info = collect_environment()

    except Exception as e:
        logger.error(f"Failed to collect environment metadata ({e}). Skipping environment metadata collection.")
        return metadata

    # === original logic preserved ===
    software_keys = {
        "os": "os",
        "python": "python_version",
        "torch": "torch_version",
        "transformers": "transformers_version",
        "accelerate": "accelerate_version",
        "amd-quark": "quark_version",
        "cuda_driver_version": "cuda_driver_version",
    }

    for key, new_key in software_keys.items():
        value = env_info.get("software", {}).get(key)
        metadata[f"environment.software.{new_key}"] = value

    for k, v in env_info.get("hardware", {}).items():
        metadata[f"environment.hardware.{k}"] = v

    metadata["command_argv"] = " ".join(sys.argv)
    return metadata


def build_experiment_dict(args: Namespace) -> dict[str, Any]:
    model_path = getattr(args, "model_dir", "")
    model = os.path.basename(model_path)
    quant_scheme = getattr(args, "quant_scheme", "")
    quant_algo = getattr(args, "quant_algo", "")
    if isinstance(quant_algo, list):
        quant_algo = "_".join(quant_algo)
    name = "/".join(
        filter(
            None,
            [
                model,
                quant_scheme,
                quant_algo,
            ],
        )
    )
    timestamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "name": name,
        "model": model,
        "settings": {
            "quantization_scheme": quant_scheme,
            "quantization_algorithm": quant_algo,
            "calibration_dataset_name": getattr(args, "dataset", "pileval"),
            "calibration_dataset_num_samples": getattr(args, "num_calib_data", 512),
            "calibration_dataset_sequence_length": getattr(args, "seq_len", 512),
            "calibration_dataset_batch_size": getattr(args, "batch_size", 1),
            "model_export_format": getattr(args, "model_export", ""),
        },
        "timestamp": timestamp,
    }


def build_metrics_list(metrics: list[Any]) -> list[Any]:
    """
    metrics = [
        ["Perplexity", 12.45],
        ["MMLU", 0.22, "acc,none: ..."],
    ]
    """
    result = []

    for item in metrics:
        if len(item) < 2:
            raise ValueError(f"Invalid metric entry: {item}")

        result.append({"name": item[0], "value": item[1], "metadata": item[2] if len(item) > 2 else ""})

    return result


def generate_experiment_report(
    args: Any,
    metrics: list[Any],
) -> ExperimentReport:
    experiment = build_experiment_dict(args)
    metadata = generate_metadata()
    metrics = build_metrics_list(metrics)
    report = ExperimentReport(
        experiment=experiment,
        metrics=metrics,
        metadata=metadata,
    )

    return report
