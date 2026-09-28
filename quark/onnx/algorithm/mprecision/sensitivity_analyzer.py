#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from onnxruntime.quantization.calibrate import TensorsData
from onnxruntime.quantization.onnx_model import ONNXModel
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger
from quark.onnx.algorithm.finetuning.onnx_evaluate import inference_model
from quark.version import __version__ as version

from .metric_funcs import MetricFn
from .mixing_strategy import MixingStrategy
from .mprecision_config import AutoMixprecisionConfig
from .subgraph_parser import SubgraphSpec

logger = ScreenLogger(__name__)


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for a single layer or subgraph.

    :param name: Layer or subgraph name.
    :param candidate_nodes: Node names covered by this candidate.
    :param score: Metric distance from the float model output when this
        candidate is promoted (lower means closer to float). This is the
        minimum score across all config candidates.
    :param all_config_scores: Per-config scores in the same order as the
        ``target_config_list`` supplied to ``MixingStrategy``.
    :param best_config_index: Index into ``target_config_list`` of the config that
        produced the minimum score.
    :param enabled: When ``False`` the candidate is excluded from the greedy
        promotion loop even though its score is recorded. Users can set this
        field to ``false`` directly in the cache JSON to pin specific layers
        or subgraphs at their original quantization precision.
    """

    name: str
    candidate_nodes: list[str]
    score: float
    all_config_scores: list[float]
    best_config_index: int
    enabled: bool = True


class SensitivityAnalyzer:
    """Measures per-layer or per-subgraph sensitivity to precision changes.

    Each candidate is temporarily promoted to the target precision, the
    model is evaluated against the float reference, and the metric
    distance is recorded. Results are returned sorted ascending by score
    (closest to float first).

    :param config: AMP configuration.
    :param metric_fn: Metric function scoring the distance between
        float and quantized outputs (lower is better).
    :param strategy: Mixing strategy used to promote/demote candidates.
    """

    def __init__(
        self,
        config: AutoMixprecisionConfig,
        metric_fn: MetricFn,
        float_out: list[list[np.ndarray[Any, Any]]],
        strategy: MixingStrategy,
    ) -> None:
        self._config = config
        self._metric_fn = metric_fn
        self._float_out = float_out
        self._strategy = strategy

    def analyze(
        self,
        quant_model: onnx.ModelProto,
        data_reader: Any,
        tensors_range: TensorsData,
        subgraph_specs: list[SubgraphSpec] | None,
    ) -> list[SensitivityResult]:
        """Run sensitivity analysis and return ranked candidates.

        When *subgraph_specs* is provided, each subgraph is scored as a
        group. Otherwise, each candidate layer is scored individually
        (layer-wise is treated as single-node subgraphs).

        :param quant_model: The quantized ONNX model.
        :param data_reader: Calibration data reader for inference.
        :param tensors_range: Data range for all quantizing tensors.
        :param subgraph_specs: Subgraph definitions for subgraph-wise
            analysis, or ``None`` for layer-wise analysis.
        :returns: Sensitivity results sorted ascending by score.
        """
        if subgraph_specs is None:
            candidates = self._candidate_nodes(quant_model)
            subgraph_specs = [
                SubgraphSpec(name=n, start_nodes=[], end_nodes=[], resolved_nodes=[n]) for n in candidates
            ]

        return self._score_specs(quant_model, data_reader, tensors_range, subgraph_specs)

    def _score_specs(
        self,
        quant_model: onnx.ModelProto,
        data_reader: Any,
        tensors_range: TensorsData,
        subgraph_specs: list[SubgraphSpec],
    ) -> list[SensitivityResult]:
        """Score each subgraph by promoting, measuring, and demoting.

        :param quant_model: The quantized model (for candidate discovery).
        :param data_reader: Calibration data reader for inference.
        :param tensors_range: Data range for all quantizing tensors.
        :param subgraph_specs: Subgraph definitions to evaluate.
        :returns: Sensitivity results sorted ascending by score.
        """
        target_config_list = self._strategy.target_config_list

        def _score_spec_worker(spec: SubgraphSpec, strategy: MixingStrategy, dr: Any) -> SensitivityResult | None:
            candidates = self._candidate_nodes(quant_model, spec.resolved_nodes)
            if not candidates:
                return None

            local_proto = onnx.ModelProto()
            local_proto.CopyFrom(quant_model)
            local_model = ONNXModel(local_proto)

            all_config_scores: list[float] = []
            for layer_config in target_config_list:
                strategy.promote(local_model, candidates, tensors_range, layer_config)
                local_model.clean_initializers()
                quant_out = inference_model(
                    local_model.model, dr, self._config.data_size, self._config.metric_output_index
                )
                all_config_scores.append(self._metric_fn(self._float_out, quant_out))
                strategy.demote(local_model, quant_model)
            best_idx = min(range(len(all_config_scores)), key=lambda i: all_config_scores[i])
            return SensitivityResult(
                name=spec.name,
                candidate_nodes=candidates,
                score=all_config_scores[best_idx],
                all_config_scores=all_config_scores,
                best_config_index=best_idx,
            )

        worker_num = min(max(self._config.worker_num, 1), multiprocessing.cpu_count())
        if worker_num > 1:
            # Each worker gets its own deep-copied strategy so that concurrent
            # promote/demote calls do not race on shared internal state.
            # Each worker also gets its own shallow-copied data reader so that
            # concurrent reset_iter/get_next calls do not race on the shared
            # iterator; the underlying _data_cache list (numpy arrays) is shared
            # read-only across all copies.
            strategy_copies = [copy.deepcopy(self._strategy) for _ in subgraph_specs]
            dr_copies = [copy.copy(data_reader) for _ in subgraph_specs]
            raw = Parallel(n_jobs=worker_num, backend="threading")(
                delayed(_score_spec_worker)(spec, strategy_copy, dr_copy)
                for spec, strategy_copy, dr_copy in tqdm(
                    zip(subgraph_specs, strategy_copies, dr_copies, strict=False),
                    desc=f"Sensitivity analysis ({worker_num} workers)",
                    unit="spec",
                    total=len(subgraph_specs),
                )
            )
        else:
            raw = [
                _score_spec_worker(spec, self._strategy, data_reader)
                for spec in tqdm(subgraph_specs, desc="Sensitivity analysis", unit="spec")
            ]

        results = [r for r in raw if r is not None]
        return sorted(results, key=lambda r: r.score)

    def _candidate_nodes(self, quant_model: onnx.ModelProto, node_names: list[str] | None = None) -> list[str]:
        """Collect candidate node names filtered by config constraints.

        Filters nodes by ``target_op_type``, ``include_layers``,
        ``exclude_layers``, and optionally restricts to a given set of
        *node_names* (used in subgraph-wise analysis).

        :param quant_model: The quantized model whose graph is scanned.
        :param node_names: If provided, only consider nodes in this set.
        :returns: List of qualifying node names.
        """
        target_ops = set(self._config.target_op_type)
        included = set(self._config.include_layers)
        excluded = set(self._config.exclude_layers)
        result = []
        for i, node in enumerate(quant_model.graph.node):
            if node.op_type not in target_ops:
                continue
            if node_names is not None and node.name not in node_names:
                continue
            if excluded and node.name in excluded:
                continue
            if included and node.name not in included:
                continue
            result.append(node.name)
        return result

    @staticmethod
    def print_sensitivity_table(results: list[SensitivityResult]) -> None:
        """Log a ranked sensitivity table to the console."""
        if results is None or len(results) == 0:
            return None

        try:
            from rich.console import Console
            from rich.table import Table

            table = Table()
            table.add_column("Rank", justify="right")
            table.add_column("Name")
            table.add_column("Score", justify="right", style="bold green1")
            table.add_column("Nodes", justify="right", style="bold green1")

            for rank, r in enumerate((r for r in results if r.enabled), 1):
                scores_str = ", ".join(f"{s:.4f}" for s in r.all_config_scores)
                table.add_row(
                    str(rank),
                    r.name,
                    f"{r.score:.4f} [{scores_str}]",
                    str(len(r.candidate_nodes)),
                )

            Console().print(table)
        except Exception:
            pass


def compute_cache_key(quant_model: onnx.ModelProto, config: AutoMixprecisionConfig) -> str:
    """Compute a fingerprint that identifies the sensitivity cache for a given model and config.

    The key covers graph topology (node names, op types, and connectivity) plus the
    quantization target configuration and op-type filter, so that any of the following
    changes will invalidate a stale cache file:

    - Model structure (added/removed/renamed nodes, rewired edges)
    - ``target_layer_config`` spec change
    - ``target_op_type`` change
    - ``include_layers`` or ``exclude_layers`` change

    Weight values are intentionally excluded: sensitivity scores depend on the graph
    shape and quantization spec, not on the numeric weight values.

    :param quant_model: The quantized ONNX model.
    :param config: The ``AutoMixprecisionConfig`` used for the current run.
    :returns: A hex SHA-256 digest string.
    """
    parts: list[str] = []

    # Graph topology: sorted (name, op_type, inputs, outputs) for each node.
    for node in sorted(quant_model.graph.node, key=lambda n: (n.name, n.op_type)):
        parts.append(f"{node.name}|{node.op_type}|{','.join(node.input)}|{','.join(node.output)}")

    # Target op types (sorted for stability).
    parts.append("target_op_type:" + ",".join(sorted(config.target_op_type)))

    # Target layer config(s) serialized to JSON (keys sorted for stability).
    target = config.target_layer_config
    if isinstance(target, list):
        config_repr = json.dumps([c.to_dict() for c in target], sort_keys=True)
    elif isinstance(target, dict):
        config_repr = json.dumps(
            {json.dumps(k.to_dict(), sort_keys=True): v for k, v in target.items()}, sort_keys=True
        )
    else:
        config_repr = json.dumps(target.to_dict() if target is not None else {}, sort_keys=True)
    parts.append("target_layer_config:" + config_repr)

    # Layer inclusion/exclusion filters (sorted for stability).
    parts.append("include_layers:" + ",".join(sorted(config.include_layers)))
    parts.append("exclude_layers:" + ",".join(sorted(config.exclude_layers)))

    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def save_sensitivity_results(results: list[SensitivityResult], path: str | Path, cache_key: str) -> None:
    """Serialize sensitivity results to a JSON file.

    :param results: The ranked sensitivity results to save.
    :param path: Destination file path.
    :param cache_key: Fingerprint (from :func:`compute_cache_key`) embedded in the
        file so that :func:`load_sensitivity_results` can detect stale caches.
    """
    payload = {
        "version": version,
        "cache_key": cache_key,
        "results": [
            {
                "name": r.name,
                "candidate_nodes": r.candidate_nodes,
                "score": r.score,
                "all_config_scores": r.all_config_scores,
                "best_config_index": r.best_config_index,
                "enabled": r.enabled,
            }
            for r in results
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2))
    logger.info(f"Saved {len(results)} sensitivity result(s) to {path}")


def load_sensitivity_results(path: str | Path, valid_key: str) -> list[SensitivityResult] | None:
    """Load sensitivity results from a JSON file.

    :param path: Source file path.
    :param valid_key: The valid cache key to compare with the stored key.
    :returns: The loaded sensitivity results, if the stored key is invalidated,
        None will be returned.
    """
    payload = json.loads(Path(path).read_text())
    results = [
        SensitivityResult(
            name=entry["name"],
            candidate_nodes=entry["candidate_nodes"],
            score=entry["score"],
            all_config_scores=entry["all_config_scores"],
            best_config_index=entry["best_config_index"],
            enabled=entry.get("enabled", True),
        )
        for entry in payload["results"]
    ]
    if payload["cache_key"] != valid_key:
        logger.warning(
            "The cached sensitivity results are stale: the stored key does not match the expected key. "
            "Please check if the model or configuration parameters have changed."
        )
        return None
    logger.info(f"Loaded {len(results)} sensitivity result(s) from {path} (written by Quark {payload['version']})")
    return results
