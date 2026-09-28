#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationMethod, TensorsData
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.tensor_quant_overrides import TensorQuantOverridesHelper
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger
from quark.onnx.algorithm.finetuning.onnx_evaluate import inference_model
from quark.onnx.quantization.quant_utils import insert_quant_nodes_at_boundaries

from .metric_funcs import MetricFn
from .mixing_strategy import MixingStrategy
from .mprecision_config import AutoMixprecisionConfig
from .sensitivity_analyzer import SensitivityResult

logger = ScreenLogger(__name__)


class MixingExecutor:
    """Greedy promotion loop that upgrades candidate ops to the target precision.

    After sensitivity analysis ranks the candidates, this executor walks
    the ranking from most-sensitive to least-sensitive, promoting each
    candidate and checking the metric.  It stops as soon as the metric
    exceeds ``metric_threshold``.  When ``metric_threshold`` is ``0`` the
    threshold is disabled and all candidates are promoted unconditionally.

    :param config: AMP configuration.
    :param strategy: Optional mixing strategy override; resolved
        automatically from *config* when ``None``.
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

        self._promoted_nodes: set[str] = set()
        self._promoted_tensors: set[str] = set()

    @property
    def promoted_nodes(self) -> set[str]:
        """Node names promoted to the target precision in the last :meth:`execute` call."""
        return self._promoted_nodes

    @property
    def promoted_tensors(self) -> set[str]:
        """Return all promoted tensors in the last :meth:`execute` call."""
        return self._promoted_tensors

    def insert_dual_quant_nodes(
        self,
        result: onnx.ModelProto,
        tensors_range: TensorsData,
        extra_options: dict[str, Any],
    ) -> onnx.ModelProto:
        """Insert Q/DQ pairs at precision boundaries when dual-quant mode is enabled.

        :param result: The mixed-precision model returned by :meth:`execute`.
        :param tensors_range: Data range for all quantizing tensors.
        :param extra_options: Extra options forwarded from the AMP pipeline.
        :returns: The model with boundary Q/DQ pairs inserted, or *result* unchanged
            if dual-quant is disabled or a conflicting option is already active.
        """
        if not self._config.dual_quant_nodes:
            return result

        if extra_options.get("EnableDualQuantNodePairs", False):
            logger.warning(
                "The extra option 'EnableDualQuantNodePairs' is enabled already, "
                "can not insert quant nodes at precision boundaries again."
            )
            return result

        logger.info("Inserting quant nodes at precision boundaries (activation tensors).")

        # For simplicity, we use the first config in the list as the global config.
        global_layer_config = self._strategy.target_config_list[0]
        if len(self._strategy.target_config_list) > 1 or self._strategy.candidate_to_config:
            logger.warning(
                "Can not handle multiple target configs for inserting QDQ pairs. "
                "Use the first config as the global config for simplicity."
            )

        activation_spec = (
            global_layer_config.activation if global_layer_config.activation else global_layer_config.input_tensors
        )
        calibration_method = getattr(activation_spec, "calibration_method", CalibrationMethod.MinMax)

        # Merge the tensor quant overrides from the extra options and the promoted tensors.
        tensor_quant_overrides_dict = extra_options.get("TensorQuantOverrides", {})
        promoted_tensors_dict: dict[str, list[dict[str, Any]]] = {name: [{}] for name in self._promoted_tensors}
        tensor_quant_overrides = TensorQuantOverridesHelper({**tensor_quant_overrides_dict, **promoted_tensors_dict})

        # Merge the nodes with mixed precision from the extra options and the promoted nodes.
        nodes_with_mixed_precision_list = extra_options.get("NodesWithMixedPrecision", [])
        nodes_with_mixed_precision = list(set(nodes_with_mixed_precision_list) | self._promoted_nodes)

        return insert_quant_nodes_at_boundaries(
            result,
            tensor_quant_overrides,
            tensors_range,
            reduce_range=False,
            calibrate_method=calibration_method,
            extra_options={"NodesWithMixedPrecision": nodes_with_mixed_precision},
        )

    def execute(
        self,
        quant_model: onnx.ModelProto,
        data_reader: Any,
        tensors_range: TensorsData,
        ranked: list[SensitivityResult],
    ) -> onnx.ModelProto:
        """Run the greedy promotion loop and return the modified model.

        :param quant_model: The quantized ONNX model to modify.
        :param data_reader: Calibration data reader (may be ``None`` to skip
            threshold-based stopping).
        :param tensors_range: Data range for all quantizing tensors,
            which is used for re-compute the quantization parameters.
        :param ranked: Sensitivity results sorted from most to
            least sensitive.
        :returns: The modified quantized model with promoted candidates.
        """
        work_proto = onnx.ModelProto()
        work_proto.CopyFrom(quant_model)
        work_model = ONNXModel(work_proto)

        self._promoted_nodes = set()
        self._promoted_tensors = set()

        for candidate in tqdm(ranked, desc="Mixing precision", unit="candidate"):
            # Skip candidates explicitly disabled by the user in the cache file.
            if not candidate.enabled:
                logger.info(f"Skipped {candidate.name}: disabled in sensitivity cache.")
                continue

            # Filter out individual nodes whose input activation Q/DQ is shared;
            # only those nodes are skipped — the rest of the candidate still promotes.
            nodes_to_promote = candidate.candidate_nodes
            if self._config.no_input_qdq_shared and candidate.candidate_nodes:
                input_name_to_nodes = work_model.input_name_to_nodes()
                eligible = []
                for node_name in candidate.candidate_nodes:
                    node = next(
                        (n for n in work_model.model.graph.node if n.name == node_name),
                        None,
                    )
                    if node is not None and node.input and len(input_name_to_nodes.get(node.input[0], [])) > 1:
                        logger.info(f"Skipped {node_name}: shared activation QDQ.")
                    else:
                        eligible.append(node_name)
                nodes_to_promote = eligible

            if not nodes_to_promote:
                continue

            # Snapshot before promoting so demote can target only this candidate.
            prev_proto = onnx.ModelProto()
            prev_proto.CopyFrom(work_model.model)

            # Get the best config for the candidate.
            if candidate.best_config_index is None or candidate.best_config_index >= len(
                self._strategy.target_config_list
            ):
                logger.warning(
                    f"Best config index {candidate.best_config_index} is out of range, using the first config as the default config."
                )
                layer_config = self._strategy.target_config_list[0]
            else:
                layer_config = self._strategy.target_config_list[candidate.best_config_index]

            # Promote the candidate nodes to the target precision.
            promoted_tensors = self._strategy.promote(work_model, nodes_to_promote, tensors_range, layer_config)
            work_model.clean_initializers()
            self._promoted_nodes |= set(nodes_to_promote)
            self._promoted_tensors |= promoted_tensors

            # Check if the metric exceeds the threshold
            quant_out = inference_model(
                work_model.model, data_reader, self._config.data_size, self._config.metric_output_index
            )

            score = self._metric_fn(self._float_out, quant_out)
            if self._config.metric_threshold is None or self._config.metric_threshold == 0:
                logger.info(f"Mixing precision promoted {candidate.name}. Score {score:.4f} (no threshold)")
            elif self._config.metric_optimize_object == "speed":
                if score > self._config.metric_threshold:
                    self._strategy.demote(work_model, prev_proto)
                    self._promoted_nodes -= set(nodes_to_promote)
                    logger.info(
                        f"Mixing precision stopped at {candidate.name}. Score {score:.4f} exceeds threshold {self._config.metric_threshold}."
                    )
                    break
                else:
                    logger.info(f"Mixing precision promoted {candidate.name}. Score {score:.4f}")
            else:
                logger.info(f"Mixing precision promoted {candidate.name}. Score {score:.4f}")
                if score <= self._config.metric_threshold:
                    logger.info(
                        f"Mixing precision reached threshold at {candidate.name}. Score {score:.4f} meets threshold {self._config.metric_threshold}."
                    )
                    break

        total_candidate_nodes = sum(len(c.candidate_nodes) for c in ranked) if ranked else 0
        logger.info(
            f"Automatic Mixed Precision has promoted {len(self._promoted_nodes)} node(s) out of {total_candidate_nodes} candidate node(s)."
        )

        work_model.topological_sort()
        return work_model.model
