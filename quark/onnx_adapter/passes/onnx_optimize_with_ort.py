#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Any

import onnx
from onnx import ModelProto
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from quark.onnx.utils.system_utils import create_tmp_dir
from quark.onnx_adapter.onnx_adapter_pass import ONNXAdapterPass
from quark.onnx_adapter.pass_config import PassConfigParam
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ONNXOptimizeWithORTPass(ONNXAdapterPass):
    """ONNX Adapter pass that optimizes an ONNX model using ONNXRuntime.

    This pass loads the model into ONNXRuntime with optimization enabled,
    triggers the graph optimization process, and then loads the optimized
    model back for further processing.
    """

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Return the default configuration for this optimization pass.

        The configuration currently supports:

        - ``optimize_with_ort``: Whether to apply ONNXRuntime optimization.
        """
        config = {
            "optimize_with_ort": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to optimize the input ONNX model with official ONNXRuntime.",
            )
        }
        config.update(self.config)
        return config

    def _onnx_optimize_with_ort(self, model: ModelProto) -> ModelProto:
        """Optimize an ONNX model using ONNXRuntime's built-in optimization pipeline.

        Steps:
        1. Serialize the model in memory.
        2. Initialize an ONNXRuntime session with basic optimizations enabled.
        3. Let ORT write out the optimized graph to a temporary file.
        4. Load and return the optimized model.

        Args:
            model: The input ONNX ModelProto to optimize.

        Returns:
            A new optimized ONNX ModelProto.
        """
        temp_dir = create_tmp_dir(prefix="onnx_adapter.optimize_with_ort.")
        temp_path = Path(temp_dir.name).joinpath("opt_model.onnx")
        model_bytes = model.SerializeToString()
        sess_option = SessionOptions()
        sess_option.optimized_model_filepath = temp_path.as_posix()
        sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        kwargs: dict[Any, Any] = {}
        _ = InferenceSession(model_bytes, sess_option, providers=["CPUExecutionProvider"], **kwargs)
        opt_model = onnx.load(temp_path)
        return opt_model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Execute this pass on the supplied model based on configuration.

        If ``optimize_with_ort`` is enabled, the model will be passed through
        ONNX Runtime for graph optimization. Otherwise, a warning is logged.

        Args:
            model: The ONNX model to process.
            config: The resolved pass configuration parameters.

        Returns:
            The optimized model, or the original model if optimization is disabled.
        """
        if "optimize_with_ort" in config and config["optimize_with_ort"]:
            model = self._onnx_optimize_with_ort(model)
        else:
            logger.warning(
                "Please ensure that the optimize_with_ort pass contains the optimize_with_ort parameter and it is True."
            )
        return model
