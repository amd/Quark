#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from collections import OrderedDict
from typing import Any

import onnx
from onnx import ModelProto

from .onnx_adapter_pass import ONNXAdapterPass
from .passes import (
    ONNXConvertBNToConvPass,
    ONNXConvertClipToReluPass,
    ONNXConvertFP16ToFP32Pass,
    ONNXConvertNCHWToNHWCPass,
    ONNXConvertOpsetVersionPass,
    ONNXConvertReduceMeanToGlobalAvgPoolPass,
    ONNXConvertSplitToSlicePass,
    ONNXCopyBiasInitPass,
    ONNXCopySharedInitPass,
    ONNXFixShapesPass,
    ONNXFoldBatchNormAfterConcatPass,
    ONNXFoldBatchNormPass,
    ONNXFuseGeluPass,
    ONNXFuseInstanceNormPass,
    ONNXFuseL2NormPass,
    ONNXFuseLayerNormPass,
    ONNXOptimizeWithORTPass,
    ONNXRemoveInputInitPass,
    ONNXSimplifyPass,
    ONNXSplitLargeKernelPoolPass,
    RemoveQDQ,
)


class Engine:
    """The engine executes the registered Passes.

    It facilitate evaluation of the output models using provided evaluation criteria and produces output model(s).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config
        self._passes_registry: dict[str, type[ONNXAdapterPass]] = OrderedDict()
        self._initialized: bool = False

    def initialize(self) -> None:
        """Initialize engine state. This should be done before running the registered passes."""

        # TODO: write code to iterate over the passes folder and call self.register with the first argument as the filename without extension as string and second argument is the object obtained from instantiating the class that inherits from ONNXAdapterPass
        # e.g. `self.register("onnx_convert_opset_version", ONNXConvertOpsetVersionPass())` for passes/onnx_convert_opset_version.py which contains `ONNXConvertOpsetVersionPass(ONNXAdapterPass)` class
        self.register("onnx_convert_opset_version", ONNXConvertOpsetVersionPass)
        self.register("onnx_convert_nchw_to_nhwc", ONNXConvertNCHWToNHWCPass)
        self.register("onnx_simplify", ONNXSimplifyPass)
        self.register("onnx_convert_bn_to_conv", ONNXConvertBNToConvPass)
        self.register("onnx_convert_clip_to_relu", ONNXConvertClipToReluPass)
        self.register("onnx_convert_fp16_to_fp32", ONNXConvertFP16ToFP32Pass)
        self.register("onnx_remove_input_init", ONNXRemoveInputInitPass)
        self.register("onnx_copy_shared_init", ONNXCopySharedInitPass)
        self.register("onnx_copy_bias_init", ONNXCopyBiasInitPass)
        self.register("onnx_fix_shapes", ONNXFixShapesPass)
        self.register("onnx_fold_batch_norm", ONNXFoldBatchNormPass)
        self.register("onnx_fold_batch_norm_after_concat", ONNXFoldBatchNormAfterConcatPass)
        self.register("onnx_optimize_with_ort", ONNXOptimizeWithORTPass)
        self.register("onnx_fuse_gelu", ONNXFuseGeluPass)
        self.register("onnx_fuse_l2_norm", ONNXFuseL2NormPass)
        self.register("onnx_fuse_layer_norm", ONNXFuseLayerNormPass)
        self.register("onnx_fuse_instance_norm", ONNXFuseInstanceNormPass)
        self.register("onnx_convert_reduce_mean_to_global_avg_pool", ONNXConvertReduceMeanToGlobalAvgPoolPass)
        self.register("onnx_split_large_kernel_pool", ONNXSplitLargeKernelPoolPass)
        self.register("onnx_convert_split_to_slice", ONNXConvertSplitToSlicePass)
        self.register("remove_qdq", RemoveQDQ)

        self._initialized = True

    def register(
        self,
        name: str,
        pass_obj: type[ONNXAdapterPass],
    ) -> None:
        """Register a pass configuration so that it could be instantiated and executed later."""
        self._passes_registry[name] = pass_obj

    def run(self, float_model: ModelProto | None = None) -> ModelProto | None:
        """Run all the registered passes on the input model and produce one or more (intermediate) output models."""
        if float_model is None:
            model = onnx.load(self._config["input_model_path"])
        else:
            model = float_model
        for pass_ in self._config["passes"]:
            pass_class = self._passes_registry[pass_]
            pass_instance = pass_class(self._config["passes"][pass_])
            # TODO: Add code that if `Engine.save_intermediate_models is True`, save the model after each pass for debugging purposes
            model = pass_instance._run_for_config(model, self._config["passes"][pass_])
        if float_model is not None:
            return model
        onnx.save(model, self._config["output_model_path"])
        return None
