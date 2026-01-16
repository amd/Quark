#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from .onnx_convert_bn_to_conv import ONNXConvertBNToConvPass
from .onnx_convert_clip_to_relu import ONNXConvertClipToReluPass
from .onnx_convert_fp16_to_fp32 import ONNXConvertFP16ToFP32Pass
from .onnx_convert_nchw_to_nhwc import ONNXConvertNCHWToNHWCPass
from .onnx_convert_opset_version import ONNXConvertOpsetVersionPass
from .onnx_convert_reduce_mean_to_global_avg_pool import ONNXConvertReduceMeanToGlobalAvgPoolPass
from .onnx_convert_split_to_slice import ONNXConvertSplitToSlicePass
from .onnx_copy_bias_init import ONNXCopyBiasInitPass
from .onnx_copy_shared_init import ONNXCopySharedInitPass
from .onnx_fix_shapes import ONNXFixShapesPass
from .onnx_fold_batch_norm import ONNXFoldBatchNormPass
from .onnx_fold_batch_norm_after_concat import ONNXFoldBatchNormAfterConcatPass
from .onnx_fuse_gelu import ONNXFuseGeluPass
from .onnx_fuse_instance_norm import ONNXFuseInstanceNormPass
from .onnx_fuse_l2_norm import ONNXFuseL2NormPass
from .onnx_fuse_layer_norm import ONNXFuseLayerNormPass
from .onnx_optimize_with_ort import ONNXOptimizeWithORTPass
from .onnx_remove_input_init import ONNXRemoveInputInitPass
from .onnx_simplify import ONNXSimplifyPass
from .onnx_split_large_kernel_pool import ONNXSplitLargeKernelPoolPass
from .remove_qdq import RemoveQDQ

__all__ = [
    "ONNXConvertOpsetVersionPass",
    "ONNXConvertNCHWToNHWCPass",
    "ONNXSimplifyPass",
    "ONNXConvertBNToConvPass",
    "ONNXConvertFP16ToFP32Pass",
    "ONNXRemoveInputInitPass",
    "ONNXCopySharedInitPass",
    "ONNXCopyBiasInitPass",
    "ONNXFixShapesPass",
    "ONNXFoldBatchNormPass",
    "ONNXOptimizeWithORTPass",
    "ONNXFuseGeluPass",
    "ONNXFuseL2NormPass",
    "ONNXFuseLayerNormPass",
    "ONNXFuseInstanceNormPass",
    "ONNXFoldBatchNormAfterConcatPass",
    "ONNXConvertReduceMeanToGlobalAvgPoolPass",
    "ONNXSplitLargeKernelPoolPass",
    "ONNXConvertClipToReluPass",
    "ONNXConvertSplitToSlicePass",
    "RemoveQDQ",
]
