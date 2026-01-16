#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Transformations pipeline for onnx model conversion."""

from typing import Any

import onnx

from .convert_transforms import (
    AddQDQToQOPTransform,
    MatMulQDQToQOPTransform,
    MulQDQToQOPTransform,
    RemoveQDQTransform,
    SigmoidQDQToQOPTransform,
)
from .model_transformer import ModelTransformer
from .transforms_pipeline import TransformsPipeline


class ConvertQDQToQOPTransformsPipeline(TransformsPipeline):
    """Convert QDQ to QOperator transformations pipeline."""

    def apply(self, model: onnx.ModelProto, candidate_nodes: Any, node_metadata: Any) -> tuple[onnx.ModelProto, Any]:
        """Implement the transforms.

        Args:
            model: Onnx model to be quantized.

        Returns:
            Conveted onnx model.
        """
        convert_transforms = [
            # ConvQDQToQOPTransform(),
            MatMulQDQToQOPTransform(),
            AddQDQToQOPTransform(),
            MulQDQToQOPTransform(),
            SigmoidQDQToQOPTransform(),
        ]
        converted_model, metadata = ModelTransformer(
            model, convert_transforms, candidate_nodes, node_metadata
        ).transform()
        return converted_model, metadata


class RemoveQDQTransformsPipeline(TransformsPipeline):
    """Remove QDQ pairs transformations pipeline."""

    def apply(self, model: onnx.ModelProto, candidate_nodes: Any, node_metadata: Any) -> tuple[onnx.ModelProto, Any]:
        """Implement the transforms.

        Args:
            model: Onnx model to be quantized.

        Returns:
            Conveted onnx model.
        """
        convert_transforms = [
            RemoveQDQTransform(),
        ]
        converted_model, metadata = ModelTransformer(
            model, convert_transforms, candidate_nodes, node_metadata
        ).transform()
        return converted_model, metadata
