#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Transformations pipeline for onnx model conversion."""

from typing import Any, Tuple

import onnx

from quark.onnx.graph_transformations import model_transformer, transforms_pipeline
from quark.onnx.optimizations import convert_transforms as convert_transforms_mod

TransformsPipeline = transforms_pipeline.TransformsPipeline


class ConvertQDQToQOPTransformsPipeline(TransformsPipeline):
    """Convert QDQ to QOperator transformations pipeline."""

    def apply(self, model: onnx.ModelProto, candidate_nodes: Any, node_metadata: Any) -> tuple[onnx.ModelProto, Any]:
        """Implement the transforms.

        Args:
            model: Onnx model to be quantized.

        Returns:
            Conveted onnx model.
        """
        configs = self.get_configs()

        convert_transforms = [
            #  convert_transforms_mod.ConvQDQToQOPTransform(),
            convert_transforms_mod.MatMulQDQToQOPTransform(),
            convert_transforms_mod.AddQDQToQOPTransform(),
            convert_transforms_mod.MulQDQToQOPTransform(),
            convert_transforms_mod.SigmoidQDQToQOPTransform(),
        ]
        converted_model, metadata = model_transformer.ModelTransformer(
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
        configs = self.get_configs()

        convert_transforms = [
            convert_transforms_mod.RemoveQDQTransform(),
        ]
        converted_model, metadata = model_transformer.ModelTransformer(
            model, convert_transforms, candidate_nodes, node_metadata
        ).transform()
        return converted_model, metadata
