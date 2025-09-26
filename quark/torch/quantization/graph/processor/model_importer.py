#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# from packaging import version
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
from onnxruntime.quantization.onnx_model import ONNXModel
from torch.fx import GraphModule
from torch.nn import Module

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["ONNXImporter", "TorchModuleImporter", "FXModuleImporter", "get_fx_model"]


class GraphImporter:
    @abstractmethod
    def apply(
        self,
        model: Any,
        args: tuple[Any],
        kwargs: dict[str, Any] | None = None,
        dynamic_shapes: Union[dict[str, Any], tuple[Any]] | None = None,
    ) -> GraphModule:
        pass


class ONNXImporter(GraphImporter):
    """
    Transper a Onnx model to torch.fx.GraphModule
    """

    # def _pre_check(self, onnx_model: ONNXModel, args: Tuple[Any], kwargs: Optional[Dict[str, Any]] = None) -> bool:
    #     '''
    #     TODO check the model whether satisfy some condition
    #     '''
    #     return True

    def apply(
        self,
        model: ONNXModel,
        args: tuple[Any],
        kwargs: dict[str, Any] | None = None,
        dynamic_shapes: Union[dict[str, Any], tuple[Any]] | None = None,
    ) -> GraphModule:
        # if not self._pre_check(model, args, kwargs):
        #     raise ValueError("This Model ******* need check")  # TODO
        return None  # type: ignore [return-value]


class TorchModuleImporter(GraphImporter):
    """
    Transper a torch.nn. model to torch.fx.GraphModule
    """

    def _pre_check(self, torch_model: Module, args: tuple[Any], kwargs: dict[str, Any] | None = None) -> bool:
        """
        TODO check the model whether satisfy some condition
            e.g device checkt, shape check etc.
        """
        return True

    def apply(
        self,
        model: Module,
        args: tuple[Any],
        kwargs: dict[str, Any] | None = None,
        dynamic_shapes: Union[dict[str, Any], tuple[Any]] | None = None,
    ) -> GraphModule:
        if not self._pre_check(model, args, kwargs):
            raise ValueError("This torch.nn.Module is not supported please check")
        graph_module: GraphModule
        graph_module = torch.export.export_for_training(
            mod=model, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
        ).module()  # type: ignore
        return graph_module


class FXModuleImporter(GraphImporter):
    """
    Transper a torch.nn. model to torch.fx.GraphModule
    """

    def _pre_check(self, fx_model: GraphModule, args: tuple[Any], kwargs: dict[str, Any] | None = None) -> bool:
        """
        TODO check the model whether satisfy some condition
            e.g device checkt, shape check etc.
        """
        return True

    def apply(
        self,
        model: GraphModule,
        args: tuple[Any],
        kwargs: dict[str, Any] | None = None,
        dynamic_shapes: Union[dict[str, Any], tuple[Any]] | None = None,
    ) -> GraphModule:
        if not self._pre_check(model, args, kwargs):
            raise ValueError("This fx model not supported")
        return model


def get_fx_model(
    model: Any,
    args: tuple[Any],
    kwargs: dict[str, Any] | None = None,
    dynamic_shapes: Union[dict[str, Any], tuple[Any]] | None = None,
) -> GraphModule:
    model_importer: GraphImporter
    if isinstance(model, GraphModule):
        model_importer = FXModuleImporter()
    elif isinstance(model, ONNXModel):
        # transfer the onnx model to fx graph
        model_importer = ONNXImporter()
    elif isinstance(model, torch.nn.Module):
        model_importer = TorchModuleImporter()
    else:
        logger.error("Unrecognized model type, can not perform quantization, Please Check")
        raise ValueError(f"Not supported model type {type(model)}")
    return model_importer.apply(model, args, kwargs, dynamic_shapes)
