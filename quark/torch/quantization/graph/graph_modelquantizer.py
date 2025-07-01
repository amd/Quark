#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark FX model Quantization API for PyTorch."""

import torch
import types
from pathlib import Path
import torch.nn as nn
import torch.fx
from torch.fx import GraphModule
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from onnxruntime.quantization.onnx_model import ONNXModel

# quark part
from quark.torch.quantization.api import ModelQuantizer  # TODO still inherit
from quark.torch.quantization.config.config import Config
# Step 1 prepare the fx graph module
from quark.torch.quantization.graph.processor.model_importer import get_fx_model
# Step 2 precheck
from quark.torch.quantization.graph.processor.pre_check_befor_quant import pre_quant_model_and_config_checks
# Step 3 pre_quant_optimize
from quark.torch.quantization.graph.optimization.model_optimization import trans_opsfunc_2_quant_module, apply_pre_hw_constrain_passes
# Step 4 annotation and insert quantizer
from quark.torch.quantization.graph.processor.node_annotate import mark_exclude_quant_node
from quark.torch.quantization.graph.processor.insert_quantizer import insert_quantizer
from quark.torch.quantization.graph.processor.processor import annotate
from quark.torch.quantization.graph.torch_utils import allow_exported_model_train_eval
# Step 5 PTQ/QAT/Other algom
# Step 6 post quant optimize
from quark.torch.quantization.graph.processor.processor import post_calib_optimize, post_quant_optimize
# Step 7 optmize before export
from quark.torch.quantization.graph.processor.processor import freeze_model
# export to onnx
from quark.torch import ModelExporter
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ['FxGraphQuantizer']
'''
================================
FxGraphQuantizer defines the overall quantization pipeline,
   All quantization step functions are defined here,
   NOTE: for code clarity, all detailed code should not realized in this file
================================
'''


class FxGraphQuantizer(ModelQuantizer):

    def __init__(self, config: Config, multi_device: bool = False) -> None:

        super().__init__(config, multi_device=multi_device)  # in init will check the config automatically

    # Step 1 prepare the fx graph module
    def _prepare_fx_model(
        self,
        model: Union[nn.Module, ONNXModel, GraphModule],
        args: Tuple[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ) -> GraphModule:
        return get_fx_model(model, args, kwargs, dynamic_shapes)

    # Step 2 precheck
    def _pre_check_before_quant(self, model: GraphModule) -> bool:
        '''
        TODO check whether the GraphModule satisfied enquirements
        All check should be packaged here
        '''
        return pre_quant_model_and_config_checks(model, self.config)

    # Step 3 pre_quant_optimize
    def _pre_quant_optimize(self, model: GraphModule, hw_constrain: bool = True) -> GraphModule:
        '''
        Add other optimization if applicable
        '''
        _ = mark_exclude_quant_node(model)  # NOTE may use exclude=self.config.exclude
        # TODO  based on config/opt config
        model = trans_opsfunc_2_quant_module(model)
        if hw_constrain:
            model = apply_pre_hw_constrain_passes(model=model)
        # TODO CLE, AQW and other algo here
        return model

    # Step 4 annotation and insert quantizer
    def _annotate_and_insert_quantizer(self, model: GraphModule) -> GraphModule:
        model = annotate(model=model, config=self.config)
        model = insert_quantizer(model)
        return model

    # Step 5 PTQ/QAT/Other algom
    def _ptq_qat_algo(
        self,
        model: GraphModule,
        calibdata: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                  DataLoader[Dict[str, torch.Tensor]]]] = None,
        trainer: Optional[Callable[[GraphModule], GraphModule]] = None,
    ) -> GraphModule:
        #
        model = self._do_calibration(model, calibdata)  # may move to another place
        model = post_calib_optimize(model)  # type: ignore [arg-type]
        model = trainer(model) if trainer is not None else model
        return model

    # Step 6 post quant optimize
    def _post_quant_optimize(self, model: GraphModule, hw_constrain: bool = True) -> GraphModule:
        '''
        Add other optimization if applicable
        '''
        # TODO optimization for post quantization
        return model

    # Step 7 optmize before export
    def _optimization_befor_export(self, model: GraphModule) -> GraphModule:
        model = self.freeze(model=model)
        model = post_quant_optimize(model=model, hw_constrain=True)  # type: ignore [arg-type]
        return model

    def _prepare_quantizable_model(self,
                                   model: Optional[Union[nn.Module, ONNXModel, GraphModule]],
                                   args: Tuple[Any],
                                   kwargs: Optional[Dict[str, Any]] = None,
                                   dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None) -> GraphModule:

        # Step 1 prepare the fx graph module
        fx_graph_model = self._prepare_fx_model(model, args, kwargs, dynamic_shapes)

        # Step 2 precheck
        if not self._pre_check_before_quant(fx_graph_model):
            raise Exception("model pre quant check failed")

        # Step 3 pre_quant_optimize
        model = self._pre_quant_optimize(fx_graph_model, hw_constrain=True)

        # Step 4 annotation and insert quantizer
        model = self._annotate_and_insert_quantizer(model)

        model.freeze_model = types.MethodType(freeze_model, model)  # type: ignore [assignment]

        model = allow_exported_model_train_eval(model)

        return model

    def quantize_model(  # type: ignore [override]
        self,
        model: Optional[Union[nn.Module, ONNXModel, GraphModule]],
        args: Tuple[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
        calibdata: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                  DataLoader[Dict[str, torch.Tensor]]]] = None,
        trainer: Optional[Callable[[GraphModule], GraphModule]] = None,
    ) -> GraphModule:

        # Step 1 prepare the fx graph module
        # Step 2 precheck
        # Step 3 pre_quant_optimize
        # Step 4 annotation and insert quantizer
        model = self._prepare_quantizable_model(model=model, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)

        # Step 5 PTQ/QAT/Other algom
        model = self._ptq_qat_algo(model, calibdata, trainer)

        # Step 6 post quant optimize
        model = self._post_quant_optimize(model)

        return model.eval()

    def export_onnx_model(self, model: GraphModule, input_args: Union[torch.Tensor, Tuple[float]],
                          export_dir: Path) -> None:
        # Step 7 post quant optimize
        freezed_model = self._optimization_befor_export(model)
        config = ExporterConfig(json_export_config=JsonExporterConfig())
        exporter = ModelExporter(config=config, export_dir=export_dir)
        exporter.export_onnx_model(freezed_model, input_args)
