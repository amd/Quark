#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization API for PyTorch."""

import torch
import torch.nn as nn
import torch.fx
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, Union, List, Tuple, Iterable, Type
from dataclasses import fields
from quark.shares.utils.import_utils import is_transformers_available
from quark.torch.quantization.config.type import QuantizationMode, Dtype, QSchemeType
from quark.torch.quantization.config.config_verification import check_and_adjust_quant_config
from quark.torch.quantization.model_transformation import process_model_transformation
from quark.torch.quantization.config.config import Config, QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.config_verification import init_quantization_config, verify_quantization_spec
from quark.torch.quantization.graph.processor.processor import prepare_quant_model, post_calib_optimize
from quark.torch.quantization.graph.processor.pre_check_befor_quant import check_supported_model_and_config
from quark.torch.quantization.graph.processor.processor import post_quant_optimize
from quark.torch.quantization.utils import set_op_by_name, get_op_by_name
from quark.torch.quantization.nn.modules.mixin import QuantMixin
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, ScaledFakeQuantize, NonScaledFakeQuantize, SequentialQuantize, enable_or_disable_quantizer
from quark.torch.quantization.utils import deep_compare, count_calibration_tokens
from quark.shares.utils.log import ScreenLogger, log_errors
from quark.shares.utils.import_utils import is_safetensors_available
from quark.torch.algorithm.api import apply_pre_quantization_optimization, apply_advanced_quant_algo, add_algorithm_config_by_model
from quark.torch.algorithm.utils.utils import clear_memory
import logging
from quark.torch.quantization.nn.modules import QuantConv2d, QuantConvTranspose2d, QuantLinear, QuantEmbedding, QuantEmbeddingBag
from quark.torch.utils.pack import create_pack_method
import quark.torch.kernel

if is_transformers_available():
    from transformers.feature_extraction_utils import BatchFeature

import os
from pathlib import Path
from collections import Counter

from quark.torch.quantization.debug import (insert_stats_hooks, collect_quantization_statistics, check_scale_stats)

if is_safetensors_available():
    from safetensors.torch import load_file

__all__ = ["ModelQuantizer", "load_params"]

logger = ScreenLogger(__name__)

QUARK_QUANT_OPS: Dict[str, Type[Union[QuantConv2d, QuantConvTranspose2d, QuantLinear, QuantEmbedding,
                                      QuantEmbeddingBag]]] = {
                                          "QuantConv2d": QuantConv2d,
                                          "QuantConvTranspose2d": QuantConvTranspose2d,
                                          "QuantLinear": QuantLinear,
                                          "QuantEmbedding": QuantEmbedding,
                                          "QuantEmbeddingBag": QuantEmbeddingBag,
                                      }


class ModelQuantizer:
    """
    Provides an API for quantizing deep learning models using PyTorch.

    This class handles the configuration and processing of the model for quantization based on user-defined parameters. It is essential to ensure that the 'config' provided has all necessary quantization parameters defined. This class assumes that the model is compatible with the quantization settings specified in 'config'.

    :param Config config: The model quantization configuration.
    """

    def __init__(self, config: Config, multi_device: bool = False) -> None:
        self.config = config
        self.is_all_dynamic: Optional[bool] = None
        self.is_weight_only: Optional[bool] = None
        self.is_act_dynamic: Optional[bool] = None
        self.is_act_contain_scale_per_tensor: Optional[bool] = None
        self._is_accelerate: Optional[bool] = None
        self.multi_device: bool = multi_device
        self.init_config()

    def set_logging_level(self) -> None:
        if self.config.log_severity_level == 0:
            ScreenLogger.set_shared_level(logging.DEBUG)
        elif self.config.log_severity_level == 1:
            ScreenLogger.set_shared_level(logging.INFO)
        elif self.config.log_severity_level == 2:
            ScreenLogger.set_shared_level(logging.WARNING)
        elif self.config.log_severity_level == 3:
            ScreenLogger.set_shared_level(logging.ERROR)
        else:
            ScreenLogger.set_shared_level(logging.CRITICAL)

    def quantize_model(
        self,
        model: nn.Module,
        dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                   DataLoader[Dict[str, torch.Tensor]], DataLoader[List["BatchFeature"]]]] = None
    ) -> nn.Module:
        """
        Quantizes the given PyTorch model to optimize its performance and reduce its size.

        The dataloader is used to provide data necessary for calibration during the quantization process. Depending on the type of data provided (either tensors directly or structured as lists or dictionaries of tensors), the function will adapt the quantization approach accordingly.

        It is important that the model and dataloader are compatible in terms of the data they expect and produce. Misalignment in data handling between the model and the dataloader can lead to errors during the quantization process.

        :param torch.nn.Module model: The PyTorch model to be quantized. This model should be already trained and ready for quantization.

        :param Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]], DataLoader[Dict[str, torch.Tensor]], DataLoader[List[BatchFeature]]]] dataloader: The ``torch.utils.data.DataLoader`` providing data that the quantization process will use for calibration. This can be a simple ``DataLoader`` returning tensors, or a more complex structure returning either a list of dictionaries or a dictionary of tensors.

        :return: The quantized version of the input model. This model is now optimized for inference with reduced size and potentially improved performance on targeted devices.
        :rtype: torch.nn.Module

        Example:

        .. code-block:: python

            # Model & Data preparation
            from torch.utils.data import DataLoader
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from quark.torch.quantization.config.config import Config
            from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
            from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver

            from quark.torch import ModelQuantizer

            model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

            quant_spec = QuantizationSpec(
                dtype=Dtype.uint4,
                observer_cls=PerGroupMinMaxObserver,
                symmetric=False,
                scale_type=ScaleType.float,
                round_method=RoundType.half_even,
                qscheme=QSchemeType.per_group,
                ch_axis=1,
                is_dynamic=False,
                group_size=128
            )
            quant_config = Config(global_quant_config=QuantizationConfig(weight=quant_spec))

            text = "Hello, how are you?"
            tokenized_outputs = tokenizer(text, return_tensors="pt")
            calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

            quantizer = ModelQuantizer(quant_config)
            quant_model = quantizer.quantize(model, calib_dataloader)
        """
        logger.info(f"Quantizing with the quantization configuration:\n{self.config}")

        # Step0-1: Pre quant device check
        self._check_model_device(model)

        # Step0-2: Enhance config
        self._generate_complete_config_by_model(model, dataloader)

        # Step1[optional]: Pre quant optimization
        model = self._apply_pre_quantization_optimization(model, dataloader)

        # Step2: Prepare quantization model for graph mode and eager mode
        model = self._prepare_model(model)

        # Step3[optional]: Apply Advanced quant algo such as gptq/awq ...
        model = self._apply_advanced_quant_algo(model, dataloader)

        # Step4[optional]: Do calibration
        model = self._do_calibration(model, dataloader)

        # Step5[optional]: Post calib optimization
        model = self._do_post_calib_optimazation(model)

        # Optionally, collect statistics on the quantization errors over the network weights/activations.
        if os.environ.get("QUARK_DEBUG", None) is not None:
            log_dir = Path(os.environ["QUARK_DEBUG"])
            log_dir.mkdir(parents=True, exist_ok=True)

            stats: Dict[str, Any] = {}
            dataloader = dataloader if not self.is_all_dynamic else None

            with insert_stats_hooks(model, stats, log_dir):
                collect_quantization_statistics(model, dataloader, stats, log_dir)

        # Check the scale of the quantized model.
        if os.getenv("QUARK_CHECK_SCALE") == "1":
            check_scale_stats(model, self.config)

        return model

    def _check_model_device(self, model: nn.Module) -> None:
        # using accelerate cause, device can not be cpu or disk, temporarily
        if hasattr(model, 'hf_device_map'):
            if not self.multi_device:
                for _, layer_device in model.hf_device_map.items():
                    if layer_device == "cpu" or layer_device == "disk":
                        # TODO: We should handle this for customers.
                        raise MemoryError(
                            "Out of memory. The available GPU memory is insufficient to load the entire model. You can try adding '--multi_device' "
                        )

            self._is_accelerate = True
        else:
            self._is_accelerate = False

    def _generate_complete_config_by_model(
        self, model: nn.Module, dataloader: Union[DataLoader[torch.Tensor], DataLoader[list[dict[str, torch.Tensor]]],
                                                  DataLoader[dict[str, torch.Tensor]], DataLoader[List["BatchFeature"]],
                                                  None]
    ) -> None:
        """
        Generates a complete configuration based on the provided model and dataloader.
        """
        self.config = add_algorithm_config_by_model(model, dataloader, self.config)

    @staticmethod
    def freeze(model: Union[nn.Module, torch.fx.GraphModule]) -> Union[nn.Module, torch.fx.GraphModule]:
        """
        Freezes the quantized model by replacing ``FakeQuantize`` modules with ``FreezedFakeQuantize`` modules.`

        In order to be able to compile a quantized model through ``torch.compile``, this method needs to be applied.

        :param torch.nn.Module model: The neural network model containing quantized layers.

        :return: The modified model with ``FakeQuantize`` modules replaced by ``FreezedFakeQuantize`` modules.
        :rtype: torch.nn.Module
        """
        logger.info("Freeze model start.")
        # ----replace FakeQuantize to FreezedFakeQuantize --------------
        named_modules = dict(model.named_modules(remove_duplicate=False))
        for name, module in named_modules.items():
            if isinstance(module, FakeQuantizeBase):
                if module.is_dynamic:
                    # TODO: Add freeze for dynamic model
                    logger.warning("Cannot freeze dynamic quantize model for now. Keep use FakeQuantize.")
                    pass
                else:
                    freezed_quantized_module = module.to_freezed_module()
                    set_op_by_name(model, name, freezed_quantized_module)

        # ----if model is quantized in fx.graph mode--------------
        if isinstance(model, torch.fx.GraphModule):
            model = model.freeze_model()
            assert isinstance(model, torch.fx.GraphModule)
            model = post_quant_optimize(model=model, hw_constrain=True)  # TODO pass argument

        logger.info("Freeze model end.")
        return model

    def _apply_pre_quantization_optimization(
        self,
        model: nn.Module,
        dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                   DataLoader[Dict[str, torch.Tensor]], DataLoader[List["BatchFeature"]]]] = None
    ) -> nn.Module:
        return apply_pre_quantization_optimization(model, self.config, dataloader=dataloader)

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        if self.config.quant_mode is QuantizationMode.eager_mode:
            return process_model_transformation(model, self.config)
        elif self.config.quant_mode is QuantizationMode.fx_graph_mode:
            # Quantization with torch.fx does not support some quantization config and some FX graphs.
            # This raises an error if the config / model used are not supported.
            check_supported_model_and_config(model, self.config)  # type: ignore [arg-type]

            return prepare_quant_model(model, self.config).eval()  # type: ignore [arg-type]

    def _apply_advanced_quant_algo(
        self,
        model: nn.Module,
        dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                   DataLoader[Dict[str, torch.Tensor]], DataLoader[List["BatchFeature"]]]] = None
    ) -> nn.Module:
        return apply_advanced_quant_algo(model, self.config, self._is_accelerate, dataloader)

    def _check_token_distribution(
        self, model: nn.Module, dataloader: Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                                  DataLoader[Dict[str, torch.Tensor]], DataLoader[List["BatchFeature"]]]
    ) -> None:
        """
        A helper function that warns when a MoE module
        received 0 token throughout the calibration process.
        """
        threshold = float(os.environ['TOKEN_DISTRIBUTION_THRESHOLD']) if os.getenv(
            "TOKEN_DISTRIBUTION_THRESHOLD") is not None else 0.0
        assert 0.0 <= threshold <= 1.0, 'threshold should be in [0.0, 1.0]'
        total_token_count = count_calibration_tokens(dataloader)
        if total_token_count == 0:
            logger.warning("No tokens found in calibration dataset. "
                           "Skipping token distribution check.")
            return

        # Get the observer token count for each module
        token_counts: Counter[str] = Counter()
        for name, module in model.named_modules():
            if isinstance(module, ScaledFakeQuantize):
                if '_input_quantizer' in name:
                    if module.observer._num_observed_tokens is not None:
                        token_counts[name.replace("._input_quantizer", "")] = module.observer._num_observed_tokens

        for module_name, token_count in token_counts.items():
            if (token_count / float(total_token_count)) <= threshold:
                logger.warning(f"The module: {module_name} "
                               f"received {token_count} tokens less than {threshold * 100:.1f}% "
                               f"of all {total_token_count} calibration tokens.")

    # when using multi_device, you must add it here or offload will fail.
    # The gpu memory used for gradients cannot be cleaned up by torch.cuda.empty_cache()
    @torch.no_grad()
    def _do_calibration(
        self,
        model: nn.Module,
        dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                   DataLoader[Dict[str, torch.Tensor]], DataLoader[List["BatchFeature"]]]] = None
    ) -> nn.Module:
        # just calib, turn off quantize
        if self.is_all_dynamic:  # TODO: to be deperated
            logger.info("Dynamic quantization, no calibration.")
        elif self.is_weight_only or (self.is_act_dynamic and not self.is_act_contain_scale_per_tensor):
            logger.info("Weight calibration start.")
            for module in model.modules():
                if isinstance(module, ScaledFakeQuantize):
                    module.enable_observer()
                    module.disable_fake_quant()

            # Simply run through the observers to set min_val, max_val, scale and zero_point buffers for the weight and bias.
            named_modules = dict(model.named_modules(remove_duplicate=False))
            for name, module in tqdm(named_modules.items()):
                if isinstance(module, QuantMixin):
                    if module._weight_quantizer is not None and isinstance(module._weight_quantizer,
                                                                           (ScaledFakeQuantize, SequentialQuantize)):
                        weight_quantizers: Union[List[ScaledFakeQuantize], SequentialQuantize] = [
                            module._weight_quantizer
                        ] if isinstance(module._weight_quantizer, ScaledFakeQuantize) else module._weight_quantizer
                        if all(quantizer.scale.numel() == 1 and quantizer.scale.item() == 1
                               for quantizer in weight_quantizers):
                            # This condition prevents layers that have already been quantized from being quantized again.
                            if module.weight.device == torch.device("meta"):
                                weight = module._hf_hook.weights_map["weight"].data
                                weight = module.get_quant_weight(weight.to(module._hf_hook.execution_device))
                                del weight
                            else:
                                _ = module.get_quant_weight(module.weight)
                    if module._bias_quantizer is not None and isinstance(module._bias_quantizer,
                                                                         (ScaledFakeQuantize, SequentialQuantize)):
                        bias_quantizers: Union[List[ScaledFakeQuantize], SequentialQuantize] = [
                            module._bias_quantizer
                        ] if isinstance(module._bias_quantizer, ScaledFakeQuantize) else module._bias_quantizer
                        if all(quantizer.scale.numel() == 1 and quantizer.scale.item() == 1
                               for quantizer in bias_quantizers):
                            if module.bias.device == torch.device("meta"):
                                bias = module._hf_hook.weights_map["bias"].data
                                _ = module.get_quant_bias(bias.to(module._hf_hook.execution_device))
                                del bias
                            else:
                                _ = module.get_quant_bias(module.bias)
                    torch.cuda.empty_cache()
            clear_memory()
            logger.info("Weight calibration end.")
        else:
            logger.info("Calibration start.")
            for module in model.modules():
                if isinstance(module, ScaledFakeQuantize):
                    module.enable_observer()
                    module.disable_fake_quant()

            assert dataloader is not None

            with torch.no_grad():
                for data in tqdm(dataloader):
                    if isinstance(data, dict):  # pragma: no cover
                        model(**data)
                    elif is_transformers_available() and isinstance(data, BatchFeature):  # pragma: no cover
                        _ = model(**data)
                    else:
                        model(data)

            self._check_token_distribution(model, dataloader)

            clear_memory()
            logger.info("Calibration end.")
        logger.info("Model quantization has been completed.")

        # step5[optional]: do evaluation, turn on quantize
        if (self.config.algo_config) and self.config.algo_config.name in ['gptq'] and hasattr(
                self.config.algo_config, "static_groups") and self.config.algo_config.static_groups is False:
            logger.warning(
                "Dynamic groups in GPTQ (static_groups=false) does not support FakeQuantize for export, turn off FakeQuantize for weight while keeping open FakeQuantize for activation in order to run evaluations."
            )
            named_modules = dict(model.named_modules(remove_duplicate=False))
            for _, module in tqdm(named_modules.items()):
                if isinstance(module, QuantMixin):
                    if module._weight_quantizer is not None:
                        enable_or_disable_quantizer(module._weight_quantizer, enable=False)

                    if module._input_quantizer is not None:
                        enable_or_disable_quantizer(module._input_quantizer, enable=True)

                    if module._output_quantizer is not None:
                        enable_or_disable_quantizer(module._output_quantizer, enable=True)
        else:
            for name, module in model.named_modules():
                if isinstance(module, ScaledFakeQuantize):
                    if module.is_dynamic and not (
                            module.is_scale_quant and module.qscheme == QSchemeType.per_tensor
                    ):  # For dynamic quantization, observer should be enable and update qparam every time.
                        module.enable_observer()
                        module.enable_fake_quant()
                    else:
                        module.disable_observer()
                        module.enable_fake_quant()
                elif isinstance(module, NonScaledFakeQuantize):
                    module.enable_fake_quant()
        return model

    def _do_post_calib_optimazation(self, model: nn.Module) -> nn.Module:
        '''
        In some case:
            1. After calibration: get weight, activation and bias scale
            2. Some hw constrain need let: bias_scale = weight_scale * act_scale
        After calibration, we need to do some optimization, and then perform QAT/export.
        '''
        if self.config.quant_mode is QuantizationMode.eager_mode:
            # remain this API TODO
            assert isinstance(model, nn.Module)
            return model
        elif self.config.quant_mode is QuantizationMode.fx_graph_mode:
            '''
            In calibration: observer will record tensor's distribution. Scale and ZP will be calculated.
            In some hardware constrain case.
                e.g. b_scale = w_scale * a_scale  (we need to modify bias_scale after calibration)
            '''
            assert isinstance(model, torch.fx.GraphModule)
            model = post_calib_optimize(model)
        return model  # type: ignore[no-any-return]

    def init_config(self) -> None:
        self.set_logging_level()  # set log level: default info
        logger.info("Configuration checking start.")
        config = self.config
        verify_quantization_spec(config)
        # TODO: Verify quant algo

        for field in fields(Config):
            if field.name in ["global_quant_config"]:
                quantization_config = getattr(config, field.name)
                _config = check_and_adjust_quant_config(quantization_config)
                setattr(self.config, field.name, _config)
                self.is_all_dynamic, self.is_weight_only, self.is_act_dynamic, self.is_act_contain_scale_per_tensor = \
                    init_quantization_config(quantization_config)
            elif field.name in ["layer_type_quant_config", "layer_quant_config"]:
                quantization_config_list = getattr(config, field.name)
                for quantization_config in quantization_config_list.values():
                    self.is_all_dynamic, self.is_weight_only, self.is_act_dynamic, self.is_act_contain_scale_per_tensor = \
                        init_quantization_config(quantization_config)

        if self.is_weight_only:
            config_parsing_result = 'weight only quantization'
        else:
            if self.is_act_dynamic:
                config_parsing_result = 'weight quantization and activation dynamic quantization'
            else:
                config_parsing_result = 'weight quantization and activation static quantization'
        logger.info(f"Configuration checking end. The configuration is effective. This is {config_parsing_result}.")


def get_name_and_info(model_info: Dict[str, Any], parent_key: str = "") -> Iterable[Tuple[str, Dict[str, Any]]]:
    for key, value in model_info.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            if value.get("type", None) is not None and value.get("weight", None) is not None:
                yield new_key, value
            else:
                yield from get_name_and_info(value, new_key)
        else:
            continue


# TODO: This function is only used in load_params, add support for SequentialQuantize later
def from_float_and_dict(float_module: nn.Module,
                        quant_info: Dict[str, Any],
                        param_dict: Dict[str, torch.Tensor],
                        layer_name: str,
                        compressed: bool = False,
                        reorder: bool = True) -> nn.Module:
    input_tensors = None
    quant_params: Dict[str, Optional[torch.Tensor]] = {}
    if quant_info.get("input_quant", None) is not None:
        input_tensors = QuantizationSpec.from_dict(quant_info["input_quant"])
        quant_params["input_scale"] = param_dict[layer_name + ".input_scale"]  # pragma: no cover
        quant_params["input_zero_point"] = param_dict[layer_name + ".input_zero_point"]  # pragma: no cover

    output_tensors = None
    if quant_info.get("output_quant", None) is not None:
        output_tensors = QuantizationSpec.from_dict(quant_info["output_quant"])
        quant_params["output_scale"] = param_dict[layer_name + ".output_scale"]
        quant_params["output_zero_point"] = param_dict[layer_name + ".output_zero_point"]

    weight_qspec: Optional[QuantizationSpec] = None
    weight_tensor = param_dict[quant_info.get("weight", None)]
    if quant_info.get("weight_quant", None) is not None:
        weight_qspec = QuantizationSpec.from_dict(quant_info["weight_quant"])
        weight_scale = param_dict[layer_name + ".weight_scale"]
        weight_zero_point = param_dict[layer_name + ".weight_zero_point"]

        if compressed:
            assert isinstance(weight_qspec, QuantizationSpec), "weight_qspec must be QuantizationSpec instance"
            assert isinstance(weight_qspec.qscheme, QSchemeType), "weight_qspec.qscheme must be QSchemeType instance"
            assert isinstance(weight_qspec.dtype, Dtype), "weight_qspec.dtype must be Dtype instance"
            pack_method = create_pack_method(qscheme=weight_qspec.qscheme.value, dtype=weight_qspec.dtype.value)
            weight_tensor = pack_method.unpack(
                weight_tensor, reorder,
                **({
                    "origin_packed_axis_size": weight_scale.shape[-1]
                } if weight_scale.shape != torch.Size([]) else {}))

            weight_tensor = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                weight_qspec.dtype.value, weight_tensor, weight_scale, weight_zero_point, weight_qspec.ch_axis,
                weight_qspec.group_size, weight_qspec.qscheme.value)

        quant_params["weight_scale"] = weight_scale
        quant_params["weight_zero_point"] = weight_zero_point

    module_config = QuantizationConfig(input_tensors=input_tensors, output_tensors=output_tensors, weight=weight_qspec)

    bias_tensor = None
    if quant_info.get("bias", None) is not None:
        bias_tensor = param_dict[quant_info.get("bias", None)]

    quant_module: nn.Module
    if quant_info['type'] in QUARK_QUANT_OPS:
        quant_module = QUARK_QUANT_OPS[quant_info['type']].from_float(
            float_module,
            module_config,
            reload=True,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
        )
    else:
        raise ValueError(f"The type {quant_info['type']} dose not support in Quark now!")
    quant_module.load_quant_params(quant_params)
    return quant_module


# TODO: add support for SequentialQuantize later
# TODO: better `reorder` doc
@log_errors
def load_params(model: Optional[nn.Module] = None,
                json_path: str = "",
                safetensors_path: str = "",
                pth_path: str = "",
                quant_mode: QuantizationMode = QuantizationMode.eager_mode,
                compressed: bool = False,
                reorder: bool = True) -> nn.Module:
    """
    Instantiates a quantized model from saved model files, which is generated from the :py:func:`quark.torch.export.api.save_params` function.

    :param torch.nn.Module model: The original Pytorch model.
    :param str json_path: The path of the saved json file. Only available for eager mode quantization.
    :param str safetensors_path: The path of the saved safetensors file. Only available for eager mode quantization.
    :param str pth_path: The path of the saved ``.pth`` file. Only available for ``fx_graph`` mode quantization.
    :param QuantizationMode quant_mode: The quantization mode. The choice includes ``"QuantizationMode.eager_mode"`` and ``"QuantizationMode.fx_graph_mode"``. Default is ``"QuantizationMode.eager_mode"``.
    :param bool compressed: Whether the quantized model to load is stored using its compressed data type, or in a "fake quantized" format (QDQ).
    :param bool reorder: Reorder.

    :return: The reloaded quantized version of the input model.
    :rtype: torch.nn.Module

    Examples:

    .. code-block:: python

        # eager mode:
        from quark.torch import load_params
        model = load_params(model, json_path=json_path, safetensors_path=safetensors_path)

    .. code-block:: python

        # fx_graph mode:
        from quark.torch.quantization.api import load_params
        model = load_params(pth_path=model_file_path, quant_mode=QuantizationMode.fx_graph_mode)

    Note:
        This function does not support dynamic quantization for now.
    """

    if quant_mode is QuantizationMode.eager_mode:
        if not is_safetensors_available():
            raise ImportError(
                "The function `load_params` with `quant_mode=QuantizationMode.eager_mode` requires the package `safetensors` to be installed, but it was not found. Please install `safetensors`."
            )

        if model is None:
            raise ValueError("Model should not be none if loading eager_mode quantized model")
        if json_path == "" or safetensors_path == "":
            raise ValueError("Json_path and safetensors_path should not be empty if loading eager_mode quantized model")
        # load model structure and parameters
        with open(json_path, "r") as file:
            model_dict = json.load(file)
        params_dict = load_file(safetensors_path)

        # verify exported model and float model have the same configuration
        model_config = model_dict["config"]
        if model_config:
            float_model_config: Dict[str, Any] = {}
            if hasattr(model.config, "to_diff_dict"):
                float_model_config = model.config.to_diff_dict()
            elif hasattr(model.config, "items"):
                float_model_config = dict(model.config.items())

            if not deep_compare(model_config, float_model_config):
                raise RuntimeError("Exported model and float model are not the same model!")
        # assert ((json.dumps(model_config) == json.dumps(float_model_config)),
        #         "Exported model and float model are not the same model!")

        logger.info("In-place OPs replacement start.")
        for name, module_info in get_name_and_info(model_dict["structure"]):
            float_module = get_op_by_name(model, name)
            if module_info["type"] in QUARK_QUANT_OPS:
                module = from_float_and_dict(float_module,
                                             module_info,
                                             params_dict,
                                             layer_name=name,
                                             compressed=compressed,
                                             reorder=reorder)
                set_op_by_name(model, name, module)
            else:
                device = float_module.weight.device
                float_module.weight.data = params_dict[module_info.get("weight", None)].to(device)
                if module_info.get("bias", None) is not None:
                    float_module.bias.data = params_dict[module_info.get("bias", None)].to(device)

        model = ModelQuantizer.freeze(model)
        logger.info("In-place OPs replacement end.")
    elif quant_mode is QuantizationMode.fx_graph_mode:
        if pth_path == "":
            raise ValueError("Pth_path should not be empty if loading eager_mode quantized model")
        loaded_quantized_ep = torch.export.load(pth_path)
        model = loaded_quantized_ep.module()

    return model
