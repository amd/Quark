#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Algorithm/Pre-Quant Optimization API for PyTorch."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from packaging import version
from typing import Dict, List, Optional, Union
from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.config.config import Config
from quark.torch.pruning.config import Config as Pruning_Config
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
from quark.torch.algorithm.utils.auto_config import is_auto_config_needed, add_auto_config
from quark.torch.algorithm.utils.utils import get_device_map, set_device_map
from quark.torch.algorithm.awq.awq import AwqProcessor
from quark.torch.algorithm.awq.smooth import SmoothQuantProcessor
from quark.torch.algorithm.rotation.rotation import RotationProcessor
from quark.torch.algorithm.quarot.quarot import QuaRotProcessor
from quark.torch.algorithm.gptq.gptq import GptqProcessor
from quark.torch.algorithm.osscar.osscar import OsscarProcessor
from quark.torch.algorithm.blockwise_tuning.blockwise_tuning import BlockwiseTuningProcessor
from quark.torch.algorithm.awq.auto_smooth import AutoSmoothQuantProcessor

logger = ScreenLogger(__name__)

__all__ = [
    "apply_pre_quantization_optimization", "apply_advanced_quant_algo", "apply_advanced_pruning_algo",
    "blockwise_tuning_algo"
]

PROCESSOR_MAP = {
    'rotation': RotationProcessor,
    'quarot': QuaRotProcessor,
    'smooth': SmoothQuantProcessor,
    'autosmoothquant': AutoSmoothQuantProcessor,
    'awq': AwqProcessor,
    'gptq': GptqProcessor,
    'osscar': OsscarProcessor,
    'blockwise_tuning': BlockwiseTuningProcessor
}


@torch.no_grad()
def apply_pre_quantization_optimization(
    model: nn.Module,
    config: Config,
    dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                               DataLoader[Dict[str, torch.Tensor]]]] = None
) -> nn.Module:
    if config.pre_quant_opt_config is not None:
        logger.info("Pre-quantization optimization start.")

        if not isinstance(config.pre_quant_opt_config, List):
            pre_quant_opts = [config.pre_quant_opt_config]
        else:
            pre_quant_opts = config.pre_quant_opt_config

        for pre_quant_opt_config in pre_quant_opts:
            pre_quant_optimizer = PROCESSOR_MAP[pre_quant_opt_config.name](model, pre_quant_opt_config, dataloader)
            pre_quant_optimizer.apply()

        logger.info("Pre-quantization optimization end.")
    return model


@torch.no_grad()
def apply_advanced_quant_algo(
    model: nn.Module,
    config: Config,
    is_accelerate: Optional[bool],
    dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                               DataLoader[Dict[str, torch.Tensor]]]] = None
) -> nn.Module:
    if config.algo_config is not None:
        logger.info("Advanced algorithm start.")

        device_map = get_device_map(model, is_accelerate)

        for module in model.modules():
            if isinstance(module, ScaledFakeQuantize):
                module.disable_fake_quant()
                module.disable_observer()

        quantizer = PROCESSOR_MAP[config.algo_config.name](model, config.algo_config, dataloader)
        quantizer.apply()

        model = set_device_map(model, device_map)

        logger.info("Advanced algorithm end.")
    return model


def add_algorithm_config_by_model(model: nn.Module,
                                  dataloader: Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                                    DataLoader[Dict[str,
                                                                    torch.Tensor]], None], config: Config) -> Config:
    # Determine the positions and need for auto configuration
    smooth_position, rotation_position, is_awq_needed = is_auto_config_needed(config)

    if not (version.parse("2.1") < version.parse(torch.__version__) < version.parse("2.5")):
        logger.warning(
            f"Lack of specific information of pre-optimization configuration. However, PyTorch version {torch.__version__} detected. Only torch versions between 2.2 and 2.4 support auto generating algorithms configuration."
        )
        return config

    # If any configuration is needed, proceed with auto configuration
    if smooth_position >= 0 or rotation_position >= 0 or is_awq_needed:
        assert dataloader is not None, "Dataloader must be provided when auto-configuration is needed."
        # Get a sample input from the dataloader
        dummy_input = next(iter(dataloader))
        # Add auto-generated configurations to the existing config
        config = add_auto_config(model, dummy_input, config, smooth_position, rotation_position, is_awq_needed)

    return config


@torch.no_grad()
def apply_advanced_pruning_algo(
    model: nn.Module,
    config: Pruning_Config,
    is_accelerate: Optional[bool],
    dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                               DataLoader[Dict[str, torch.Tensor]]]] = None
) -> nn.Module:
    if config.algo_config is not None:
        logger.info("Advanced pruning algorithm start.")

        device_map = get_device_map(model, is_accelerate)

        pruner = PROCESSOR_MAP[config.algo_config.name](model, config.algo_config, dataloader)
        pruner.apply()

        model = set_device_map(model, device_map)

        logger.info("Advanced pruning algorithm end.")
    return model


def blockwise_tuning_algo(
    fp_model: nn.Module,
    model: nn.Module,
    config: Pruning_Config,
    is_accelerate: Optional[bool],
    dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                               DataLoader[Dict[str, torch.Tensor]]]] = None
) -> nn.Module:

    if config.blockwise_tuning_config is not None:
        logger.info("Blockwise tuning algorithm start.")

        device_map = get_device_map(model, is_accelerate)

        pruner = PROCESSOR_MAP[config.blockwise_tuning_config.name](fp_model, model, config.blockwise_tuning_config,
                                                                    dataloader)
        pruner.apply()

        model = set_device_map(model, device_map)

        logger.info("Blockwise tuning algorithm end.")

    return model
