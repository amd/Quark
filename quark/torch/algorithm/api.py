#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Algorithm/Pre-Quant Optimization API for PyTorch."""

from typing import Any

import torch
import torch.nn as nn
from packaging import version
from torch.utils.data import DataLoader

from quark.common.utils.import_utils import is_transformers_available
from quark.common.utils.log import ScreenLogger
from quark.experimental.torch.twobitscalar.twobitscalar import TwoBitScalarProcessor
from quark.torch.algorithm.awq.auto_smooth import AutoSmoothQuantProcessor
from quark.torch.algorithm.awq.awq import AwqProcessor
from quark.torch.algorithm.awq.smooth import SmoothQuantProcessor
from quark.torch.algorithm.blockwise_joint_tuning.processor import BlockwiseJointTuningProcessor
from quark.torch.algorithm.blockwise_tuning.blockwise_tuning import BlockwiseTuningProcessor
from quark.torch.algorithm.config import BaseAlgoConfig
from quark.torch.algorithm.depth_pruning.layer_importance import LayerImportancePrunerProcessor
from quark.torch.algorithm.gptaq.gptaq import GptaqProcessor
from quark.torch.algorithm.gptq.gptq import GptqProcessor
from quark.torch.algorithm.osscar.osscar import OsscarProcessor
from quark.torch.algorithm.qronos.qronos import QronosProcessor
from quark.torch.algorithm.rotation.rotation import RotationProcessor
from quark.torch.algorithm.svdquant.svdquant import SVDQuantProcessor
from quark.torch.algorithm.utils.auto_config import add_auto_config, is_auto_config_needed
from quark.torch.algorithm.utils.utils import get_device_map, set_device_map
from quark.torch.pruning.config import PConfig as Pruning_Config
from quark.torch.quantization.config.config import QConfig
from quark.torch.quantization.tensor_quantize import NonScaledFakeQuantize, ScaledFakeQuantize

if is_transformers_available():
    from transformers.feature_extraction_utils import BatchFeature

logger = ScreenLogger(__name__)

__all__ = ["apply_advanced_quant_algo", "apply_advanced_pruning_algo", "blockwise_tuning_algo", "get_processor"]

# `type[Any]` rather than `type[BaseAlgoProcessor]`: the entries share the processor protocol but
# not all of them subclass the ABC, and the blockwise ones take a fourth constructor argument.
PROCESSOR_MAP: dict[str, type[Any]] = {
    "rotation": RotationProcessor,
    "quarot": RotationProcessor,
    "smooth": SmoothQuantProcessor,
    "autosmoothquant": AutoSmoothQuantProcessor,
    "awq": AwqProcessor,
    "gptq": GptqProcessor,
    "gptaq": GptaqProcessor,
    "qronos": QronosProcessor,
    "osscar": OsscarProcessor,
    "blockwise_tuning": BlockwiseTuningProcessor,
    "blockwise_joint_tuning": BlockwiseJointTuningProcessor,
    "layer_importance_depth_pruning": LayerImportancePrunerProcessor,
    "svdquant": SVDQuantProcessor,
    "twobitscalar": TwoBitScalarProcessor,
}


def get_processor(name: str) -> type[Any]:
    """Return the processor class that runs the algorithm called ``name``.

    The registry is consulted before ``PROCESSOR_MAP``, so a ``QuarkAlgorithm`` claiming a core
    algorithm's name takes over dispatch for it — the same precedence the config loader uses.

    :param str name: The algorithm name, as it appears in the config's ``name`` field.
    :return: The processor class to instantiate.
    :rtype: type[Any]
    :raises KeyError: If neither the registry nor ``PROCESSOR_MAP`` knows the name.
    """
    # Imported lazily: an algorithm module imports the processors this module imports, so a
    # module-level import here risks a cycle.
    from quark.torch.algorithm.registry import ALGORITHM_REGISTRY

    algorithm = ALGORITHM_REGISTRY.get(name)
    if algorithm is not None:
        return algorithm.algo_processor

    return PROCESSOR_MAP[name]


@torch.no_grad()
def apply_advanced_quant_algo(
    model: nn.Module,
    config: QConfig,
    is_accelerate: bool | None,
    dataloader: DataLoader[torch.Tensor]
    | DataLoader[list[dict[str, torch.Tensor]]]
    | DataLoader[dict[str, torch.Tensor]]
    | DataLoader[list["BatchFeature"]]
    | None = None,
) -> nn.Module:
    # apply algorithms sequentially
    if config.algo_config is not None and len(config.algo_config) > 0:
        for module in model.modules():
            if isinstance(module, ScaledFakeQuantize | NonScaledFakeQuantize):
                module.disable_fake_quant()
                module.disable_observer()

        logger.info("Advanced algorithm start.")

        for i in range(len(config.algo_config)):
            device_map = get_device_map(model, is_accelerate)

            logger.info(f"Applying {config.algo_config[i].name} processing/algorithm...")
            processor = get_processor(config.algo_config[i].name)(model, config.algo_config[i], dataloader)
            processor.apply()

            model = set_device_map(model, device_map)

        logger.info("Advanced algorithm end.")

    return model


def add_algorithm_config_by_model(
    model: nn.Module,
    dataloader: DataLoader[torch.Tensor]
    | DataLoader[list[dict[str, torch.Tensor]]]
    | DataLoader[dict[str, torch.Tensor]]
    | DataLoader[list["BatchFeature"]]
    | None,
    config: QConfig,
) -> QConfig:
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
    is_accelerate: bool | None,
    dataloader: DataLoader[torch.Tensor]
    | DataLoader[list[dict[str, torch.Tensor]]]
    | DataLoader[dict[str, torch.Tensor]]
    | None = None,
) -> nn.Module:
    if config.algo_config is not None:
        logger.info("Advanced pruning algorithm start.")

        device_map = get_device_map(model, is_accelerate)

        pruner = get_processor(config.algo_config.name)(model, config.algo_config, dataloader)
        pruner.apply()

        model = set_device_map(model, device_map)

        logger.info("Advanced pruning algorithm end.")
    return model


def _get_blockwise_processor_map() -> dict[str, type]:
    """Processors reachable only through :func:`blockwise_tuning_algo` -- NOT the shared
    ``PROCESSOR_MAP`` that ``apply_advanced_quant_algo`` uses for the standard
    ``QConfig``/``ModelQuantizer.quantize_model()`` path, which only supplies 3 constructor args.

    ``AutoRoundProcessor`` is imported here, not at module top: ``quark.experimental.torch.
    autoround.wrapper`` imports ``quark.torch.quantization.config.type``, which (via
    ``quark.torch``'s own ``__init__``) re-enters ``quark.torch.algorithm.api`` while this module
    is still initializing -- a real circular import if ``AutoRoundProcessor`` were imported
    eagerly at module load time. Deferring the import to call time (after all modules have
    finished loading) breaks the cycle.
    """
    from quark.experimental.torch.autoround.autoround import AutoRoundProcessor

    return {
        "autoround": AutoRoundProcessor,
        **PROCESSOR_MAP,
    }


def blockwise_tuning_algo(
    fp_model: nn.Module,
    model: nn.Module,
    blockwise_tuning_config: BaseAlgoConfig,
    is_accelerate: bool | None,
    dataloader: DataLoader[torch.Tensor]
    | DataLoader[list[dict[str, torch.Tensor]]]
    | DataLoader[dict[str, torch.Tensor]]
    | None = None,
) -> nn.Module:
    if blockwise_tuning_config is not None:
        logger.info("Blockwise tuning algorithm start.")

        device_map = get_device_map(model, is_accelerate)

        blockwise_processor_map = _get_blockwise_processor_map()
        processor = blockwise_processor_map[blockwise_tuning_config.name](
            fp_model, model, blockwise_tuning_config, dataloader
        )

        processor.apply()

        model = set_device_map(model, device_map)

        logger.info("Blockwise tuning algorithm end.")

    return model
