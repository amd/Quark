#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Peuning API for PyTorch."""

import torch
import torch.nn as nn
import torch.fx
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union, List
from quark.torch.pruning.model_transformation import process_model_pruning
from quark.torch.pruning.utils import pre_process_tuning
from quark.torch.pruning.config import Config
from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.api import apply_advanced_pruning_algo, blockwise_tuning_algo
import logging

__all__ = ["ModelPruner"]

logger = ScreenLogger(__name__)


class ModelPruner:
    """
    Provides an API for pruning deep learning models using PyTorch. This class handles the configuration and processing of the model for pruning based on user-defined parameters. It is essential to ensure that the 'config' provided has all necessary pruning parameters defined. This class assumes that the model is compatible with the pruning settings specified in 'config'.

    Args:
        config (Config): Configuration object containing settings for pruning.

    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._is_accelerate: Optional[bool] = None
        self.set_logging_level()  # set log level: default info

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

    def pruning_model(
        self,
        model: nn.Module,
        dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                   DataLoader[Dict[str, torch.Tensor]]]] = None
    ) -> nn.Module:
        """
        This function aims to prune the given PyTorch model to optimize its performance and reduce its size. This function accepts a model and a torch dataloader. The dataloader is used to provide data necessary for calibration during the pruning process. Depending on the type of data provided (either tensors directly or structured as lists or dictionaries of tensors), the function will adapt the pruning approach accordingly.It's important that the model and dataloader are compatible in terms of the data they expect and produce. Misalignment in data handling between the model and the dataloader can lead to errors during the pruning process.

        Parameters:
            model (nn.Module): The PyTorch model to be pruning. This model should be already trained and ready for pruning.
            dataloader (Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]], DataLoader[Dict[str, torch.Tensor]]]):
                The DataLoader providing data that the pruning process will use for calibration. This can be a simple DataLoader returning
                tensors, or a more complex structure returning either a list of dictionaries or a dictionary of tensors.

        Returns:
            nn.Module: The pruned version of the input model. This model is now optimized for inference with reduced size and potentially improved
            performance on targeted devices.

        """
        # Step0: Pre pruning device check
        self._check_model_device(model)

        # Step1[optional]: Pre-processing for blockwise tuning ...
        fp_model = self._pre_process_tuning(model)

        # Step2: Apply Advanced pruning algo such as osscar ...
        model = self._apply_advanced_pruning_algo(model, dataloader)

        # Step3: pruning model ...
        model = self._post_process_model(model)

        # Step4[optional]: pruning model ...
        model = self._blockwise_tuning_algo(fp_model, model, dataloader)

        return model

    def _check_model_device(self, model: nn.Module) -> None:
        # using accelerate cause, device can not be cpu or disk, temporarily
        if hasattr(model, 'hf_device_map'):
            for _, layer_device in model.hf_device_map.items():
                if layer_device == "cpu" or layer_device == "disk":
                    raise MemoryError(
                        f"Out of memory. The available GPU memory is insufficient to load the entire model. Portions of the model have been assigned to '{layer_device}', "
                        "but Quark does not support loading models simultaneously across GPU, CPU and disk. Please consider freeing up resources or reducing memory usage."
                    )

            self._is_accelerate = True
        else:
            self._is_accelerate = False

    def _pre_process_tuning(self, model: nn.Module) -> nn.Module:
        return pre_process_tuning(model, self.config, self._is_accelerate)

    def _apply_advanced_pruning_algo(
        self,
        model: nn.Module,
        dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                   DataLoader[Dict[str, torch.Tensor]]]] = None
    ) -> nn.Module:
        return apply_advanced_pruning_algo(model, self.config, self._is_accelerate, dataloader)

    def _post_process_model(self, model: nn.Module) -> nn.Module:
        return process_model_pruning(model, self.config, self._is_accelerate)

    def _blockwise_tuning_algo(
        self,
        fp_model: nn.Module,
        model: nn.Module,
        dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                   DataLoader[Dict[str, torch.Tensor]]]] = None
    ) -> nn.Module:
        return blockwise_tuning_algo(fp_model, model, self.config, self._is_accelerate, dataloader)
