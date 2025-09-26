#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch
from torch import nn

from quark.shares.utils.import_utils import is_accelerate_available, is_safetensors_available, is_transformers_available
from quark.shares.utils.log import ScreenLogger
from quark.torch.export.utils import (
    _build_quantized_model,
    _convert_quantized_model,
    _handle_multi_device_loading,
    _untie_parameters,
)

if TYPE_CHECKING and is_transformers_available():
    from transformers import PreTrainedModel, PreTrainedTokenizer  # type: ignore[attr-defined]

if is_safetensors_available():
    from safetensors.torch import load_file

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
logger = ScreenLogger(__name__)


def export_hf_model(
    model: "PreTrainedModel", export_dir: Union[str, Path], tokenizer: Optional["PreTrainedTokenizer"] = None
) -> None:
    """
    This function is used to export models in Hugging Face safetensors format.
    """

    logger.info("Start exporting huggingface_format quantized model ...")
    # Save model to safetensors.
    model.save_pretrained(export_dir)  # type: ignore[attr-defined]

    # Optionally, save the tokenizer from the original model.
    if tokenizer is not None:
        tokenizer.save_pretrained(export_dir)

    logger.info(f"hf_format quantized model exported to {export_dir} successfully.")


def import_hf_model(
    model_importer: "ModelImporter",  # type: ignore [name-defined]
    model: nn.Module,
    model_info_dir: str,
) -> nn.Module:
    """
    Load the model file, perform preprocessing and post-processing, load weights into the model.
    """
    if not is_safetensors_available():
        raise ImportError(
            "The function `import_hf_model` requires the package `safetensors` to be installed, but it was not found. Please install `safetensors`."
        )
    checkpoint_weights = _load_weights_from_safetensors(model_info_dir)

    model_config = model_importer.get_model_config()
    model = _build_quantized_model(model, model_config, checkpoint_weights)

    if is_accelerate_available():
        _untie_parameters(model, checkpoint_weights)
    # The module here is qparamlinear, the float module has been removed, the internal weight is already a quantized dtype like fp8 and is assigned to each GPU or meta by device.
    model_state_dict = model.state_dict()

    # In case we are loading the quantized weights into a model that is not on meta device,
    # we re-use the original device the weights were placed on, as `assign=True` is used later.
    # This is helpful e.g. in case the original model was dispatched to multiple
    # devices ahead of time with `accelerate`.
    for name, param in model_state_dict.items():
        if name not in checkpoint_weights:
            raise ValueError(f"The loaded checkpoint misses the key {name} present in the model weights.")

        if param.device.type != "meta":
            checkpoint_weights[name] = checkpoint_weights[name].to(param.device)

    # Handle multi-device loading if enabled
    if model_importer.multi_device and is_accelerate_available():
        _handle_multi_device_loading(model, checkpoint_weights)

    model.load_state_dict(checkpoint_weights, assign=True)
    model = _convert_quantized_model(model, model_config)

    logger.info("hf_format quantized model imported successfully.")
    return model


def _load_weights_from_safetensors(model_info_dir: str) -> dict[str, torch.Tensor]:
    """
    Load the state dict from safetensor file with safetensors.torch.load_file, possibly from multiple safetensors files in case of sharded model.
    """
    model_state_dict: dict[str, torch.Tensor] = {}
    safetensors_dir = Path(model_info_dir)
    safetensors_path = safetensors_dir / SAFE_WEIGHTS_NAME
    safetensors_index_path = safetensors_dir / SAFE_WEIGHTS_INDEX_NAME
    if safetensors_path.exists():
        # In this case, the weights are in a single `model.safetensors` file.
        model_state_dict = load_file(str(safetensors_path))
    elif safetensors_index_path.exists():
        # In this case, the weights are split in several `.safetensors` files.
        with open(str(safetensors_index_path)) as file:
            safetensors_indices = json.load(file)
        safetensors_files = [value for _, value in safetensors_indices["weight_map"].items()]
        safetensors_files = list(set(safetensors_files))
        for filename in safetensors_files:
            filepath = safetensors_dir / filename
            model_state_dict.update(load_file(str(filepath)))
    else:
        raise FileNotFoundError(
            f"Neither {str(safetensors_path)} nor {str(safetensors_index_path)} were found. Please check that the model path specified {str(safetensors_dir)} is correct."
        )
    return model_state_dict
