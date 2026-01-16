#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from quark.shares.utils.import_utils import is_safetensors_available, is_transformers_available
from quark.shares.utils.log import ScreenLogger
from quark.torch.export.utils import (
    get_state_dict_for_export,
)

if TYPE_CHECKING and is_transformers_available():
    from transformers import PreTrainedModel, PreTrainedTokenizer  # type: ignore[attr-defined]

if is_safetensors_available():
    from safetensors.torch import load_file

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
logger = ScreenLogger(__name__)


def export_hf_model(
    model: "PreTrainedModel", export_dir: str | Path, tokenizer: Optional["PreTrainedTokenizer"] = None
) -> None:
    """
    This function is used to export models in Hugging Face safetensors format.
    """

    logger.info("Start exporting huggingface_format quantized model ...")

    state_dict = get_state_dict_for_export(model)

    # Save model to safetensors.
    model.save_pretrained(export_dir, state_dict=state_dict)  # type: ignore[attr-defined]

    # Optionally, save the tokenizer from the original model.
    if tokenizer is not None:
        tokenizer.save_pretrained(export_dir)

    logger.info(f"hf_format quantized model exported to {export_dir} successfully.")


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
