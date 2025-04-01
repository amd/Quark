#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import List, Optional, Dict, Any
import torch
import json
from accelerate.utils.modeling import find_tied_parameters
from torch import nn
from transformers import AutoTokenizer
from quark.torch.quantization.config.config import Config
from quark.torch.export.config.config import ExporterConfig
from quark.shares.utils.log import ScreenLogger
from quark.torch import ModelExporter
from quark.torch.export.api import ModelImporter
from safetensors.torch import load_file as safe_load_file
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../examples/torch/language_modeling')))
from llm_utils.model_preparation import get_model_type

PT_WEIGHTS_NAME = "model_state_dict.pth"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
logger = ScreenLogger(__name__)

def export_hf_model(model: nn.Module, export_config: ExporterConfig, model_dir: str, export_dir: str, quant_config: Config, custom_mode: str) -> None:
    '''
    This function is used to export models in huggingface format.
    '''
    if quant_config is None:
        raise ValueError("quant_config should not be None when exporting huggingface format files")
    exporter = ModelExporter(config=export_config, export_dir=export_dir)
    export_path = exporter.export_dir
    logger.info("Start exporting huggingface_format quantized model ...")

    # add_export_info_for_hf=True means export info of quark will be added in config.json, see the description of the get_export_model function
    model = exporter.get_export_model(model, quant_config=quant_config, custom_mode=custom_mode, add_export_info_for_hf=True)
    # save tokenizer and model safetensors
    _save_hf_model_info(model, model_dir, export_path)
    # The export_func replaces some of the model's submodules and modifies the contents of the config, so restore them.
    exporter.reset_model(model=model)
    logger.info(f"hf_format quantized model exported to {export_path} successfully.")

def _save_hf_model_info(model, model_dir, export_path):
    '''
    Call save_pretrained to save the model file.
    '''
    model.save_pretrained(export_path)
    try:
        # TODO: Having trust_remote_code=True by default in our codebase is dangerous.
        model_type = get_model_type(model)
        use_fast = True if model_type in ["grok", "cohere", "olmo"] else False
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=use_fast)
        tokenizer.save_pretrained(export_path)
    except Exception as e:
        logger.error(f"An error occurred when saving tokenizer: {e}.  You can try to save the tokenizer manually")


def import_hf_model(model: nn.Module, model_info_dir: str):
    '''
    Load the model file, perform preprocessing and post-processing, load weights into the model.
    '''
    logger.info("Start importing hf_format quantized model ...")
    importer = ModelImporter(model_info_dir=model_info_dir)
    model_config = importer.get_model_config()
    model_state_dict = _load_hf_state_dict(model_info_dir)
    model = importer.import_model(model, model_config, model_state_dict)
    _untie_parameters(model, model_state_dict)
    model.load_state_dict(model_state_dict)
    logger.info("hf_format quantized model imported successfully.")
    return model

def _load_hf_state_dict(model_info_dir: str) -> Dict[str, torch.Tensor]:
    '''
    Load the state dict from safetensor file by load_file of safetensors.torch.
    '''
    model_state_dict: Dict[str, torch.Tensor] = {}
    safetensors_dir = Path(model_info_dir)
    safetensors_path = safetensors_dir / SAFE_WEIGHTS_NAME
    safetensors_index_path = safetensors_dir / SAFE_WEIGHTS_INDEX_NAME
    if safetensors_path.exists():
        model_state_dict = safe_load_file(str(safetensors_path))
    # is_shard
    elif safetensors_index_path.exists():
        with open(str(safetensors_index_path), "r") as file:
            safetensors_indices = json.load(file)
        safetensors_files = [value for _, value in safetensors_indices["weight_map"].items()]
        safetensors_files = list(set(safetensors_files))
        for filename in safetensors_files:
            filepath = safetensors_dir / filename
            model_state_dict.update(safe_load_file(str(filepath)))
    else:
        raise FileNotFoundError(f"Neither {str(safetensors_path)} nor {str(safetensors_index_path)} were found. Please check that the model path specified {str(safetensors_dir)} is correct.")
    return model_state_dict

def _untie_parameters(model: nn.Module, model_state_dict: Dict[str, Any]) -> None:
    '''
    Some parameters share weights, such as embedding and lm_head, and when exporting with `PretrainedModel.save_pretrained`
    only one of them will be exported, so need to copy the parameters.
    '''
    # TODO: Only embedding for now, need to solve other cases, such as encoder-decoder tied
    tied_param_groups = find_tied_parameters(model)
    if len(tied_param_groups) > 0:
        if len(tied_param_groups) > 1 or "lm_head.weight" not in tied_param_groups[0]:
            raise ValueError(
                f"Your have tied_param_groups: {tied_param_groups}, temporarily does not support the case where tied_param is not 'lm_head and embedding'"
            )
        missing_key: List[str] = []
        tied_param_value: Optional[torch.Tensor] = None
        for tied_param_name in tied_param_groups[0]:
            if tied_param_name in model_state_dict.keys():
                tied_param_value = model_state_dict[tied_param_name]
            else:
                missing_key.append(tied_param_name)
        if tied_param_value is not None:
            for tied_param_key in missing_key:
                model_state_dict[tied_param_key] = tied_param_value
        else:
            raise ValueError("Cannot assign a value to tied_params because tied_param_value is None")
