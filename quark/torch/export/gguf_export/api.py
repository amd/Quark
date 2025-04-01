#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from pathlib import Path
from .gguf_model_writer import ModelWriter
from typing import Union, Dict, Any, Tuple
from .gguf_model_converter import GGUFModelConverter
import json


def convert_exported_model_to_gguf(model_name: str, json_path: Union[str, Path], safetensor_path: Union[str, Path],
                                   tokenizer_dir: Union[str, Path], output_file_path: Union[str, Path]) -> None:
    """This function is used to convert quark exported model to gguf model.

    Args:
        model_name (str): name of this model which will be written to gguf field `general.name`
        json_path (Union[str, Path]): Quark exported model consists of a `.json` file and a `.safetensors` file.
            This arguments indicates the path of `.json` file
        safetensor_path (Union[str, Path]): Path of `.safetensors` file.
        tokenizer_dir (Union[str, Path]): Tokenizer needs to be encoded into gguf model.
            This argument specifies the directory path of tokenizer which contains tokenizer.json, tokenizer_config.json and/or tokenizer.model
        output_file_path (str): The path of generated gguf model.
    """
    tokenizer_dir = Path(tokenizer_dir)
    output_file_path = Path(output_file_path)
    json_path = Path(json_path)
    safetensor_path = Path(safetensor_path)
    with open(json_path) as f:
        config = json.load(f)["config"]
    model_writer = ModelWriter.from_model_architecture(config["architectures"][0])(model_name=model_name,
                                                                                   json_path=json_path,
                                                                                   safetensor_path=safetensor_path,
                                                                                   tokenizer_dir=tokenizer_dir,
                                                                                   fname_out=output_file_path)
    model_writer.set_gguf_parameters()
    model_writer.set_vocab()
    model_writer.write()


def insert_quant_info_from_gguf(model_name: str, model_info: Dict[str, Any], param_info: Dict[str, torch.Tensor],
                                gguf_path: str) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    gguf_path = Path(gguf_path)
    model_converter = GGUFModelConverter(model_name, model_info, param_info, gguf_path)
    model_converter.convert()
    return model_converter.model_info, model_converter.param_info
