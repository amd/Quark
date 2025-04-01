#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import tempfile
import os

from quark.torch.quantization.config.config import load_pre_optimization_config_from_file, load_quant_algo_config_from_file, _load_pre_optimization_config_from_dict, _load_quant_algo_config_from_dict

def create_temp_file(content: str):
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
    temp_file.write(content)
    temp_file.close()
    return temp_file.name

def test_load_pre_optimization_config_from_file():
    file_path = create_temp_file(
        '{"name":"smooth", "scaling_layers":[{"prev_op": "self_attn_layer_norm"}], "model_decoder_layers": "model.decoder.layers"}'
    )
    try:
        loaded_config = load_pre_optimization_config_from_file(file_path)
        assert loaded_config.model_decoder_layers == "model.decoder.layers"
    finally:
        os.unlink(file_path)


def test_load_quant_algo_config_from_file():
    file_path = create_temp_file(
        '{"name":"awq", "scaling_layers":[{"prev_op": "self_attn_layer_norm"}], "model_decoder_layers": "model.decoder.layers"}'
    )
    try:
        loaded_config = load_quant_algo_config_from_file(file_path)
        assert loaded_config.model_decoder_layers == "model.decoder.layers"
    finally:
        os.unlink(file_path)


def test_load_pre_optimization_config_from_dict():
    config_dict = {
        "name": "smooth",
        "scaling_layers": [{"prev_op": "self_attn_layer_norm"}],
        "model_decoder_layers": "model.decoder.layers",
    }
    loaded_config = _load_pre_optimization_config_from_dict(config_dict)
    assert loaded_config.model_decoder_layers == "model.decoder.layers"


def test_load_quant_algo_config_from_dict():
    config_dict = {
        "name": "awq",
        "scaling_layers": [{"prev_op": "self_attn_layer_norm"}],
        "model_decoder_layers": "model.decoder.layers",
    }
    loaded_config = _load_quant_algo_config_from_dict(config_dict)
    assert loaded_config.model_decoder_layers == "model.decoder.layers"
