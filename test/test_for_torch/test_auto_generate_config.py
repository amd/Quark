#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
import json
import os
import re

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.testing_utils import torch_device
from quark.testing import skip_if_no_gpu
from quark.torch.algorithm.utils.auto_config import EasyGraph

logger = ScreenLogger(__name__)


def get_golden_pattern_list(config_path):
    with open(config_path, encoding="utf-8") as file:
        golden_config_data = json.load(file)
    golden_patterns = []
    for scaling_layer in golden_config_data["scaling_layers"]:
        layer_patterns = []
        pattern_prev_op = rf"{golden_config_data['model_decoder_layers']}\.\d+\.{scaling_layer['prev_op']}"
        layer_patterns.append(pattern_prev_op)
        for sub_layer in scaling_layer["layers"]:
            pattern_layers = rf"{golden_config_data['model_decoder_layers']}\.\d+\.{sub_layer}"
            layer_patterns.append(pattern_layers)
            print("\t\tlayers:", pattern_layers)
        golden_patterns.append(layer_patterns)

    if "additional_scaling_layers" in golden_config_data:
        for scaling_layer in golden_config_data["additional_scaling_layers"]:
            layer_patterns = [scaling_layer["prev_op"]]
            for sub_layer in scaling_layer["layers"]:
                layer_patterns.append(sub_layer)
                print("\t\tlayers:", pattern_layers)
            golden_patterns.append(layer_patterns)

    return golden_patterns


def find_awq_json_files(directory):
    awq_json_files = []
    for _, _, files in os.walk(directory):
        for file in files:
            awq_json_files.append(file)
    return awq_json_files


def golden_match_generate_config(golden_pattern_list, generate_config_dict):
    """
    make sure all golden case can find in generate_config
    """
    for golden_case in golden_pattern_list:
        match_golden_case_flag = False
        for generate_pair in generate_config_dict["scaling_layers"]:
            # match prev_op
            if re.match(golden_case[0], generate_pair["prev_op"]):
                # match layer_patten
                if len(golden_case[1:]) == len(generate_pair["layers"]):
                    tmp_layers = copy.deepcopy(generate_pair["layers"])
                    for layer_patten in golden_case[1:]:
                        for layer in tmp_layers:
                            if re.match(layer_patten, layer):
                                # layer
                                tmp_layers.remove(layer)
                                break
                    if len(tmp_layers) == 0:
                        match_golden_case_flag = True
                    else:
                        print("not match all layer", tmp_layers)
                        raise AssertionError()
        if match_golden_case_flag:
            print("match success", golden_case)
        else:
            print("not match", golden_case)
            raise AssertionError()


def generate_config_match_golden(golden_pattern_list, generate_config_dict):
    """
    make sure all generate_config cases can be found in the golden pattern list
    """
    for generate_pair in generate_config_dict["scaling_layers"]:
        match_generate_config_case_flag = False
        for golden_case in golden_pattern_list:
            # match prev_op
            if re.match(golden_case[0], generate_pair["prev_op"]):
                # Check if the layers match
                if len(golden_case[1:]) == len(generate_pair["layers"]):
                    tmp_layers = copy.deepcopy(generate_pair["layers"])
                    for golden_case_layer_patten in golden_case[1:]:
                        for layer in tmp_layers:
                            if re.match(golden_case_layer_patten, layer):
                                # layer matched
                                tmp_layers.remove(layer)
                                break
                    # If all layers match
                    if len(tmp_layers) == 0:
                        match_generate_config_case_flag = True
                        break
                    else:
                        print("not all layers matched", tmp_layers)
                        raise AssertionError()
        if match_generate_config_case_flag:
            print("match success", generate_pair)
        else:
            print("not match", generate_pair)
            raise AssertionError()


def get_model(
    ckpt_path: str, data_type: str = "auto", device: str = "cuda", multi_gpu: bool = False
) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    config.num_hidden_layers = 2
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager").half()
    model = model.to(torch_device)
    model.eval()
    assert isinstance(model, torch.nn.Module)
    return model


def generate_config():
    model_id_list = [
        "facebook/opt-125m",
        "Qwen/Qwen1.5-0.5B",
    ]

    generate_config_dir = "generate_config_dir"
    for model_id in model_id_list:
        model = get_model(model_id)
        input_data = torch.randint(0, 100, [1, 512], dtype=torch.int64).to(torch_device)
        model(input_data)
        if "Llama-2-7b" in model_id or "Qwen" in model_id:
            eg = EasyGraph(model, input_data, True)
            rotation_config = eg.get_rotation_config()
            with open("rotations.json", "w", encoding="utf-8") as f:
                json.dump(rotation_config, f, ensure_ascii=False, indent=4)
        else:
            eg = EasyGraph(model, input_data)
        parameterized_pair_config = eg.get_parameterized_pair_config()

        if not os.path.exists(generate_config_dir):
            os.makedirs(generate_config_dir)

        with open(os.path.join(generate_config_dir, model_id.replace("/", "_") + ".json"), "w") as f:
            f.write(json.dumps(parameterized_pair_config))


@skip_if_no_gpu  # TODO (tfernand): What is this all about? Many tests require GPU and have no @skip_if_no_gpu decorator
def test_compare_generate_config_with_golden_config():
    generate_config()
    golden_config_dir = "golden_config_dir"
    generate_config_dir = "generate_config_dir"
    golden_config_path_list = find_awq_json_files(golden_config_dir)
    for golden_path in golden_config_path_list:
        print("\n\n\nmodel:", golden_path)
        golden_pattern_list = get_golden_pattern_list(os.path.join(golden_config_dir, golden_path))
        with open(os.path.join(generate_config_dir, golden_path[:-5] + ".json"), encoding="utf-8") as file:
            generate_config_dict = json.load(file)
        generate_config_match_golden(golden_pattern_list, generate_config_dict)
        golden_match_generate_config(golden_pattern_list, generate_config_dict)

    # check rotation
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "rotations_golden.json"), encoding="utf-8"
    ) as file:
        rotations_golden = file.read()
    with open("./rotations.json", encoding="utf-8") as file:
        rotations_generate = file.read()
    assert eval(rotations_golden) == eval(rotations_generate)
