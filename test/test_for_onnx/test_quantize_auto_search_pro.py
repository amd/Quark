#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import logging
import os
import random
import shutil
import time
import unittest
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn as nn

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import AutoSearchPro, generate_all_configs, get_auto_search_config
from quark.onnx.quantization.auto_search.auto_search_pro import replace_keys, validate_keys
from quark.onnx.quantization.output_eval import calculate_l1_distance, calculate_ssim

input_tensor = np.array(
    [
        [
            [
                [100.26921557, 10.79500909, 0.6102178, 0.04375664],
                [100.06221361, 10.98258356, 0.38635129, 0.06492238],
                [100.49631707, 10.35442799, 0.51719146, 0.52100111],
                [100.04145599, 10.88960236, 0.50627326, 0.57204613],
            ],
            [
                [0.99185097, 0.93582153, 0.13174529, 0.42896287],
                [0.14552133, 0.02538564, 0.0732355, 0.25725371],
                [0.09856916, 0.43015628, 0.55679755, 0.66560074],
                [0.9439425, 0.45701841, 0.86791293, 0.64728276],
            ],
            [
                [0.29159685, 0.79021383, 0.3117182, 0.11342342],
                [0.16660495, 0.46426165, 0.31348552, 0.143383],
                [0.96454802, 0.63258874, 0.30295267, 0.96720039],
                [0.29879457, 0.79916527, 0.02905061, 0.20115725],
            ],
        ]
    ]
).astype(np.float32)


base_config = {
    "search_space": {
        "activation": ["Int8Spec"],
        "activation_params": {
            "symmetric": [True, False],
            "calibration_method": ["MinMax", "Percentile", "LayerwisePercentile"],
            "only_if": "activation",
        },
        "weight": ["Int8Spec"],
        "weight_params": {
            "symmetric": [True],
            "calibration_method": ["MinMax"],
            "only_if": "weight",
        },
        "algorithms": ["adaquant", "adaround"],
        "adaquant_params": {
            "num_iterations": [10, 20],
            "learning_rate": [1e-6, 1e-3],
            "optim_device": ["cuda:0"],
            "infer_device": ["cuda:0"],
            "data_size": [1, 10],
            "only_if": {"algorithms": "adaquant"},
        },
        "cle_algo": [
            "cle",
        ],
        "cle_params": {"cle_steps": [1], "only_if": {"cle_algo": "cle"}},
        "specific_layer_config": [None],
        "layer_type_config": [None],
        "exclude": [None],
        "use_external_data_format": [False],
        "OptimizeModel": [False],
    },
    "n_trials": 20,
    "n_jobs": 1,
    "output_dir": "./output",
    "temp_dir": "./temp_dir",
    "search_algo": "TPE",
    "search_evaluator": None,  # Custom or built-in function
    "direction": "minimize",
    "base_framework": "onnx",
    "model_input": None,
    "calib_data_reader": None,
    "load_study_if_exists": False,
    "eval_data_reader": None,
    "two_stage_search": False,
    "plot_results": False,
}


class DataReader:
    def __init__(self, input_tensor):
        self.data = [input_tensor, 100.0 * input_tensor + 10.0]
        self.input_name = "input"
        self.index = 0
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: nhwc_data} for nhwc_data in self.data])
        return next(self.enum_data, None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {self.input_name: self.data[idx]}

    def rewind(self):
        self.index = 0


class DoubleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1, 1)

        with torch.no_grad():
            self.conv2.weight *= 100.0

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = DoubleConvModel()

    dummy_input = torch.randn(1, 3, 4, 4)

    onnx_model_path = Path(output_dir, "double_conv_model.onnx").as_posix()
    onnx_quantized_model_path = Path(output_dir, "double_conv_model_quantized.onnx").as_posix()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )

    print(f"Model has been saved to {onnx_model_path}")
    return onnx_model_path, onnx_quantized_model_path


def prepare_data():
    data_reader = DataReader(input_tensor)
    return data_reader


def infer_quantized_model(quantized_model_path):
    sess = onnxruntime.InferenceSession(quantized_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def mock_evaluator(onnx_path):
    random.seed(int(time.time() * 1000))
    return random.random()


def logger_config(log_path: str = "./auto_search.log", logging_name: str = "auto search") -> logging.Logger:
    logger = logging.getLogger(logging_name)

    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(log_path, encoding="UTF-8")
    handler.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


class TestTensorQuantize(unittest.TestCase):
    def test_repalce_dict_keys(self):
        reference_dict = {"algorithms": "adaround", "cle_algo": "cle"}
        input_dict = {"algorithm": "adaround", "cle_alg": "cle"}
        logger_ins = logger_config()
        matched_res = validate_keys(list(reference_dict.keys()), list(input_dict.keys()), logger_ins)
        replace_res = replace_keys(input_dict, matched_res)
        self.assertEqual(replace_res, reference_dict)

    def test_calculate_l1_distance(self):
        input_array = np.array([3.0, 4.0, 5.0, 6.0])
        ref_array = np.array([2.0, 3.0, 4.0, 5.0])
        output = calculate_l1_distance(input_array, ref_array)
        golden = np.array(4.0).astype(np.float32)
        self.assertEqual(output, golden)

    def test_calculate_ssim(self):
        input_array = np.array([3.0, 4.0, 5.0, 6.0])
        ref_array = np.array([2.0, 3.0, 4.0, 5.0])
        output = calculate_ssim(input_array, ref_array)
        golden = np.array(0.9743).astype(np.float32)
        self.assertAlmostEqual(output, golden, delta=0.0001)

    @use_temporary_directory
    def test_auto_search_pro(self, tmpdir: str):
        temp_dir = tmpdir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)

        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        quant_config = copy.deepcopy(base_config)
        quant_config["calib_data_reader"] = data_reader
        quant_config["model_input"] = input_model_path
        quant_config["search_evaluator"] = mock_evaluator
        quant_config["search_space"]["adaquant_params"] = {
            "num_iterations": [10, 20],
            "only_if": {"algorithms": "adaquant"},
        }
        quant_config["search_space"]["adaround_params"] = {
            "num_iterations": [10, 20],
            "only_if": {"algorithms": "adaround"},
        }
        quant_config["output_dir"] = os.path.join(temp_dir, "output")
        quant_config["temp_dir"] = os.path.join(temp_dir, "temp_dir")

        auto_search_pro_ins = AutoSearchPro(quant_config)
        _ = auto_search_pro_ins.run()

    @use_temporary_directory
    def test_generate_all_configs(self, tmpdir: str):
        search_space_all = generate_all_configs(copy.deepcopy(base_config["search_space"]))
        self.assertEqual(len(search_space_all), 54)

    @use_temporary_directory
    def test_grid_search(self, tmpdir: str):
        temp_dir = tmpdir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)

        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        quant_config = copy.deepcopy(base_config)
        quant_config["calib_data_reader"] = data_reader
        quant_config["model_input"] = input_model_path
        quant_config["search_algo"] = "Grid"
        quant_config["search_space"]["algorithms"] = [None]
        quant_config["output_dir"] = os.path.join(temp_dir, "output")
        quant_config["temp_dir"] = os.path.join(temp_dir, "temp_dir")

        all_configs = generate_all_configs(quant_config["search_space"])
        auto_search_pro_ins = AutoSearchPro(quant_config)
        best_params = auto_search_pro_ins.run()
        self.assertEqual(best_params, all_configs[3])

    @use_temporary_directory
    def test_two_stage_search(self, tmpdir: str):
        temp_dir = tmpdir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)

        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()
        gt_param = {
            "algorithms": "adaquant",
            "activation": "Int8Spec",
            "weight": "Int8Spec",
            "cle_algo": "cle",
            "adaquant_params": {"num_iterations": 1, "learning_rate": 1e-06, "data_size": 2},
            "activation_params": {"symmetric": False, "calibration_method": "MinMax"},
            "weight_params": {"symmetric": True, "calibration_method": "MinMax"},
            "cle_params": {"cle_steps": 1},
        }

        quant_config = copy.deepcopy(base_config)
        quant_config["calib_data_reader"] = data_reader
        quant_config["model_input"] = input_model_path
        quant_config["search_algo"] = "Grid"
        quant_config["two_stage_search"] = True
        quant_config["search_space"]["algorithms"] = ["adaquant"]
        quant_config["search_space"]["adaquant_params"] = {
            "num_iterations": [1],
            "learning_rate": [1e-6, 1e-3],
            "data_size": [2],
            "only_if": {"algorithms": "adaquant"},
        }
        quant_config["output_dir"] = os.path.join(temp_dir, "output")
        quant_config["temp_dir"] = os.path.join(temp_dir, "temp_dir")

        auto_search_pro_ins = AutoSearchPro(quant_config)
        best_params = auto_search_pro_ins.run()
        self.assertEqual(best_params, gt_param)

    @use_temporary_directory
    def test_get_auto_search_config(self, tmpdir: str):
        temp_dir = tmpdir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)

        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        quant_config_name = "XINT8_SEARCH"
        quant_config = get_auto_search_config(quant_config_name)
        quant_config = copy.deepcopy(quant_config)

        quant_config["calib_data_reader"] = data_reader
        quant_config["model_input"] = input_model_path

        quant_config["search_algo"] = "TPE"
        quant_config["search_space"]["algorithms"] = ["adaquant"]
        quant_config["search_space"]["adaquant_params"] = {
            "NumIterLR": [[10, 1e-4], [20, 1e-3]],
            "only_if": {"algorithms": "adaquant"},
        }

        quant_config["output_dir"] = os.path.join(temp_dir, "output")
        quant_config["temp_dir"] = os.path.join(temp_dir, "temp_dir")

        auto_search_pro_ins = AutoSearchPro(quant_config)
        auto_search_pro_ins.run()


if __name__ == "__main__":
    unittest.main()
