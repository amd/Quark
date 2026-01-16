#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import logging
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from onnxruntime.quantization.quant_utils import QuantType

from quark.onnx import CalibrationMethod, Config, ModelQuantizer, PowerOfTwoMethod
from quark.onnx.quantization.auto_search.auto_search_v1 import (
    AutoSearch,
    AutoSearchConfig,
    SearchSpace,
    cos_metric,
    l1_metric,
    l2_metric,
    logger_config,
    psnr_metric,
    split_config_levels,
    ssim_metric,
)
from quark.onnx.quantization.config.custom_config import U8S8_AAWS_CONFIG
from quark.shares.utils.testing_utils import use_temporary_directory

auto_search_config = AutoSearchConfig()
# level1
level1_config = copy.deepcopy(auto_search_config.search_space)
del level1_config["extra_options"]
# leve1 + level2
level12_config = copy.deepcopy(auto_search_config.search_space)
del level12_config["extra_options"]["FastFinetune"]
# level1 + level3
level13_config = copy.deepcopy(auto_search_config.search_space)
temp_dict = copy.deepcopy(level13_config["extra_options"]["FastFinetune"])
del level13_config["extra_options"]
level13_config["extra_options"] = {"FastFinetune": temp_dict}
# level1 + level2 + level3
level123_config = copy.deepcopy(auto_search_config.search_space)

input_tensor = np.array(
    [
        [
            [
                [0.26921557, 0.79500909, 0.6102178, 0.04375664],
                [0.06221361, 0.98258356, 0.38635129, 0.06492238],
                [0.49631707, 0.35442799, 0.51719146, 0.52100111],
                [0.04145599, 0.88960236, 0.50627326, 0.57204613],
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


class EmptyConfig:
    search_evaluator = None


class DataReader:
    def __init__(self, input_tensor):
        self.data = [input_tensor, input_tensor]
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
        super(DoubleConvModel, self).__init__()
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


def prepare_config():
    config_copy = copy.deepcopy(U8S8_AAWS_CONFIG)
    config_copy.include_cle = True
    config_copy.extra_options["ReplaceClip6Relu"] = True
    config_copy.extra_options["CLESteps"] = 2
    config_copy.extra_options["CLEScaleAppendBias"] = True
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_fastft_config():
    config_copy = copy.deepcopy(U8S8_AAWS_CONFIG)
    config_copy.include_cle = True
    config_copy.extra_options["ReplaceClip6Relu"] = True
    config_copy.extra_options["CLESteps"] = 2
    config_copy.extra_options["CLEScaleAppendBias"] = True
    config_copy.include_fast_ft = True
    quant_config = Config(global_quant_config=config_copy)
    return quant_config


def prepare_data():
    data_reader = DataReader(input_tensor)
    return data_reader


def prepare_quantizer(quant_config):
    quantizer = ModelQuantizer(quant_config)
    return quantizer


def quantize_static(quantizer, input_model_path, output_model_path, data_reader):
    quantizer.quantize_model(input_model_path, output_model_path, data_reader)
    print("Quantized the ONNX model and saved it at:", output_model_path)
    return output_model_path


def infer_quantized_model(quantized_model_path):
    sess = onnxruntime.InferenceSession(quantized_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_data = input_tensor
    output = sess.run([output_name], {input_name: input_data})
    print(f"Model output: {output}")
    return output


def tensor_quantize(output_dir):
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = prepare_data()
    quant_config = prepare_config()
    quantizer = prepare_quantizer(quant_config)
    quantized_model_path = quantize_static(quantizer, input_model_path, output_model_path, data_reader)
    output = infer_quantized_model(quantized_model_path)
    return output


class TestTensorQuantize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])

    def test_l2_metric(self):
        input_array = np.array([3.0, 4.0, 5.0, 6.0])
        ref_array = np.array([2.0, 3.0, 4.0, 5.0])
        output = l2_metric(input_array, ref_array)
        golden = np.array(1.0).astype(np.float32)
        self.assertEqual(output, golden)

    def test_l1_metric(self):
        input_array = np.array([3.0, 4.0, 5.0, 6.0])
        ref_array = np.array([2.0, 3.0, 4.0, 5.0])
        output = l1_metric(input_array, ref_array)
        golden = np.array(1.0).astype(np.float32)
        self.assertEqual(output, golden)

    def test_cos_metric(self):
        input_array = np.array([3.0, 4.0, 5.0, 6.0])
        ref_array = np.array([2.0, 3.0, 4.0, 5.0])
        output = cos_metric(input_array, ref_array)
        golden = np.array(0.9989).astype(np.float32)
        self.assertAlmostEqual(output, golden, delta=0.0001)

    def test_psnr_metric(self):
        input_array = np.array([3.0, 4.0, 5.0, 6.0])
        ref_array = np.array([2.0, 3.0, 4.0, 5.0])
        output = psnr_metric(input_array, ref_array)
        golden = np.array(48.1308).astype(np.float32)
        self.assertAlmostEqual(output, golden, delta=0.0001)

    def test_ssim_metric(self):
        input_array = np.array([3.0, 4.0, 5.0, 6.0])
        ref_array = np.array([2.0, 3.0, 4.0, 5.0])
        output = ssim_metric(input_array, ref_array)
        golden = np.array(0.9743).astype(np.float32)
        self.assertAlmostEqual(output, golden, delta=0.0001)

    def test_split_config_levels(self):
        l1_config, l2_config, l3_config = split_config_levels(level1_config)
        self.assertEqual(len(l1_config), 7)
        self.assertEqual(len(l2_config), 0)
        self.assertEqual(len(l3_config), 0)
        del l1_config, l2_config, l3_config

        l1_config, l2_config, l3_config = split_config_levels(level12_config)
        self.assertEqual(len(l1_config), 7)
        self.assertEqual(len(l2_config), 6)
        self.assertEqual(len(l3_config), 0)
        del l1_config, l2_config, l3_config

        l1_config, l2_config, l3_config = split_config_levels(level13_config)
        self.assertEqual(len(l1_config), 7)
        self.assertEqual(len(l2_config), 0)
        self.assertEqual(len(l3_config), 5)
        del l1_config, l2_config, l3_config

        l1_config, l2_config, l3_config = split_config_levels(level123_config)
        self.assertEqual(len(l1_config), 7)
        self.assertEqual(len(l2_config), 6)
        self.assertEqual(len(l3_config), 5)
        del l1_config, l2_config, l3_config

    @use_temporary_directory
    def test_logger_config(self, tmpdir: str):
        logger = logger_config(log_path=Path(tmpdir, "./auto_search.log").as_posix(), logging_name="auto_search")
        self.assertEqual(type(logger), logging.Logger)
        del logger

    def test_search_space_remove_invalid(self):
        # test level1 + level2
        my_search_conf = {
            "calibrate_method": [PowerOfTwoMethod.MinMSE, CalibrationMethod.MinMax],
            "activation_type": ["QInt8", "QInt16"],
            "weight_type": [
                "QInt8",
            ],
            "extra_options": {"Percentile": [99.999, 99.9999], "CalibMovingAverage": [True, False]},
        }
        my_search_space = SearchSpace(my_search_conf)
        res = my_search_space.get_all_configs()
        valid_res = my_search_space.remove_invalid_configs(res)
        self.assertEqual(len(valid_res), 6)
        del my_search_conf
        del my_search_space
        del res
        del valid_res

        my_search_conf = {
            "calibrate_method": [CalibrationMethod.Percentile],
            "activation_type": ["QInt8"],
            "weight_type": [
                "QInt8",
            ],
            "include_sq": [False],
            "extra_options": {"CalibMovingAverageConstant": [0.01, 0.001], "SmoothAlpha": [0.5, 0.6]},
        }
        my_search_space = SearchSpace(my_search_conf)
        res = my_search_space.get_all_configs()
        valid_res = my_search_space.remove_invalid_configs(res)
        self.assertEqual(len(valid_res), 1)
        del my_search_conf
        del my_search_space
        del res
        del valid_res

        # level1 + level3
        my_search_conf = {
            "calibrate_method": [CalibrationMethod.MinMax],
            "activation_type": ["QInt8"],
            "weight_type": [
                "QInt8",
            ],
            "include_fast_ft": [False],
            "extra_options": {
                "FastFinetune": {
                    "DataSize": [500, 1000],
                    "NumIterations": [100, 1000],
                    "OptimAlgorithm": ["adaround", "adaquant"],
                    "LearningRate": [0.01, 0.001, 0.0001],
                }
            },
        }
        my_search_space = SearchSpace(my_search_conf)
        res = my_search_space.get_all_configs()
        valid_res = my_search_space.remove_invalid_configs(res)
        self.assertEqual(len(valid_res), 1)
        del my_search_conf
        del my_search_space
        del res
        del valid_res

        my_search_conf = {
            "calibrate_method": [CalibrationMethod.MinMax],
            "activation_type": ["QInt8"],
            "weight_type": [
                "QInt8",
            ],
            "extra_options": {
                "FastFinetune": {
                    "DataSize": [500, 1000],
                    "NumIterations": [100, 1000],
                    "OptimAlgorithm": ["adaround", "adaquant"],
                    "LearningRate": [0.01, 0.001, 0.0001],
                }
            },
        }
        my_search_space = SearchSpace(my_search_conf)
        res = my_search_space.get_all_configs()
        valid_res = my_search_space.remove_invalid_configs(res)
        self.assertEqual(len(valid_res), 1)
        del my_search_conf
        del my_search_space
        del res
        del valid_res

        # level1 + level3
        my_search_conf = {
            "calibrate_method": [CalibrationMethod.MinMax],
            "activation_type": ["QInt8"],
            "weight_type": [
                "QInt8",
            ],
            "include_fast_ft": [True],
            "extra_options": {
                "FastFinetune": {
                    "DataSize": [500, 1000],
                    "NumIterations": [100, 1000],
                    "OptimAlgorithm": ["adaround", "adaquant"],
                    "LearningRate": [0.01, 0.001, 0.0001],
                }
            },
        }
        my_search_space = SearchSpace(my_search_conf)
        res = my_search_space.get_all_configs()
        valid_res = my_search_space.remove_invalid_configs(res)
        self.assertEqual(len(valid_res), 24)
        del my_search_conf
        del my_search_space
        del res
        del valid_res

        # level1 + level2+ level3
        my_search_conf = {
            "calibrate_method": [CalibrationMethod.MinMax],
            "activation_type": ["QInt8"],
            "weight_type": [
                "QInt8",
            ],
            "include_fast_ft": [True],
            "extra_options": {
                "CalibMovingAverage": [False],
                "CalibMovingAverageConstant": [0.01, 0.001],
                "SmoothAlpha": [0.5, 0.6],
                "FastFinetune": {
                    "DataSize": [500, 1000],
                    "NumIterations": [100, 1000],
                    "OptimAlgorithm": ["adaround", "adaquant"],
                    "LearningRate": [0.01, 0.001, 0.0001],
                },
            },
        }
        my_search_space = SearchSpace(my_search_conf)
        res = my_search_space.get_all_configs()
        valid_res = my_search_space.remove_invalid_configs(res)
        self.assertEqual(len(valid_res), 24)
        del my_search_conf
        del my_search_space
        del res
        del valid_res

        # level1 + level3
        my_search_conf = {
            "calibrate_method": [CalibrationMethod.MinMax],
            "activation_type": ["QInt8"],
            "weight_type": [
                "QInt8",
            ],
            "include_fast_ft": [True],
            "extra_options": {
                "Percentile": [99.9999],
                "FastFinetune": {
                    "DataSize": [500, 1000],
                    "NumIterations": [100, 1000],
                    "OptimAlgorithm": ["adaround", "adaquant"],
                    "LearningRate": [0.01, 0.001, 0.0001],
                },
            },
        }
        my_search_space = SearchSpace(my_search_conf)
        res = my_search_space.get_all_configs()
        valid_res = my_search_space.remove_invalid_configs(res)
        self.assertEqual(len(valid_res), 24)
        del my_search_conf
        del my_search_space
        del res
        del valid_res

    def test_search_space(self):
        search_space = SearchSpace(level1_config)
        res = search_space.get_all_configs()
        self.assertEqual(len(res), 384)
        del search_space

        search_space = SearchSpace(level12_config)
        res = search_space.get_all_configs()
        self.assertEqual(len(res), 384 * 64)
        del search_space

        search_space = SearchSpace(level13_config)
        res = search_space.get_all_configs()
        self.assertEqual(len(res), 384 * 24)
        del search_space

        search_space = SearchSpace(level123_config)
        res = search_space.get_all_configs()
        self.assertEqual(len(res), 384 * 64 * 24)
        del search_space

    def test_valid_config_keys(
        self,
    ):
        level123_config["nonexist_key"] = ["nonexist_l1"]
        level123_config["extra_options"]["nonexist_key2"] = ["nonexist_l2"]
        level123_config["extra_options"]["FastFinetune"]["nonexist_key3"] = ["nonexist_l3"]
        search_space = SearchSpace(level123_config)
        res = search_space.get_all_configs()
        self.assertEqual(len(res), 384 * 64 * 24)
        del search_space

    # level1 test auto search
    @use_temporary_directory
    def test_auto_search_l1(self, tmpdir: str):
        temp_dir = Path(tmpdir, "./test_auto_search_l1").as_posix()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
        auto_search_config = AutoSearchConfig()

        l1_autosearch_config = copy.deepcopy(auto_search_config)
        l1_autosearch_config.search_space = {
            "calibrate_method": [
                PowerOfTwoMethod.MinMSE,
                CalibrationMethod.MinMax,
            ],
            "activation_type": [QuantType.QInt8],
            "weight_type": [QuantType.QInt8],
            "include_cle": [True, False],
        }
        l1_autosearch_config.search_cache_dir = Path(temp_dir, "cache_dir").as_posix()
        l1_autosearch_config.search_output_dir = Path(temp_dir, "output_dir").as_posix()
        l1_autosearch_config.search_log_path = Path(temp_dir, "auto_search.log").as_posix()
        l1_autosearch_config.search_metric_tolerance = 1000.0
        l1_autosearch_config.search_stop_condition["find_n_candidates"] = 2
        l1_autosearch_config.search_metric = "l1"

        quantize_config = prepare_config()
        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        auto_search_instance = AutoSearch(
            config=quantize_config,
            auto_search_config=l1_autosearch_config,
            model_input=input_model_path,
            eval_dataloader=None,
            calibration_data_reader=data_reader,
            calibration_data_path=None,
        )

        res = auto_search_instance.search_model()
        self.assertEqual(len(res), 2)
        del auto_search_config
        del auto_search_instance
        del l1_autosearch_config

    # level1 + l2 test auto search
    @use_temporary_directory
    def test_auto_search_l12(self, tmpdir: str):
        temp_dir = Path(tmpdir, "./test_auto_search_l12").as_posix()
        if os.path.exists(temp_dir):
            pass
        else:
            os.mkdir(temp_dir)
        auto_search_config = AutoSearchConfig()

        l12_autosearch_config = copy.deepcopy(auto_search_config)
        l12_autosearch_config.search_space = {
            "calibrate_method": [
                CalibrationMethod.MinMax,
            ],
            "activation_type": [QuantType.QInt8],
            "weight_type": [QuantType.QInt8],
            "include_cle": [True, False],
            "extra_options": {
                "CalibMovingAverage": [True],
                "CalibMovingAverageConstant": [0.01, 0.001],
            },
        }
        l12_autosearch_config.search_cache_dir = Path(temp_dir, "cache_dir").as_posix()
        l12_autosearch_config.search_output_dir = Path(temp_dir, "output_dir").as_posix()
        l12_autosearch_config.search_log_path = Path(temp_dir, "auto_search.log").as_posix()
        l12_autosearch_config.search_metric_tolerance = 1000.0
        l12_autosearch_config.search_stop_condition["find_n_candidates"] = 1
        l12_autosearch_config.search_algo = "random"

        quantize_config = prepare_config()
        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        auto_search_instance = AutoSearch(
            config=quantize_config,
            auto_search_config=l12_autosearch_config,
            model_input=input_model_path,
            eval_dataloader=None,
            calibration_data_reader=data_reader,
            calibration_data_path=None,
        )

        res = auto_search_instance.search_model()
        self.assertEqual(len(res), 1)
        del auto_search_config
        del auto_search_instance
        del l12_autosearch_config

    # level1 + l3
    @use_temporary_directory
    def test_auto_search_l13(self, tmpdir: str):
        temp_dir = Path(tmpdir, "./test_auto_search_l13").as_posix()
        if os.path.exists(temp_dir):
            pass
        else:
            os.mkdir(temp_dir)
        auto_search_config = AutoSearchConfig()

        l12_autosearch_config = copy.deepcopy(auto_search_config)
        l12_autosearch_config.search_space = {
            "calibrate_method": [
                CalibrationMethod.MinMax,
            ],
            "activation_type": [QuantType.QInt8],
            "weight_type": [QuantType.QInt8],
            "include_cle": [False],
            "extra_options": {
                "FastFinetune": {
                    "DataSize": [5],
                    "NumIterations": [10],
                    "OptimAlgorithm": ["adaround"],
                    "LearningRate": [0.01],
                }
            },
        }
        l12_autosearch_config.search_cache_dir = Path(temp_dir, "cache_dir").as_posix()
        l12_autosearch_config.search_output_dir = Path(temp_dir, "output_dir").as_posix()
        l12_autosearch_config.search_log_path = Path(temp_dir, "auto_search.log").as_posix()
        l12_autosearch_config.search_metric_tolerance = 1000.0
        l12_autosearch_config.search_stop_condition["time_limit"] = 10.0
        l12_autosearch_config.search_algo = "self_defined"

        quantize_config = prepare_config()
        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        auto_search_instance = AutoSearch(
            config=quantize_config,
            auto_search_config=l12_autosearch_config,
            model_input=input_model_path,
            eval_dataloader=None,
            calibration_data_reader=data_reader,
            calibration_data_path=None,
        )

        res = auto_search_instance.search_model()
        self.assertEqual(len(res), 1)
        del auto_search_config
        del auto_search_instance
        del l12_autosearch_config

    # level1 + l2 + l3 test auto search
    @use_temporary_directory
    def test_auto_search_l123(self, tmpdir: str):
        temp_dir = Path(tmpdir, "./test_auto_search_l123").as_posix()
        if os.path.exists(temp_dir):
            pass
        else:
            os.mkdir(temp_dir)
        auto_search_config = AutoSearchConfig()

        l12_autosearch_config = copy.deepcopy(auto_search_config)
        l12_autosearch_config.search_space = {
            "calibrate_method": [
                CalibrationMethod.MinMax,
            ],
            "activation_type": [QuantType.QInt8],
            "weight_type": [QuantType.QInt8],
            "include_cle": [False],
            "extra_options": {
                "CalibMovingAverage": [True],
                "CalibMovingAverageConstant": [0.001],
                "FastFinetune": {
                    "DataSize": [5],
                    "NumIterations": [10],
                    "OptimAlgorithm": ["adaround"],
                    "LearningRate": [0.01],
                },
            },
        }

        l12_autosearch_config.search_cache_dir = Path(temp_dir, "cache_dir").as_posix()
        l12_autosearch_config.search_output_dir = Path(temp_dir, "output_dir").as_posix()
        l12_autosearch_config.search_log_path = Path(temp_dir, "auto_search.log").as_posix()
        l12_autosearch_config.search_metric_tolerance = 1000.0
        l12_autosearch_config.search_stop_condition["time_limit"] = 10.0
        l12_autosearch_config.search_algo = "self_defined"

        quantize_config = prepare_config()
        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        auto_search_instance = AutoSearch(
            config=quantize_config,
            auto_search_config=l12_autosearch_config,
            model_input=input_model_path,
            eval_dataloader=None,
            calibration_data_reader=data_reader,
            calibration_data_path=None,
        )

        res = auto_search_instance.search_model()
        self.assertEqual(len(res), 1)
        del auto_search_config
        del auto_search_instance
        del l12_autosearch_config

        # level1 test auto search

    @use_temporary_directory
    def test_auto_search_empty(self, tmpdir: str):
        temp_dir = Path(tmpdir, "./test_auto_search_empty").as_posix()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
        auto_search_config = EmptyConfig()

        quantize_config = prepare_config()
        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        auto_search_instance = AutoSearch(
            config=quantize_config,
            auto_search_config=auto_search_config,
            model_input=input_model_path,
            eval_dataloader=None,
            calibration_data_reader=data_reader,
            calibration_data_path=None,
        )

        res = auto_search_instance.all_configs
        self.assertEqual(len(res), 384 * 64 * 24)
        del auto_search_config
        del auto_search_instance

    @use_temporary_directory
    def test_auto_search_numiter_lr(self, tmpdir: str):
        temp_dir = Path(tmpdir, "./test_auto_search_numiter_lr").as_posix()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
        auto_search_config = AutoSearchConfig()

        l3_autosearch_config = copy.deepcopy(auto_search_config)
        l3_autosearch_config.search_space = {
            "include_fast_ft": [True],
            "extra_options": {"FastFinetune": {"NumIterLR": [[100, 1e-4], [200, 1e-5]]}},
        }
        l3_autosearch_config.search_cache_dir = Path(temp_dir, "cache_dir").as_posix()
        l3_autosearch_config.search_output_dir = Path(temp_dir, "output_dir").as_posix()
        l3_autosearch_config.search_log_path = Path(temp_dir, "auto_search.log").as_posix()
        l3_autosearch_config.search_metric_tolerance = 1000.0
        l3_autosearch_config.search_stop_condition["find_n_candidates"] = 1

        quantize_config = prepare_config()
        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        auto_search_instance = AutoSearch(
            config=quantize_config,
            auto_search_config=l3_autosearch_config,
            model_input=input_model_path,
            eval_dataloader=None,
            calibration_data_reader=data_reader,
            calibration_data_path=None,
        )

        res = auto_search_instance.search_model()
        self.assertEqual(len(res), 1)
        del auto_search_config
        del auto_search_instance
        del l3_autosearch_config

    # search space define None in the default space
    @use_temporary_directory
    def test_auto_search_space_outside(self, tmpdir: str):
        temp_dir = Path(tmpdir, "./test_auto_search_space_outside").as_posix()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
        auto_search_config = AutoSearchConfig()

        l1_autosearch_config = copy.deepcopy(auto_search_config)
        l1_autosearch_config.search_space = None
        l1_autosearch_config.search_cache_dir = Path(temp_dir, "cache_dir").as_posix()
        l1_autosearch_config.search_output_dir = Path(temp_dir, "output_dir").as_posix()
        l1_autosearch_config.search_log_path = Path(temp_dir, "auto_search.log").as_posix()
        l1_autosearch_config.search_metric_tolerance = 1000.0
        l1_autosearch_config.search_stop_condition["find_n_candidates"] = 2
        l1_autosearch_config.search_metric = "l1"

        quantize_config = prepare_config()
        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        auto_search_instance = AutoSearch(
            config=quantize_config,
            auto_search_config=l1_autosearch_config,
            model_input=input_model_path,
            eval_dataloader=None,
            calibration_data_reader=data_reader,
            calibration_data_path=None,
        )

        search_space_outside = {
            "calibrate_method": [
                PowerOfTwoMethod.MinMSE,
                CalibrationMethod.MinMax,
            ],
            "activation_type": [QuantType.QInt8],
            "weight_type": [QuantType.QInt8],
            "include_cle": [True, False],
        }
        search_space_defined_outside = auto_search_instance.build_all_configs(search_space_outside)
        auto_search_instance.all_configs = search_space_defined_outside

        res = auto_search_instance.search_model()
        self.assertEqual(len(res), 2)
        del auto_search_config
        del auto_search_instance
        del l1_autosearch_config

    # test auto search: two stage search
    @use_temporary_directory
    def test_auto_search_two_stage_search(self, tmpdir: str):
        temp_dir = Path(tmpdir, "./test_auto_search_temp").as_posix()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)
        auto_search_config = AutoSearchConfig()

        l1_autosearch_config = copy.deepcopy(auto_search_config)
        l1_autosearch_config.search_space = {
            "calibrate_method": [
                PowerOfTwoMethod.MinMSE,
                CalibrationMethod.MinMax,
            ],
            "activation_type": [QuantType.QInt8],
            "weight_type": [QuantType.QInt8],
            "include_cle": [True, False],
            "extra_options": {
                "FastFinetune": {
                    "DataSize": [5],
                    "NumIterations": [10],
                    "OptimAlgorithm": ["adaround"],
                    "LearningRate": [0.01],
                }
            },
        }
        l1_autosearch_config.search_cache_dir = Path(temp_dir, "cache_dir").as_posix()
        l1_autosearch_config.search_output_dir = Path(temp_dir, "output_dir").as_posix()
        l1_autosearch_config.search_log_path = Path(temp_dir, "auto_search.log").as_posix()
        l1_autosearch_config.search_metric_tolerance = 1000.0
        l1_autosearch_config.search_stop_condition["find_n_candidates"] = 2
        l1_autosearch_config.two_stage_search = True

        quantize_config = prepare_fastft_config()
        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        auto_search_instance = AutoSearch(
            config=quantize_config,
            auto_search_config=l1_autosearch_config,
            model_input=input_model_path,
            eval_dataloader=None,
            calibration_data_reader=data_reader,
            calibration_data_path=None,
        )

        res = auto_search_instance.search_model()
        self.assertEqual(len(res), 1)
        del auto_search_config
        del auto_search_instance
        del l1_autosearch_config

    @use_temporary_directory
    def test_advanced_fastft_search(self, tmpdir: str):
        temp_dir = Path(tmpdir, "./test_advanced_fastft_search").as_posix()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)

        auto_search_config = AutoSearchConfig()
        advanced_fast_ft_config = copy.deepcopy(auto_search_config)
        advanced_fast_ft_config.search_cache_dir = Path(temp_dir, "cache_dir").as_posix()
        advanced_fast_ft_config.search_output_dir = Path(temp_dir, "output_dir").as_posix()
        advanced_fast_ft_config.search_log_path = Path(temp_dir, "auto_search.log").as_posix()
        advanced_fast_ft_config.advanced_fastft_search = True
        advanced_fast_ft_config.n_jobs = 1
        advanced_fast_ft_config.search_devices = "cpu"
        advanced_fast_ft_config.advanced_fastft_params = {
            "sampler_algo": "TPE",
            "direction": "minimize",
            "n_trials": 10,
        }
        advanced_fast_ft_config.search_space = {
            "extra_options": {
                "FastFinetune": {
                    "NumIterations": [100, 200],
                    "LearningRate": [1e-3, 1e-7],
                }
            }
        }

        quantize_config = prepare_fastft_config()
        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        auto_search_instance = AutoSearch(
            config=quantize_config,
            auto_search_config=advanced_fast_ft_config,
            model_input=input_model_path,
            eval_dataloader=None,
            calibration_data_reader=data_reader,
            calibration_data_path=None,
        )
        auto_search_instance.search_model()
        del auto_search_config
        del advanced_fast_ft_config
        del auto_search_instance
        del quantize_config
        del data_reader

    @use_temporary_directory
    def test_advanced_fastft_search2(self, tmpdir: str):
        temp_dir = Path(tmpdir, "./test_advanced_fastft_search2").as_posix()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)

        def fake_custom_evaluator(onnx_path):
            return 1.0

        auto_search_config = AutoSearchConfig()
        advanced_fast_ft_config = copy.deepcopy(auto_search_config)
        advanced_fast_ft_config.search_cache_dir = Path(temp_dir, "cache_dir").as_posix()
        advanced_fast_ft_config.search_output_dir = Path(temp_dir, "output_dir").as_posix()
        advanced_fast_ft_config.search_log_path = Path(temp_dir, "auto_search.log").as_posix()
        advanced_fast_ft_config.advanced_fastft_search = True
        advanced_fast_ft_config.search_evaluator = fake_custom_evaluator
        advanced_fast_ft_config.n_jobs = 2
        advanced_fast_ft_config.search_devices = "cuda:0,1,2"
        advanced_fast_ft_config.advanced_fastft_params = {"sampler_algo": "TPE", "direction": "minimize", "n_trials": 1}
        advanced_fast_ft_config.search_space = {
            "extra_options": {
                "FastFinetune": {
                    "NumIterations": [100, 200],
                    "LearningRate": [1e-3, 1e-7],
                }
            }
        }

        quantize_config = prepare_fastft_config()
        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        auto_search_instance = AutoSearch(
            config=quantize_config,
            auto_search_config=advanced_fast_ft_config,
            model_input=input_model_path,
            eval_dataloader=None,
            calibration_data_reader=data_reader,
            calibration_data_path=None,
        )
        auto_search_instance.search_model()
        del auto_search_config
        del advanced_fast_ft_config
        del auto_search_instance
        del quantize_config
        del data_reader

    @use_temporary_directory
    def test_advanced_fastft_search3(self, tmpdir: str):
        temp_dir = Path(tmpdir, "./test_advanced_fastft_search3").as_posix()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)

        def fake_custom_evaluator(onnx_path):
            return 1.0

        auto_search_config = AutoSearchConfig()
        advanced_fast_ft_config = copy.deepcopy(auto_search_config)
        advanced_fast_ft_config.search_cache_dir = Path(temp_dir, "cache_dir").as_posix()
        advanced_fast_ft_config.search_output_dir = Path(temp_dir, "output_dir").as_posix()
        advanced_fast_ft_config.search_log_path = Path(temp_dir, "auto_search.log").as_posix()
        advanced_fast_ft_config.advanced_fastft_search = True
        advanced_fast_ft_config.search_evaluator = fake_custom_evaluator
        advanced_fast_ft_config.n_jobs = 2
        advanced_fast_ft_config.search_metric_tolerance = 0.0
        advanced_fast_ft_config.search_devices = "cuda:0,1,2"
        advanced_fast_ft_config.advanced_fastft_params = {
            "sampler_algo": "TPE_wrong",
            "direction": "minimize",
            "n_trials": 1,
        }
        advanced_fast_ft_config.search_space = {
            "include_fast_ft": [True],
            "extra_options": {
                "FastFinetune": {
                    "NumIterations": [100, 200],
                }
            },
        }

        quantize_config = prepare_fastft_config()
        input_model_path, _ = prepare_model(temp_dir)
        data_reader = prepare_data()

        auto_search_instance = AutoSearch(
            config=quantize_config,
            auto_search_config=advanced_fast_ft_config,
            model_input=input_model_path,
            eval_dataloader=None,
            calibration_data_reader=data_reader,
            calibration_data_path=None,
        )
        auto_search_instance.search_model()
        del auto_search_config
        del advanced_fast_ft_config
        del auto_search_instance
        del quantize_config
        del data_reader


if __name__ == "__main__":
    unittest.main()
