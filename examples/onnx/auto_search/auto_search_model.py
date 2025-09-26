#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import copy
import os

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType

from quark.onnx import ExtendedQuantFormat, ExtendedQuantType, LayerWiseMethod, PowerOfTwoMethod, auto_search
from quark.onnx.quantization.config import Config, get_default_config


class AutoSearchConfig_Default:
    # for s8s8 & s16s8 aaws/asws
    search_space: dict[str, any] = {
        "calibrate_method": [
            CalibrationMethod.MinMax,
            CalibrationMethod.Percentile,
            LayerWiseMethod.LayerWisePercentile,
        ],
        "activation_type": [
            QuantType.QInt8,
            QuantType.QInt16,
        ],
        "weight_type": [
            QuantType.QInt8,
        ],
        "include_cle": [False],
        "include_fast_ft": [False],
        "extra_options": {
            "ActivationSymmetric": [True, False],
            "WeightSymmetric": [True],
            "CalibMovingAverage": [False, True],
            "CalibMovingAverageConstant": [0.01],
        },
    }

    search_space_advanced: dict[str, any] = {
        "calibrate_method": [
            CalibrationMethod.MinMax,
            CalibrationMethod.Percentile,
            LayerWiseMethod.LayerWisePercentile,
        ],
        "activation_type": [
            QuantType.QInt8,
            QuantType.QInt16,
        ],
        "weight_type": [
            QuantType.QInt8,
        ],
        "include_cle": [False, True],
        "include_fast_ft": [False, True],
        "extra_options": {
            "ActivationSymmetric": [True, False],
            "WeightSymmetric": [True],
            "CalibMovingAverage": [
                False,
                True,
            ],
            "CalibMovingAverageConstant": [0.01],
            "FastFinetune": {
                "DataSize": [
                    200,
                ],
                "NumIterations": [1000, 5000, 10000],
                "OptimAlgorithm": ["adaround"],
                "LearningRate": [0.1, 0.01],
                # 'OptimDevice': ['cuda:0'],
                # 'InferDevice': ['cuda:0'],
                "EarlyStop": [False],
            },
        },
    }

    search_space_advanced2: dict[str, any] = {
        "calibrate_method": [
            CalibrationMethod.MinMax,
            CalibrationMethod.Percentile,
            LayerWiseMethod.LayerWisePercentile,
        ],
        "activation_type": [
            QuantType.QInt8,
            QuantType.QInt16,
        ],
        "weight_type": [
            QuantType.QInt8,
        ],
        "include_cle": [False, True],
        "include_fast_ft": [False, True],
        "extra_options": {
            "ActivationSymmetric": [True, False],
            "WeightSymmetric": [True],
            "CalibMovingAverage": [
                False,
                True,
            ],
            "CalibMovingAverageConstant": [0.01],
            "FastFinetune": {
                "DataSize": [
                    200,
                ],
                "NumIterations": [1000, 5000, 10000],
                "OptimAlgorithm": ["adaquant"],
                "LearningRate": [1e-5, 1e-6],
                # 'OptimDevice': ['cuda:0'],
                # 'InferDevice': ['cuda:0'],
                "EarlyStop": [False],
            },
        },
    }

    # for XINT8
    search_space_XINT8: dict[str, any] = {
        "calibrate_method": [PowerOfTwoMethod.MinMSE],
        "activation_type": [QuantType.QUInt8],
        "weight_type": [
            QuantType.QInt8,
        ],
        "enable_npu_cnn": [True],
        "include_cle": [False],
        "include_fast_ft": [False],
        "extra_options": {
            "ActivationSymmetric": [True],
        },
    }

    search_space_XINT8_advanced: dict[str, any] = {
        "calibrate_method": [PowerOfTwoMethod.MinMSE],
        "activation_type": [
            QuantType.QUInt8,
        ],
        "weight_type": [
            QuantType.QInt8,
        ],
        "enable_npu_cnn": [True],
        "include_cle": [False, True],
        "include_fast_ft": [True],
        "extra_options": {
            "ActivationSymmetric": [
                True,
            ],
            "WeightSymmetric": [True],
            "CalibMovingAverage": [
                False,
                True,
            ],
            "CalibMovingAverageConstant": [0.01],
            "FastFinetune": {
                "DataSize": [
                    200,
                ],
                "NumIterations": [1000],
                "OptimAlgorithm": ["adaround"],
                "LearningRate": [
                    0.1,
                ],
                # 'OptimDevice': ['cuda:0'],
                # 'InferDevice': ['cuda:0'],
                "EarlyStop": [False],
            },
        },
    }

    search_space_XINT8_advanced2: dict[str, any] = {
        "calibrate_method": [PowerOfTwoMethod.MinMSE],
        "activation_type": [
            QuantType.QUInt8,
        ],
        "weight_type": [
            QuantType.QInt8,
        ],
        "enable_npu_cnn": [True],
        "include_cle": [False, True],
        "include_fast_ft": [True],
        "extra_options": {
            "ActivationSymmetric": [
                True,
            ],
            "WeightSymmetric": [True],
            "CalibMovingAverage": [
                False,
                True,
            ],
            "CalibMovingAverageConstant": [0.01],
            "FastFinetune": {
                "DataSize": [
                    200,
                ],
                "NumIterations": [5000],
                "OptimAlgorithm": ["adaquant"],
                "LearningRate": [
                    1e-5,
                ],
                # 'OptimDevice': ['cuda:0'],
                # 'InferDevice': ['cuda:0'],
                "EarlyStop": [False],
            },
        },
    }

    # for BF16
    search_space_bf16: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax],
        "activation_type": [ExtendedQuantType.QBFloat16],
        "weight_type": [ExtendedQuantType.QBFloat16],
        "quant_format": [ExtendedQuantFormat.QDQ],
        "include_cle": [False],
        "include_fast_ft": [False],
    }

    search_space_bf16_advanced: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax],
        "activation_type": [ExtendedQuantType.QBFloat16],
        "weight_type": [ExtendedQuantType.QBFloat16],
        "quant_format": [ExtendedQuantFormat.QDQ],
        "include_cle": [False],
        "include_fast_ft": [True],
        "extra_options": {
            "FastFinetune": {
                "DataSize": [1000],
                "FixedSeed": [1705472343],
                "BatchSize": [2],
                "NumIterations": [1000],
                "LearningRate": [0.00001],
                "OptimAlgorithm": ["adaquant"],
                # 'OptimDevice': ['cuda:0'],
                # 'InferDevice': ['cuda:0'],
                "EarlyStop": [False],
            }
        },
    }

    #  for BFP16
    search_space_bfp16: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax],
        "activation_type": [ExtendedQuantType.QBFP],
        "weight_type": [ExtendedQuantType.QBFP],
        "quant_format": [ExtendedQuantFormat.QDQ],
        "include_cle": [False],
        "include_fast_ft": [False],
        "extra_options": {
            "BFPAttributes": [
                {
                    "bfp_method": "to_bfp",
                    "axis": 1,
                    "bit_width": 16,
                    "block_size": 8,
                    "rounding_mode": 2,
                }
            ]
        },
    }

    search_space_bfp16_advanced: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax],
        "activation_type": [ExtendedQuantType.QBFP],
        "weight_type": [ExtendedQuantType.QBFP],
        "quant_format": [ExtendedQuantFormat.QDQ],
        "include_cle": [False],
        "include_fast_ft": [True],
        "extra_options": {
            "BFPAttributes": [
                {
                    "bfp_method": "to_bfp",
                    "axis": 1,
                    "bit_width": 16,
                    "block_size": 8,
                    "rounding_mode": 2,
                }
            ],
            "FastFinetune": {
                "DataSize": [1000],
                "FixedSeed": [1705472343],
                "BatchSize": [2],
                "NumIterations": [1000],
                "LearningRate": [0.00001],
                "OptimAlgorithm": ["adaquant"],
                # 'OptimDevice': ['cuda:0'],
                # 'InferDevice': ['cuda:0'],
                "EarlyStop": [False],
            },
        },
    }

    search_metric: str = "L2"
    search_algo: str = "grid_search"  # candidates: "grid_search", "random"
    search_evaluator = None
    search_metric_tolerance: float = 0.60001
    search_cache_dir: str = "./"
    search_output_dir: str = "./"
    search_log_path: str = "./auto_search.log"

    search_stop_condition: dict[str, any] = {
        "find_n_candidates": 1,
        "iteration_limit": 10000,
        "time_limit": 1000000.0,  # unit: second
    }


class ImageDataReader:
    def __init__(self, model_path: str, calibration_image_folder: str):
        self.enum_data = None
        self.data_list = self._preprocess_images(calibration_image_folder)
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = session.get_inputs()[0].name

        self.datasize = len(self.data_list)

    def _preprocess_images(self, image_folder: str):
        data_list = []
        img_names = [
            f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".JPEG")
        ]
        for name in img_names:
            input_image = cv2.imread(os.path.join(image_folder, name))
            input_image = cv2.resize(input_image, (640, 640))
            input_data = np.array(input_image).astype(np.float32)
            # Customer Pre-Process
            input_data = input_data.transpose(2, 0, 1)
            input_size = input_data.shape
            if input_size[1] > input_size[2]:
                input_data = input_data.transpose(0, 2, 1)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = input_data / 255.0
            data_list.append(input_data)

        return data_list

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: data} for data in self.data_list])
        return next(self.enum_data, None)

    def __getitem__(self, idx):
        return {self.input_name: self.data_list[idx]}

    def __len__(
        self,
    ):
        return self.datasize

    def rewind(self):
        self.enum_data = None

    def reset(self):
        self.enum_data = None


def main(args: argparse.Namespace) -> None:
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = args.input_model_path

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = args.dataset_path

    # get auto search config
    if args.auto_search_config == "default_auto_search":
        auto_search_config = AutoSearchConfig_Default()
    else:
        auto_search_config = args.default_auto_search

    # Get quantization configuration
    quant_config = get_default_config(args.config)
    config_copy = copy.deepcopy(quant_config)
    config_copy.calibrate_method = CalibrationMethod.MinMax
    config = Config(global_quant_config=config_copy)
    print(f"The configuration for quantization is {config}")

    # Create auto search instance
    auto_search_ins = auto_search.AutoSearch(
        config=config,
        auto_search_config=auto_search_config,
        model_input=input_model_path,
        calibration_data_reader=ImageDataReader(
            input_model_path,
            calibration_dataset_path,
        ),
    )

    # build search space
    # To reduce computational load for this demo, we have commented out the other predefined search spaces. Users are welcome to modify them based on their needs

    # fixed point
    # space1 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8)
    # space2 = auto_search_ins.build_all_configs(auto_search_config.search_space)
    # space3 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8_advanced)
    # space4 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8_advanced2)
    space5 = auto_search_ins.build_all_configs(auto_search_config.search_space_advanced)
    space6 = auto_search_ins.build_all_configs(auto_search_config.search_space_advanced2)

    # bf16 and bfp16
    # space7 = auto_search_ins.build_all_configs(auto_search_config.search_space_bf16)
    # space8 = auto_search_ins.build_all_configs(auto_search_config.search_space_bfp16)
    # space9 = auto_search_ins.build_all_configs(auto_search_config.search_space_bf16_advanced)
    # space10 = auto_search_ins.build_all_configs(auto_search_config.search_space_bfp16_advanced)
    # auto_search_ins.all_configs = space1 + space2 + space3 + space4 + space5 + space6 + space7 + space8 + space9 + space10
    auto_search_ins.all_configs = space5 + space6

    # Excute the auto search process
    auto_search_ins.search_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_model_path",
        help="Specify the input model to be quantized",
        type=str,
        default="./yolov8n.onnx",
        required=True,
    )
    parser.add_argument("--dataset_path", help="The path of the dataset for calibration", type=str, required=True)
    parser.add_argument(
        "--device",
        help="The device type of executive provider, it can be set to 'cpu', 'rocm' or 'cuda'",
        type=str,
        default="cpu",
    )
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="S8S8_AAWS")
    parser.add_argument(
        "--auto_search_config",
        help="The configuration for auto search quantizaiton setting",
        type=str,
        default="default_auto_search",
    )

    args = parser.parse_args()

    main(args)
