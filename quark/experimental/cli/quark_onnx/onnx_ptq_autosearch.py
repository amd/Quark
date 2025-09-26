#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import argparse
import copy

from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType

from quark.experimental.cli import base_cli
from quark.onnx import auto_search
from quark.onnx.quantization.config import Config, get_default_config

from .helper_utils import NpyDataReader


class AutoSearchConfig_Default:
    search_space: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.Entropy],
        "activation_type": [QuantType.QInt8],
        "weight_type": [QuantType.QInt8],
        "include_cle": [False],
        "include_fast_ft": [True],
        "extra_options": {
            "CalibMovingAverage": [
                True,
            ],
            "CalibMovingAverageConstant": [0.01],
            "FastFinetune": {
                "DataSize": [
                    5,
                ],
                "NumIterations": [10, 50],
                "OptimAlgorithm": ["adaround"],
                "LearningRate": [
                    0.1,
                    0.01,
                ],
            },
        },
    }

    search_metric: str = "L2"
    search_algo: str = "grid_search"  # candidates: "grid_search", "random"
    search_evaluator = None
    search_metric_tolerance: float = 1.00
    search_cache_dir: str = "./"
    search_output_dir: str = "./"
    search_log_path: str = "./auto_search.log"

    search_stop_condition: dict[str, any] = {
        "find_n_candidates": 2,
        "iteration_limit": 10000,
        "time_limit": 1000000.0,  # unit: second
    }


class OnnxAutoSearch_CLI(base_cli.BaseQuarkCLICommand):
    """
    CLI interface for Quark ONNX's AutoSearch feature which can explore many different configurations to help find the best configuration.
    """

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser):
        parser.add_argument("--model_name", help="Specify the input model name to be quantized", required=True)
        parser.add_argument("--input_model_path", help="Specify the input model to be quantized", required=True)
        parser.add_argument(
            "--output_model_path",
            help="Specify the path to save the quantized model",
            type=str,
            default="",
            required=False,
        )
        parser.add_argument(
            "--calibration_dataset_path",
            help="The path of the dataset for calibration",
            type=str,
            default="",
            required=True,
        )
        parser.add_argument("--num_calib_data", help="Number of samples for calibration", type=int, default=1000)
        parser.add_argument("--batch_size", help="Batch size for calibration", type=int, default=1)
        parser.add_argument(
            "--workers", help="Number of worker threads used during calib data loading.", type=int, default=1
        )
        parser.add_argument(
            "--device",
            help="The device type of executive provider, it can be set to 'cpu', 'rocm' or 'cuda'",
            type=str,
            default="cpu",
        )
        parser.add_argument(
            "--config", help="The configuration for quantization", type=str, default="S8S8_AAWS_ADAROUND"
        )
        parser.add_argument(
            "--auto_search_config",
            help="The configuration for auto search quantization setting",
            type=str,
            default="default_auto_search",
        )

    def run(self):
        args = self.args

        # `input_model_path` is the path to the original, unquantized ONNX model.
        input_model_path = args.input_model_path

        # `output_model_path` is the path where the quantized model will be saved.
        output_model_path = args.output_model_path

        # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
        calibration_dataset_path = args.calibration_dataset_path

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
        dr = NpyDataReader(calibration_image_folder=calibration_dataset_path, model_path=input_model_path, data_size=10)

        # Create auto search instance
        auto_search_ins = auto_search.AutoSearch(
            config=config,
            auto_search_config=auto_search_config,
            model_input=input_model_path,
            model_output=output_model_path,
            calibration_data_reader=dr,
        )

        # Excute the auto search process
        auto_search_ins.search_model()
