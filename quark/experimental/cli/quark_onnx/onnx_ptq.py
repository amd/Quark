#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Wrapper for the "quantize-onnx" subcommand.
# This basically comprises a combo of the files called quantize_quark.py from the examples/onnx/.... examples.
import argparse
import copy

import onnxruntime as ort
from onnxruntime.quantization.calibrate import CalibrationMethod

from quark.experimental.cli import base_cli
from quark.onnx import LayerWiseMethod, ModelQuantizer
from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.config.custom_config import get_default_config

from .helper_utils import ImageDataReader, get_calib_dataset, get_model_input_name, is_number, parse_subgraphs_list

DEFAULT_ADAROUND_PARAMS = {
    "DataSize": 1000,
    "FixedSeed": 1705472343,
    "BatchSize": 2,
    "NumIterations": 1000,
    "LearningRate": 0.1,
    "OptimAlgorithm": "adaround",
    "OptimDevice": "cpu",
    "InferDevice": "cpu",
    "EarlyStop": True,
}

DEFAULT_ADAQUANT_PARAMS = {
    "DataSize": 1000,
    "FixedSeed": 1705472343,
    "BatchSize": 2,
    "NumIterations": 1000,
    "LearningRate": 0.00001,
    "OptimAlgorithm": "adaquant",
    "OptimDevice": "cpu",
    "InferDevice": "cpu",
    "EarlyStop": True,
}


class OnnxPTQ_CLI(base_cli.BaseQuarkCLICommand):
    """
    This class is the Quark CLI onnx-ptq subcommand.

    Design
    ------
    This like quark/experimental/cli/torch_llm_ptq.py is built as a superset of all the ONNX examples from the Quark Examples package.
    Many of these examples were superficially similar i.e. named quantize_quark.py and had many of the same arguments.
    Where possible they have been consolidated and the interface has been kept as close as possible to the example version
    However, some scripts make this difficult e.g. one might implicitly only work on image models while another would only work on language models. Where possible this CLI tries to still accomodate this but where not, new arguments have had to be added.
    """

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser):
        parser.add_argument("--adaquant", action="store_true")
        parser.add_argument("--adaround", action="store_true")
        parser.add_argument(
            "--auto_search_config",
            help="The configuration for auto search quantization setting",
            type=str,
            default="default_auto_search",
        )
        parser.add_argument("--cle", "--include_cle", action="store_true")
        parser.add_argument("--use_moving_average", action="store_true", help="Using EMA when calibrating")

        parser.add_argument(
            "--calib_method", help="Specify the calibration method", type=str, default="", required=False
        )
        parser.add_argument("--calib_dataset_name", help="Specify a specific calibration dataset to use", default=None)
        parser.add_argument(
            "--calib_data_path",
            "--calibration_dataset_path",
            help="Specify the calibration data path for quantization",
            default=None,
        )
        parser.add_argument(
            "--calib_image_dim",
            type=int,
            help="Dimension, in pixels, to resize calibration image to. Give one side of square. e.g.'224' for 224x224 Resnet50, and '640' for 640x640 yolov8n.",
            default=224,
        )
        parser.add_argument("--num_calib_data", help="Number of samples for calibration", type=int, default=1000)
        parser.add_argument("--config", help="The configuration for quantization", type=str, default="XINT8")
        parser.add_argument(
            "--device",
            help="The device type of executive provider, it can be set to 'cpu', 'rocm' or 'cuda'",
            type=str,
            default="cpu",
        )

        parser.add_argument("--exclude_nodes", help="The names of excluding nodes", type=str, default="")
        parser.add_argument("--exclude_subgraphs", help="The lists of excluding subgraphs", type=str, default="")
        parser.add_argument("--input_model_path", help="Specify the input model to be quantized", required=True)
        parser.add_argument("--learning_rate", help="The learing_rate for fastfinetune", type=float, default=0.1)
        parser.add_argument("--num_iters", help="The number of iterations for fastfinetune", type=int, default=1000)
        parser.add_argument(
            "--output_model_path",
            help="Specify the path to save the quantized model",
            type=str,
            default="quantized.onnx",
        )
        parser.add_argument("--save_as_external_data", action="store_true")

        parser.add_argument("--model_name", help="The name of the model", type=str, default=None)
        parser.add_argument("--batch_size", help="Batch size for calibration", type=int, default=1)
        parser.add_argument(
            "--workers", help="Number of worker threads used during calib data loading.", type=int, default=1
        )

        parser.add_argument("--include_sq", action="store_true", help="Optimize the models using SmoothQuant")
        parser.add_argument("--sq_alpha", help="Define the alpha for smooth quant", type=float, default=0.5)

        parser.add_argument("--use_gptq", action="store_true", help="Optimize the models using GPTQ")

        parser.add_argument("--include_rotation", action="store_true", help="Optimize the models using rotation")
        parser.add_argument("--hidden_size", help="Dim of the R1 rotation tensor", type=int, default=4096)
        parser.add_argument("--num_head", help="Number of heads in the QKV proj", type=int, default=32)
        parser.add_argument(
            "--use_random_had", action="store_true", help="Activate that randomly generate Hadamard matrix"
        )
        parser.add_argument(
            "--r_config_path",
            help="Specify the path to load the rotation configuration",
            type=str,
            default="",
            required=False,
        )
        parser.add_argument(
            "--use_crypto", action="store_true", help="Perform quantization in memory and encrypt any temporary files"
        )

    def run(self):
        args = self.args

        quant_config = get_default_config(args.config)
        config_copy = copy.deepcopy(quant_config)

        quant_config.crypto_mode = args.use_crypto

        config_copy.use_external_data_format = args.save_as_external_data

        config_copy.optimize_model = False
        config_copy.extra_options["RemoveInputInit"] = False
        config_copy.extra_options["SimplifyModel"] = False
        config_copy.extra_options["CopySharedInit"] = None
        config_copy.extra_options["OpTypesToExcludeOutputQuantization"] = ["MatMul", "Gemm"]

        if args.adaround or args.adaquant:
            config_copy.include_fast_ft = True
            if args.adaround:
                config_copy.extra_options["FastFinetune"] = DEFAULT_ADAROUND_PARAMS
            if args.adaquant:
                config_copy.extra_options["FastFinetune"] = DEFAULT_ADAQUANT_PARAMS
            if args.learning_rate:
                config_copy.extra_options["FastFinetune"]["LearningRate"] = args.learning_rate
            if args.num_iters:
                config_copy.extra_options["FastFinetune"]["NumIterations"] = args.num_iters
        config_copy.include_cle = args.cle

        if args.exclude_nodes:
            exclude_nodes = args.exclude_nodes.split(";")
            exclude_nodes = [node_name.strip() for node_name in exclude_nodes]
            config_copy.nodes_to_exclude = exclude_nodes

        if args.exclude_subgraphs:
            exclude_subgraphs = parse_subgraphs_list(args.exclude_subgraphs)
            config_copy.subgraphs_to_exclude = exclude_subgraphs

        config_copy.include_sq = args.include_sq

        if args.include_sq:
            config_copy.extra_options["SmoothAlpha"] = args.sq_alpha

        config_copy.extra_options["OpTypesToExcludeOutputQuantization"] = ["MatMul", "Gemm"]

        if args.config == "INT8_TRANSFORMER_DEFAULT":
            config_copy.extra_options["UseGPTQ"] = args.use_gptq
        elif args.config == "MATMUL_NBITS":
            config_copy.extra_options["MatMulNBitsParams"]["AccuracyLevel"] = 0
            config_copy.extra_options["MatMulNBitsParams"]["Algorithm"] = "GPTQ"

        config_copy.extra_options["GPTQParams"] = {
            "MSE": False,
            "GroupSize": 128,
            "ActOrder": False,
            "PerChannel": True,
        }

        config_copy.extra_options["CalibMovingAverage"] = args.use_moving_average

        config_copy.include_rotation = args.include_rotation
        if args.include_rotation:
            config_copy.extra_options["RMatrixDim"] = args.hidden_size
            config_copy.extra_options["RConfigPath"] = args.r_config_path
            config_copy.extra_options["UseRandomHad"] = args.use_random_had
            # should be disabled when using rotation
            config_copy.include_cle = False

        config_copy.extra_options["ActivationSymmetric"] = True  # Must activate
        config_copy.extra_options["MatMulConstBOnly"] = True  # Must activate

        if args.calib_method == "minmax":
            config_copy.calibrate_method = CalibrationMethod.MinMax
        elif args.calib_method == "layerwise_percentile":
            config_copy.calibrate_method = LayerWiseMethod.LayerWisePercentile
        elif "percentile" in args.calib_method:
            config_copy.calibrate_method = CalibrationMethod.Percentile
            # ratio is encoded in the method name
            if args.calib_method != "percentile":
                percentile_ratio = args.calib_method.split("+")[-1]  # get num
                assert is_number(percentile_ratio)
                percentile_ratio = float(percentile_ratio)
                config_copy.extra_options["Percentile"] = percentile_ratio
                print(f"Use percentile calibration with ratio {percentile_ratio}.")

        # need to get input names to help construct the calibration data reader
        session = ort.InferenceSession(args.input_model_path)
        input_names = [input.name for input in session.get_inputs()]

        if args.calib_dataset_name:
            calib_datareader = get_calib_dataset(
                args.input_model_path,
                args.calib_dataset_name,
                args.num_calib_data,
                args.device,
                args.hidden_size,
                args.num_head,
                input_names,
            )
        else:
            model_input_name = get_model_input_name(args.input_model_path)
            if args.calib_data_path is not None:
                calib_datareader = ImageDataReader(
                    calibration_image_folder=args.calib_data_path,
                    input_name=model_input_name,
                    model_name=args.model_name,
                    image_dim=args.calib_image_dim,
                )
            else:
                calib_datareader = None

        quant_config = Config(global_quant_config=config_copy)
        quantizer = ModelQuantizer(quant_config)  # Note(Anton) this is quark.onnx.ModelQuantizer()
        quantizer.quantize_model(args.input_model_path, args.output_model_path, calib_datareader)
