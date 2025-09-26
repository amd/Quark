#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys

sys.path.append("..")
import argparse
from typing import Any, List, Union

import numpy
import torch
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model
from utils.onnx_validate import load_loader

from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config.algorithm import AutoMixprecisionConfig
from quark.onnx.quantization.config.config import QConfig
from quark.onnx.quantization.config.data_type import Int8
from quark.onnx.quantization.config.spec import Int8Spec, Int16Spec, QLayerConfig


def post_process_top1(output: torch.tensor) -> float:
    _, preds_top1 = torch.max(output, 1)
    return preds_top1


def getAccuracy_top1(preds: Union[torch.tensor, list], targets: Union[torch.tensor, list]) -> float:
    assert len(preds) == len(targets)
    assert len(preds) > 0
    count = 0
    for i in range(len(preds)):
        pred = preds[i]
        target = targets[i]
        if pred == target:
            count += 1
    return count / len(preds)


def top1_acc(results: list[Union[torch.tensor, list[Any]]]) -> float:
    """
    Calculate the top1 accuracy of the model.
    :param results: the result of the model
    :return: the top1 accuracy
    """
    timm_model_name = model_name
    calib_data_path = calibration_dataset_path

    timm_model = create_model(
        timm_model_name,
        pretrained=False,
    )

    data_config = resolve_data_config(model=timm_model, use_test_size=True)

    loader = create_loader(
        create_dataset("", calib_data_path),
        input_size=data_config["input_size"],
        batch_size=20,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=2,
        crop_pct=data_config["crop_pct"],
    )
    target = []
    for _, labels in loader:
        target.extend(labels.data.tolist())
    outputs_top1 = post_process_top1(torch.tensor(numpy.squeeze(numpy.array(results))))
    top1_acc = getAccuracy_top1(outputs_top1, target)
    return round(top1_acc, 2)


class CalibrationDataReader:
    def __init__(self, dataloader):
        super().__init__()
        self.iterator = iter(dataloader)

    def get_next(self) -> dict:
        try:
            return {"input": next(self.iterator)[0].numpy()}
        except Exception:
            return None


def main(args: argparse.Namespace) -> None:
    # `model_name` is the name of the original, unquantized ONNX model.
    global model_name
    model_name = args.model_name

    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = args.input_model_path

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = args.output_model_path

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    global calibration_dataset_path
    calibration_dataset_path = args.calibration_dataset_path

    # `dr` (Data Reader) is an instance of ResNet50DataReader, which is a utility class that
    # reads the calibration dataset and prepares it for the quantization process.
    data_loader = load_loader(model_name, calibration_dataset_path, args.batch_size, args.workers)
    dr = CalibrationDataReader(data_loader)

    # Get quantization configuration
    if args.config == "S16S16_MIXED_S8S8":
        activation_spec = Int16Spec()
        weight_spec = Int16Spec()
        algo_config = [
            AutoMixprecisionConfig(
                l2_target=None,
                top1_acc_target=0.02,
                evaluate_function=top1_acc,
                act_target_quant_type=Int8,
                weight_target_quant_type=Int8,
                output_index=0,
            )
        ]
    else:
        activation_spec = Int8Spec()
        weight_spec = Int8Spec()
        activation_spec.set_symmetric(False)
        algo_config = []

    config = QConfig(
        global_config=QLayerConfig(activation=activation_spec, weight=weight_spec),
        algo_config=algo_config,
        Percentile=99.9999,
        Int32Bias=False,
        Int16Bias=False,
    )
    print(f"The configuration for quantization is {config}")

    # Create an ONNX quantizer
    quantizer = ModelQuantizer(config)

    # Quantize the ONNX model
    quantizer.quantize_model(input_model_path, output_model_path, dr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", help="Specify the input model name to be quantized", required=True)
    parser.add_argument("--input_model_path", help="Specify the input model to be quantized", required=True)
    parser.add_argument(
        "--output_model_path", help="Specify the path to save the quantized model", type=str, default="", required=False
    )
    parser.add_argument(
        "--calibration_dataset_path",
        help="The path of the dataset for calibration",
        type=str,
        default="",
        required=False,
    )
    parser.add_argument(
        "--include_mixed_precision", action="store_true", help="Optimize the models using mixed_precision"
    )
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
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="S16S16_MIXED_S8S8")

    args = parser.parse_args()

    main(args)
