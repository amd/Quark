#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

from quark.onnx import AutoSearchPro

base_config = {
    "search_space": {
        "activation": ["Int8Spec"],
        "activation_params": {
            "symmetric": [False],
            "calibration_method": ["MinMax", "Percentile", "LayerwisePercentile"],
            "scale_type": ["Float32"],
            "quant_granularity": ["Tensor"],
            "only_if": "activation",
        },
        "weight": ["Int8Spec"],
        "weight_params": {
            "symmetric": [True],
            "calibration_method": ["MinMax"],
            "only_if": "weight",
        },
        "algorithms": ["adaround", "adaquant"],
        "cle_algo": ["cle"],
        "cle_params": {"cle_steps": [-1, 1], "only_if": {"cle_algo": "cle"}},
        "adaquant_params": {
            "num_iterations": {"type": "int", "low": 100, "high": 200, "step": 100},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
            "optim_device": ["cuda:0"],
            "infer_device": ["cuda:0"],
            "data_size": [100],
            "batch_size": [2],
            "early_stop": [True],
            "only_if": {"algorithms": "adaquant"},
        },
        "adaround_params": {
            "num_iterations": {"type": "int", "low": 100, "high": 200, "step": 100},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 1e-1, "log": True},
            "optim_device": ["cuda:0"],
            "infer_device": ["cuda:0"],
            "only_if": {"algorithms": "adaround"},
        },
        "specific_layer_config": [None],
        "layer_type_config": [None],
        "exclude": [None],
        "use_external_data_format": [False],
        "OptimizeModel": [True],
    },
    "n_trials": 20,
    "n_jobs": 1,
    "output_dir": "./output",
    "temp_dir": "./temp_dir",
    "search_algo": "TPE",
    "search_evaluator": None,  # Custom or built-in function
    "search_metric": "L2",
    "direction": "minimize",
    "base_framework": "onnx",
    "study_storage_db": "auto_search.db",
    "load_study_if_exists": True,
    "study_name": "AutoSearch",
    "model_input": None,
    "calib_data_reader": None,
    "eval_data_reader": None,
    "two_stage_search": True,
    "plot_results": False,
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

    def reset(self):
        self.enum_data = None


def main(args: argparse.Namespace) -> None:
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = args.input_model_path

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_datareader = ImageDataReader(input_model_path, args.dataset_path)

    # get auto search config
    search_config = base_config
    search_config["model_input"] = args.input_model_path
    search_config["calib_data_reader"] = calibration_datareader
    # search_config["search_algo"] = "Grid"
    search_config["two_stage_search"] = True

    # Get quantization configuration
    auto_search_pro_ins = AutoSearchPro(search_config)
    best_params = auto_search_pro_ins.run()

    print(f"The best configuration for quantization is {best_params}")


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

    args = parser.parse_args()

    main(args)
