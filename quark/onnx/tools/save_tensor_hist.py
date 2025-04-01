#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
'''
A tool for showing the activation distribution of a model.

    Example : python -m quark.onnx.tools.save_tensor_hist --input_model [INPUT_MODEL_PATH] --data_path [CALIB_DATA_PATH]  --output_path [OUTPUT_PATH]

'''

import os
from quark.shares.utils.log import ScreenLogger
import argparse
import numpy as np
import onnx
import tempfile
import pathlib
import onnxruntime as ort
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from quark.onnx.calibrate import create_calibrator_float_scale
from quark.onnx.quant_utils import (CachedDataReader, RandomDataReader, check_and_create_path)
from onnxruntime.quantization.calibrate import (CalibrationDataReader, CalibrationMethod, CalibraterBase)

from typing import List, Dict, Any, Optional, Iterator
from numpy.typing import NDArray

logger = ScreenLogger(__name__)

preprocess_range = "[-1,1]"


# Raw data means binary format which had been pre-processed
def load_raw_data(data_path: str, file_names: List[str], input_shape: List[int]) -> Dict[str, NDArray[np.float32]]:
    data_dict = {}
    for file_name in file_names:
        with open(os.path.join(data_path, file_name), "rb") as f:
            raw_data = f.read()
            data_array = np.frombuffer(raw_data, dtype=np.float32)
            data_dict[file_name] = np.reshape(data_array, input_shape)
    return data_dict


# Npy data means the data was stored in numpy array format
def load_npy_data(data_path: str, file_names: List[str], input_shape: List[int]) -> Dict[str, NDArray[Any]]:
    data_dict = {}
    for file_name in file_names:
        npy_data = np.load(os.path.join(data_path, file_name))
        npy_data = npy_data.transpose(1, 2, 0)
        input_data = np.expand_dims(npy_data, axis=0)
        assert (list(input_data.shape) == input_shape
                and "{} data shape {} does not match expected {}".format(file_name, input_data.shape, input_shape))
        data_dict[file_name] = input_data
    return data_dict


# Img data means image files and need pre-processing
# - Loaded image's shape : (H, W, C)
# - Model input's shape : [N, C, H, W] or [N, H, W, C]
def load_img_data(data_path: str, file_names: List[str], input_shape: List[int]) -> Dict[str, NDArray[np.float32]]:
    import cv2
    print(f"load image data and pre-process with range {preprocess_range} expected shape {input_shape}")

    data_dict = {}

    for file_name in file_names:
        img_data = cv2.imread(os.path.join(data_path, file_name))

        if input_shape[3] == 3:
            input_shape_copy = [input_shape[0], input_shape[3], input_shape[1],
                                input_shape[2]]  # [N, H, W, C] -> [N, C, H, W]
        else:
            input_shape_copy = input_shape  # [N, C, H, W]

        if (img_data.shape[0] != input_shape_copy[2] or img_data.shape[1] != input_shape_copy[3]):
            img_data = cv2.resize(img_data, (input_shape_copy[2], input_shape_copy[3]))

        if input_shape[1] == 3:
            img_data = img_data.transpose(2, 0, 1)
        input_data = img_data.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        if preprocess_range == "[-1,1]":
            input_data = (input_data / 255. - 0.5) * 2.
        elif preprocess_range == "[0,1]":
            input_data = input_data / 255.

        data_dict[file_name] = input_data

    return data_dict


# Used PIL instead of opencv
def load_img_data2(data_path: str, file_names: List[str], input_shape: List[int]) -> Dict[str, NDArray[np.float32]]:
    from PIL import Image
    print(f"load image data and pre-process with range {preprocess_range} expected shape {input_shape}")

    data_dict = {}

    for file_name in file_names:
        input_image = Image.open(os.path.join(data_path, file_name))

        if input_shape[3] == 3:
            input_shape_copy = [input_shape[0], input_shape[3], input_shape[1],
                                input_shape[2]]  # [N, H, W, C] -> [N, C, H, W]
        else:
            input_shape_copy = input_shape  # [N, C, H, W]

        if (input_image.size[1] != input_shape_copy[2] or  # Image.size = (W, H)
                input_image.size[0] != input_shape_copy[3]):
            input_image_new = input_image.resize((input_shape_copy[2], input_shape_copy[3]))

        input_data = np.array(input_image_new).astype(np.float32)
        if input_shape[1] == 3:
            input_data = input_data.transpose(2, 0, 1)
        input_data = np.expand_dims(input_data, axis=0)

        if preprocess_range == "[-1,1]":
            input_data = (input_data / 255. - 0.5) * 2.
        elif preprocess_range == "[0,1]":
            input_data = input_data / 255.

        data_dict[file_name] = input_data

    return data_dict


# Load data from data path and support raw data, npy data and image data,
# return a dict, key is file name and value is numpy arrary
def load_data(data_path: str, input_shape: List[int]) -> Dict[str, NDArray[np.float32]]:
    files = [f for f in os.listdir(data_path) if (f.endswith('.png') or f.endswith('.jpg'))]
    if files != []:
        print("Loading image data from {}".format(data_path))
        return load_img_data2(data_path, files, input_shape)
    else:
        files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        if files != []:
            print("Loading npy data from {}".format(data_path))
            return load_npy_data(data_path, files, input_shape)
        else:
            files = [
                f for f in os.listdir(data_path) if (f.endswith('.bin') or f.endswith('.raw') or f.endswith('.data'))
            ]
            if files != []:
                print("Loading raw data from {}".format(data_path))
                return load_raw_data(data_path, files, input_shape)
            else:
                raise RuntimeError("Not found data in {}".format(data_path))


# Load raw data according to input name
def load_raw_data_by_input_name(data_path: str, file_names: List[str], input_shape: List[int], model_path: str,
                                input_name: str) -> Dict[str, NDArray[np.float32]]:
    origin_name = input_name

    data_dict = {}
    for file_name in file_names:
        if len(file_name) > len(origin_name) and file_name[0:len(origin_name)] == origin_name and file_name[len(
                origin_name)] == "_":
            with open(os.path.join(data_path, file_name), "rb") as f:
                raw_data = f.read()
                data_array = np.frombuffer(raw_data, dtype=np.float32)
                data_dict[file_name[len(origin_name):]] = np.reshape(data_array, input_shape)
    return data_dict


# Load data according to input name
def load_data_by_input_name(data_path: str, input_shape: List[int], model_path: str,
                            input_name: str) -> Dict[str, NDArray[np.float32]]:
    files = [f for f in os.listdir(data_path) if (f.endswith('.bin') or f.endswith('.raw') or f.endswith('.data'))]
    if files != []:
        print("Loading raw data from {} for input {}".format(data_path, input_name))
        return load_raw_data_by_input_name(data_path, files, input_shape, model_path, input_name)
    else:
        raise RuntimeError("Not found data in {} for input {}".format(data_path, input_name))


class HistDataReader(RandomDataReader):

    def __init__(self, model_path: str, data_path: str, input_shape: Dict[str, List[int]] = {}):
        """
        :param model_path : Full path of the input model.
        :param data_path  : Full path of the input data.
        :param input_shape: If dynamic axes of inputs require specific value, users should provide its shapes.
                            The basic format of shape for single input is `list(int)` or `tuple(int)`,
                            and all dimensions should have concrete values (batch dimensions can be set to 1).
                            For example, input_shape=[1, 3, 224, 224] or input_shape=(1, 3, 224, 224).
                            If the model has multiple inputs, it can be fed in `list(shape)` format,
                            where the list order is the same as the onnxruntime got inputs.
                            For example, input_shape=[[1, 1, 224, 224], [1, 2, 224, 224]] for 2 inputs.
                            Moreover, it is possible to use `dict{name:shape}` to specify a certain input,
                            for example, input_shape={"image":[1, 3, 224, 224]} for the input named "image".
        """

        self._data_path = data_path
        self.enum_data_iter: Optional[Iterator[Dict[str, NDArray[np.float32]]]] = None
        self.data_dict: Dict[str, List[NDArray[np.float32]]] = {}
        super().__init__(model_path, input_shape)

    def get_next(self) -> Optional[Dict[str, NDArray[np.float32]]]:
        """
        Get next feed data
        :return: feed dict for the model
        """
        if self.enum_data_iter is None:
            so = ort.SessionOptions()
            session = ort.InferenceSession(self._model_path, so, providers=['CPUExecutionProvider'])

            for input_index, input_node in enumerate(session.get_inputs()):
                input_name = self._get_input_name(input_node)
                input_shape = self._parse_input_shape(input_index, input_name)
                if input_shape == [] or input_shape is None:
                    input_shape = self._get_input_shape(input_node)
                input_type = self._get_input_type(input_node)

                # load data from data path
                data_dict: Dict[str, NDArray[np.float32]] = {}

                if len(session.get_inputs()) > 1 or len(  # for audio models
                        session.get_outputs()) >= 5:  # for model K1
                    data_dict = load_data_by_input_name(self._data_path, input_shape, self._model_path, input_name)
                else:
                    data_dict = load_data(self._data_path, input_shape)

                if len(data_dict) <= 0:
                    raise RuntimeError("Load data from the path {} failed for input{} {}".format(
                        self._data_path, input_index, input_name))
                else:
                    print("Load data from the path {} for input{} with {} samples ".format(
                        self._data_path, input_index, len(data_dict)))

                # save to data_dict
                for key, value in data_dict.items():
                    if value.dtype is not input_type:
                        value = value.astype(input_type)
                    if key in self.data_dict:
                        self.data_dict[key].append(value)
                    else:
                        self.data_dict[key] = [value]

                print("Real input name {} shape {} type {} ".format(input_name, input_shape, input_type))

            self.enum_data_list = []

            for arrays in self.data_dict.values():
                enum_data = {}
                for i, arr in enumerate(arrays):
                    name = self._get_input_name(session.get_inputs()[i])
                    enum_data[name] = arr
                self.enum_data_list.append(enum_data)

            self.enum_data_iter = iter(self.enum_data_list)

        return next(self.enum_data_iter, None)


def save_figure(calibrator: CalibraterBase, saved_path: Optional[str] = None) -> None:
    if saved_path is None:
        saved_path = "./"

    print("The tensors hist saved path: {}".format(saved_path))

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=len(calibrator.collector.histogram_dict), desc="Saving Histograms")

    for tensor_name, tensor_value in calibrator.collector.histogram_dict.items():
        # Replace the tensor_name
        tensor_name = tensor_name.replace("/", "_")
        tensor_name = tensor_name.replace(".", "_")
        tensor_name = tensor_name.replace(":", "_")

        # Extract histogram data
        tensor_freq = tensor_value[0]
        tensor_bins = tensor_value[1]

        # Plot the histogram
        bar_width = tensor_bins[1] - tensor_bins[0]
        plt.bar(tensor_bins[:-1], tensor_freq, width=bar_width)
        plt.text(tensor_bins[-1], tensor_freq[-1], str(tensor_freq[-1]), ha='center', va='bottom')

        # Construct the file path to save the histogram
        model_hist_path = Path(saved_path).joinpath(tensor_name + ".png").as_posix()

        # Add title and labels
        plt.title(tensor_name)
        plt.xlabel('Values')
        plt.ylabel('Frequency')

        # Save the histogram
        plt.savefig(model_hist_path)

        # Clear the current figure and close it to release resources
        plt.clf()
        plt.close()

        # Update progress bar
        progress_bar.update(1)

    # Close progress bar
    progress_bar.close()

    print("The tensor hist saved done")


# Generate the percentile calibrator
# Collect all data then save tensors to picture
# Reset the DataReader
def save_tensor_hist_figure(input_model_path: str,
                            dr: CalibrationDataReader,
                            output_figure_path: Optional[str] = None) -> None:

    # Need to reload & save the file if the input_model_path does not have write permissions
    origin_input_model = onnx.load(input_model_path)
    tmp_path = tempfile.TemporaryDirectory(prefix="vai.tools.")
    tmp_model = Path(tmp_path.name).joinpath("converted.onnx").as_posix()
    onnx.save(origin_input_model, tmp_model)
    # Generate the calibrator
    # Need to save the augmented_model.onnx to tmp_path
    calibrator = create_calibrator_float_scale(
        Path(tmp_model),
        None,
        augmented_model_path=Path(tmp_path.name).joinpath("augmented_model.onnx").as_posix(),
        calibrate_method=CalibrationMethod.Percentile,
        use_external_data_format=False,
        execution_providers=['CPUExecutionProvider'],
        extra_options={"symmetric": False},
    )
    # Warp the DataReader
    cached_data_reader = CachedDataReader(dr)
    # Collect data and save data to histogram_dict
    calibrator.collect_data(cached_data_reader)

    logger.info("Saving the tensor histogram")

    save_figure(calibrator, output_figure_path)
    # Reset the CachedDataReader
    cached_data_reader.reset_iter()


# generate the output tensor histogram
def get_tensor_hist() -> None:
    parser = argparse.ArgumentParser(
        f"{os.path.basename(__file__)}:{get_tensor_hist.__name__}",
        description="""
                                    Generated the output tensor histogram
                                    Provide input_model path and DataReader path, output_path""",
    )

    parser.add_argument("--input_model",
                        type=pathlib.Path,
                        help="Provide path to ONNX model to generate histogram.",
                        required=True)

    parser.add_argument("--data_path", type=pathlib.Path, help="Provide the data reader path.", required=True)

    parser.add_argument("--output_path", type=pathlib.Path, help="Provide path to write histogram figure.")

    args = parser.parse_args()

    abs_path = check_and_create_path(args.output_path)

    # generate the default DataReader
    dr = HistDataReader(args.input_model, args.data_path, {})

    save_tensor_hist_figure(args.input_model, dr, abs_path)


if __name__ == '__main__':
    get_tensor_hist()
