#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime
from onnx.onnx_ml_pb2 import NodeProto
from onnxruntime.quantization.calibrate import CalibrationDataReader

from quark.onnx.quant_utils import create_infer_session_for_onnx_model
from quark.shares.utils.log import ScreenLogger, log_errors

logger = ScreenLogger(__name__)


class CachedDataReader(CalibrationDataReader):  # type: ignore
    """
    A CalibrationDataReader cached input data from the user provided data reader.
    """

    @log_errors
    def __init__(
        self,
        dr: CalibrationDataReader,
        data_size: int | None = None,
        convert_nchw_to_nhwc: bool = False,
        quantize_fp16: bool = False,
    ):
        """
        :param dr : Original calibration data reader
        """
        self.data_size = data_size
        self.convert_nchw_to_nhwc = convert_nchw_to_nhwc
        self._data_cache = []
        self.quantize_fp16 = quantize_fp16
        n = 1
        while True:
            inputs = dr.get_next()
            if not inputs or self.data_size is not None and n > self.data_size:
                break
            n = n + 1
            if self.quantize_fp16:
                new_inputs = {}
                for input_name, input_array in inputs.items():
                    if input_array.dtype == np.float32:
                        new_inputs[input_name] = input_array.astype(np.float16)
                    else:
                        new_inputs[input_name] = input_array
                inputs = new_inputs
            if self.convert_nchw_to_nhwc:
                for input_name, input_array in inputs.items():
                    shape_tuple = input_array.shape
                    if len(shape_tuple) != 4:
                        logger.info(
                            f"Expected 4-dimension output shape but got {shape_tuple}, skip the nchw to nhwc conversion."
                        )
                        continue

                    C, H, W = shape_tuple[1:]
                    if not all(isinstance(_, int) for _ in [C, H, W]):
                        logger.warning(
                            f"Expected integer output shape but got [{C}, {H}, {W}], skip the nchw to nhwc conversion."
                        )
                        continue

                    if not (H > C and W > C):
                        logger.warning(f"Expected H,W > C but got [{C}, {H}, {W}].")
                    inputs[input_name] = np.transpose(input_array, axes=(0, 2, 3, 1))
            self._data_cache.append(inputs)

        if len(self._data_cache) == 0:
            raise ValueError("No data in the input calibration data reader")
        else:
            logger.debug(f"Obtained calibration data with {len(self._data_cache)} iters")

        self.enum_data_dicts = iter(self._data_cache)

    def __len__(self) -> int:
        """
        Get the num of inputs
        """
        return len(self._data_cache)

    def reset_iter(self) -> None:
        """
        Recreate the iter so that it can iterate again
        """
        self.enum_data_dicts = iter(self._data_cache)
        logger.debug("Reset the iter of the data reader once")

    def get_next(self) -> dict[str, np.ndarray[Any, Any]] | None:
        """
        Get next feed data
        :return: feed dict for the model
        """
        return next(self.enum_data_dicts, None)


class RandomDataReader(CalibrationDataReader):  # type: ignore
    """
    A CalibrationDataReader using random data for rapid quantiation.
    """

    def __init__(
        self,
        model_input: Union[str, Path, onnx.ModelProto],
        input_shape: dict[str, list[int]] = {},
        input_data_range: dict[str, list[int]] | None = None,
    ):
        """
        :param Union[str, Path, onnx.ModelProto] model_input: Full path or ModelProto of the input model.
        :param input_shape: If dynamic axes of inputs require specific value, users should provide its shapes.
                            The basic format of shape for single input is `list(int)` or `tuple(int)`,
                            and all dimensions should have concrete values (batch dimensions can be set to 1).
                            For example, input_shape=[1, 3, 224, 224] or input_shape=(1, 3, 224, 224).
                            If the model has multiple inputs, it can be fed in `list(shape)` format,
                            where the list order is the same as the onnxruntime got inputs.
                            For example, input_shape=[[1, 1, 224, 224], [1, 2, 224, 224]] for 2 inputs.
                            Moreover, it is possible to use `dict{name:shape}` to specify a certain input,
                            for example, input_shape={"image":[1, 3, 224, 224]} for the input named "image".
        :param input_data_range: How to deal with input data range in the generated random data.
                            Default is none which means ignore data type, otherwise consider data type.
        """
        self._model_input = model_input
        self._input_shape = input_shape
        self._input_data_range: dict[str, list[int]] | None = input_data_range

        self.enum_data_dicts: Iterator[dict[str, np.ndarray[Any, Any]]] | None = None
        self.batch_size = 1

    def _parse_input_shape(self, input_index: int, input_name: str) -> Any:
        """
        Parse input shape of model from user's input
        :param input_index: the input index in session.get_inputs()
        :param input_name: the input name string
        :return: input shape required for the input node
        """

        def _deal_shape_value(list_or_tuple_shape: Union[int, list[int], Any]) -> Any:
            if not isinstance(list_or_tuple_shape, (list, tuple)):
                logger.warning(f"Invalid input shape {list_or_tuple_shape}")
                return []

            input_shape = []
            for index, shape in enumerate(list_or_tuple_shape):
                if isinstance(shape, int) and shape > 0:
                    input_shape.append(shape)
                else:
                    if index == 0:
                        input_shape.append(self.batch_size)
                    else:
                        logger.warning(f"Invalid input shape {list_or_tuple_shape} in #{index} : {shape}")
                        return []
            return input_shape

        if isinstance(self._input_shape, dict):
            if input_name in self._input_shape.keys():
                return _deal_shape_value(self._input_shape[input_name])
            elif self._input_shape != {}:
                raise ValueError(
                    f'Input name "{input_name}" is not found in RandomDataReaderInputShape. Please check whether '
                    'the parameter config.global_quant_config.extra_options["RandomDataReaderInputShape"] is correct.'
                )
        else:
            raise TypeError("The RandomDataReaderInputShape must be a Dict[str, List[int]] type!")

        return []

    def _get_input_name(self, input_node: NodeProto) -> str:
        """
        :param input_node: the input node
        :return: name of the input node
        """
        input_name = input_node.name
        return input_name

    def _get_input_shape(self, input_node: NodeProto) -> list[int]:
        """
        :param input_node: the input node
        :return: input shape of the input node
        """
        input_shape = []

        if len(input_node.shape):
            for index, shape in enumerate(input_node.shape):
                if isinstance(shape, int) and shape > 0:
                    input_shape.append(shape)
                else:
                    if index == 0:
                        input_shape.append(self.batch_size)  # default batch size
                    elif index == 1:
                        if len(input_node.shape) == 2:
                            input_shape.append(16)  # maybe sequence length
                        elif len(input_node.shape) == 4:
                            input_shape.append(3)  # maybe image channel
                        else:
                            input_shape.append(1)
                    elif index == 2:
                        if len(input_node.shape) == 4:
                            input_shape.append(32)  # maybe image height
                        else:
                            input_shape.append(1)
                    elif index == 3:
                        if len(input_node.shape) == 4:
                            input_shape.append(32)  # maybe image width
                        else:
                            input_shape.append(1)
                    else:
                        input_shape.append(1)  # unknown or None

        if input_shape == []:
            # workaround empty shape
            return [self.batch_size]
        else:
            return input_shape

    def _get_input_type(self, input_node: NodeProto) -> Any:
        """
        :param input_node: the input node
        :return: data type of the input node
        """
        input_type: Union[Any, None] = None

        if "tensor(int8)" in input_node.type:
            input_type = np.int8
        elif "tensor(uint8)" in input_node.type:
            input_type = np.uint8
        elif "tensor(int16)" in input_node.type:
            input_type = np.int16
        elif "tensor(uint16)" in input_node.type:
            input_type = np.uint16
        elif "tensor(int32)" in input_node.type:
            input_type = np.int32
        elif "tensor(uint32)" in input_node.type:
            input_type = np.uint32
        elif "tensor(int64)" in input_node.type:
            input_type = np.int64
        elif "tensor(uint64)" in input_node.type:
            input_type = np.uint64
        elif "tensor(float16)" in input_node.type:
            input_type = np.float16
        elif "tensor(float)" in input_node.type:
            input_type = np.float32
        elif "tensor(double)" in input_node.type:
            input_type = np.float64
        elif "tensor(bool)" in input_node.type:
            input_type = np.bool_

        return input_type

    @log_errors
    def get_next(self) -> dict[str, np.ndarray[Any, Any]] | None:
        """
        Get next feed data
        :return: feed dict for the model
        """
        if self.enum_data_dicts is None:
            so = onnxruntime.SessionOptions()
            # TODO: register_custom_ops_library(so)
            session = create_infer_session_for_onnx_model(self._model_input, so)

            enum_data: dict[str, np.ndarray[Any, Any]] = {}
            for input_index, input_node in enumerate(session.get_inputs()):
                input_name = self._get_input_name(input_node)
                input_shape = self._parse_input_shape(input_index, input_name)
                if input_shape == [] or input_shape is None:
                    input_shape = self._get_input_shape(input_node)
                input_type = self._get_input_type(input_node)

                if input_shape is not None:
                    np.random.seed(42)
                    if "tensor(string)" in input_node.type:
                        input_data = np.chararray(tuple(input_shape))
                    else:
                        if self._input_data_range is None:
                            input_data = np.random.random(input_shape).astype(input_type)
                        else:
                            if input_name not in self._input_data_range.keys():
                                raise ValueError(
                                    f'Input name "{input_name}" is not found in RandomDataReaderInputDataRange. Please check whether '
                                    'the parameter config.global_quant_config.extra_options["RandomDataReaderInputDataRange"] is correct.'
                                )
                            range_pair: list[int] = self._input_data_range[input_name]  # Upper bound will be reached.
                            low: int = range_pair[0]
                            high: int = range_pair[1]
                            if "uint" in input_node.type or "int" in input_node.type:
                                input_data = np.random.randint(low, high=high + 1, size=input_shape).astype(input_type)
                            else:
                                input_data = np.random.uniform(low, high=high, size=input_shape).astype(input_type)
                else:
                    raise ValueError(
                        f"Unsupported input name {input_node.name} shape {input_node.shape} type {input_node.type} "
                    )
                enum_data[input_name] = input_data
                logger.info(f"Random input name {input_name} shape {input_shape} type {input_type} ")
            self.enum_data_dicts = iter([enum_data])

        return next(self.enum_data_dicts, None)


# PathData reader
class PathDataReader(CalibrationDataReader):  # type: ignore
    """
    A CalibrationDataReader loading data from specified paths for model calibration.
    """

    def __init__(
        self, model_input: Union[str, Path, onnx.ModelProto], data_path: str, input_shape: list[Any] = []
    ) -> None:
        """
        :param Union[str, Path, onnx.ModelProto] model_path: Full path of the input model.
        :param str data_path: Full path of the input data.
        :param List[Any] input_shape: List or dictionary specifying the input shapes. Defaults to ``[]``.
        """
        self._model_input = model_input
        self._data_path = data_path
        self._input_shape = input_shape

        self.data_dict: dict[str, list[np.ndarray[Any, Any]]] = {}
        self.enum_data_list: list[dict[str, np.ndarray[Any, Any]]] = []
        self.enum_data_iter: Iterator[dict[str, np.ndarray[Any, Any]]] | None = None

        self.batch_size = 1

    def _parse_input_shape(self, input_index: int, input_name: str) -> Any:
        """
        Parse input shape of model from user's input
        :param input_index: the input index in session.get_inputs()
        :param input_name: the input name string
        :return: input shape required for the input node
        """

        def _deal_shape_value(
            list_or_tuple_shape: Union[list[int], tuple[int], list[list[int]], list[Any], Any],
        ) -> Any:
            if not isinstance(list_or_tuple_shape, (list, tuple)):
                logger.warning(f"Invalid input shape {list_or_tuple_shape}")
                return []

            input_shape = []
            for index, shape in enumerate(list_or_tuple_shape):
                if isinstance(shape, int) and shape > 0:
                    input_shape.append(shape)
                else:
                    if index == 0:
                        input_shape.append(self.batch_size)
                    else:
                        logger.warning(f"Invalid input shape {list_or_tuple_shape} in #{index} : {shape}")
                        return []
            return input_shape

        if isinstance(self._input_shape, dict):
            if input_name in self._input_shape.keys():
                return _deal_shape_value(self._input_shape[input_name])
        elif all(isinstance(n, (list, tuple)) for n in self._input_shape):
            if input_index < len(self._input_shape):
                return _deal_shape_value(self._input_shape[input_index])
        else:
            return _deal_shape_value(self._input_shape)

        return []

    def _get_input_name(self, input_node: NodeProto) -> str:
        """
        :param input_node: the input node
        :return: name of the input node
        """
        input_name = input_node.name
        return input_name

    def _get_input_shape(self, input_node: NodeProto) -> list[int | None]:
        """
        :param input_node: the input node
        :return: input shape of the input node
        """
        input_shape: list[Any] = []

        if len(input_node.shape):
            for index, shape in enumerate(input_node.shape):
                if isinstance(shape, int) and shape > 0:
                    input_shape.append(shape)
                else:
                    input_shape.append(None)

        return input_shape

    def _get_input_type(self, input_node: NodeProto) -> Any:
        """
        :param input_node: the input node
        :return: data type of the input node
        """
        input_type: Any = None

        if "tensor(int8)" in input_node.type:
            input_type = np.int8
        elif "tensor(uint8)" in input_node.type:
            input_type = np.uint8
        elif "tensor(int16)" in input_node.type:
            input_type = np.int16
        elif "tensor(uint16)" in input_node.type:
            input_type = np.uint16
        elif "tensor(int32)" in input_node.type:
            input_type = np.int32
        elif "tensor(uint32)" in input_node.type:
            input_type = np.uint32
        elif "tensor(int64)" in input_node.type:
            input_type = np.int64
        elif "tensor(uint64)" in input_node.type:
            input_type = np.uint64
        elif "tensor(float16)" in input_node.type:
            input_type = np.float16
        elif "tensor(float)" in input_node.type:
            input_type = np.float32
        elif "tensor(double)" in input_node.type:
            input_type = np.float64
        elif "tensor(bool)" in input_node.type:
            input_type = np.bool_

        return input_type

    @log_errors
    def load_npy_data(
        self, data_path: str, file_names: list[str], input_shape: Union[list[int], tuple[int, ...]]
    ) -> dict[str, np.ndarray[Any, Any]]:
        data_dict: dict[str, np.ndarray[Any, Any]] = {}

        for file_name in file_names:
            file_path = os.path.join(data_path, file_name)
            npy_data = np.load(file_path)

            if npy_data.ndim == 1 and None not in input_shape and npy_data.size == np.prod(input_shape):
                npy_data = np.reshape(npy_data, input_shape)
                logger.debug(
                    f"Detected npy_data with 1 dimension. Reshaped to input node shape. Please check npy shape for file '{file_path}'."
                )

            if npy_data.ndim != len(input_shape):
                if npy_data.ndim + 1 == len(input_shape):
                    npy_data = np.expand_dims(npy_data, axis=0)
                    logger.debug(
                        f"Detected npy_data shape {npy_data.shape} with ndim {npy_data.ndim}. "
                        f"Expanded dimensions to match input shape length {len(input_shape)} for file '{file_path}'. "
                        "Please check npy shape."
                    )
                else:
                    raise ValueError(
                        f"Provided npy data from file '{file_path}' and model input shape do not match in number of dimensions."
                    )

            for i, (npy_dim, input_node_dim) in enumerate(zip(npy_data.shape, input_shape, strict=False)):
                if isinstance(input_node_dim, int) and npy_dim != input_node_dim:
                    raise ValueError(
                        f"Shape mismatch in file '{file_path}': provided npy data and model input "
                        f"shape do not match (expected {input_shape}, got {npy_data.shape})."
                    )

            data_dict[file_name] = npy_data

        return data_dict

    # Load data from data path and support raw data, npy data and image data,
    # return a dict, key is file name and value is numpy arrary
    def load_data(
        self, data_path: str, input_shape: Union[list[int], tuple[int, ...]], input_name: str
    ) -> dict[str, np.ndarray[Any, Any]]:
        files = [f for f in os.listdir(data_path) if f.endswith(".npy")]
        if files != []:
            logger.info(f"Loading npy data from {data_path}")
            return self.load_npy_data(data_path, files, input_shape)
        else:
            data_dir_path = os.path.join(data_path, input_name)
            files = [f for f in os.listdir(data_dir_path) if f.endswith(".npy")]
            if files != []:
                logger.info(f"Loading npy data from {data_dir_path}")
                return self.load_npy_data(data_dir_path, files, input_shape)
            else:
                raise FileNotFoundError(f"Not found data in {data_path}")

    @log_errors
    def get_next(self) -> dict[str, np.ndarray[Any, Any]] | None:
        """
        Get next feed data
        :return: feed dict for the model
        """
        if self.enum_data_iter is None:
            so = onnxruntime.SessionOptions()
            # TODO: register_custom_ops_library(so)
            session = create_infer_session_for_onnx_model(self._model_input, so)

            # load data from data path
            for input_index, input_node in enumerate(session.get_inputs()):
                input_name = self._get_input_name(input_node)

                input_shape = self._parse_input_shape(input_index, input_name)
                if not input_shape:
                    input_shape = self._get_input_shape(input_node)
                    if not input_shape:
                        raise ValueError("Cannot get the input shape of the model.")
                input_type = self._get_input_type(input_node)

                data_dict = {}
                data_dict = self.load_data(self._data_path, input_shape, input_name)

                if len(data_dict) <= 0:
                    raise ValueError(
                        f"Load data from the path {self._data_path} failed for input{input_index} {input_name}"
                    )

                else:
                    logger.info(
                        f"Load data from the path {self._data_path} for input{input_index} with {len(data_dict)} samples "
                    )

                # save to data_dict
                for key, value in data_dict.items():
                    if value.dtype is not input_type:
                        value = value.astype(input_type)
                    if key in self.data_dict:
                        self.data_dict[key].append(value)
                    else:
                        self.data_dict[key] = [value]

                logger.info(f"Real input name {input_name} shape {input_shape} type {input_type} ")

            self.enum_data_list = []

            for arrays in self.data_dict.values():
                enum_data = {}
                for i, arr in enumerate(arrays):
                    name = self._get_input_name(session.get_inputs()[i])
                    enum_data[name] = arr
                self.enum_data_list.append(enum_data)

            self.enum_data_iter = iter(self.enum_data_list)

        return next(self.enum_data_iter, None)


@log_errors
def get_data_reader(
    model_input: Union[str, Path, onnx.ModelProto],
    calibration_data_reader: CalibrationDataReader | None = None,
    calibration_data_path: str | None = None,
    extra_options: dict[str, Any] = {},
) -> CalibrationDataReader:
    """This function is used to determine data reader based on user inputs."""

    data_reader = calibration_data_reader

    if calibration_data_reader is not None and calibration_data_path is not None:
        logger.warning(
            "Both calibration_data_reader and calibration_data_path are provided, will use the former for calibration."
        )

    elif calibration_data_reader is None and calibration_data_path is not None:
        logger.info(f"Since calibration_data_reader is None, will use {calibration_data_path} to create data reader.")
        data_reader = PathDataReader(model_input, calibration_data_path)

    elif calibration_data_reader is None and calibration_data_path is None:
        if not extra_options.get("UseRandomData", False):
            raise ValueError(
                "A calibration data reader is required for quantization, but none was provided. "
                "Please provide a calibration data reader or path, or alternatively enable random data reader "
                "for calibration by setting config.global_quant_config.extra_options['UseRandomData'] to True."
            )
        else:
            logger.info("No calibration data reader or path provided, will use random data reader.")
            data_reader = RandomDataReader(
                model_input,
                input_shape=extra_options.get("RandomDataReaderInputShape", {}),
                input_data_range=extra_options.get("RandomDataReaderInputDataRange"),
            )

    assert data_reader is not None, "No data reader provided!"

    return data_reader
