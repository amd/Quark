#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import itertools
from quark.shares.utils.log import ScreenLogger, log_errors
import tempfile
import platform
from enum import Enum
from pathlib import Path
from datetime import datetime
import math
import numpy as np
import onnx
import os
import sys
import csv
from typing import Any, Union, Tuple, List, Dict, Optional, Iterator
from onnx import onnx_pb as onnx_proto
import onnx.helper as helper
from onnx.reference import ReferenceEvaluator
from onnx.onnx_ml_pb2 import TensorProto, NodeProto, ModelProto, GraphProto
from onnx import numpy_helper, shape_inference
import onnxruntime as ort
from onnxruntime import __version__ as OrtVersion
from onnxruntime.quantization.calibrate import (CalibrationDataReader, CalibrationMethod)
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import (QuantType, QUANT_OP_NAME, DEQUANT_OP_NAME)
from onnxruntime import SessionOptions
from packaging import version as pv
from quark.version import __version__ as versions
from quark.version import git_version as commitid

try:
    from onnx.reference.custom_element_types import float8e4m3fn
except ImportError:
    float8e4m3fn = None  # type: ignore

# INT4 np.dtypes added in ONNX 1.16. These map to np.int8/np.uint8 because numpy
# does not support sub-byte types.
try:
    from onnx.reference.custom_element_types import int4, uint4
    from onnx.reference import custom_element_types
except ImportError:
    int4 = None  # type: ignore
    uint4 = None  # type: ignore

try:
    # The custom op library may not have been compiled
    from quark.onnx.operators.custom_ops import _COP_DOMAIN
    from quark.onnx.operators.custom_ops import get_library_path as vai_library_path
except Exception:
    # Try to import from original path but may raise an error when call get_library_path
    from quark.onnx.operators.custom_ops import _COP_DOMAIN
    vai_library_path = None  # type: ignore

logger = ScreenLogger(__name__)

__producer__ = "quark.onnx"
__version__ = '{}+{}'.format(versions, commitid)

VAI_DOMAIN = "ai.onnx.contrib"  # domain for vai ops that inherited from python class
COP_DOMAIN = _COP_DOMAIN  # domain for custom ops that implemented using c api
COP_QUANT_OP_NAME = "VitisQuantizeLinear"
COP_DEQUANT_OP_NAME = "VitisDequantizeLinear"
FIX_OP_NAME = "FixNeuron"
BFPFIX_OP_NAME = "BFPFixNeuron"
MXFIX_OP_NAME = "MXFixNeuron"
CUSTOM_VAI_DOMAIN = "com.vai.quantize"

QUANT_OP_TYPES = [QUANT_OP_NAME, COP_QUANT_OP_NAME]
DEQUANT_OP_TYPES = [DEQUANT_OP_NAME, COP_DEQUANT_OP_NAME]

HARD_SIGMOID_SCALE = (2731. / 16384.) / (1. / 6.)
annotate_op_type = ["Conv", "Add", "MaxPool", "AveragePool", "GlobalAveragePool", "MatMul", "Gemm", "ConvTranspose"]
avg_pool_op_type = ["AveragePool", "GlobalAveragePool"]
remove_qdq_op_type: List[str] = []

FIX_OP_DEFAULT_ATTRS = {
    'bit_width': 8,
    'pos': 0,
}
BFPFIX_OP_DEFAULT_ATTRS = {
    'bfp_method': "to_bfp",
    'axis': 1,
    'bit_width': 16,
    'block_size': 8,
    'rounding_mode': 0,
    'sub_block_size': 2,
    'sub_block_shift_bits': 1,
    'convert_to_bfloat_before_bfp': 0,
}
MXFIX_OP_DEFAULT_ATTRS = {
    'element_dtype': "int8",
    'axis': 1,
    'block_size': 32,
    'rounding_mode': 0,
}


def check_and_create_path(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"The path '{path}' didn't exist, so it has been created.")
    else:
        logger.info(f"The path '{path}' already exists.")
    return path


def is_ort_version_below(target_version: str) -> bool:
    """
    This function checks whether the current version of ONNX Runtime (ORT) is below a specified version.

    Args:
        target_version (str): The version to compare against the current ORT version.

    Returns:
        True if the current ORT version is less than the target version, False otherwise.
    """
    current_version = pv.parse(OrtVersion)  # Assuming OrtVersion is a string representing the current ORT version.
    return current_version < pv.parse(target_version)


def register_custom_ops_library(session_options: SessionOptions) -> None:
    try:
        session_options.register_custom_ops_library(vai_library_path())
    except Exception:
        logger.warning("Due to mismatch of dependent libraries, the custom op library "
                       "from the pre-built wheel package failed to register with ORT. "
                       "Please try to re-build the quark.onnx's lib in current environment.")


class Int16Method(Enum):
    MinMax = 0


class PowerOfTwoMethod(Enum):
    NonOverflow = 0
    MinMSE = 1


class VitisQuantType(Enum):
    QInt8 = 1
    QUInt8 = 2
    QInt16 = 3
    QUInt16 = 4
    QInt4 = 5
    QUInt4 = 6
    QInt32 = 7
    QUInt32 = 8
    QFloat16 = 9
    QBFloat16 = 10
    QBFP = 11
    QMX = 12

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_string(t: str) -> Any:
        try:
            return VitisQuantType[t]
        except KeyError:
            raise ValueError()

    @property
    def tensor_type(self) -> Any:
        if self == VitisQuantType.QUInt8:
            return TensorProto.UINT8
        if self == VitisQuantType.QInt8:
            return TensorProto.INT8
        if self == VitisQuantType.QUInt16:
            return TensorProto.UINT16
        if self == VitisQuantType.QInt16:
            return TensorProto.INT16
        if self == VitisQuantType.QInt32:
            return TensorProto.INT32
        if self == VitisQuantType.QUInt32:
            return TensorProto.UINT32
        if self == VitisQuantType.QFloat16:
            return TensorProto.FLOAT16
        if self == VitisQuantType.QBFloat16:
            return TensorProto.BFLOAT16
        if self == VitisQuantType.QBFP or self == VitisQuantType.QMX:
            return TensorProto.UNDEFINED
        raise ValueError(f"Unexpected value qtype={self!r}.")


class VitisQuantFormat(Enum):
    QDQ = 2
    FixNeuron = 3
    BFPFixNeuron = 4
    MXFixNeuron = 5

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_string(format: str) -> Any:
        try:
            return VitisQuantFormat[format]
        except KeyError:
            raise ValueError()


DType = Union[np.dtype[np.int8], np.dtype[np.uint8], np.dtype[np.int16], np.dtype[np.uint16], np.dtype[np.int32],
              np.dtype[np.uint32], np.dtype[np.float16], None, Any]
ONNX_TYPE_TO_NP_TYPE: Dict[int, Optional[DType]] = {
    onnx_proto.TensorProto.INT8: np.dtype("int8"),
    onnx_proto.TensorProto.UINT8: np.dtype("uint8"),
    onnx_proto.TensorProto.INT16: np.dtype("int16"),
    onnx_proto.TensorProto.UINT16: np.dtype("uint16"),
    onnx_proto.TensorProto.INT32: np.dtype("int32"),
    onnx_proto.TensorProto.UINT32: np.dtype("uint32"),
    onnx_proto.TensorProto.FLOAT16: np.dtype("float16"),
    # This is mismatched conversion,
    # numpy does not support yet
    onnx_proto.TensorProto.BFLOAT16: np.dtype("float16"),
    onnx_proto.TensorProto.FLOAT8E4M3FN: float8e4m3fn,  # type ignore
    onnx_proto.TensorProto.INT4: int4,  # type ignore
    onnx_proto.TensorProto.UINT4: uint4,  # type ignore
    # This is for the new data types BFP and MX
    onnx_proto.TensorProto.UNDEFINED: np.dtype("float32"),  # type ignore
}


def create_range_dict(dtype_ranges: Dict[str, Tuple[int, int]]) -> Any:

    result = {}
    for dtype, range_pair in dtype_ranges.items():
        tensor_proto_dtype = getattr(onnx_proto.TensorProto, dtype)

        if dtype.lower() not in ['int4', 'uint4']:
            np_dtype = getattr(np, dtype.lower())
        else:
            np_dtype = getattr(custom_element_types, dtype.lower())

        array_pair = (np.array(range_pair[0], dtype=np_dtype), np.array(range_pair[1], dtype=np_dtype))
        result[tensor_proto_dtype] = array_pair
    return result


dtype_ranges = {
    "UINT8": (0, 255),
    "INT8": (-128, 127),
    "UINT16": (0, 65535),
    "INT16": (-32768, 32767),
    "UINT4": (0, 15),
    "INT4": (-8, 7),
    "UINT32": (0, 2**32 - 1),
    "INT32": (-2**31, 2**31 - 1),
}

symmetric_ranges = {
    "INT8": (-127, 127),
    "INT16": (-32767, 32767),
    "INT32": (-(2**31 - 1), 2**31 - 1),
}

reduced_ranges = {
    "UINT8": (0, 127),
    "INT8": (-64, 64),
    "UINT16": (0, 32767),
    "INT16": (-16384, 16384),
    "UINT4": (0, 7),
    "INT4": (-4, 3),
    "UINT32": (0, 2**31 - 1),
    "INT32": (-2**30, 2**30),
}

ONNX_INT_TYPE_RANGE = create_range_dict(dtype_ranges)
ONNX_INT_TYPE_SYMMETRIC_RANGE = create_range_dict(symmetric_ranges)
ONNX_INT_TYPE_REDUCED_RANGE = create_range_dict(reduced_ranges)

ONNX_WBIT_QTYPES_LIST = [
    onnx_proto.TensorProto.UINT16,
    onnx_proto.TensorProto.INT16,
    onnx_proto.TensorProto.UINT32,
    onnx_proto.TensorProto.INT32,
    onnx_proto.TensorProto.FLOAT16,
    onnx_proto.TensorProto.BFLOAT16,
]

ONNX_FP_QTYPES_LIST = [
    onnx_proto.TensorProto.FLOAT16,
    onnx_proto.TensorProto.BFLOAT16,
]

ONNX_BFP_QTYPES_LIST = [
    onnx_proto.TensorProto.UNDEFINED,
]


def _check_type(*args: Any, zero_point_index: int = -1) -> Any:
    new_args: List[np.ndarray[Any, Any]] = []
    for i, a in enumerate(args):
        if np.issubdtype(type(a), np.number):
            new_args.append(np.array(a))
        elif isinstance(a, np.ndarray):
            new_args.append(a)
        else:
            raise TypeError(f"arg {i} is not an array: {a}")
        if i == zero_point_index:
            v = new_args[-1]
    return tuple(new_args) if len(new_args) > 1 else new_args[0]


@log_errors
def get_tensor_type_from_qType(quant_type: Union[QuantType, VitisQuantType]) -> int:
    if quant_type == QuantType.QUInt8 or quant_type == VitisQuantType.QUInt8:
        return TensorProto.UINT8
    if quant_type == QuantType.QInt8 or quant_type == VitisQuantType.QInt8:
        return TensorProto.INT8
    if quant_type == QuantType.QUInt16 or quant_type == VitisQuantType.QUInt16:
        return TensorProto.UINT16
    if quant_type == QuantType.QInt16 or quant_type == VitisQuantType.QInt16:
        return TensorProto.INT16
    if quant_type == VitisQuantType.QUInt32:
        return TensorProto.UINT32
    if quant_type == VitisQuantType.QInt32:
        return TensorProto.INT32
    if quant_type == VitisQuantType.QFloat16:
        return TensorProto.FLOAT16
    if quant_type == VitisQuantType.QBFloat16:
        return TensorProto.BFLOAT16
    if quant_type == VitisQuantType.QBFP or quant_type == VitisQuantType.QMX:
        return TensorProto.UNDEFINED
    raise ValueError(f"Unexpected value qtype={quant_type!r}.")


@log_errors
def get_qmin_qmax_for_qType(qType: int, reduce_range: bool = False, symmetric: bool = False) -> Any:
    """
    Return qmin and qmax, the minimum and maximum value representable by the given qType
    :parameter qType: Integer or Floating Point Type
    :return: qmin, qmax
    """
    if qType in ONNX_BFP_QTYPES_LIST:
        return (np.array(-3.4e38, dtype=np.float32), np.array(3.4e38, dtype=np.float32))

    if qType in ONNX_FP_QTYPES_LIST:
        if qType == onnx_proto.TensorProto.FLOAT16:
            return (np.array(-65504.0, dtype=np.float32), np.array(65504.0, dtype=np.float32))
        elif qType == onnx_proto.TensorProto.BFLOAT16:
            if reduce_range:
                # For narrow-bit floating point data types, to utilize the dense area near zero,
                # we use a reduced range cooperated with scaling, which could avoid overflow also
                return (np.array(-2.0, dtype=np.float32), np.array(2.0, dtype=np.float32))
            else:
                return (np.array(-3.38953139e38, dtype=np.float32), np.array(3.38953139e38, dtype=np.float32))
        else:
            raise NotImplementedError(f"This function does not support the qType {qType}.")

    qrange = None

    if reduce_range:
        qrange = ONNX_INT_TYPE_REDUCED_RANGE.get(qType)
    elif symmetric and qType in ONNX_INT_TYPE_SYMMETRIC_RANGE:
        qrange = ONNX_INT_TYPE_SYMMETRIC_RANGE[qType]
    else:
        qrange = ONNX_INT_TYPE_RANGE.get(qType)

    if not qrange:
        raise ValueError(
            f"Unexpected data type {qType} requested. Only INT4, UINT4, INT8, UINT8, INT16, and UINT16 are supported.")

    return qrange


def get_qrange_for_qType(qType: int, reduce_range: bool = False, symmetric: bool = False) -> Any:
    """
    Helper function to get the quantization range for a type.
        parameter qType: quantization type.
        return: quantization range.
    """
    qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range, symmetric=symmetric)
    return qmax - qmin


def quantize_nparray(qType: Any,
                     arr: np.ndarray[Any, Any],
                     scale: np.ndarray[Any, Any],
                     zero_point: float,
                     low: Optional[float] = None,
                     high: Optional[float] = None) -> Any:
    if qType in ONNX_BFP_QTYPES_LIST:
        return arr

    assert (
        qType in ONNX_TYPE_TO_NP_TYPE
    ), f"Unexpected data type {qType} requested. Only INT4, UINT4, INT8, UINT8, INT16, UINT16, FLOAT16, and BFLOAT16 are supported."

    if qType in ONNX_FP_QTYPES_LIST:
        arr_fp32 = arr.astype(np.float32) / scale + zero_point
        onnx_model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Cast", ["X"], ["Y"], to=qType)],
                "qu",
                [helper.make_tensor_value_info("X", onnx_proto.TensorProto.FLOAT, None)],
                [helper.make_tensor_value_info("Y", qType, None)],
            ))
        ref = ReferenceEvaluator(onnx_model)
        return ref.run(None, {"X": arr_fp32})[0]
    else:
        dtype = ONNX_TYPE_TO_NP_TYPE[qType]
        (qmin, qmax) = get_qmin_qmax_for_qType(qType, reduce_range=False, symmetric=True)

        cliplow = max(qmin, low) if low is not None else qmin
        cliphigh = min(qmax, high) if high is not None else qmax
        arr_fp32 = np.asarray((arr.astype(np.float32) / scale).round() + zero_point)
        np.clip(arr_fp32, cliplow, cliphigh, out=arr_fp32)
        return _check_type(arr_fp32.astype(dtype))


class CachedDataReader(CalibrationDataReader):  # type: ignore
    """
    A CalibrationDataReader cached input data from the user provided data reader.
    """

    @log_errors
    def __init__(self,
                 dr: CalibrationDataReader,
                 data_size: Optional[int] = None,
                 convert_nchw_to_nhwc: bool = False,
                 quantize_fp16: bool = False):
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
            if not inputs:
                break
            elif self.data_size is not None and n > self.data_size:
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
                            f"Expected integer output shape but got [{C}, {H}, {W}], skip the nchw to nhwc conversion.")
                        continue

                    if not (H > C and W > C):
                        logger.warning(f"Expected H,W > C but got [{C}, {H}, {W}].")
                    inputs[input_name] = np.transpose(input_array, axes=(0, 2, 3, 1))
            self._data_cache.append(inputs)

        if len(self._data_cache) == 0:
            raise ValueError("No data in the input calibration data reader")
        else:
            logger.info("Obtained calibration data with {} iters".format(len(self._data_cache)))

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

    def get_next(self) -> Optional[Dict[str, np.ndarray[Any, Any]]]:
        """
        Get next feed data
        :return: feed dict for the model
        """
        return next(self.enum_data_dicts, None)


class RandomDataReader(CalibrationDataReader):  # type: ignore
    """
    A CalibrationDataReader using random data for rapid quantiation.
    """

    def __init__(self,
                 model_path: str,
                 input_shape: Dict[str, List[int]] = {},
                 input_data_range: Optional[Dict[str, List[int]]] = None):
        """
        :param model_path : Full path of the input model.
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
        self._model_path = model_path
        self._input_shape = input_shape
        self._input_data_range: Optional[Dict[str, List[int]]] = input_data_range

        self.enum_data_dicts: Optional[Iterator[Dict[str, np.ndarray[Any, Any]]]] = None
        self.batch_size = 1

    def _parse_input_shape(self, input_index: int, input_name: str) -> Any:
        """
        Parse input shape of model from user's input
        :param input_index: the input index in session.get_inputs()
        :param input_name: the input name string
        :return: input shape required for the input node
        """

        def _deal_shape_value(list_or_tuple_shape: Union[int, List[int], Any]) -> Any:
            if not isinstance(list_or_tuple_shape, (list, tuple)):
                logger.warning("Invalid input shape {}".format(list_or_tuple_shape))
                return []

            input_shape = []
            for index, shape in enumerate(list_or_tuple_shape):
                if isinstance(shape, int) and shape > 0:
                    input_shape.append(shape)
                else:
                    if index == 0:
                        input_shape.append(self.batch_size)
                    else:
                        logger.warning("Invalid input shape {} in #{} : {}".format(list_or_tuple_shape, index, shape))
                        return []
            return input_shape

        if isinstance(self._input_shape, dict):
            if input_name in self._input_shape.keys():
                return _deal_shape_value(self._input_shape[input_name])
            elif self._input_shape != {}:
                raise ValueError(
                    f"Input name \"{input_name}\" is not found in RandomDataReaderInputShape. Please check whether the parameter config.global_quant_config.extra_options[\"RandomDataReaderInputShape\"] is correct."
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

    def _get_input_shape(self, input_node: NodeProto) -> List[int]:
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

        if 'tensor(int8)' in input_node.type:
            input_type = np.int8
        elif 'tensor(uint8)' in input_node.type:
            input_type = np.uint8
        elif 'tensor(int16)' in input_node.type:
            input_type = np.int16
        elif 'tensor(uint16)' in input_node.type:
            input_type = np.uint16
        elif 'tensor(int32)' in input_node.type:
            input_type = np.int32
        elif 'tensor(uint32)' in input_node.type:
            input_type = np.uint32
        elif 'tensor(int64)' in input_node.type:
            input_type = np.int64
        elif 'tensor(uint64)' in input_node.type:
            input_type = np.uint64
        elif 'tensor(float16)' in input_node.type:
            input_type = np.float16
        elif 'tensor(float)' in input_node.type:
            input_type = np.float32
        elif 'tensor(double)' in input_node.type:
            input_type = np.float64
        elif 'tensor(bool)' in input_node.type:
            input_type = np.bool_

        return input_type

    @log_errors
    def get_next(self) -> Optional[Dict[str, np.ndarray[Any, Any]]]:
        """
        Get next feed data
        :return: feed dict for the model
        """
        if self.enum_data_dicts is None:
            so = ort.SessionOptions()
            # TODO: To deal with onnxruntime_extension with ort1.17
            # so.register_custom_ops_library(ext_lib_path())
            # register_custom_ops_library(so)
            session = ort.InferenceSession(self._model_path, so, providers=['CPUExecutionProvider'])
            enum_data: Dict[str, np.ndarray[Any, Any]] = {}
            for input_index, input_node in enumerate(session.get_inputs()):
                input_name = self._get_input_name(input_node)
                input_shape = self._parse_input_shape(input_index, input_name)
                if input_shape == [] or input_shape is None:
                    input_shape = self._get_input_shape(input_node)
                input_type = self._get_input_type(input_node)

                if input_shape is not None:
                    if 'tensor(string)' in input_node.type:
                        input_data = np.chararray(tuple(input_shape))
                    else:
                        if self._input_data_range is None:
                            input_data = np.random.random(input_shape).astype(input_type)
                        else:
                            if input_name not in self._input_data_range.keys():
                                raise ValueError(
                                    f"Input name \"{input_name}\" is not found in RandomDataReaderInputDataRange. Please check whether the parameter config.global_quant_config.extra_options[\"RandomDataReaderInputDataRange\"] is correct."
                                )
                            range_pair: List[int] = self._input_data_range[input_name]  # Upper bound will be reached.
                            low: int = range_pair[0]
                            high: int = range_pair[1]
                            if 'uint' in input_node.type:
                                input_data = np.random.randint(low, high=high + 1, size=input_shape).astype(input_type)
                            elif 'int' in input_node.type:
                                input_data = np.random.randint(low, high=high + 1, size=input_shape).astype(input_type)
                            else:
                                input_data = np.random.uniform(low, high=high, size=input_shape).astype(input_type)
                else:
                    raise ValueError("Unsupported input name {} shape {} type {} ".format(
                        input_node.name, input_node.shape, input_node.type))
                enum_data[input_name] = input_data
                logger.info("Random input name {} shape {} type {} ".format(input_name, input_shape, input_type))
            self.enum_data_dicts = iter([enum_data])

        return next(self.enum_data_dicts, None)


# PathData reader
class PathDataReader(CalibrationDataReader):  # type: ignore
    """
    A CalibrationDataReader loading data from specified paths for model calibration.
    """

    def __init__(self, model_path: str, data_path: str, input_shape: List[Any] = []) -> None:
        """
        :param model_path : Full path of the input model.
        :param data_path  : Full path of the input data.
        :param input_shape: List or dictionary specifying the input shapes.
        """
        self._model_path = model_path
        self._data_path = data_path
        self._input_shape = input_shape

        self.data_dict: Dict[str, List[np.ndarray[Any, Any]]] = {}
        self.enum_data_list: List[Dict[str, np.ndarray[Any, Any]]] = []
        self.enum_data_iter: Optional[Iterator[Dict[str, np.ndarray[Any, Any]]]] = None

        self.batch_size = 1

    def _parse_input_shape(self, input_index: int, input_name: str) -> Any:
        """
        Parse input shape of model from user's input
        :param input_index: the input index in session.get_inputs()
        :param input_name: the input name string
        :return: input shape required for the input node
        """

        def _deal_shape_value(
                list_or_tuple_shape: Union[List[int], Tuple[int], List[List[int]], List[Any], Any]) -> Any:
            if not isinstance(list_or_tuple_shape, (list, tuple)):
                logger.warning("Invalid input shape {}".format(list_or_tuple_shape))
                return []

            input_shape = []
            for index, shape in enumerate(list_or_tuple_shape):
                if isinstance(shape, int) and shape > 0:
                    input_shape.append(shape)
                else:
                    if index == 0:
                        input_shape.append(self.batch_size)
                    else:
                        logger.warning("Invalid input shape {} in #{} : {}".format(list_or_tuple_shape, index, shape))
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

    def _get_input_shape(self, input_node: NodeProto) -> List[Optional[int]]:
        """
        :param input_node: the input node
        :return: input shape of the input node
        """
        input_shape: List[Any] = []

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

        if 'tensor(int8)' in input_node.type:
            input_type = np.int8
        elif 'tensor(uint8)' in input_node.type:
            input_type = np.uint8
        elif 'tensor(int16)' in input_node.type:
            input_type = np.int16
        elif 'tensor(uint16)' in input_node.type:
            input_type = np.uint16
        elif 'tensor(int32)' in input_node.type:
            input_type = np.int32
        elif 'tensor(uint32)' in input_node.type:
            input_type = np.uint32
        elif 'tensor(int64)' in input_node.type:
            input_type = np.int64
        elif 'tensor(uint64)' in input_node.type:
            input_type = np.uint64
        elif 'tensor(float16)' in input_node.type:
            input_type = np.float16
        elif 'tensor(float)' in input_node.type:
            input_type = np.float32
        elif 'tensor(double)' in input_node.type:
            input_type = np.float64
        elif 'tensor(bool)' in input_node.type:
            input_type = np.bool_

        return input_type

    @log_errors
    def load_npy_data(self, data_path: str, file_names: List[str],
                      input_shape: Union[List[int], Tuple[int, ...]]) -> Dict[str, np.ndarray[Any, Any]]:
        data_dict: Dict[str, np.ndarray[Any, Any]] = {}

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
                        "Please check npy shape.")
                else:
                    raise ValueError(
                        f"Provided npy data from file '{file_path}' and model input shape do not match in number of dimensions."
                    )

            for i, (npy_dim, input_node_dim) in enumerate(zip(npy_data.shape, input_shape)):
                if isinstance(input_node_dim, int) and npy_dim != input_node_dim:
                    raise ValueError(f"Shape mismatch in file '{file_path}': provided npy data and model input "
                                     f"shape do not match (expected {input_shape}, got {npy_data.shape}).")

            data_dict[file_name] = npy_data

        return data_dict

    # Load data from data path and support raw data, npy data and image data,
    # return a dict, key is file name and value is numpy arrary
    def load_data(self, data_path: str, input_shape: Union[List[int], Tuple[int, ...]],
                  input_name: str) -> Dict[str, np.ndarray[Any, Any]]:
        files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        if files != []:
            logger.info("Loading npy data from {}".format(data_path))
            return self.load_npy_data(data_path, files, input_shape)
        else:
            data_dir_path = os.path.join(data_path, input_name)
            files = [f for f in os.listdir(data_dir_path) if f.endswith('.npy')]
            if files != []:
                logger.info("Loading npy data from {}".format(data_dir_path))
                return self.load_npy_data(data_dir_path, files, input_shape)
            else:
                raise FileNotFoundError("Not found data in {}".format(data_path))

    @log_errors
    def get_next(self) -> Optional[Dict[str, np.ndarray[Any, Any]]]:
        """
        Get next feed data
        :return: feed dict for the model
        """
        if self.enum_data_iter is None:
            so = ort.SessionOptions()
            try:
                from onnxruntime_extensions import get_library_path as ext_lib_path
                from quark.onnx.operators.custom_ops import get_library_path as vai_lib_path
                so.register_custom_ops_library(ext_lib_path())
                so.register_custom_ops_library(vai_lib_path())
            except Exception:
                logger.warning("Could not register the custom op libraries, "
                               "since your onnxruntime is higher than 1.16.0 "
                               "or GLIBC does not match in your environment.")
            session = ort.InferenceSession(self._model_path, so, providers=['CPUExecutionProvider'])

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
                    raise ValueError("Load data from the path {} failed for input{} {}".format(
                        self._data_path, input_index, input_name))

                else:
                    logger.info("Load data from the path {} for input{} with {} samples ".format(
                        self._data_path, input_index, len(data_dict)))

                # save to data_dict
                for key, value in data_dict.items():
                    if value.dtype is not input_type:
                        value = value.astype(input_type)
                    if key in self.data_dict:
                        self.data_dict[key].append(value)
                    else:
                        self.data_dict[key] = [value]

                logger.info("Real input name {} shape {} type {} ".format(input_name, input_shape, input_type))

            self.enum_data_list = []

            for arrays in self.data_dict.values():
                enum_data = {}
                for i, arr in enumerate(arrays):
                    name = self._get_input_name(session.get_inputs()[i])
                    enum_data[name] = arr
                self.enum_data_list.append(enum_data)

            self.enum_data_iter = iter(self.enum_data_list)

        return next(self.enum_data_iter, None)


def infer_shape(model: ModelProto) -> ModelProto:
    """
    :param model: the source model
    :return: the target model contains inferred shape
    """
    inferred_onnx_model = shape_inference.infer_shapes(model)
    return inferred_onnx_model


def get_datatype_shape(tensor: TensorProto) -> Tuple[str, List[Any]]:
    """
    :param tensor: the input tensor
    :return: datatype and shape of the tensor
    """
    elem_type_num = tensor.type.tensor_type.elem_type
    data_type = TensorProto.DataType.Name(elem_type_num).lower()
    data_type = data_type if data_type != 'float' else 'float32'
    dims = tensor.type.tensor_type.shape.dim
    n = len(dims)
    shape = [dims[i].dim_value if dims[i].dim_value else -1 for i in range(n)]
    return (data_type, shape)


@log_errors
def dump_model(model: Union[str, onnx.ModelProto],
               dump_data_reader: Optional[object] = None,
               random_data_reader_input_shape: Dict[str, List[int]] = {},
               dump_float: bool = False,
               output_dir: str = './dump_results') -> None:
    """
    This function dumps the simulation results of the quantized model,
    including weights and activation results.
    :param model: the input model
    :param dump_data_reader: data reader for dumpping
    :param random_data_reader_input_shape: if use internal random data reader,
           this is used to configure input node's shape
    :param dump_float: dump results of the float model or not
    :param output_dir: output directory for results
    """
    if isinstance(model, str):
        model_path = model
        model = onnx.load(model)
    else:
        raise ValueError("The model requires a string of the model path")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # modify_output_nodes
    fn_node_pos = {}
    has_fixneuron = False
    for n in model.graph.node:
        if n.op_type == "FixNeuron":
            fn_node_pos[n.output[0]] = 2**int(n.attribute[1].s)
            has_fixneuron = True
    if not has_fixneuron:
        if not dump_float:
            raise ValueError(
                "No FixNeuron node detected in the model, the results of the quantized tensor values will not be saved. "
                "Please use the parameter quant_format=VitisQuantFormat.FixNeuron to quantize the float model.")
            return
        else:
            logger.warning(
                "No FixNeuron node detected in the model, the results of the quantized tensor values will not be saved. "
                "Please use the parameter quant_format=VitisQuantFormat.FixNeuron to quantize the float model "
                "if you want to dump the quantized tensor value.")
            logger.info("The float output results of each node in the model will be saved. ")
    node_output = []
    model.graph.ClearField("output")
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
            node_output.append(output)
    tmp_dump_model = str(Path(output_dir) / "./tmp_dump_model.onnx")
    onnx.save(model, tmp_dump_model)

    so = ort.SessionOptions()
    # TODO: To deal with onnxruntime_extension with ort1.17
    # so.register_custom_ops_library(ext_lib_path())
    # register_custom_ops_library(so)
    sess = ort.InferenceSession(tmp_dump_model, so, providers=['CPUExecutionProvider'])

    if dump_data_reader is None:
        dump_data_reader = RandomDataReader(model_path, input_shape=random_data_reader_input_shape)

    if isinstance(dump_data_reader, CalibrationDataReader):
        input = dump_data_reader.get_next()
        if not input:
            raise ValueError("dump_data_reader returned None, "
                             "please confirm if the dump_data_reader is correct")
        else:
            logger.info("Dumping activations and weights...")
            results_outputs = sess.run(None, input)
            for node, res in zip(node_output, results_outputs):
                filename = os.path.join(output_dir, node.replace('/', '_'))
                res = res.flatten()
                if node in fn_node_pos:
                    res_q = np.round(res * fn_node_pos[node])
                    res_q = res_q.clip(-128, 127)
                    res_q.astype(np.int8).tofile(filename + ".bin")
                    np.savetxt(filename + ".txt", res_q.astype(np.int8), fmt="%s", delimiter=",")
                if dump_float:
                    res.tofile(filename + '_float.bin')
                    np.savetxt(filename + "_float.txt", res, fmt="%s", delimiter=",")
            os.remove(tmp_dump_model)
    else:
        raise ValueError("dump_data_reader is used for the dumping process. "
                         "It should be an instance of CalibrationDataReader.")


def is_approximately_equal(a: float, b: float, epsilon: float = 1e-6) -> bool:
    """
    :param a: scalar input
    :param b: scalar input
    :param epsilon: difference tolerance
    :return: equal or not
    """
    if a is None or b is None:
        return False
    return abs(a - b) < epsilon


def check_reduce_mean_condition(model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
    """
    Check conditions for Reduce Mean operation in ONNX graph nodes.

    :param model: ONNX model
    :param node: ONNX node
    :return: True if conditions for Reduce Mean are satisfied, False otherwise
    """
    has_axes_attr = any(attr.name == 'axes' for attr in node.attribute)
    has_axes_2_3_attr = any(attr.name == 'axes' and len(attr.ints) == 2 and attr.ints == [2, 3]
                            for attr in node.attribute)
    has_keepdims_attr = any(attr.name == 'keepdims' for attr in node.attribute)
    has_keepdims_1_attr = any(attr.name == 'keepdims' and attr.i == 1 for attr in node.attribute)

    if has_axes_attr:
        if has_axes_2_3_attr and (not has_keepdims_attr or has_keepdims_1_attr):
            return True
    # Handling opset >= 18 for Reduce Mean
    elif (not has_keepdims_attr or has_keepdims_1_attr) and len(node.input) == 2:
        for init in model.graph.initializer:
            if init.name == node.input[1]:
                axes = onnx.numpy_helper.to_array(init).tolist()
                if axes == [2, 3]:
                    return True

    return False


def check_hard_sigmoid_condition(node: onnx.NodeProto) -> bool:
    """
    :param node: node object
    :return: hard sigmoid or not
    """
    has_beta_attr = any(attr.name == 'beta' for attr in node.attribute)
    has_beta_0_5_attr = any(attr.name == 'beta' and is_approximately_equal(attr.f, 0.5) for attr in node.attribute)
    has_alpha_attr = any(attr.name == 'alpha' and is_approximately_equal(attr.f, 1. / 6.) for attr in node.attribute)
    if (not has_beta_attr or has_beta_0_5_attr) and has_alpha_attr:
        return True
    return False


def is_leaky_relu_with_alpha(node: onnx.NodeProto, alpha_value: float = 0.1) -> bool:
    """
    :param node: node object
    :param alpha_value: DPU supported alpha value
    :return: the Leaky ReLU node has a approximately alpha or not
    """
    if node.op_type == "LeakyRelu":
        for attr in node.attribute:
            if attr.name == "alpha" and is_approximately_equal(attr.f, alpha_value):
                return True
    return False


def is_clip_with_min_max(model: onnx.ModelProto,
                         node: onnx.NodeProto,
                         min_value: float = 0.0,
                         max_value: float = 6.0) -> bool:
    """
    :param model: model object
    :param node: node object
    :param min_value: supported minimum value of Clip
    :param max_value: supported maximum value of Clip
    :return: the Clip node has supported min and max value or not
    """
    if node.op_type == "Clip" and len(node.input) == 3:
        min_input = node.input[1]
        max_input = node.input[2]

        for init in model.graph.initializer:
            if init.name == min_input:
                try:
                    min = onnx.numpy_helper.to_array(init).item()
                except Exception:
                    continue
                if is_approximately_equal(min, min_value):
                    for init2 in model.graph.initializer:
                        if init2.name == max_input:
                            try:
                                max = onnx.numpy_helper.to_array(init2).item()
                            except Exception:
                                continue
                            if is_approximately_equal(max, max_value):
                                return True

    return False


def is_node_needs_annotated(model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
    """
    :param model: model object
    :param node: node object
    :return: the node needs annotated or not
    """
    if node.op_type == "Clip" and node.op_type in remove_qdq_op_type:
        if is_clip_with_min_max(model, node, 0, 6) or is_clip_with_min_max(model, node, 0, 1):
            return True
    elif node.op_type in remove_qdq_op_type:
        return True
    return False


def get_tensor_to_consumer(model: onnx.ModelProto) -> Dict[str, List[onnx.NodeProto]]:
    onnx_model = ONNXModel(model)
    tensor_to_consumer = {}
    for node in onnx_model.model.graph.node:
        for input in node.input:
            if input not in tensor_to_consumer:
                tensor_to_consumer[input] = [node]
            else:
                tensor_to_consumer[input].append(node)
    for init in onnx_model.model.graph.initializer:
        if init.name not in tensor_to_consumer:
            tensor_to_consumer[init.name] = [init]
        else:
            tensor_to_consumer[init.name].append(init)
    return tensor_to_consumer


def get_annotate_tensors(model: onnx.ModelProto) -> List[str]:
    """
    Find patterns in the model where qdq needs to be removed, and then return the corresponding tensor names
    annotate_tensors refers to the tensors associated with the input of the qdq that need to be removed
    :param model: model object
    :return: the annotate tensors
    """
    matching_output_tensor = []
    pad_output_tensor = []
    tensor_to_consumer = get_tensor_to_consumer(model)
    for node in model.graph.node:
        if node.op_type in annotate_op_type and node.output[0] in tensor_to_consumer:
            if len(tensor_to_consumer[node.output[0]]) == 1:
                matching_output_tensor.append(node.output[0])
        elif node.op_type == "Pad" and node.output[0] in tensor_to_consumer:
            if len(tensor_to_consumer[node.output[0]]) == 1:
                pad_output_tensor.append(node.output[0])

    annotate_tensors = []
    for node in model.graph.node:
        if ((is_node_needs_annotated(model, node) and node.input[0] in matching_output_tensor)
                or (node.op_type in avg_pool_op_type and node.input[0] in pad_output_tensor)):
            annotate_tensors.append(node.input[0])
    return annotate_tensors


def get_qdq_to_remove(model: onnx.ModelProto,
                      relu_input: List[str]) -> Tuple[List[onnx.NodeProto], List[onnx.NodeProto], Dict[str, str]]:
    """
    Return the names of nodes to be removed and a dictionary for converting input tensors
    :param model: model object
    :param relu_input: the ReLU node inputs list
    :return: de-quantize & quantize nodes to remove and node mapping dict
    """
    q_nodes_to_remove = []
    dq_nodes_to_remove = []
    q_nodes_output_to_remove = []
    input_node_mapping = {}
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "VitisQuantizeLinear") and node.input[0] in relu_input:
            input_node_mapping[node.input[0]] = node.output[0]
            q_nodes_to_remove.append(node)
            q_nodes_output_to_remove.append(node.output[0])
    for node in model.graph.node:
        if node.op_type in ("DequantizeLinear", "VitisDequantizeLinear") and node.input[0] in q_nodes_output_to_remove:
            for k, v in input_node_mapping.items():
                if v == node.input[0]:
                    input_node_mapping[k] = node.output[0]
            dq_nodes_to_remove.append(node)
    return dq_nodes_to_remove, q_nodes_to_remove, input_node_mapping


def customqdq_to_contribqdq(model_path: str, use_external_data_format: bool) -> None:
    """
    Convert the custom QDQs to the contrib QDQs in the model
    :param model_path: the model path
    :return: None
    """
    from onnxruntime.quantization.quant_utils import QUANT_OP_NAME, DEQUANT_OP_NAME, ms_domain

    OpMapping = {
        COP_QUANT_OP_NAME: QUANT_OP_NAME,
        COP_DEQUANT_OP_NAME: DEQUANT_OP_NAME,
    }
    OpDomain = ms_domain
    OpQuantType = (onnx.TensorProto.INT4, onnx.TensorProto.UINT4, onnx.TensorProto.INT8, onnx.TensorProto.UINT8,
                   onnx.TensorProto.INT16, onnx.TensorProto.UINT16, onnx.TensorProto.INT32)
    model = onnx.load(model_path)
    onnx_model = ONNXModel(model)

    total_num = 0
    converted_num = 0

    for node in onnx_model.model.graph.node:
        if node.op_type not in OpMapping:
            continue

        zp_init = onnx_model.get_initializer(node.input[2])
        if zp_init is not None and zp_init.data_type in OpQuantType:
            if zp_init.data_type == onnx.TensorProto.INT32:
                if node.op_type == COP_QUANT_OP_NAME:
                    continue  # QuantizeLinear does not support int32 quantization
                elif np.count_nonzero(onnx.numpy_helper.to_array(zp_init)) != 0:
                    continue  # DequantizeLinear does not support non-zero zero points

            node.op_type = OpMapping[node.op_type]
            node.domain = OpDomain
            converted_num += 1

        total_num += 1

    if converted_num > 0:
        logger.info(f"Converted {converted_num}/{total_num} custom QDQs to contributed QDQs")
        onnx_model.save_model_to_file(model_path, use_external_data_format=use_external_data_format)


def remove_nodes(model: onnx.ModelProto, nodes_list: List[Any]) -> onnx.ModelProto:
    """
    Delete nodes according to the nodes in the list
    :param model: model object
    :param nodes_list: nodes list to remove
    :return: the model that has removed some nodes
    """
    for node in nodes_list:
        model.graph.node.remove(node)
    return model


def remove_initializers(model: ModelProto, init_list: List[str]) -> ModelProto:
    """
    Delete initializers according to the initializer in the list
    :param model: model object
    :param init_list: initializer's name list to remove
    :return: the model that has removed some initializers
    """
    for init in init_list:
        for i in model.graph.initializer:
            if init == i.name:
                model.graph.initializer.remove(i)
                break
        for input in model.graph.input:
            if input.name == init:
                model.graph.input.remove(input)
                break
    return model


def modified_annotate_input(model: ModelProto, input_node_mapping: Dict[str, str]) -> ModelProto:
    """
    Modify the input of ReLU to the output of annotate op, and delete QDQ
    :param model: model object
    :param input_node_mapping: input node mapping dict
    :return: the modified model
    """

    for node in model.graph.node:
        # Clip might get skipped due to parameter quantization, so handle it separately
        if is_node_needs_annotated(model, node) or node.op_type in avg_pool_op_type + ["Clip"]:
            for k, v in input_node_mapping.items():
                if v == node.input[0]:
                    node.input[0] = k
    return model


def scale2pos(scale: float) -> int:
    """
    Obtain the fixed-point position corresponding to the scale.
    To avoid generating infinity during computations,
    the range of scale is limited.
    :param scale: the scale
    :return: the fixed-point position
    """
    scale = min(max(scale, float(2**-127)), float(2**127))
    return int(np.rint(-np.log2(scale)))


def pos2scale(pos: int) -> float:
    """
    Obtain the scale corresponding to the fixed-point position.
    :param scale: the fixed-point position
    :return: the scale
    """
    return float(np.power(2.0, -pos))


@log_errors
def compute_scale_zp(rmin: np.ndarray[Any, Any],
                     rmax: np.ndarray[Any, Any],
                     qmin: np.ndarray[Any, Any],
                     qmax: np.ndarray[Any, Any],
                     element_type: int,
                     method: PowerOfTwoMethod,
                     symmetric: bool = False,
                     use_pof2s: bool = True) -> Any:
    """Calculate the scale s and zero point z for the quantization relation
    r = s(q-z), where r are the original values and q are the corresponding
    quantized values.

    r and z are calculated such that every value within [rmin,rmax] has an
    approximate representation within [qmin,qmax]. In addition, qmin <= z <=
    qmax is enforced. If the symmetric flag is set to True, the interval
    [rmin,rmax] is symmetrized to [-absmax, +absmax], where
    absmax = max(abs(rmin), abs(rmax)).

    :parameter rmin: minimum value of r
    :parameter rmax: maximum value of r
    :parameter qmin: minimum value representable by the target quantization data type
    :parameter qmax: maximum value representable by the target quantization data type
    :return: zero and scale [z, s]

    """

    if qmin > 0 or qmax < 0:
        raise ValueError(f"qmin and qmax must meet requirement: qmin <= 0 <= qmax while qmin:{qmin}, qmmax:{qmax}")

    # Adjust rmin and rmax such that 0 is included in the range. This is
    # required to make sure zero can be represented by the quantization data
    # type (i.e. to make sure qmin <= zero_point <= qmax)
    rmin = np.minimum(rmin, np.array(0, dtype=rmin.dtype))
    rmax = np.maximum(rmax, np.array(0, dtype=rmax.dtype))

    # Ensure that rmax-rmin is less than or equal to sys.float_info.max
    if rmin == -np.inf:
        logger.warning("rmin is set to -inf, replacing with a very small value.")
        rmin = np.full_like(rmin, -sys.float_info.max / 2)
    if rmax == np.inf:
        logger.warning("rmax is set to inf, replacing with a very large value.")
        rmax = np.full_like(rmax, sys.float_info.max / 2)

    if symmetric:
        absmax = np.maximum(np.abs(rmin), np.abs(rmax))
        rmin = -absmax
        rmax = +absmax

    assert qmin <= qmax, f"qmin={rmin} > qmax={rmax}"
    dr = np.array(rmax - rmin, dtype=np.float64)
    dq = np.array(qmax, dtype=np.float64) - np.array(qmin, dtype=np.float64)
    scale = np.array(dr / dq)
    if np.isnan(scale):
        raise ValueError("NaN detected, please check the correctness of the model")
    assert scale >= 0, "scale isse"
    if scale < np.finfo(rmax.dtype).tiny:
        scale = np.array(1.0, dtype=rmax.dtype)
        zero_point = np.array(0, dtype=qmin.dtype)
    else:
        zero_point = np.array(np.round(qmin - rmin / scale), dtype=qmin.dtype)
        scale = scale.astype(rmax.dtype)

    if method in CalibrationMethod:
        if symmetric and element_type == onnx_proto.TensorProto.UINT8 and zero_point == 127:
            zero_point = np.array(128, dtype=qmin.dtype)
        return [zero_point, scale]
    # Power-of-2 scale calculation
    elif method in PowerOfTwoMethod:
        if use_pof2s is False:
            return [zero_point, scale]
        pos = scale2pos(scale.item())
        pof2_scale = np.array(pos2scale(pos), dtype=scale.dtype)
        new_rmin = np.minimum((qmin.astype(np.float32) - zero_point.astype(np.float32)) * pof2_scale,
                              np.array(0, dtype=rmin.dtype))
        new_zero_point = np.array(np.round(qmin - new_rmin / pof2_scale), dtype=qmin.dtype)
        # To meet hardware's requirements
        if symmetric and element_type == onnx_proto.TensorProto.UINT8 and new_zero_point == 127:
            new_zero_point = np.array(128, dtype=qmin.dtype)
        return [new_zero_point, pof2_scale]
    elif method in Int16Method:
        M, N, diff = find_int16_scale(scale.item())
        int16_scale: Union[np.ndarray[Any, Any], float] = np.array(M / 2**N, dtype=scale.dtype)
        logger.debug(f"Find the {M} / 2 ** {N} that is closest to scale {scale}"
                     f"with the difference being {diff}")
        if int16_scale < np.finfo(np.float32).tiny:
            int16_scale = 1 / 2**14

        new_rmin = np.minimum((qmin.astype(np.float32) - zero_point.astype(np.float32)) * int16_scale,
                              np.array(0, dtype=rmin.dtype))
        new_zero_point = np.array(np.round(qmin - new_rmin / int16_scale), dtype=qmin.dtype)
        if symmetric and element_type == onnx_proto.TensorProto.UINT8 and new_zero_point == 127:
            new_zero_point = np.array(128, dtype=qmin.dtype)

        return [new_zero_point, int16_scale]
    else:
        return [zero_point, scale]


@log_errors
def compute_scale_zp_fp(rmin: np.ndarray[Any, Any],
                        rmax: np.ndarray[Any, Any],
                        qmin: np.ndarray[Any, Any],
                        qmax: np.ndarray[Any, Any],
                        element_type: int,
                        method: CalibrationMethod,
                        symmetric: bool = True,
                        use_scaling: bool = False) -> List[Any]:
    """Calculate the scale and zero point for a float type.

    :param rmin: minimum value of r
    :param rmax: maximum value of r
    :param element_type: the element data type of the tensor to quantize
    :return: zero and scale [z, s]
    """
    if element_type not in ONNX_FP_QTYPES_LIST + ONNX_BFP_QTYPES_LIST:
        raise ValueError(f"Quantization to element_type={element_type} not implemented.")

    # Adjust rmin and rmax such that 0 is included in the range. This is
    # required to make sure zero can be represented by the quantization data
    # type (i.e. to make sure qmin <= zero_point <= qmax)
    rmin = np.minimum(rmin, np.array(0, dtype=rmin.dtype))
    rmax = np.maximum(rmax, np.array(0, dtype=rmax.dtype))

    # Ensure that rmax-rmin is less than or equal to sys.float_info.max
    if rmin == -np.inf:
        logger.warning("rmin is set to -inf, replacing with a very small value.")
        rmin = np.full_like(rmin, -sys.float_info.max / 2)
    if rmax == np.inf:
        logger.warning("rmax is set to inf, replacing with a very large value.")
        rmax = np.full_like(rmax, sys.float_info.max / 2)

    if symmetric:
        absmax = np.maximum(np.abs(rmin), np.abs(rmax))
        rmin = -absmax
        rmax = +absmax

    assert qmin <= qmax, f"qmin={rmin} > qmax={rmax}"
    dr = np.array(rmax.astype(np.float64) - rmin.astype(np.float64), dtype=np.float64)
    dq = np.array(qmax, dtype=np.float64) - np.array(qmin, dtype=np.float64)
    scale = np.array(dr / dq) if use_scaling else np.array(1.0, dtype=np.float32)
    if np.isnan(scale):
        raise ValueError("NaN detected, please check the correctness of the model")
    assert scale >= 0, "scale issue"
    if scale < np.finfo(rmax.dtype).tiny:
        scale = np.array(1.0, dtype=rmax.dtype)
        zero_point = np.array(0, dtype=scale.dtype)
    else:
        scale = scale.astype(rmax.dtype)
        if symmetric:
            zero_point = np.array(0, dtype=scale.dtype)
        else:
            zero_point = np.array(np.round(qmin - rmin / scale), dtype=scale.dtype)

    if method not in CalibrationMethod:
        logger.warning("Suggest using methods from CalibrationMethod as it only supports float scale.")

    return [zero_point, scale]


def dequantize_data(data: np.ndarray[Any, Any], scale: np.ndarray[Any, Any], zero_point: np.ndarray[Any, Any]) -> Any:
    """
    :param data: the input data
    :param scale: the scale for quantization
    :param zero_point: the zero point for quantization
    :return: the de-quantized data
    """
    data = data.astype(np.float32)
    deq_arr = (data - zero_point.astype(np.float32)) * scale
    return deq_arr.astype(np.float32)


def quantize_data_pof2s(data: np.ndarray[Any, Any],
                        qType: int,
                        symmetric: bool,
                        reduce_range: bool = False,
                        rmin_real_range: Optional[float] = None,
                        rmin_override: Optional[np.ndarray[Any, Any]] = None,
                        rmax_override: Optional[np.ndarray[Any, Any]] = None,
                        method: PowerOfTwoMethod = PowerOfTwoMethod.NonOverflow,
                        pos_range: int = 5,
                        use_pof2s: bool = True,
                        use_scaling: bool = False) -> Any:
    """
    :param data: data to quantize
    :param qType: data type to quantize to. Supported types UINT8/16 and INT8/16
    :param symmetric: whether symmetric quantization is used or not. This is applied to INT8/16.
    :return: minimum, maximum, zero point, scale, and quantized weights

    To pack weights, we compute a linear transformation

    - when data `type == uint8` mode, from `[rmin, rmax]` -> :math:`[0, 2^{b-1}]` and
    - when data `type == int8`, from `[-m , m]` -> :math:`[-(2^{b-1}-1), 2^{b-1}-1]` where
        `m = max(abs(rmin), abs(rmax))`

    and add necessary intermediate nodes to trasnform quantized weight to full weight using the equation

    :math:`r = S(q-z)`, where

    - *r*: real original value
    - *q*: quantized value
    - *S*: scale
    - *z*: zero point
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Weight must be given as an array not {type(data)}.")
    if rmin_override is not None:
        rmin = rmin_override
    else:
        rmin = data.min() if len(data) else 0.0
    if rmax_override is not None:
        rmax = rmax_override
    else:
        rmax = data.max() if len(data) else 0.0
    rmin = np.array(rmin, dtype=data.dtype)
    rmax = np.array(rmax, dtype=data.dtype)
    zero_point = 0
    scale = np.array(1.0, dtype=data.dtype)

    if qType in ONNX_FP_QTYPES_LIST + ONNX_BFP_QTYPES_LIST:
        reduce_range = use_scaling  # If scale the activation, it will use a reduced range
        qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range=reduce_range)
        zero_point, scale = compute_scale_zp_fp(rmin,
                                                rmax,
                                                qmin,
                                                qmax,
                                                qType,
                                                method,
                                                symmetric=symmetric,
                                                use_scaling=use_scaling)
        quantized_data = quantize_nparray(qType, np.asarray(data), scale, zero_point)
        return _check_type(rmin, rmax, zero_point, scale, quantized_data, zero_point_index=2)

    qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range, symmetric=symmetric)
    zero_point, scale = compute_scale_zp(rmin,
                                         rmax,
                                         qmin,
                                         qmax,
                                         qType,
                                         method,
                                         symmetric=symmetric,
                                         use_pof2s=use_pof2s)

    quantized_data = quantize_nparray(qType, np.asarray(data), scale, zero_point)

    if method == PowerOfTwoMethod.NonOverflow:
        return _check_type(rmin, rmax, zero_point, scale, quantized_data, zero_point_index=2)
    elif method == PowerOfTwoMethod.MinMSE:
        scale_mse = scale
        zp_mse = zero_point
        quantized_data_mse = quantized_data
        diff_min = float("inf")
        for i in range(pos_range):
            new_scale = pos2scale(scale2pos(scale) + i - 1)
            new_scale = np.array(new_scale, dtype=data.dtype)
            rmin = (qmin.astype(np.float32) - zero_point.astype(np.float32)) * new_scale

            new_quantized_data = quantize_nparray(qType, np.asarray(data), new_scale, zp_mse)
            diff = np.sum((dequantize_data(new_quantized_data, new_scale, zp_mse) - np.asarray(data))**2)
            if diff < diff_min:
                diff_min = diff
                scale_mse = new_scale
                quantized_data_mse = new_quantized_data

        rmin_mse = (qmin.astype(np.float32) - zp_mse.astype(np.float32)) * scale_mse
        rmax_mse = (qmax.astype(np.float32) - zp_mse.astype(np.float32)) * scale_mse
        return _check_type(rmin_mse, rmax_mse, zp_mse, scale_mse, quantized_data_mse, zero_point_index=2)

    elif method == Int16Method.MinMax:
        return _check_type(rmin, rmax, zero_point, scale, quantized_data, zero_point_index=2)
    else:
        return _check_type(rmin, rmax, zero_point, scale, quantized_data, zero_point_index=2)


def save_tensor_hist_fig(calibrator: Any, dr: Any, extra_options: Dict[str, Any]) -> None:
    is_save_hist = False
    if "SaveTensorHistFig" in extra_options and extra_options["SaveTensorHistFig"]:
        is_save_hist = True
    if not is_save_hist:
        return
    calibrator.collect_data(dr)
    if not hasattr(calibrator, 'collector') or not calibrator.collector or not calibrator.collector.histogram_dict:
        logger.warning("This calib Method not support tensor histogram")
        return

    import matplotlib.pyplot as plt
    with tempfile.TemporaryDirectory(prefix="ort.hist.") as hist_tmp_dir:
        hist_tmp_dir = "./tensor_hist"
        check_and_create_path(hist_tmp_dir)
        hist_tmp_dir = os.path.abspath(hist_tmp_dir)
        logger.info("The Tensor Hist: {}".format(hist_tmp_dir))
        percentile_dict = calibrator.collector.compute_percentile()
        for tensor_name, tensor_value in calibrator.collector.histogram_dict.items():
            percentile_min = percentile_dict[tensor_name][0].item()
            percentile_max = percentile_dict[tensor_name][1].item()
            tensor_name = tensor_name.replace("/", "_")
            tensor_name = tensor_name.replace(".", "_")
            tensor_name = tensor_name.replace(":", "_")
            tensor_bins = tensor_value[1]
            tensor_freq = tensor_value[0]
            bar_width = tensor_bins[1] - tensor_bins[0]
            plt.bar(tensor_bins[:-1], tensor_freq, width=bar_width)

            model_hist_path = Path(hist_tmp_dir).joinpath(tensor_name).as_posix()
            min_value = tensor_value[2]
            max_value = tensor_value[3]
            plt.title(tensor_name)
            plt.axvline(x=max_value, color='r', linestyle='--', linewidth=2)
            plt.axvline(x=percentile_max, color='r', linestyle='--', linewidth=2)
            plt.axvline(x=min_value, color='r', linestyle='--', linewidth=2)
            plt.axvline(x=percentile_min, color='r', linestyle='--', linewidth=2)
            plt.xlabel(
                f'Value Max:{max_value:.4f}; PerMax:{percentile_max:.4f} Min:{min_value:.4f}; PerMin:{percentile_min:.4f}'
            )
            plt.ylabel('Frequency')
            plt.savefig(model_hist_path)

            plt.close()


def get_exclude_nodes(model_path: str, input_nodes: Union[List[str], None], output_nodes: Union[List[str],
                                                                                                None]) -> List[str]:
    """
    Return the nodes to be excluded based on the given input and output nodes.
    :param model_path: the model path
    :param input_nodes: the nodes to start quantizing
    :param zero_point: the nodes to terminate quantizing
    :return: the nodes excluded from quantization
    """

    def update_exclude_input_nodes(exclude_nodes: List[str], name_list: List[str], name: str,
                                   input_nodes: List[str]) -> List[str]:
        index = name_list.index(name)
        exclude_nodes_i = name_list[:index]
        exclude_nodes = list(set(exclude_nodes) | set(exclude_nodes_i))
        exclude_nodes = list(set(exclude_nodes) - set(input_nodes))
        return exclude_nodes

    def update_exclude_output_nodes(exclude_nodes: List[str], name_list: List[str], name: str,
                                    output_nodes: List[str]) -> List[str]:
        index = name_list.index(name) + 1
        exclude_nodes_o = name_list[index:]
        exclude_nodes = list(set(exclude_nodes) | set(exclude_nodes_o))
        exclude_nodes = list(set(exclude_nodes) - set(output_nodes))
        return exclude_nodes

    model = ONNXModel(onnx.load(model_path))
    model.topological_sort()

    model_input_to_node: Dict[str, List[str]] = {}
    model_output_to_node: Dict[str, List[str]] = {}
    name_list: List[str] = []
    exclude_nodes: List[str] = []

    for i in model.model.graph.input:
        model_input_to_node[i.name] = []
    for o in model.model.graph.output:
        model_output_to_node[o.name] = []
    for n in model.model.graph.node:
        for i in n.input:
            for k, v in model_input_to_node.items():
                if i == k:
                    model_input_to_node[k].append(n.name)
        for o in n.output:
            for k, v in model_output_to_node.items():
                if o == k:
                    model_output_to_node[k].append(n.name)
        name_list.append(n.name)

    if input_nodes:
        for name in input_nodes:
            if name in name_list:
                exclude_nodes = update_exclude_input_nodes(exclude_nodes, name_list, name, input_nodes)
            elif name in model_input_to_node:
                for n in model_input_to_node[name]:
                    exclude_nodes = update_exclude_input_nodes(exclude_nodes, name_list, n, model_input_to_node[name])
            elif name in model_output_to_node:
                for n in model_output_to_node[name]:
                    exclude_nodes = update_exclude_input_nodes(exclude_nodes, name_list, n, model_output_to_node[name])
            else:
                logger.warning(
                    f"Fail to find the {name} in {model_path}, the input_nodes {input_nodes} did not take effect, please check input_nodes parameter"
                )

    if output_nodes:
        for name in output_nodes:
            if name in name_list:
                exclude_nodes = update_exclude_output_nodes(exclude_nodes, name_list, name, output_nodes)
            elif name in model_output_to_node:
                for n in model_output_to_node[name]:
                    exclude_nodes = update_exclude_output_nodes(exclude_nodes, name_list, n, model_output_to_node[name])
            elif name in model_input_to_node:
                for n in model_input_to_node[name]:
                    exclude_nodes = update_exclude_output_nodes(exclude_nodes, name_list, n, model_input_to_node[name])
            else:
                logger.warning(
                    f"Fail to find the {name} in {model_path}, the input_nodes {input_nodes} did not take effect, please check input_nodes parameter"
                )
    return exclude_nodes


def get_matmul_nodes_without_weights(model_path: str) -> List[str]:
    model = ONNXModel(onnx.load(model_path))
    model.topological_sort()

    initializer_names = {init.name for init in model.model.graph.initializer}

    matmul_without_weights_nodes_name = []

    for node in model.model.graph.node:
        if node.op_type == 'MatMul':
            _, input2 = node.input
            if input2 not in initializer_names:
                matmul_without_weights_nodes_name.append(node.name)

    return matmul_without_weights_nodes_name


@log_errors
def run_onnx_model(model_path: str, data_reader: Any) -> None:
    """
    Check if the input ONNX can run successfully
    :param model_path: the model path
    :param data_reader: the data reader for feeding data
    """
    try:
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        inputs = data_reader.get_next()
        output = sess.run(None, inputs)
        if output:
            logger.info(f"The input ONNX model {model_path} can run inference successfully")
        else:
            logger.warning(f"Fail to run inference, please check the {model_path} and the 'calibration_data_reader'.")
    except Exception as e:
        raise ValueError(
            f"Fail to run inference for {model_path}. Exception: {e}. Please check the {model_path} and the 'calibration_data_reader'."
        )


@log_errors
def check_onnx_model(model_path: str) -> None:
    """
    Check if the input ONNX can create InferenceSession successfully
    :param model_path: the model path
    """
    try:
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        logger.info(f"The input ONNX model {model_path} can create InferenceSession successfully")

    except Exception as e:
        raise ValueError(
            f"Fail to create InferenceSession for {model_path}. Exception: {e}. Please check the {model_path}.")


def check_model_quantizable(model: ModelProto, op_types_to_quantize: Optional[List[str]],
                            nodes_to_exclude: List[str]) -> bool:
    """
    Check if the model can be quantized.
    """
    value_infos = {vi.name: vi for vi in model.graph.value_info}
    value_infos.update({ot.name: ot for ot in model.graph.output})
    value_infos.update({it.name: it for it in model.graph.input})
    initializer = {init.name for init in model.graph.initializer}

    tensors_to_calibrate = set()
    tensor_type_to_calibrate = {TensorProto.FLOAT, TensorProto.FLOAT16}

    for node in model.graph.node:
        if (not op_types_to_quantize or node.op_type in op_types_to_quantize) and node.name not in nodes_to_exclude:
            for tensor_name in itertools.chain(node.input, node.output):
                if tensor_name in value_infos:
                    vi = value_infos[tensor_name]
                    if (vi.type.HasField("tensor_type") and (vi.type.tensor_type.elem_type in tensor_type_to_calibrate)
                            and (tensor_name not in initializer)):
                        tensors_to_calibrate.add(tensor_name)

    if len(tensors_to_calibrate) == 0:
        return False

    return True


def dpu_leaky_relu_alpha(x: float) -> float:
    """
    This function implements a DPU-specific Leaky ReLU activation with alpha value correction.
    """
    rounded_value = round(x * 256)
    return rounded_value / 256.


def get_model_node_name_dict(model: ModelProto) -> Dict[str, NodeProto]:
    model_node_name_dict: Dict[str, NodeProto] = {}
    for node in model.node:
        if node.name and not model_node_name_dict.get(node.name):
            model_node_name_dict[node.name] = node
        else:
            if not node.name and node.output[0]:
                model_node_name_dict[node.output[0]] = node
            else:
                logger.warning(f"the node name:{node.name} is not exist in model_node_name_dict.")
    return model_node_name_dict


def get_model_weight_name_dict(model: ModelProto) -> Dict[str, TensorProto]:
    model_weight_name_dict: Dict[str, TensorProto] = {}
    for wgt in model.initializer:
        if not model_weight_name_dict.get(wgt.name):
            model_weight_name_dict[wgt.name] = wgt
        else:
            logger.warning(f"the weight name:{wgt.name} is exist in model_weight_name_dict.")
    return model_weight_name_dict


@log_errors
def get_model_node_output_node_name_dict(model: ModelProto) -> Dict[str, str]:
    model_node_output_node_name_dict: Dict[str, str] = {}
    # handle all node
    for node in model.node:
        # the node.output is support multi
        for out in node.output:
            if not model_node_output_node_name_dict.get(out):
                model_node_output_node_name_dict[out] = node.output[0]
            else:
                raise ValueError(
                    f"the node output var name:{node.output} is exist in model_node_output_node_name_dict.")
    return model_node_output_node_name_dict


def get_node_input_var(node: NodeProto) -> Any:
    if len(node.input) > 0:
        return node.input


def get_node_input_node_name(node: NodeProto, model_output_name_dict: Dict[str, str],
                             model_weight_name_dict: Dict[str, TensorProto]) -> Tuple[List[str], List[TensorProto]]:
    inputs = get_node_input_var(node)
    node_input_node_name = []
    node_weights_bias_node_name = []
    for var in inputs:
        if var in model_output_name_dict.keys():
            node_input_node_name.append(model_output_name_dict[var])
        elif var in model_weight_name_dict.keys():
            node_weights_bias_node_name.append(model_weight_name_dict[var])
        else:
            logger.debug(f"the node: {var} is input or output")
    return node_input_node_name, node_weights_bias_node_name


@log_errors
def get_node_from_node_name(name: str, model_output_node_dict: Dict[str, NodeProto]) -> Any:
    if model_output_node_dict.get(name):
        return model_output_node_dict[name]
    else:
        raise ValueError(f"cann't get node:{name} from name.")


def get_weight_from_weight_name(name: str, model_weight_node_dict: Dict[str, TensorProto]) -> Any:
    if model_weight_node_dict.get(name):
        return model_weight_node_dict[name]
    else:
        logger.warning(f"cann't get weight:{name} from name.")


def get_weights_node_of_node(node: NodeProto, model_output_name_dict: Dict[str, str],
                             model_weights_node_dict: Dict[str, TensorProto]) -> List[TensorProto]:
    _, all_weights_name = get_node_input_node_name(node, model_output_name_dict, model_weights_node_dict)
    weights_nodes = []
    for weight in all_weights_name:
        if (weight):
            weights_nodes.append(weight)
    return weights_nodes


def get_output_nodes_of_node(node: NodeProto, model: GraphProto) -> List[NodeProto]:
    output_nodes_list = []
    for output in node.output:
        for one_node in model.node:
            for one_node_in in one_node.input:
                if one_node_in == output:
                    if one_node and one_node.name not in output_nodes_list:
                        output_nodes_list.append(one_node)
                    else:
                        logger.info(f"the output_node:{one_node.name} already in list")
    return output_nodes_list


def get_clip_min_max(model: ModelProto, clip_node: NodeProto) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Get clip min and max value from Clip node.
    :param model: onnx model instance
    :param clip_node: target Clip node
    :return: the min, max value and para type
             The meaning of para type is:
             None - unknown
             0 - attribute
             1 - initializer
             2 - other nodes
    """

    def _get_from_initializer(model: ModelProto, name: str) -> Any:
        for init in model.graph.initializer:
            if init.name == name:
                return onnx.numpy_helper.to_array(init)
        return None

    def _get_from_attribute(node: NodeProto) -> Any:
        for attr in node.attribute:
            if attr.name == 'value':
                if attr.t.data_type == 1:
                    return list(attr.t.float_data)[0]
                else:
                    return list(attr.t.int32_data)[0]
        return None

    def _get_from_other_node(model: ModelProto, name: str) -> Any:
        for node in model.graph.node:
            if node.op_type == 'Identity' and name in node.output:
                return _get_from_initializer(model, node.input[0])
            if node.op_type == 'Constant' and name in node.output:
                return _get_from_attribute(node)
        return None

    min_value = None
    max_value = None
    if clip_node.op_type != 'Clip':
        return min_value, max_value, None

    # Get from attributes
    for attr in clip_node.attribute:
        if attr.name == 'min':
            min_value = attr.f
        if attr.name == 'max':
            max_value = attr.f

    if min_value is not None or max_value is not None:
        return min_value, max_value, 0

    # Get from initializers
    if len(clip_node.input) > 1:
        min_value = _get_from_initializer(model, clip_node.input[1])
    if len(clip_node.input) > 2:
        max_value = _get_from_initializer(model, clip_node.input[2])

    if min_value is not None or max_value is not None:
        return min_value, max_value, 1

    # Try to get from other nodes
    if len(clip_node.input) > 1:
        min_value = _get_from_other_node(model, clip_node.input[1])
    if len(clip_node.input) > 2:
        max_value = _get_from_other_node(model, clip_node.input[2])

    if min_value is not None or max_value is not None:
        return min_value, max_value, 2

    return min_value, max_value, None


def check_relu_like_node(model: ModelProto, node: NodeProto) -> bool:
    """
    Check if the node is a relu-like node
    :param model: the model instance
    :param node: the node to check
    :return: True if it is
    """
    if node.op_type == 'Relu':
        return True
    elif node.op_type == 'Clip':
        min_value, _, _ = get_clip_min_max(model, node)
        if min_value == 0:
            return True
    return False


def print_quantize_info(model_input: str, model_output: str, calibration_data_reader: str,
                        calibration_data_path: Union[str, None], quant_format: Union[Any, VitisQuantFormat],
                        input_nodes: Union[List[str], None], output_nodes: Union[List[str], None],
                        op_types_to_quantize: Union[List[str], None], extra_op_types_to_quantize: Union[List[str],
                                                                                                        None],
                        per_channel: bool, reduce_range: bool, activation_type: Union[Any, VitisQuantType],
                        weight_type: Union[Any, VitisQuantType], nodes_to_quantize: List[str],
                        nodes_to_exclude: List[str], subgraphs_to_exclude: List[Tuple[List[str]]], optimize_model: bool,
                        use_external_data_format: bool, calibrate_method: Union[Any, PowerOfTwoMethod, Int16Method],
                        execution_providers: Union[List[str], None], enable_npu_cnn: bool, enable_npu_transformer: bool,
                        specific_tensor_precision: bool, debug_mode: bool, convert_fp16_to_fp32: bool,
                        convert_nchw_to_nhwc: bool, include_cle: bool, include_sq: bool, include_rotation: bool,
                        include_fast_ft: bool, extra_options: Dict[str, Any]) -> None:
    """
    print os_cpu, time, tool_version, quantized_configuration information.
    """

    def _print_time_info() -> None:
        """
        print time information.
        """
        now = datetime.now()
        print("[QUARK_INFO]: Time information:")
        print(now)

    def _print_os_cpu_info() -> None:
        """
        print os_cpu information.
        """
        system_info = platform.system()
        node_info = platform.node()
        release_info = platform.release()
        version_info = platform.version()
        machine_info = platform.machine()
        processor_info = platform.processor()
        print("[QUARK_INFO]: OS and CPU information:")
        print("{:>50}".format("system ---"), system_info)
        print("{:>50}".format("node ---"), node_info)
        print("{:>50}".format("release ---"), release_info)
        print("{:>50}".format("version ---"), version_info)
        print("{:>50}".format("machine ---"), machine_info)
        print("{:>50}".format("processor ---"), processor_info)

    def _print_tools_version_info() -> None:
        """
        print tools version information.
        """
        python_version = platform.python_version()
        onnx_version = onnx.__version__  # type: ignore[attr-defined]
        onnxruntime_version = ort.__version__
        quark_onnx_version = __version__
        print("[QUARK_INFO]: Tools version information:")
        print("{:>50}".format("python ---"), python_version)
        print("{:>50}".format("onnx ---"), onnx_version)
        print("{:>50}".format("onnxruntime ---"), onnxruntime_version)
        print("{:>50}".format("quark.onnx ---"), quark_onnx_version)

    def _print_quantized_config_info() -> None:
        """
        print quantized configuration information.
        """
        print("[QUARK_INFO]: Quantized Configuration information:")
        print("{:>50}".format("model_input ---"), model_input)
        print("{:>50}".format("model_output ---"), model_output)
        print("{:>50}".format("calibration_data_reader ---"), calibration_data_reader)
        print("{:>50}".format("calibration_data_path ---"), calibration_data_path)
        print("{:>50}".format("quant_format ---"), quant_format)
        print("{:>50}".format("input_nodes ---"), input_nodes)
        print("{:>50}".format("output_nodes ---"), output_nodes)
        print("{:>50}".format("op_types_to_quantize ---"), op_types_to_quantize)
        print("{:>50}".format("extra_op_types_to_quantize ---"), extra_op_types_to_quantize)
        print("{:>50}".format("per_channel ---"), per_channel)
        print("{:>50}".format("reduce_range ---"), reduce_range)
        print("{:>50}".format("activation_type ---"), activation_type)
        print("{:>50}".format("weight_type ---"), weight_type)
        print("{:>50}".format("nodes_to_quantize ---"), nodes_to_quantize)
        print("{:>50}".format("nodes_to_exclude ---"), nodes_to_exclude)
        print("{:>50}".format("subgraphs_to_exclude ---"), subgraphs_to_exclude)
        print("{:>50}".format("optimize_model ---"), optimize_model)
        print("{:>50}".format("use_external_data_format ---"), use_external_data_format)
        print("{:>50}".format("calibrate_method ---"), calibrate_method)
        print("{:>50}".format("execution_providers ---"), execution_providers)
        print("{:>50}".format("enable_npu_cnn ---"), enable_npu_cnn)
        print("{:>50}".format("enable_npu_transformer ---"), enable_npu_transformer)
        print("{:>50}".format("specific_tensor_precision ---"), specific_tensor_precision)
        print("{:>50}".format("debug_mode ---"), debug_mode)
        print("{:>50}".format("convert_fp16_to_fp32 ---"), convert_fp16_to_fp32)
        print("{:>50}".format("convert_nchw_to_nhwc ---"), convert_nchw_to_nhwc)
        print("{:>50}".format("include_cle ---"), include_cle)
        print("{:>50}".format("include_sq ---"), include_sq)
        print("{:>50}".format("include_rotation ---"), include_rotation)
        print("{:>50}".format("include_fast_ft ---"), include_fast_ft)
        print("{:>50}".format("extra_options ---"), extra_options)

    try:
        _print_time_info()
        _print_os_cpu_info()
        _print_tools_version_info()
        _print_quantized_config_info()
    except Exception as e:
        pass


def print_quantize_dynamic_info(model_input: Union[str, Path, onnx.ModelProto], model_output: Union[str, Path],
                                op_types_to_quantize: Union[List[str], None], per_channel: bool, reduce_range: bool,
                                weight_type: Union[Any, VitisQuantType], nodes_to_quantize: List[str],
                                nodes_to_exclude: List[str], subgraphs_to_exclude: List[Tuple[List[str]]],
                                use_external_data_format: bool, debug_mode: bool, extra_options: Dict[str,
                                                                                                      Any]) -> None:
    """
    print os_cpu, time, tool_version, quantized_configuration information.
    """

    def _print_time_info() -> None:
        """
        print time information.
        """
        now = datetime.now()
        print("[QUARK_INFO]: Time information:")
        print(now)

    def _print_os_cpu_info() -> None:
        """
        print os_cpu information.
        """
        system_info = platform.system()
        node_info = platform.node()
        release_info = platform.release()
        version_info = platform.version()
        machine_info = platform.machine()
        processor_info = platform.processor()
        print("[QUARK_INFO]: OS and CPU information:")
        print("{:>50}".format("system ---"), system_info)
        print("{:>50}".format("node ---"), node_info)
        print("{:>50}".format("release ---"), release_info)
        print("{:>50}".format("version ---"), version_info)
        print("{:>50}".format("machine ---"), machine_info)
        print("{:>50}".format("processor ---"), processor_info)

    def _print_tools_version_info() -> None:
        """
        print tools version information.
        """
        python_version = platform.python_version()
        onnx_version = onnx.__version__  # type: ignore[attr-defined]
        onnxruntime_version = ort.__version__
        quark_onnx_version = __version__
        print("[QUARK_INFO]: Tools version information:")
        print("{:>50}".format("python ---"), python_version)
        print("{:>50}".format("onnx ---"), onnx_version)
        print("{:>50}".format("onnxruntime ---"), onnxruntime_version)
        print("{:>50}".format("quark.onnx ---"), quark_onnx_version)

    def _print_quantized_config_info() -> None:
        """
        print quantized configuration information.
        """
        print("[QUARK_INFO]: Quantized Configuration information:")
        print("{:>50}".format("model_input ---"), model_input)
        print("{:>50}".format("model_output ---"), model_output)
        print("{:>50}".format("op_types_to_quantize ---"), op_types_to_quantize)
        print("{:>50}".format("per_channel ---"), per_channel)
        print("{:>50}".format("reduce_range ---"), reduce_range)
        print("{:>50}".format("weight_type ---"), weight_type)
        print("{:>50}".format("nodes_to_quantize ---"), nodes_to_quantize)
        print("{:>50}".format("nodes_to_exclude ---"), nodes_to_exclude)
        print("{:>50}".format("subgraphs_to_exclude ---"), subgraphs_to_exclude)
        print("{:>50}".format("use_external_data_format ---"), use_external_data_format)
        print("{:>50}".format("debug_mode ---"), debug_mode)
        print("{:>50}".format("extra_options ---"), extra_options)

    try:
        _print_time_info()
        _print_os_cpu_info()
        _print_tools_version_info()
        _print_quantized_config_info()
    except Exception as e:
        pass


def find_int16_scale(x: float) -> Tuple[float, float, float]:
    """
    Given a float value, find the closest value corresponding to  M and 2**N,
    where the range of M and 2**N is within the representation range of int16 and uint16.
    """
    if x == 0:
        return 0, 0, 0

    closest_m = 0
    closest_n = 0
    closest_diff = float('inf')

    # Loop through possible values of n and m
    for n in range(0, 17):  # Adjust the range as needed
        m_fs = x * 2**n
        if m_fs < -2**15 or m_fs > 2**15 - 1:
            continue
        m_floor = math.floor(m_fs)
        m_ceil = math.ceil(m_fs)
        for m in [m_floor, m_ceil]:  # Adjust the range as needed
            value = m / 2**n
            diff = abs(value - x)
            if diff < closest_diff:
                closest_m = m
                closest_n = n
                closest_diff = diff

    return closest_m, closest_n, closest_diff


def remove_initializer_from_input(model: ModelProto) -> ModelProto:

    if model.ir_version < 4:
        logger.warning(
            "Model with ir_version below 4 requires to include initilizer in graph input, change ir_version to 7")
        model.ir_version = 7

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    return model


def fp32_nodes(input_model_path: str) -> Dict[str, int]:
    try:
        fp32_nodes_dict = {}
        fp32_model = onnx.load(input_model_path)
        onnx_model = ONNXModel(fp32_model)

        for node in onnx_model.model.graph.node:
            if node.op_type not in fp32_nodes_dict:
                fp32_nodes_dict[node.op_type] = 0
            fp32_nodes_dict[node.op_type] += 1

        return fp32_nodes_dict

    except Exception as e:
        return {}


def print_fp32_nodes(fp32_nodes_dict: Dict[str, int], output_model_path: str) -> None:
    try:
        fp32_nodes_list = list(fp32_nodes_dict.keys())

        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table()
        table.add_column('Op Type')
        table.add_column('Float Model', style='bold green1')

        for node_op_type in fp32_nodes_list:
            node_fp32_count = fp32_nodes_dict[node_op_type]
            table.add_row(node_op_type, str(node_fp32_count))
        table.add_section()
        table.add_row("Quantized model path", output_model_path)

        print(
            "The operation types and their corresponding quantities of the input float model is shown in the table below."
        )
        console.print(table)

    except Exception as e:
        pass


# using data for sub_model to inference
def inference_sub_model_with_data(input_model: onnx.ModelProto, start_node_map: Dict[str, List[float]],
                                  end_node_list: List[str]) -> Tuple[List[float], str]:
    node_name_map = get_model_node_name_dict(input_model.graph)
    start_node_tensor = []
    end_node_tensor = []
    start_tensor_map = {}
    for start_node_name, start_node_input_tensor_val in start_node_map.items():
        start_node = node_name_map[start_node_name]
        one_tensor = start_node.input[0]
        start_node_tensor.append(one_tensor)
        start_tensor_map[one_tensor] = start_node_input_tensor_val
    for end_node_name in end_node_list:
        end_node = node_name_map[end_node_name]
        end_node_tensor.append(end_node.output[0])

    sub_model_path = tempfile.TemporaryDirectory(prefix="vai.submodel.")
    opt_model_output = Path(sub_model_path.name).joinpath("all.onnx").as_posix()
    sub_model_output = Path(sub_model_path.name).joinpath("sub_model.onnx").as_posix()
    onnx.save_model(input_model, opt_model_output, save_as_external_data=False)
    onnx.utils.extract_model(opt_model_output, sub_model_output, start_node_tensor, end_node_tensor, False)
    session = ort.InferenceSession(sub_model_output, providers=['CUDAExecutionProvider'])
    start_tensor_one_batch = {}
    end_tensor_list = []
    for key in start_tensor_map.keys():
        values = start_tensor_map[key]
        for bs in range(len(values)):
            start_tensor_one_batch[key] = values[bs]
            end_tensor_one_tensor = session.run(end_node_tensor, start_tensor_one_batch)
            end_tensor_list.append(end_tensor_one_tensor[0])
    return end_tensor_list, sub_model_output


def extract_sub_model(model_path: str, start_tensors: List[str], end_tensors: List[str]) -> onnx.ModelProto:
    sub_model_path = tempfile.TemporaryDirectory(prefix="vai.submodel.")
    sub_model_output = Path(sub_model_path.name).joinpath("sub_model.onnx").as_posix()
    onnx.utils.extract_model(model_path, sub_model_output, start_tensors, end_tensors, check_model=False)
    sub_model = onnx.load(sub_model_output)
    return sub_model


# feed the input for model inference
# return the output value
def get_intermedia_output(model: onnx.ModelProto, input_feed_dict: Dict[str, List[float]],
                          output_tensors: List[str]) -> Any:
    sub_model_path = tempfile.TemporaryDirectory(prefix="vai.submodel.")
    sub_model_output = Path(sub_model_path.name).joinpath("sub_model.onnx").as_posix()
    onnx.save(model, sub_model_output)
    session = ort.InferenceSession(sub_model_output, providers=['CUDAExecutionProvider'])
    start_tensor_one_batch = {}
    end_tensor_list = []
    for key in input_feed_dict.keys():
        values = input_feed_dict[key]
        for bs in range(len(values)):
            start_tensor_one_batch[key] = values[bs]
            end_tensor_one_tensor = session.run(output_tensors, start_tensor_one_batch)
            end_tensor_list.append(end_tensor_one_tensor[0])
    output_tensors_val = np.array(end_tensor_list)
    return output_tensors_val


def get_batch_size(model: onnx.ModelProto) -> Any:
    input_shape = model.graph.input[0].type.tensor_type.shape
    batch_size = input_shape.dim[0].dim_value if input_shape.dim[0].dim_value != 0 else input_shape.dim[0].dim_param
    return batch_size


def make_batch_size_fixed(model: onnx.ModelProto, batch_size: int = 1) -> onnx.ModelProto:
    if isinstance(batch_size, int):
        for i in range(len(model.graph.input)):
            model.graph.input[i].type.tensor_type.shape.dim[0].ClearField("dim_param")
            model.graph.input[i].type.tensor_type.shape.dim[0].dim_value = batch_size
        for i in range(len(model.graph.output)):
            model.graph.output[i].type.tensor_type.shape.dim[0].ClearField("dim_param")
            model.graph.output[i].type.tensor_type.shape.dim[0].dim_value = batch_size
        for i in range(len(model.graph.value_info)):
            if len(model.graph.value_info[i].type.tensor_type.shape.dim) > 1:
                model.graph.value_info[i].type.tensor_type.shape.dim[0].dim_value = batch_size
    return model


def make_batch_size_dynamic(model: onnx.ModelProto, bs: int) -> Any:
    onnx_model = ONNXModel(model)
    model = onnx_model.model

    for i in range(len(model.graph.input)):
        model.graph.input[i].type.tensor_type.shape.dim[0].dim_value = bs
    for i in range(len(model.graph.output)):
        model.graph.output[i].type.tensor_type.shape.dim[0].dim_value = bs
    for i in range(len(model.graph.value_info)):
        if len(model.graph.value_info[i].type.tensor_type.shape.dim) > 1:
            model.graph.value_info[i].type.tensor_type.shape.dim[0].dim_value = bs
    for node in model.graph.node:
        if node.op_type == 'Reshape':
            reshape_input_name = node.input[1]
            for tensor in model.graph.initializer:
                if tensor.name == reshape_input_name:
                    tensor_array = onnx.numpy_helper.to_array(tensor)
                    tensor_array_shape = list(tensor_array)
                    tensor_array_shape[0] = bs
                    new_tensor_array = np.array(tensor_array_shape, dtype=np.int64)
                    new_tensor = onnx.numpy_helper.from_array(new_tensor_array, tensor.name)
                    model.graph.initializer.extend([new_tensor])
                    onnx_model.remove_initializer(tensor)
    return onnx_model.model


def infer_custom_op_shape(model: onnx.ModelProto) -> onnx.ModelProto:
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    int_max = 2**31 - 1
    auto_merge = True
    guess_output_rank = True
    verbose = 0
    shape_infer = SymbolicShapeInference(int_max, auto_merge, guess_output_rank, verbose)
    infer_onnx_file = 'sym_shape_infer_temp.onnx'
    has_file = os.path.isfile(infer_onnx_file)
    try:
        model = shape_infer.infer_shapes(model)
    except Exception:
        if not has_file and os.path.isfile(infer_onnx_file):
            os.remove(infer_onnx_file)

    input = model.graph.input
    output = model.graph.output
    initializer = model.graph.initializer
    value_info = model.graph.value_info
    vimap = {value_info.name: value_info for value_info in value_info}
    imap = {initializer.name: initializer for initializer in initializer}
    vimap.update({input.name: input for input in input})
    vimap.update({output.name: output for output in output})
    for out in output:
        model.graph.value_info.extend([out])
    need_infer = True
    cnt = 5
    while (need_infer):
        for node in model.graph.node:
            if node.op_type in ['VitisQuantizeLinear', 'QuantizeLinear']:
                input_name = node.input[0]
                zp_name = node.input[2]
                output_name = node.output[0]
                if input_name in vimap and output_name not in vimap:
                    shape_info = vimap[input_name].type.tensor_type.shape.dim
                    shape_list = [int(dim.dim_value) for dim in shape_info]
                    output_tensor = onnx.helper.make_tensor_value_info(output_name, imap[zp_name].data_type, shape_list)
                    model.graph.value_info.extend([output_tensor])
                elif output_name in vimap and input_name not in vimap:
                    shape_info = vimap[output_name].type.tensor_type.shape.dim
                    shape_list = [int(dim.dim_value) for dim in shape_info]
                    input_tensor = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, shape_list)
                    model.graph.value_info.extend([input_tensor])
                elif input_name in imap and output_name not in vimap:
                    shape_list = imap[input_name].dims
                    output_tensor = onnx.helper.make_tensor_value_info(output_name, imap[zp_name].data_type, shape_list)
                    model.graph.value_info.extend([output_tensor])
            elif node.op_type in ['VitisDequantizeLinear', 'DequantizeLinear']:
                input_name = node.input[0]
                zp_name = node.input[2]
                output_name = node.output[0]
                if input_name in vimap and output_name not in vimap:
                    shape_info = vimap[input_name].type.tensor_type.shape.dim
                    shape_list = [int(dim.dim_value) for dim in shape_info]
                    output_tensor = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape_list)
                    model.graph.value_info.extend([output_tensor])
                elif output_name in vimap and input_name not in vimap:
                    shape_info = vimap[output_name].type.tensor_type.shape.dim
                    shape_list = [int(dim.dim_value) for dim in shape_info]
                    input_tensor = onnx.helper.make_tensor_value_info(input_name, imap[zp_name].data_type, shape_list)
                    model.graph.value_info.extend([input_tensor])
                elif input_name in imap and output_name not in vimap:
                    shape_list = imap[input_name].dims
                    output_tensor = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape_list)
                    model.graph.value_info.extend([output_tensor])
            elif node.op_type in [FIX_OP_NAME, BFPFIX_OP_NAME, MXFIX_OP_NAME]:
                input_name = node.input[0]
                output_name = node.output[0]
                if input_name in vimap and output_name not in vimap:
                    shape_info = vimap[input_name].type.tensor_type.shape.dim
                    shape_list = [int(dim.dim_value) for dim in shape_info]
                    output_tensor = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape_list)
                    model.graph.value_info.extend([output_tensor])
                elif output_name in vimap and input_name not in vimap:
                    shape_info = vimap[output_name].type.tensor_type.shape.dim
                    shape_list = [int(dim.dim_value) for dim in shape_info]
                    input_tensor = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, shape_list)
                    model.graph.value_info.extend([input_tensor])
                elif input_name in imap and output_name not in vimap:
                    shape_list = imap[input_name].dims
                    output_tensor = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape_list)
                    model.graph.value_info.extend([output_tensor])
            elif node.op_type == 'VitisInstanceNormalization':
                input_name = node.input[0]
                output_name = node.output[0]
                if input_name in vimap and output_name not in vimap:
                    shape_info = vimap[input_name].type.tensor_type.shape.dim
                    shape_list = [int(dim.dim_value) for dim in shape_info]
                    output_tensor = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape_list)
                    model.graph.value_info.extend([output_tensor])
                elif output_name in vimap and input_name not in vimap:
                    shape_info = vimap[output_name].type.tensor_type.shape.dim
                    shape_list = [int(dim.dim_value) for dim in shape_info]
                    input_tensor = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, shape_list)
                    model.graph.value_info.extend([input_tensor])
        vimap.update({value_info.name: value_info for value_info in value_info})
        cnt = cnt - 1
        if cnt == 0:
            need_infer = False

    return model


def skip_node_with_inf_tensor(model: onnx.ModelProto) -> List[str]:
    tensor_to_node_dict = {}
    init_name_to_init_dict = {}
    inf_list = [np.inf, -np.inf]
    node_with_inf_tensor_list = []
    onnx_model = ONNXModel(model)
    for node in onnx_model.model.graph.node:
        for input_tensor in node.input:
            if input_tensor not in tensor_to_node_dict:
                tensor_to_node_dict[input_tensor] = [node]
            else:
                tensor_to_node_dict[input_tensor].append(node)
    for init in onnx_model.model.graph.initializer:
        init_name_to_init_dict[init.name] = init
    for init_name in init_name_to_init_dict:
        init = init_name_to_init_dict[init_name]
        if np.array_equal(onnx.numpy_helper.to_array(init), np.inf) or np.array_equal(
                onnx.numpy_helper.to_array(init), -np.inf):
            for node_with_inf_tensor in tensor_to_node_dict[init_name]:
                node_with_inf_tensor_list.append(node_with_inf_tensor.name)
    return node_with_inf_tensor_list


def add_or_update_opset_import(model: onnx.ModelProto, domain: str, version: int) -> None:
    for opset in model.opset_import:
        if opset.domain == domain:
            if opset.version < version:
                opset.version = version
            return

    model.opset_import.append(helper.make_operatorsetid(domain, version))


class ONNXQuantizedModel(object):

    def __init__(self, model: onnx.ModelProto) -> None:

        self.model = model
        self.onnx_model = ONNXModel(model)

        self.in_name_to_nodes = self.onnx_model.input_name_to_nodes()
        self.out_name_to_node = self.onnx_model.output_name_to_node()

    def _find_node_input_qdq(self, node: NodeProto,
                             tensor_name: str) -> Tuple[Union[NodeProto, None], Union[NodeProto, None]]:
        """ Find qdq nodes on input tensor, dq always exits but q may be folded """
        if tensor_name not in self.out_name_to_node:
            logger.debug(f"input {tensor_name} of {node.name} came from initializer")
            return None, None

        dq_candidate = self.out_name_to_node[tensor_name]
        if dq_candidate.op_type not in DEQUANT_OP_TYPES:
            logger.debug(f"input {tensor_name} of {node.name} was not quantized")
            return None, None
        elif dq_candidate.input[0] not in self.out_name_to_node:
            logger.debug(f"input {tensor_name} of {node.name} has a folded Q")
            return dq_candidate, None

        q_candidate = self.out_name_to_node[dq_candidate.input[0]]
        if q_candidate.op_type not in QUANT_OP_TYPES:
            logger.warning(f"input {tensor_name} of {node.name} lost a Q")
            return dq_candidate, None

        return dq_candidate, q_candidate  # Note that DQ came first

    def _find_node_output_qdq(self, node: NodeProto,
                              tensor_name: str) -> Tuple[Union[NodeProto, None], Union[NodeProto, None]]:
        """ Find qdq nodes on output tensor  """
        if tensor_name not in self.in_name_to_nodes:
            logger.debug(f"output {tensor_name} of {node.name} was a isolate node")
            return None, None

        assert (len(self.in_name_to_nodes[tensor_name]) == 1)

        q_candidate = self.in_name_to_nodes[tensor_name][0]
        if q_candidate.op_type not in QUANT_OP_TYPES:
            logger.debug(f"output {tensor_name} of {node.name} was not quantized")
            return None, None
        elif q_candidate.output[0] not in self.in_name_to_nodes:
            logger.debug(f"input {tensor_name} of {node.name} lost a DQ")
            return q_candidate, None

        dq_candidate = self.in_name_to_nodes[q_candidate.output[0]][0]
        if dq_candidate.op_type not in DEQUANT_OP_TYPES:
            logger.warning(f"input {tensor_name} of {node.name} lost a DQ")
            return q_candidate, None

        return q_candidate, dq_candidate  # Note that Q came first

    def find_target_op_type_qdqs(self, target_op_type: List[str]) -> Dict[str, Any]:
        """ Get the qdqs on all inputs and outputs of the target node,
            which is the first node with a target op type.
        """
        node_struct: Dict[str, Any] = {'node': None, 'input_qdqs': [], 'output_qdqs': []}

        for node in self.model.graph.node:
            if node.op_type in target_op_type:
                node_struct['node'] = node

                input_qdqs = []  # This contains weight/bias qdqs
                for tensor_name in node.input:
                    dq, q = self._find_node_input_qdq(node, tensor_name)
                    input_qdqs.append((dq, q))
                node_struct['input_qdqs'] = input_qdqs

                output_qdqs = []
                for tensor_name in node.output:
                    q, dq = self._find_node_output_qdq(node, tensor_name)
                    output_qdqs.append((dq, q))
                node_struct['output_qdqs'] = output_qdqs

                break  # Note that only the first node of specified op type

        return node_struct

    def find_target_node_qdqs(self, target_node: NodeProto) -> Dict[str, Any]:
        """ Get the qdqs on all inputs and outputs of the target node.
        """
        node_struct: Dict[str, Any] = {
            'node': None,
            'input_qdqs': [],
            'output_qdqs': [],
        }

        for node in self.model.graph.node:
            if node == target_node:
                node_struct['node'] = node

                input_qdqs = []  # This contains weight/bias qdqs
                for tensor_name in node.input:
                    dq, q = self._find_node_input_qdq(node, tensor_name)
                    input_qdqs.append((dq, q))
                node_struct['input_qdqs'] = input_qdqs

                output_qdqs = []
                for tensor_name in node.output:
                    q, dq = self._find_node_output_qdq(node, tensor_name)
                    output_qdqs.append((dq, q))
                node_struct['output_qdqs'] = output_qdqs

                break  # Got the target node and break

        return node_struct


def check_weights_in_node(model: ModelProto, node: NodeProto) -> bool:
    weights_in_node = False
    initializer_names = {init.name for init in model.graph.initializer}
    for input_ in node.input:
        if input_ in initializer_names:
            weights_in_node = True
    return weights_in_node


def check_ir_version(model_path: str) -> bool:
    model = onnx.load(model_path)
    ir_version = model.ir_version
    return ir_version >= 4


def check_opset_version(model_path: str) -> bool:
    model = onnx.load(model_path)
    opset_version: int = model.opset_import[0].version
    return opset_version >= 10


def check_qdq_model(model_path: str) -> bool:
    model = onnx.load(model_path)
    nodes = [node.op_type for node in model.graph.node]
    qdq_ops = {'QuantizeLinear', 'DequantizeLinear', 'VitisQuantizeLinear', 'VitisDequantizeLinear', 'BFPFixNeuron'}
    is_qdq_model = any(op in qdq_ops for op in nodes)
    return is_qdq_model


def check_extra_quant_op_types(model_path: str, extra_op_types_to_quantize: list[str]) -> None:
    model = onnx.load(model_path)
    model_op_types = {node.op_type for node in model.graph.node}
    absent_op_types = [op_type for op_type in extra_op_types_to_quantize if op_type not in model_op_types]

    if absent_op_types:
        logger.warning(f"The model does not contain the following op types: {', '.join(absent_op_types)}")


def print_quantized_info(quantized_model_path: str, debug_mode: bool) -> None:
    try:
        data_type_dict = {
            0: "",
            1: 'FLOAT',
            2: 'UINT8',
            3: 'INT8',
            4: 'UINT16',
            5: 'INT16',
            6: 'INT32',
            7: 'INT64',
            8: 'STR',
            9: 'BOOL',
            10: 'FLOAT16',
            11: 'DOUBLE',
            12: 'UINT32',
            13: 'UINT64',
            16: 'BFLOAT16',
            17: 'FP8E4M3',
            18: 'FP8E4M3UZ',
            19: 'FP8E5M2',
            20: 'FP8E5M2UZ',
            21: 'UINT4',
            22: 'INT4',
            23: 'FP4E2M1',
            40: 'BFP'
        }
        qdq_ops = {'QuantizeLinear', 'DequantizeLinear', 'VitisQuantizeLinear', 'VitisDequantizeLinear', 'BFPFixNeuron'}

        op_type_with_weights_bias = [
            "MatMul", "Conv", "ConvTranspose", "Gemm", "LayerNormalization", "EmbedLayerNormalization",
            "InstanceNormalization", "PRelu"
        ]
        quantized_data = []

        quantized_model = onnx.load(quantized_model_path)
        onnx_model = ONNXModel(quantized_model)

        tensor_to_node_dict = {}
        tensor_to_init_dict = {}
        for node in onnx_model.model.graph.node:
            for output in node.output:
                tensor_to_node_dict[output] = node
        for init in onnx_model.model.graph.initializer:
            tensor_to_init_dict[init.name] = init

        nodes_quantized_info_list = []

        for node in onnx_model.model.graph.node:
            if len(node.input) >= 1:
                if node.input[0] in tensor_to_node_dict and tensor_to_node_dict[
                        node.input[0]].op_type == 'DequantizeLinear':
                    act_dq_data_type = 0
                    weights_dq_data_type = 0
                    bias_dq_data_type = 0
                    act_dq_node = tensor_to_node_dict[node.input[0]]
                    weights_dq_node = None
                    bias_dq_node = None
                    if len(node.input) >= 2 and node.input[1] in tensor_to_node_dict:
                        weights_dq_node = tensor_to_node_dict[node.input[1]]
                    if len(node.input) >= 3 and node.input[2] in tensor_to_node_dict:
                        bias_dq_node = tensor_to_node_dict[node.input[2]]
                    act_init = tensor_to_init_dict[act_dq_node.input[2]]
                    act_dq_data_type = act_init.data_type
                    weights_init = None
                    bias_init = None
                    if weights_dq_node is not None and node.op_type in op_type_with_weights_bias and check_weights_in_node(
                            onnx_model.model, weights_dq_node):
                        weights_init = tensor_to_init_dict[weights_dq_node.input[2]]
                        weights_dq_data_type = weights_init.data_type
                    if bias_dq_node is not None and node.op_type in op_type_with_weights_bias and check_weights_in_node(
                            onnx_model.model, bias_dq_node):
                        bias_init = tensor_to_init_dict[bias_dq_node.input[2]]
                        bias_dq_data_type = bias_init.data_type
                    nodes_quantized_info_list.append(
                        [node.name, node.op_type, act_dq_data_type, weights_dq_data_type, bias_dq_data_type])
                elif len(node.input) >= 2 and node.input[1] in tensor_to_node_dict and tensor_to_node_dict[
                        node.input[1]].op_type == 'DequantizeLinear':
                    act_dq_data_type = 0
                    weights_dq_data_type = 0
                    bias_dq_data_type = 0
                    act_dq_node = None
                    weights_dq_node = None
                    bias_dq_node = None
                    if node.input[0] in tensor_to_node_dict:
                        act_dq_node = tensor_to_node_dict[node.input[0]]
                    if len(node.input) >= 2 and node.input[1] in tensor_to_node_dict:
                        weights_dq_node = tensor_to_node_dict[node.input[1]]
                    if len(node.input) >= 3 and node.input[2] in tensor_to_node_dict:
                        bias_dq_node = tensor_to_node_dict[node.input[2]]
                    act_init = None
                    weights_init = None
                    bias_init = None
                    if act_dq_node is not None and node.op_type in op_type_with_weights_bias and check_weights_in_node(
                            onnx_model.model, act_dq_node):
                        if len(act_dq_node.input) >= 3 and act_dq_node.input[2] in tensor_to_init_dict:
                            act_init = tensor_to_init_dict[act_dq_node.input[2]]
                            act_dq_data_type = act_init.data_type
                    if weights_dq_node is not None and node.op_type in op_type_with_weights_bias and check_weights_in_node(
                            onnx_model.model, weights_dq_node):
                        if len(weights_dq_node.input) >= 3 and weights_dq_node.input[2] in tensor_to_init_dict:
                            weights_init = tensor_to_init_dict[weights_dq_node.input[2]]
                        weights_dq_data_type = weights_init.data_type
                    if bias_dq_node is not None and node.op_type in op_type_with_weights_bias and check_weights_in_node(
                            onnx_model.model, bias_dq_node):
                        if len(bias_dq_node.input) >= 3 and bias_dq_node.input[2] in tensor_to_init_dict:
                            bias_init = tensor_to_init_dict[bias_dq_node.input[2]]
                        bias_dq_data_type = bias_init.data_type
                    nodes_quantized_info_list.append(
                        [node.name, node.op_type, act_dq_data_type, weights_dq_data_type, bias_dq_data_type])
                if node.input[0] in tensor_to_node_dict and tensor_to_node_dict[
                        node.input[0]].op_type == 'BFPFixNeuron':
                    act_dq_node = tensor_to_node_dict[node.input[0]]
                    weights_dq_node = None
                    bias_dq_node = None
                    if len(node.input) >= 2 and node.input[1] in tensor_to_node_dict:
                        weights_dq_node = tensor_to_node_dict[node.input[1]]
                    if len(node.input) >= 3 and node.input[2] in tensor_to_node_dict:
                        bias_dq_node = tensor_to_node_dict[node.input[2]]
                    act_dq_data_type = 0
                    weights_dq_data_type = 0
                    bias_dq_data_type = 0
                    if act_dq_node is not None and act_dq_node.op_type == "BFPFixNeuron":
                        act_dq_data_type = 40
                    if weights_dq_node is not None and weights_dq_node.op_type == "BFPFixNeuron":
                        weights_dq_data_type = 40
                    if bias_dq_node is not None and bias_dq_node.op_type == "BFPFixNeuron":
                        bias_dq_data_type = 40
                    nodes_quantized_info_list.append(
                        [node.name, node.op_type, act_dq_data_type, weights_dq_data_type, bias_dq_data_type])
                if node.input[0] in tensor_to_node_dict and tensor_to_node_dict[
                        node.input[0]].op_type == 'VitisDequantizeLinear':
                    act_dq_node = tensor_to_node_dict[node.input[0]]
                    weights_dq_node = None
                    bias_dq_node = None
                    if len(node.input) >= 2 and node.input[1] in tensor_to_node_dict:
                        weights_dq_node = tensor_to_node_dict[node.input[1]]
                    if len(node.input) >= 3 and node.input[2] in tensor_to_node_dict:
                        bias_dq_node = tensor_to_node_dict[node.input[2]]
                    act_dq_data_type = 0
                    weights_dq_data_type = 0
                    bias_dq_data_type = 0
                    act_init = tensor_to_init_dict[act_dq_node.input[2]]
                    act_dq_data_type = act_init.data_type
                    weights_init = None
                    bias_init = None
                    if weights_dq_node is not None and node.op_type in op_type_with_weights_bias and check_weights_in_node(
                            onnx_model.model, weights_dq_node):
                        weights_init = tensor_to_init_dict[weights_dq_node.input[2]]
                        weights_dq_data_type = weights_init.data_type
                    if bias_dq_node is not None and node.op_type in op_type_with_weights_bias and check_weights_in_node(
                            onnx_model.model, bias_dq_node):
                        bias_init = tensor_to_init_dict[bias_dq_node.input[2]]
                        bias_dq_data_type = bias_init.data_type
                    nodes_quantized_info_list.append(
                        [node.name, node.op_type, act_dq_data_type, weights_dq_data_type, bias_dq_data_type])
                elif len(node.input) >= 2 and node.input[1] in tensor_to_node_dict and tensor_to_node_dict[
                        node.input[1]].op_type == 'VitisDequantizeLinear':
                    act_dq_node = None
                    weights_dq_node = None
                    bias_dq_node = None
                    if node.input[0] in tensor_to_node_dict:
                        act_dq_node = tensor_to_node_dict[node.input[0]]
                    if len(node.input) >= 2 and node.input[1] in tensor_to_node_dict:
                        weights_dq_node = tensor_to_node_dict[node.input[1]]
                    if len(node.input) >= 3 and node.input[2] in tensor_to_node_dict:
                        bias_dq_node = tensor_to_node_dict[node.input[2]]
                    act_dq_data_type = 0
                    weights_dq_data_type = 0
                    bias_dq_data_type = 0
                    act_init = tensor_to_init_dict[act_dq_node.input[2]]
                    act_dq_data_type = act_init.data_type
                    weights_init = None
                    bias_init = None
                    if weights_dq_node is not None and node.op_type in op_type_with_weights_bias and check_weights_in_node(
                            onnx_model.model, weights_dq_node):
                        weights_init = tensor_to_init_dict[weights_dq_node.input[2]]
                        weights_dq_data_type = weights_init.data_type
                    if bias_dq_node is not None and node.op_type in op_type_with_weights_bias and check_weights_in_node(
                            onnx_model.model, bias_dq_node):
                        bias_init = tensor_to_init_dict[bias_dq_node.input[2]]
                        bias_dq_data_type = bias_init.data_type
                    nodes_quantized_info_list.append(
                        [node.name, node.op_type, act_dq_data_type, weights_dq_data_type, bias_dq_data_type])
                else:
                    if node.op_type not in qdq_ops:
                        act_dq_data_type = 1
                        weights_dq_data_type = 0
                        bias_dq_data_type = 0
                        if len(node.input) >= 2 and node.op_type in op_type_with_weights_bias:
                            weights_dq_data_type = 1
                        if len(node.input) >= 3 and node.op_type in op_type_with_weights_bias:
                            bias_dq_data_type = 1
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table()
        table.add_column('Node Name')
        table.add_column('Op Type')
        table.add_column('Activation', style='bold green1')
        table.add_column('Weights', style='bold green1')
        table.add_column('Bias', style='bold green1')
        quantized_data.append(['Node Name', 'Op Type', 'Activation', 'Weights', 'Bias'])

        for node_quantized_info in nodes_quantized_info_list:
            table.add_row(node_quantized_info[0], node_quantized_info[1], data_type_dict[node_quantized_info[2]],
                          data_type_dict[node_quantized_info[3]], data_type_dict[node_quantized_info[4]])
            quantized_data.append([
                node_quantized_info[0], node_quantized_info[1], data_type_dict[node_quantized_info[2]],
                data_type_dict[node_quantized_info[3]], data_type_dict[node_quantized_info[4]]
            ])
        if debug_mode:
            print("The quantized information for all nodes is shown in the table below.")
            console.print(table)

        op_types_dict: Any = {}
        for node_quantized_info in nodes_quantized_info_list:
            op_type = node_quantized_info[1]
            if op_type not in op_types_dict:
                op_types_dict[op_type] = {"act": {}, "weights": {}, "bias": {}}
            if data_type_dict[node_quantized_info[2]] not in op_types_dict[op_type]["act"]:
                op_types_dict[op_type]["act"][data_type_dict[node_quantized_info[2]]] = 0
            if data_type_dict[node_quantized_info[3]] not in op_types_dict[op_type]["weights"]:
                op_types_dict[op_type]["weights"][data_type_dict[node_quantized_info[3]]] = 0
            if data_type_dict[node_quantized_info[4]] not in op_types_dict[op_type]["bias"]:
                op_types_dict[op_type]["bias"][data_type_dict[node_quantized_info[4]]] = 0
            op_types_dict[op_type]["act"][data_type_dict[node_quantized_info[2]]] += 1
            op_types_dict[op_type]["weights"][data_type_dict[node_quantized_info[3]]] += 1
            op_types_dict[op_type]["bias"][data_type_dict[node_quantized_info[4]]] += 1

        console = Console()

        table = Table()
        table.add_column('Op Type')
        table.add_column('Activation', style='bold green1')
        table.add_column('Weights', style='bold green1')
        table.add_column('Bias', style='bold green1')
        quantized_data.append([])
        quantized_data.append(['Op Type', 'Activation', 'Weights', 'Bias'])

        for op_type in op_types_dict.keys():
            act_list = []
            weights_list = []
            bias_list = []
            for data_type in op_types_dict[op_type]['act'].keys():
                if data_type != "":
                    act_list.append(data_type + "(" + str(op_types_dict[op_type]['act'][data_type]) + ")")
            act_list.sort()
            act_str = " ".join(act_list)
            for data_type in op_types_dict[op_type]['weights'].keys():
                if data_type != "":
                    weights_list.append(data_type + "(" + str(op_types_dict[op_type]['weights'][data_type]) + ")")
            weights_list.sort()
            weights_str = " ".join(weights_list)
            for data_type in op_types_dict[op_type]['bias'].keys():
                if data_type != "":
                    bias_list.append(data_type + "(" + str(op_types_dict[op_type]['bias'][data_type]) + ")")
            bias_list.sort()
            bias_str = " ".join(bias_list)
            table.add_row(op_type, act_str, weights_str, bias_str)
            quantized_data.append([op_type, act_str, weights_str, bias_str])
        if not debug_mode:
            print("The quantized information for all operation types is shown in the table below.")
            print(
                "The discrepancy between the operation types in the quantized model and the float model is due to the application of graph optimization."
            )
            console.print(table)

        with open('quantized_info.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(quantized_data)

    except Exception as e:
        pass


def get_shape_from_tensor(tensor: onnx.TensorProto) -> List[int]:
    shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in tensor.type.tensor_type.shape.dim]
    return shape


def convert_fp16_scale_to_fp32(model_output: str, use_external_data_format: bool) -> None:
    model = onnx.load(model_output)

    for tensor in model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT16:
            logger.info(f"Converting initializer {tensor.name} from FP16 to FP32.")

            float16_data = onnx.numpy_helper.to_array(tensor)
            float32_data = float16_data.astype(np.float32)

            new_tensor = onnx.numpy_helper.from_array(float32_data, tensor.name)

            model.graph.initializer.remove(tensor)
            model.graph.initializer.append(new_tensor)

    for node in model.graph.node:
        for i, input_name in enumerate(node.input):
            for tensor in model.graph.initializer:
                if tensor.name == input_name and tensor.data_type == onnx.TensorProto.FLOAT16:
                    logger.info(f"Converting input {tensor.name} of node {node.name} from FP16 to FP32.")

                    float16_data = onnx.numpy_helper.to_array(tensor)
                    float32_data = float16_data.astype(np.float32)

                    new_tensor = onnx.numpy_helper.from_array(float32_data, tensor.name)

                    model.graph.initializer.remove(tensor)
                    model.graph.initializer.append(new_tensor)

        for i, output_name in enumerate(node.output):
            output_type = None
            for output in model.graph.value_info:
                if output.name == output_name and output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
                    logger.info(f"Converting output {output.name} of node {node.name} from FP16 to FP32.")

                    output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

        for attr in node.attribute:
            if attr.name == 'to' and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT
                logger.info(f"Converting attributes of node {node.name} from FP16 to FP32.")

            if attr.name == 'value' and attr.t.data_type == TensorProto.FLOAT16:
                old_data = attr.t.raw_data
                new_data = onnx.numpy_helper.to_array(attr.t).astype('float32')
                attr.t.data_type = TensorProto.FLOAT
                attr.t.raw_data = new_data.tobytes()
                logger.info(f"Converting attributes of node {node.name} from FP16 to FP32.")

    new_nodes = []
    input_to_node_map: Dict[str, List[NodeProto]] = {}
    for node in model.graph.node:
        for input_ in node.input:
            if input_ not in input_to_node_map:
                input_to_node_map[input_] = []
            input_to_node_map[input_].append(node)
    output_to_node_map: Dict[str, List[NodeProto]] = {}
    for node in model.graph.node:
        for output_ in node.output:
            if output_ not in output_to_node_map:
                output_to_node_map[output_] = []
            output_to_node_map[output_].append(node)

    for input_tensor in model.graph.input:
        if input_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            cast_node_name = f"{input_tensor.name}_Cast"
            cast_node = onnx.helper.make_node('Cast',
                                              inputs=[input_tensor.name],
                                              outputs=[cast_node_name],
                                              to=onnx.TensorProto.FLOAT)
            new_nodes.append(cast_node)

            shape = get_shape_from_tensor(input_tensor)
            new_input = onnx.helper.make_tensor_value_info(cast_node_name, onnx.TensorProto.FLOAT, shape)
            for after_input_node in input_to_node_map[input_tensor.name]:
                after_input_node.input[0] = cast_node_name

    for output_tensor in model.graph.output:
        if output_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            cast_node_name = f"{output_tensor.name}_Cast"
            cast_node = onnx.helper.make_node('Cast',
                                              inputs=[cast_node_name],
                                              outputs=[output_tensor.name],
                                              to=onnx.TensorProto.FLOAT16)
            new_nodes.append(cast_node)

            shape = get_shape_from_tensor(output_tensor)
            new_output = onnx.helper.make_tensor_value_info(cast_node_name, onnx.TensorProto.FLOAT16, shape)
            for before_output_node in output_to_node_map[output_tensor.name]:
                before_output_node.output[0] = cast_node_name

    model.graph.node.extend(new_nodes)

    onnx.save_model(model, model_output, save_as_external_data=use_external_data_format)


def get_eltwise_op(model_path: str) -> List[str]:
    eltwise_op_types = ["Mul", "Add", "Sub", "Div", "Min", "Max"]
    model = onnx.load(model_path)
    eltwise_tensors = []
    for node in model.graph.node:
        if node.op_type in eltwise_op_types:
            for inp in node.input:
                eltwise_tensors.append(inp)
    return eltwise_tensors


def get_opset_version(model: onnx.ModelProto) -> Any:
    ai_onnx_domain = [opset for opset in model.opset_import if not opset.domain or opset.domain == "ai.onnx"]
    if len(ai_onnx_domain) != 1:
        raise ValueError("Failed to find proper ai.onnx domain")
    opset_version = ai_onnx_domain[0].version
    return opset_version


def get_fake_tensor_range(model: onnx.ModelProto) -> Dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    fake_dict = {}
    for node in model.graph.node:
        for input_ in node.input:
            fake_dict[input_] = (np.array([0.0]).astype(np.float32), np.array([1.0]).astype(np.float32))
        for output_ in node.output:
            fake_dict[output_] = (np.array([0.0]).astype(np.float32), np.array([1.0]).astype(np.float32))
    return fake_dict


def convert_nparray(qType: Any, arr: np.ndarray[Any, Any]) -> Any:
    onnx_model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Cast", ["X"], ["Y"], to=qType)],
            "qu",
            [helper.make_tensor_value_info("X", onnx_proto.TensorProto.FLOAT, None)],
            [helper.make_tensor_value_info("Y", qType, None)],
        ))
    ref = ReferenceEvaluator(onnx_model)
    return ref.run(None, {"X": arr})[0]


def convert_to_bf16(model: ModelProto, qType: Any, original_data_type: int = 1) -> ModelProto:
    remove_init_list = []
    add_init_list = []
    for init in model.graph.initializer:
        if init.data_type == original_data_type:
            float_init = onnx.numpy_helper.to_array(init)
            bfloat16_init = convert_nparray(qType, float_init)

            q_weight_initializer = onnx.TensorProto()
            q_weight_initializer.data_type = qType
            q_weight_initializer.dims.extend(init.dims)
            q_weight_initializer.name = init.name
            q_weight_initializer.raw_data = bfloat16_init.flatten().copy().tobytes()
            remove_init_list.append(init)
            add_init_list.append(q_weight_initializer)
    for init in remove_init_list:
        model.graph.initializer.remove(init)
    for q_weight_initializer in add_init_list:
        model.graph.initializer.append(q_weight_initializer)

    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == original_data_type:
                    array = numpy_helper.to_array(attr.t)
                    bfloat16_array = convert_nparray(qType, array)
                    new_tensor = numpy_helper.from_array(bfloat16_array)
                    new_tensor.data_type = qType
                    attr.t.CopyFrom(new_tensor)

    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == original_data_type:
                    attr.i = 16

    add_node_list = []

    input_list = []
    for input_tensor in model.graph.input:
        if input_tensor.type.tensor_type.elem_type == original_data_type:
            input_list.append(input_tensor.name)
    for node in model.graph.node:
        for i in range(len(node.input)):
            input_ = node.input[i]
            if input_ in input_list:
                node.input[i] = input_ + "_cast"
                cast_node = onnx.helper.make_node("Cast",
                                                  inputs=[input_],
                                                  outputs=[input_ + "_cast"],
                                                  to=onnx_proto.TensorProto.BFLOAT16)
                add_node_list.append(cast_node)

    input_to_node: Dict[str, List[NodeProto]] = {}
    for node in model.graph.node:
        for input_ in node.input:
            if input_ not in input_to_node:
                input_to_node[input_] = []
            input_to_node[input_].append(node)

    output_list = []
    for output_tensor in model.graph.output:
        if output_tensor.type.tensor_type.elem_type == original_data_type:
            output_list.append(output_tensor.name)
    for node in model.graph.node:
        for i in range(len(node.output)):
            output_ = node.output[i]
            if output_ in output_list:
                node.output[i] = output_ + "_cast"
                if original_data_type == 1:
                    cast_node = onnx.helper.make_node("Cast",
                                                      inputs=[output_ + "_cast"],
                                                      outputs=[output_],
                                                      to=onnx_proto.TensorProto.FLOAT)
                elif original_data_type == 10:
                    cast_node = onnx.helper.make_node("Cast",
                                                      inputs=[output_ + "_cast"],
                                                      outputs=[output_],
                                                      to=onnx_proto.TensorProto.FLOAT16)
                add_node_list.append(cast_node)
                if output_ in input_to_node:
                    for after_node in input_to_node[output_]:
                        for j in range(len(after_node.input)):
                            input_ = after_node.input[j]
                            if input_ == output_:
                                after_node.input[j] = output_ + "_cast"

    for cast_node in add_node_list:
        model.graph.node.append(cast_node)

    return model


def match_exclude_subgraphs(input_model_path: Union[str, Path, ModelProto],
                            subgraphs: List[Tuple[List[str]]]) -> List[str]:

    def _dfs(node: NodeProto, exclude_nodes_list: List[str], start_nodes_list: List[str],
             output2node_dict: Dict[str, NodeProto]) -> None:
        exclude_nodes_list.append(node.name)
        for input_ in node.input:
            if input_ not in output2node_dict:
                return
            else:
                if output2node_dict[input_].name in start_nodes_list:
                    exclude_nodes_list.append(output2node_dict[input_].name)
                    return
                else:
                    exclude_nodes_list.append(node.name)
                    _dfs(output2node_dict[input_], exclude_nodes_list, start_nodes_list, output2node_dict)

    model = onnx.load(input_model_path)

    name2node_dict = {}
    for node in model.graph.node:
        name2node_dict[node.name] = node

    onnx_model = ONNXModel(model)
    output2node_dict = onnx_model.output_name_to_node()

    exclude_nodes_list: List[str] = []
    for subgraph in subgraphs:
        start_nodes_list: List[str] = []
        end_nodes_list: List[str] = []
        start_nodes_list, end_nodes_list = subgraph[0], subgraph[1]  # type: ignore
        exclude_nodes_list.extend(start_nodes_list)
        exclude_nodes_list.extend(end_nodes_list)
        for end_node_name in end_nodes_list:
            father_node = name2node_dict[end_node_name]
            _dfs(father_node, exclude_nodes_list, start_nodes_list, output2node_dict)
    exclude_nodes_list = list(set(exclude_nodes_list))
    return exclude_nodes_list


def check_model_is_fp16(input_model_path: str) -> bool:
    fp32_data_type = 1
    fp16_data_type = 10
    model = onnx.load(input_model_path)
    fp32_flag = 0
    fp16_flag = 0

    for input_tensor in model.graph.input:
        if input_tensor.type.tensor_type.elem_type == fp32_data_type:
            fp32_flag += 1
        elif input_tensor.type.tensor_type.elem_type == fp16_data_type:
            fp16_flag += 1

    for output_tensor in model.graph.output:
        if output_tensor.type.tensor_type.elem_type == fp32_data_type:
            fp32_flag += 1
        elif output_tensor.type.tensor_type.elem_type == fp16_data_type:
            fp16_flag += 1

    for initializer in model.graph.initializer:
        if initializer.data_type == fp32_data_type:
            fp32_flag += 1
        elif initializer.data_type == fp16_data_type:
            fp16_flag += 1

    if fp32_flag == 0 and fp16_flag > 0:
        return True
    else:
        return False
