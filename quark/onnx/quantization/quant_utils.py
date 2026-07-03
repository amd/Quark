#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import itertools
import math
import os
import re
import types
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx.helper as helper
from onnx import numpy_helper, shape_inference
from onnx import onnx_pb as onnx_proto
from onnx.onnx_ml_pb2 import GraphProto, ModelProto, NodeProto, TensorProto
from onnx.reference import ReferenceEvaluator
from onnxruntime.quantization.calibrate import CalibrationMethod, TensorsData
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import (
    DEQUANT_OP_NAME,
    QUANT_OP_NAME,
    QuantType,
)
from onnxruntime.quantization.quant_utils import load_model_with_shape_infer as ort_load_model_with_shape_infer
from onnxruntime.quantization.tensor_quant_overrides import TensorQuantOverridesHelper
from packaging import version as pv

from quark import __version__ as versions
from quark.common.utils.log import ScreenLogger, log_errors
from quark.onnx.calibration.methods import ExtendedCalibrationMethod, Int16Method, PowerOfTwoMethod
from quark.onnx.operators.custom_ops import (
    _COP_BFP_OP_NAME,
    _COP_DEQUANT_OP_NAME,
    _COP_DOMAIN,
    _COP_IN_OP_NAME,
    _COP_LSTM_OP_NAME,
    _COP_MX_OP_NAME,
    _COP_QUANT_OP_NAME,
    _COP_VERSION,
)
from quark.onnx.utils.system_utils import create_tmp_dir


def is_version_below(package: types.ModuleType, target_version: str) -> bool:
    """
    This function checks whether the package is below a specified version.

    Args:
        package (class ModuleType): The package name, such as onnx or onnxruntime, etc.
        target_version (str): The version to compare against the current package's version.

    Returns:
        True if the current version is less than the target version, False otherwise.
    """
    if not isinstance(package, types.ModuleType):
        raise TypeError(f"The package argument expects class ModuleType type, but you have {type(package)}")
    return pv.parse(package.__version__) < pv.parse(target_version)


if is_version_below(onnx, "1.19.0"):
    try:
        from onnx.reference.custom_element_types import float8e4m3fn  # type: ignore
    except ImportError:
        float8e4m3fn = None  # type: ignore

    # INT4 np.dtypes added in ONNX 1.16. These map to np.int8/np.uint8 because numpy
    # does not support sub-byte types.
    try:
        from onnx.reference import custom_element_types  # type: ignore
        from onnx.reference.custom_element_types import int4, uint4  # type: ignore
    except ImportError:
        int4 = None  # type: ignore
        uint4 = None  # type: ignore
else:
    import ml_dtypes
    from ml_dtypes import float8_e4m3fn as float8e4m3fn
    from ml_dtypes import int4, uint4

logger = ScreenLogger(__name__)

__producer__ = "quark.onnx"
__version__ = f"{versions}"  # The version includes commitid already

COP_DOMAIN = _COP_DOMAIN  # domain for custom ops that implemented using c api
COP_QUANT_OP_NAME = _COP_QUANT_OP_NAME
COP_DEQUANT_OP_NAME = _COP_DEQUANT_OP_NAME
COP_IN_OP_NAME = _COP_IN_OP_NAME
COP_LSTM_OP_NAME = _COP_LSTM_OP_NAME
COP_BFP_OP_NAME = _COP_BFP_OP_NAME
COP_MX_OP_NAME = _COP_MX_OP_NAME
COP_VERSION = _COP_VERSION

QUANT_OP_TYPES = [QUANT_OP_NAME, COP_QUANT_OP_NAME]
DEQUANT_OP_TYPES = [DEQUANT_OP_NAME, COP_DEQUANT_OP_NAME]
FN_OP_TYPES = [COP_BFP_OP_NAME, COP_MX_OP_NAME]

HARD_SIGMOID_SCALE = (2731.0 / 16384.0) / (1.0 / 6.0)
annotate_op_type = ["Conv", "Add", "MaxPool", "AveragePool", "GlobalAveragePool", "MatMul", "Gemm", "ConvTranspose"]
avg_pool_op_type = ["AveragePool", "GlobalAveragePool"]
remove_qdq_op_type: list[str] = []

BFP_OP_DEFAULT_ATTRS = {
    "bfp_method": "to_bfp",
    "axis": 1,
    "bit_width": 16,
    "block_size": 8,
    "rounding_mode": 0,
    "sub_block_size": 2,
    "sub_block_shift_bits": 1,
    "convert_to_bfloat_before_bfp": 0,
}
MX_OP_DEFAULT_ATTRS = {
    "element_dtype": "int8",
    "axis": 1,
    "block_size": 32,
    "rounding_mode": 0,
}


def calculate_mse(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    mse = np.mean((x - y) ** 2)
    mse = np.array(mse)
    return mse


def compute_minmse(
    data: np.ndarray[Any, Any],
    qType: int,
    method: ExtendedCalibrationMethod | None = None,
    symmetric: bool = False,
    minmse_mode: str = "Percentile",
    reduce_range: bool = False,
) -> Any:
    qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range, symmetric)

    rmins = []
    rmaxs = []
    mses = []
    zero_points = []
    scales: list[Any] = []

    if minmse_mode == "Percentile":
        for percentile in [99.9, 99.99, 99.999, 99.9999]:
            rmin = np.percentile(data, (100 - percentile) / 2)
            rmax = np.percentile(data, 100 - (100 - percentile) / 2)

            zero_point, scale = compute_scale_zp(
                rmin, rmax, qmin, qmax, qType, method, symmetric=symmetric, use_pof2s=False
            )
            if scale in scales:
                continue

            quantized_data = quantize_nparray(qType, data, scale, zero_point)
            dequant_data = dequantize_data(quantized_data, scale, zero_point)
            mse = calculate_mse(dequant_data, data)
            mses.append(mse)
            rmins.append(rmin)
            rmaxs.append(rmax)
            zero_points.append(zero_point)
            scales.append(scale)

    elif minmse_mode in ["HistCenter", "All"]:
        bins = min(2048, data.size)
        counts, edges = np.histogram(data, bins=bins)
        centers = (edges[1:] + edges[:-1]) / 2

        start_bin = int(0.5 * bins)
        stride = 1
        for center_i in range(start_bin, bins, stride):
            left_center = int((bins - center_i) / 2)
            right_center = int(bins - (bins - center_i) / 2)
            rmin = centers[left_center]
            rmax = centers[right_center]

            zero_point, scale = compute_scale_zp(
                rmin, rmax, qmin, qmax, qType, method, symmetric=symmetric, use_pof2s=False
            )

            if scale in scales:
                continue

            if minmse_mode == "HistCenter":
                quantized_data = quantize_nparray(qType, centers, scale, zero_point)
                dequant_data = dequantize_data(quantized_data, scale, zero_point)
                mse = ((dequant_data - centers) ** 2 * counts).mean()
            elif minmse_mode == "All":
                quantized_data = quantize_nparray(qType, data, scale, zero_point)
                dequant_data = dequantize_data(quantized_data, scale, zero_point)
                mse = calculate_mse(data, dequant_data)

            mses.append(mse)
            rmins.append(rmin)
            rmaxs.append(rmax)
            zero_points.append(zero_point)
            scales.append(scale)

    argmin = np.argmin(mses)
    rmin = rmins[argmin]
    rmax = rmaxs[argmin]
    zero_point = zero_points[argmin]
    scale = scales[argmin]
    quantized_data = quantize_nparray(qType, data, scale, zero_point)
    return rmin, rmax, zero_point, scale, quantized_data


class ExtendedQuantType(Enum):
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
            return ExtendedQuantType[t]
        except KeyError as e:  # pragma: no cover
            raise ValueError() from e

    @property
    def tensor_type(self) -> Any:
        if self == ExtendedQuantType.QUInt8:
            return TensorProto.UINT8
        if self == ExtendedQuantType.QInt8:
            return TensorProto.INT8
        if self == ExtendedQuantType.QUInt16:
            return TensorProto.UINT16
        if self == ExtendedQuantType.QInt16:
            return TensorProto.INT16
        if self == ExtendedQuantType.QInt32:
            return TensorProto.INT32
        if self == ExtendedQuantType.QUInt32:
            return TensorProto.UINT32
        if self == ExtendedQuantType.QFloat16:
            return TensorProto.FLOAT16
        if self == ExtendedQuantType.QBFloat16:
            return TensorProto.BFLOAT16
        if self == ExtendedQuantType.QBFP or self == ExtendedQuantType.QMX:
            return TensorProto.UNDEFINED
        raise ValueError(f"Unexpected value qtype={self!r}.")


# This is a deprecated class
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
        except KeyError as e:  # pragma: no cover
            raise ValueError() from e


class ExtendedQuantFormat(Enum):
    QOperator = 0
    QDQ = 1

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_string(f: str) -> Any:
        try:
            return ExtendedQuantFormat[f]
        except KeyError as e:  # pragma: no cover
            raise ValueError() from e


# This is a deprecated class
class VitisQuantFormat(Enum):
    QDQ = 2
    FixNeuron = 3
    BFPFixNeuron = 4
    MXFixNeuron = 5

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_string(f: str) -> Any:
        try:
            return VitisQuantFormat[f]
        except KeyError as e:  # pragma: no cover
            raise ValueError() from e


DType = (
    np.dtype[np.int8]
    | np.dtype[np.uint8]
    | np.dtype[np.int16]
    | np.dtype[np.uint16]
    | np.dtype[np.int32]
    | np.dtype[np.uint32]
    | np.dtype[np.float16]
    | None
    | Any
)
ONNX_TYPE_TO_NP_TYPE: dict[int, DType | None] = {
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


def create_range_dict(dtype_ranges: dict[str, tuple[int, int]]) -> Any:
    result = {}
    for dtype, range_pair in dtype_ranges.items():
        tensor_proto_dtype = getattr(onnx_proto.TensorProto, dtype)

        if dtype.lower() not in ["int4", "uint4"]:
            np_dtype = getattr(np, dtype.lower())
        else:
            if is_version_below(onnx, "1.19.0"):
                np_dtype = getattr(custom_element_types, dtype.lower())
            else:
                np_dtype = getattr(ml_dtypes, dtype.lower())

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
    "INT32": (-(2**31), 2**31 - 1),
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
    "INT32": (-(2**30), 2**30),
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
    new_args: list[np.ndarray[Any, Any]] = []
    for i, a in enumerate(args):
        if np.issubdtype(type(a), np.number):
            new_args.append(np.array(a))
        elif isinstance(a, np.ndarray):
            new_args.append(a)
        else:
            raise TypeError(f"arg {i} is not an array: {a}")
    return tuple(new_args) if len(new_args) > 1 else new_args[0]


@log_errors
def get_tensor_type_from_qType(quant_type: QuantType | ExtendedQuantType) -> int:
    if quant_type == QuantType.QUInt8 or quant_type == ExtendedQuantType.QUInt8:
        return TensorProto.UINT8
    if quant_type == QuantType.QInt8 or quant_type == ExtendedQuantType.QInt8:
        return TensorProto.INT8
    if quant_type == QuantType.QUInt16 or quant_type == ExtendedQuantType.QUInt16:
        return TensorProto.UINT16
    if quant_type == QuantType.QInt16 or quant_type == ExtendedQuantType.QInt16:
        return TensorProto.INT16
    if quant_type == ExtendedQuantType.QUInt32:
        return TensorProto.UINT32
    if quant_type == ExtendedQuantType.QInt32:
        return TensorProto.INT32
    if quant_type == ExtendedQuantType.QFloat16:
        return TensorProto.FLOAT16
    if quant_type == ExtendedQuantType.QBFloat16:
        return TensorProto.BFLOAT16
    if quant_type == ExtendedQuantType.QBFP or quant_type == ExtendedQuantType.QMX:
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
            f"Unexpected data type {qType} requested. Only INT4, UINT4, INT8, UINT8, INT16, and UINT16 are supported."
        )

    return qrange


def get_qrange_for_qType(qType: int, reduce_range: bool = False, symmetric: bool = False) -> Any:
    """
    Helper function to get the quantization range for a type.
        parameter qType: quantization type.
        return: quantization range.
    """
    qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range, symmetric=symmetric)
    return qmax - qmin


def quantize_nparray(
    qType: Any,
    arr: np.ndarray[Any, Any],
    scale: np.ndarray[Any, Any],
    zero_point: float,
    low: float | None = None,
    high: float | None = None,
) -> Any:
    if qType in ONNX_BFP_QTYPES_LIST:
        return arr

    assert qType in ONNX_TYPE_TO_NP_TYPE, (
        f"Unexpected data type {qType} requested. Only INT4, UINT4, INT8, UINT8, INT16, UINT16, FLOAT16, and BFLOAT16 are supported."
    )

    if qType in ONNX_FP_QTYPES_LIST:
        arr_fp32 = arr.astype(np.float32) / scale + zero_point
        onnx_model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Cast", ["X"], ["Y"], to=qType)],
                "qu",
                [helper.make_tensor_value_info("X", onnx_proto.TensorProto.FLOAT, None)],
                [helper.make_tensor_value_info("Y", qType, None)],
            )
        )
        ref = ReferenceEvaluator(onnx_model)
        return ref.run(None, {"X": arr_fp32})[0]  # type: ignore
    else:
        dtype = ONNX_TYPE_TO_NP_TYPE[qType]
        (qmin, qmax) = get_qmin_qmax_for_qType(qType, reduce_range=False, symmetric=True)

        cliplow = max(qmin, low) if low is not None else qmin
        cliphigh = min(qmax, high) if high is not None else qmax
        arr_fp32 = np.asarray((arr.astype(np.float32) / scale).round() + zero_point)
        np.clip(arr_fp32, cliplow, cliphigh, out=arr_fp32)
        return _check_type(arr_fp32.astype(dtype))


def load_model_with_shape_infer(model_path: Path) -> ModelProto:
    try:
        model = ort_load_model_with_shape_infer(model_path)
    except Exception:
        logger.warning("Shape infer failed, falling back to onnx.load.")
        model = onnx.load(model_path)
    return model


def save_and_reload_model_with_shape_infer(model: ModelProto) -> ModelProto:
    model_copy = copy.deepcopy(model)
    quant_tmp_dir = create_tmp_dir(prefix="quark_onnx.utils.")
    model_path = Path(quant_tmp_dir).joinpath("model.onnx")
    onnx.save_model(model_copy, model_path.as_posix(), save_as_external_data=True)
    return load_model_with_shape_infer(model_path)


def infer_shape(model: ModelProto) -> ModelProto:
    """
    :param model: the source model
    :return: the target model contains inferred shape
    """
    if model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
        inferred_model = save_and_reload_model_with_shape_infer(model)
    else:
        inferred_model = shape_inference.infer_shapes(model)
    return inferred_model  # type: ignore


def get_datatype_shape(tensor: TensorProto) -> tuple[str, list[Any]]:
    """
    :param tensor: the input tensor
    :return: datatype and shape of the tensor
    """
    elem_type_num = tensor.type.tensor_type.elem_type
    data_type = TensorProto.DataType.Name(elem_type_num).lower()
    data_type = data_type if data_type != "float" else "float32"
    dims = tensor.type.tensor_type.shape.dim
    n = len(dims)
    shape = [dims[i].dim_value if dims[i].dim_value else -1 for i in range(n)]
    return (data_type, shape)


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
    has_axes_attr = any(attr.name == "axes" for attr in node.attribute)
    has_axes_2_3_attr = any(
        attr.name == "axes" and len(attr.ints) == 2 and attr.ints == [2, 3] for attr in node.attribute
    )
    has_keepdims_attr = any(attr.name == "keepdims" for attr in node.attribute)
    has_keepdims_1_attr = any(attr.name == "keepdims" and attr.i == 1 for attr in node.attribute)

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
    has_beta_attr = any(attr.name == "beta" for attr in node.attribute)
    has_beta_0_5_attr = any(attr.name == "beta" and is_approximately_equal(attr.f, 0.5) for attr in node.attribute)
    has_alpha_attr = any(attr.name == "alpha" and is_approximately_equal(attr.f, 1.0 / 6.0) for attr in node.attribute)
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


def is_clip_with_min_max(
    model: onnx.ModelProto, node: onnx.NodeProto, min_value: float = 0.0, max_value: float = 6.0
) -> bool:
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
        # Make sure whether the Clip node can be considered as ReLU or ReLU6
        if is_clip_with_min_max(model, node, 0, 6) or is_clip_with_min_max(model, node, 0, 1):
            return True
    elif node.op_type in remove_qdq_op_type:
        return True
    return False


def get_all_tensor_names(model: onnx.ModelProto) -> set[str]:
    """Return every tensor name referenced in ``model``'s main graph.

    The returned set is the union of:

    * initializer names
    * all node inputs and outputs
    * graph inputs and outputs

    Useful for validating user-supplied tensor references (e.g. keys in
    ``TensorQuantOverrides``) against the actual graph. Note: this does not
    recurse into subgraphs (``If``/``Loop``/``Scan`` bodies); callers that need
    that should walk the subgraphs themselves.

    :param model: ONNX ModelProto to inspect.
    :return: set of tensor names present in the model's main graph.
    """
    names: set[str] = {init.name for init in model.graph.initializer}
    for node in model.graph.node:
        names.update(node.input)
        names.update(node.output)
    names.update(vi.name for vi in model.graph.input)
    names.update(vi.name for vi in model.graph.output)
    return names


def get_tensor_to_consumer(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
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


def get_annotate_tensors(model: onnx.ModelProto) -> list[str]:
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
        if (is_node_needs_annotated(model, node) and node.input[0] in matching_output_tensor) or (
            node.op_type in avg_pool_op_type and node.input[0] in pad_output_tensor
        ):
            annotate_tensors.append(node.input[0])
    return annotate_tensors


def get_qdq_to_remove(
    model: onnx.ModelProto, annotate_tensors: list[str], remove_fused_qdq: bool = False
) -> tuple[list[onnx.NodeProto], list[onnx.NodeProto], dict[str, str]]:
    """
    Return the names of nodes to be removed and a dictionary for converting input tensors
    :param model: model object
    :param annotate_tensors: the annotate tensors
    :param remove_fused_qdq: whether to remove fused QDQ nodes, such as BFPQuantizeDequantize and MXQuantizeDequantize
    :return: dequantize & quantize nodes to remove and node mapping dict
    """
    q_nodes_to_remove = []
    dq_nodes_to_remove = []
    q_nodes_output_to_remove = []
    input_node_mapping = {}
    for node in model.graph.node:
        if node.op_type in QUANT_OP_TYPES or (remove_fused_qdq and node.op_type in FN_OP_TYPES):
            if node.input[0] in annotate_tensors:
                input_node_mapping[node.input[0]] = node.output[0]
                q_nodes_to_remove.append(node)
                if node.op_type in QUANT_OP_TYPES:
                    q_nodes_output_to_remove.append(node.output[0])
    for node in model.graph.node:
        if node.op_type in DEQUANT_OP_TYPES and node.input[0] in q_nodes_output_to_remove:
            for k, v in input_node_mapping.items():
                if v == node.input[0]:
                    input_node_mapping[k] = node.output[0]
            dq_nodes_to_remove.append(node)
    return dq_nodes_to_remove, q_nodes_to_remove, input_node_mapping


def remove_nodes(model: onnx.ModelProto, nodes_list: list[Any]) -> onnx.ModelProto:
    """
    Delete nodes according to the nodes in the list
    :param model: model object
    :param nodes_list: nodes list to remove
    :return: the model that has removed some nodes
    """
    for node in nodes_list:
        model.graph.node.remove(node)
    return model


def remove_initializers(model: ModelProto, init_list: list[str]) -> ModelProto:
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


def modified_annotate_input(model: ModelProto, input_node_mapping: dict[str, str]) -> ModelProto:
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
def compute_scale_zp(
    rmin: np.ndarray[Any, Any],
    rmax: np.ndarray[Any, Any],
    qmin: np.ndarray[Any, Any],
    qmax: np.ndarray[Any, Any],
    element_type: int,
    method: PowerOfTwoMethod | Int16Method | ExtendedCalibrationMethod | None,
    symmetric: bool = False,
    use_pof2s: bool = True,
) -> Any:
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
    if rmin == -np.inf or rmin < -np.finfo(np.float32).max / 2:
        logger.warning("rmin is set to -inf, replacing with a very small value.")
        rmin = np.full_like(rmin, -np.finfo(np.float32).max / 2)
    if rmax == np.inf or rmax > np.finfo(np.float32).max / 2:
        logger.warning("rmax is set to inf, replacing with a very large value.")
        rmax = np.full_like(rmax, np.finfo(np.float32).max / 2)

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

    if isinstance(method, CalibrationMethod):
        if symmetric and element_type == onnx_proto.TensorProto.UINT8 and zero_point == 127:
            zero_point = np.array(128, dtype=qmin.dtype)
        return [zero_point, scale]
    # Power-of-2 scale calculation
    elif isinstance(method, PowerOfTwoMethod):
        if use_pof2s is False:
            return [zero_point, scale]
        pos = scale2pos(scale.item())
        pof2_scale = np.array(pos2scale(pos), dtype=scale.dtype)
        new_rmin = np.minimum(
            (qmin.astype(np.float32) - zero_point.astype(np.float32)) * pof2_scale, np.array(0, dtype=rmin.dtype)
        )
        new_zero_point = np.array(np.round(qmin - new_rmin / pof2_scale), dtype=qmin.dtype)
        # To meet hardware's requirements
        if symmetric and element_type == onnx_proto.TensorProto.UINT8 and new_zero_point == 127:
            new_zero_point = np.array(128, dtype=qmin.dtype)
        return [new_zero_point, pof2_scale]
    elif isinstance(method, Int16Method):
        M, N, diff = find_int16_scale(scale.item())
        int16_scale: np.ndarray[Any, Any] | float = np.array(M / 2**N, dtype=scale.dtype)
        logger.debug(f"Find the {M} / 2 ** {N} that is closest to scale {scale}with the difference being {diff}")
        if int16_scale < np.finfo(np.float32).tiny:
            int16_scale = 1 / 2**14

        new_rmin = np.minimum(
            (qmin.astype(np.float32) - zero_point.astype(np.float32)) * int16_scale, np.array(0, dtype=rmin.dtype)
        )
        new_zero_point = np.array(np.round(qmin - new_rmin / int16_scale), dtype=qmin.dtype)
        if symmetric and element_type == onnx_proto.TensorProto.UINT8 and new_zero_point == 127:
            new_zero_point = np.array(128, dtype=qmin.dtype)

        return [new_zero_point, int16_scale]
    elif isinstance(method, ExtendedCalibrationMethod):
        return [zero_point, scale]
    else:
        return [zero_point, scale]


@log_errors
def compute_scale_zp_fp(
    rmin: np.ndarray[Any, Any],
    rmax: np.ndarray[Any, Any],
    qmin: np.ndarray[Any, Any],
    qmax: np.ndarray[Any, Any],
    element_type: int,
    method: CalibrationMethod,
    symmetric: bool = True,
    use_scaling: bool = False,
) -> list[Any]:
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
    if rmin == -np.inf or rmin < -np.finfo(np.float32).max / 2:
        logger.warning("rmin is set to -inf, replacing with a very small value.")
        rmin = np.full_like(rmin, -np.finfo(np.float32).max / 2)
    if rmax == np.inf or rmax > np.finfo(np.float32).max / 2:
        logger.warning("rmax is set to inf, replacing with a very large value.")
        rmax = np.full_like(rmax, np.finfo(np.float32).max / 2)

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

    if method not in CalibrationMethod and scale != np.array(1.0, dtype=np.float32):
        logger.warning("Suggest using methods from CalibrationMethod as it only supports float scale.")

    return [zero_point, scale]


def dequantize_data(data: np.ndarray[Any, Any], scale: np.ndarray[Any, Any], zero_point: np.ndarray[Any, Any]) -> Any:
    """
    :param data: the input data
    :param scale: the scale for quantization
    :param zero_point: the zero point for quantization
    :return: the dequantized data
    """
    data = data.astype(np.float32)
    deq_arr = (data - zero_point.astype(np.float32)) * scale
    return deq_arr.astype(np.float32)


def quantize_data(
    data: np.ndarray[Any, Any],
    qType: int,
    symmetric: bool,
    reduce_range: bool = False,
    min_real_range: float | None = None,
    rmin_override: np.ndarray[Any, Any] | None = None,
    rmax_override: np.ndarray[Any, Any] | None = None,
    method: PowerOfTwoMethod | Int16Method | ExtendedCalibrationMethod = PowerOfTwoMethod.NonOverflow,
    weight_method: CalibrationMethod | ExtendedCalibrationMethod | None = None,
    minmse_mode: str | None = "Percentile",
    pos_range: int = 5,
    use_pof2s: bool = True,
    use_scaling: bool = False,
    is_partial_cal: bool = False,
) -> Any:
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

    if weight_method == ExtendedCalibrationMethod.MinMSE:
        if minmse_mode not in ["Percentile", "HistCenter", "All"]:
            if minmse_mode is not None:
                logger.warning(
                    f"Unsupported weight method '{minmse_mode}'. Supported methods are 'Percentile', 'HistCenter', and 'All'. Defaulting to 'Percentile'."
                )
            minmse_mode = "Percentile"
        rmin, rmax, zero_point, scale, quantized_data = compute_minmse(
            data, qType, weight_method, symmetric, minmse_mode, reduce_range
        )
        return _check_type(rmin, rmax, zero_point, scale, quantized_data, zero_point_index=2)
    elif weight_method == CalibrationMethod.MinMax:
        pass

    if rmin_override is not None and rmin_override.size > 0:
        rmin_value = float(rmin_override[0])
    else:
        rmin_value = data.min() if len(data) else 0.0
    if rmax_override is not None and rmax_override.size > 0:
        rmax_value = float(rmax_override[0])
    else:
        rmax_value = data.max() if len(data) else 0.0
    rmin = np.array(rmin_value, dtype=data.dtype)
    rmax = np.array(rmax_value, dtype=data.dtype)
    zero_point = 0
    scale = np.array(1.0, dtype=data.dtype)

    if qType in ONNX_FP_QTYPES_LIST + ONNX_BFP_QTYPES_LIST:
        # For floating-point quant types, use_scaling means reducing the numeric range
        qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range=use_scaling)
        zero_point, scale = compute_scale_zp_fp(
            rmin, rmax, qmin, qmax, qType, method, symmetric=symmetric, use_scaling=use_scaling
        )
        quantized_data = quantize_nparray(qType, np.asarray(data), scale, zero_point)
        return _check_type(rmin, rmax, zero_point, scale, quantized_data, zero_point_index=2)

    qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range, symmetric=symmetric)
    zero_point, scale = compute_scale_zp(
        rmin, rmax, qmin, qmax, qType, method, symmetric=symmetric, use_pof2s=use_pof2s
    )

    quantized_data = quantize_nparray(qType, np.asarray(data), scale, zero_point)

    if method == PowerOfTwoMethod.NonOverflow:
        return _check_type(rmin, rmax, zero_point, scale, quantized_data, zero_point_index=2)
    elif method == PowerOfTwoMethod.MinMSE and is_partial_cal:
        zp_mse = zero_point
        minmse_diffs = []
        minmse_scales = []
        minmse_zps = []
        for i in range(pos_range):
            new_scale = np.array(pos2scale(scale2pos(scale) + i - 1), dtype=data.dtype)
            rmin = (qmin.astype(np.float32) - zero_point.astype(np.float32)) * new_scale

            new_quantized_data = quantize_nparray(qType, np.asarray(data), new_scale, zp_mse)
            diff = np.sum((dequantize_data(new_quantized_data, new_scale, zp_mse) - np.asarray(data)) ** 2)
            minmse_diffs.append(diff)
            minmse_scales.append(new_scale)
            minmse_zps.append(zp_mse)

        return (minmse_diffs, minmse_scales, minmse_zps, qmin, qmax)
    elif method == PowerOfTwoMethod.MinMSE and not is_partial_cal:
        scale_mse = scale
        zp_mse = zero_point
        quantized_data_mse = quantized_data
        diff_min = float("inf")
        for i in range(pos_range):
            new_scale = np.array(pos2scale(scale2pos(scale) + i - 1), dtype=data.dtype)
            rmin = (qmin.astype(np.float32) - zero_point.astype(np.float32)) * new_scale

            new_quantized_data = quantize_nparray(qType, np.asarray(data), new_scale, zp_mse)
            diff = np.sum((dequantize_data(new_quantized_data, new_scale, zp_mse) - np.asarray(data)) ** 2)
            if diff < diff_min:
                diff_min = diff
                scale_mse = new_scale
                quantized_data_mse = new_quantized_data

        rmin_mse = (qmin.astype(np.float32) - zp_mse.astype(np.float32)) * scale_mse
        rmax_mse = (qmax.astype(np.float32) - zp_mse.astype(np.float32)) * scale_mse
        return _check_type(
            rmin_mse.astype(data.dtype),
            rmax_mse.astype(data.dtype),
            zp_mse,
            scale_mse,
            quantized_data_mse,
            zero_point_index=2,
        )
    elif method == Int16Method.MinMax:
        return _check_type(rmin, rmax, zero_point, scale, quantized_data, zero_point_index=2)
    else:
        return _check_type(rmin, rmax, zero_point, scale, quantized_data, zero_point_index=2)


def get_exclude_nodes(
    input_model: str | Path | onnx.ModelProto,
    input_nodes: list[str] | None,
    output_nodes: list[str] | None,
) -> list[str]:
    """
    Return the nodes to be excluded based on the given input and output nodes.
    :param input_model: the model path or ModelProto
    :param input_nodes: the nodes to start quantizing
    :param zero_point: the nodes to terminate quantizing
    :return: the nodes excluded from quantization
    """

    def update_exclude_input_nodes(
        exclude_nodes: list[str], name_list: list[str], name: str, input_nodes: list[str]
    ) -> list[str]:
        index = name_list.index(name)
        exclude_nodes_i = name_list[:index]
        exclude_nodes = list(set(exclude_nodes) | set(exclude_nodes_i))
        exclude_nodes = list(set(exclude_nodes) - set(input_nodes))
        return exclude_nodes

    def update_exclude_output_nodes(
        exclude_nodes: list[str], name_list: list[str], name: str, output_nodes: list[str]
    ) -> list[str]:
        index = name_list.index(name) + 1
        exclude_nodes_o = name_list[index:]
        exclude_nodes = list(set(exclude_nodes) | set(exclude_nodes_o))
        exclude_nodes = list(set(exclude_nodes) - set(output_nodes))
        return exclude_nodes

    model = input_model if isinstance(input_model, onnx.ModelProto) else onnx.load(input_model)
    onnx_model = ONNXModel(model)
    onnx_model.topological_sort()

    model_input_to_node: dict[str, list[str]] = {}
    model_output_to_node: dict[str, list[str]] = {}
    name_list: list[str] = []
    exclude_nodes: list[str] = []

    for i in onnx_model.model.graph.input:
        model_input_to_node[i.name] = []
    for o in onnx_model.model.graph.output:
        model_output_to_node[o.name] = []
    for n in onnx_model.model.graph.node:
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
                    f"Fail to find the {name} in the model, the input_nodes {input_nodes} did not take effect, please check input_nodes parameter"
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
                    f"Fail to find the {name} in the model, the input_nodes {input_nodes} did not take effect, please check input_nodes parameter"
                )
    return exclude_nodes


def get_matmul_nodes_without_weights(input_model: str | Path | onnx.ModelProto) -> list[str]:
    model = input_model if isinstance(input_model, onnx.ModelProto) else onnx.load(input_model)
    onnx_model = ONNXModel(model)
    onnx_model.topological_sort()

    initializer_names = {init.name for init in onnx_model.model.graph.initializer}

    matmul_without_weights_nodes_name = []

    for node in onnx_model.model.graph.node:
        if node.op_type == "MatMul":
            _, input2 = node.input
            if input2 not in initializer_names:
                matmul_without_weights_nodes_name.append(node.name)

    return matmul_without_weights_nodes_name


def check_model_quantizable(
    model: ModelProto, op_types_to_quantize: list[str] | None, nodes_to_exclude: list[str]
) -> bool:
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
                    if (
                        vi.type.HasField("tensor_type")
                        and (vi.type.tensor_type.elem_type in tensor_type_to_calibrate)
                        and (tensor_name not in initializer)
                    ):
                        tensors_to_calibrate.add(tensor_name)

    if len(tensors_to_calibrate) == 0:
        return False

    return True


def dpu_leaky_relu_alpha(x: float) -> float:
    """
    This function implements a DPU-specific Leaky ReLU activation with alpha value correction.
    """
    rounded_value = round(x * 256)
    return rounded_value / 256.0


def get_model_node_name_dict(model: ModelProto) -> dict[str, NodeProto]:
    model_node_name_dict: dict[str, NodeProto] = {}
    for node in model.node:
        if node.name and not model_node_name_dict.get(node.name):
            model_node_name_dict[node.name] = node
        else:
            if not node.name and node.output[0]:
                model_node_name_dict[node.output[0]] = node
            else:
                logger.warning(f"the node name:{node.name} is not exist in model_node_name_dict.")
    return model_node_name_dict


def get_model_weight_name_dict(model: ModelProto) -> dict[str, TensorProto]:
    model_weight_name_dict: dict[str, TensorProto] = {}
    for wgt in model.initializer:
        if not model_weight_name_dict.get(wgt.name):
            model_weight_name_dict[wgt.name] = wgt
        else:
            logger.warning(f"the weight name:{wgt.name} is exist in model_weight_name_dict.")
    return model_weight_name_dict


@log_errors
def get_model_node_output_node_name_dict(model: ModelProto) -> dict[str, str]:
    model_node_output_node_name_dict: dict[str, str] = {}
    # handle all node
    for node in model.node:
        # the node.output is support multi
        for out in node.output:
            if out == "":
                continue
            elif not model_node_output_node_name_dict.get(out):
                model_node_output_node_name_dict[out] = node.output[0]
            else:
                raise ValueError(
                    f"the node output var name:{node.output} is exist in model_node_output_node_name_dict."
                )
    return model_node_output_node_name_dict


def get_node_input_var(node: NodeProto) -> Any:
    if len(node.input) > 0:
        return node.input


def get_node_input_node_name(
    node: NodeProto, model_output_name_dict: dict[str, str], model_weight_name_dict: dict[str, TensorProto]
) -> tuple[list[str], list[TensorProto]]:
    inputs = get_node_input_var(node)
    node_input_node_name = []
    node_weights_bias_node_name = []
    for var in inputs:
        if var in model_output_name_dict:
            node_input_node_name.append(model_output_name_dict[var])
        elif var in model_weight_name_dict:
            node_weights_bias_node_name.append(model_weight_name_dict[var])
        else:
            logger.debug(f"the node: {var} is input or output")
    return node_input_node_name, node_weights_bias_node_name


@log_errors
def get_node_from_node_name(name: str, model_output_node_dict: dict[str, NodeProto]) -> Any:
    if model_output_node_dict.get(name):
        return model_output_node_dict[name]
    else:
        raise ValueError(f"cann't get node:{name} from name.")


def get_weight_from_weight_name(name: str, model_weight_node_dict: dict[str, TensorProto]) -> Any:
    if model_weight_node_dict.get(name):
        return model_weight_node_dict[name]
    else:
        logger.warning(f"cann't get weight:{name} from name.")


def get_weights_node_of_node(
    node: NodeProto, model_output_name_dict: dict[str, str], model_weights_node_dict: dict[str, TensorProto]
) -> list[TensorProto]:
    _, all_weights_name = get_node_input_node_name(node, model_output_name_dict, model_weights_node_dict)
    weights_nodes = []
    for weight in all_weights_name:
        if weight:
            weights_nodes.append(weight)
    return weights_nodes


def get_output_nodes_of_node(node: NodeProto, model: GraphProto) -> list[NodeProto]:
    output_nodes_list = []
    for output in node.output:
        for one_node in model.node:
            if output in one_node.input and one_node.name not in output_nodes_list:
                output_nodes_list.append(one_node)
            elif output in one_node.input:
                logger.info(f"the output_node:{one_node.name} already in list")
    return output_nodes_list


def get_clip_min_max(model: ModelProto, clip_node: NodeProto) -> tuple[float | None, float | None, int | None]:
    """
    Get clip min and max value from Clip node.

    :param model: onnx model instance
    :param clip_node: target Clip node

    :return: the min, max value and para type The meaning of para type is:

        * ``None``: unknown.
        * ``0``: attribute.
        * ``1``: initializer.
        * ``2``: other nodes.
    """

    def _get_from_initializer(model: ModelProto, name: str) -> Any:
        for init in model.graph.initializer:
            if init.name == name:
                return onnx.numpy_helper.to_array(init).tolist()
        return None

    def _get_from_attribute(node: NodeProto) -> Any:
        for attr in node.attribute:
            if attr.name == "value":
                if attr.t.data_type == 1:
                    return list(attr.t.float_data)[0]
                else:
                    return list(attr.t.int32_data)[0]
        return None

    def _get_from_other_node(model: ModelProto, name: str) -> Any:
        for node in model.graph.node:
            if node.op_type == "Identity" and name in node.output:
                return _get_from_initializer(model, node.input[0])
            if node.op_type == "Constant" and name in node.output:
                return _get_from_attribute(node)
        return None

    min_value = None
    max_value = None
    if clip_node.op_type != "Clip":
        return min_value, max_value, None

    # Get from attributes
    for attr in clip_node.attribute:
        if attr.name == "min":
            min_value = attr.f
        if attr.name == "max":
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
    if node.op_type == "Relu":
        return True
    elif node.op_type == "Clip":
        min_value, *_ = get_clip_min_max(model, node)
        if min_value == 0:
            return True
    return False


def find_int16_scale(x: float) -> tuple[float, float, float]:
    """
    Given a float value, find the closest value corresponding to  M and 2**N,
    where the range of M and 2**N is within the representation range of int16 and uint16.
    """
    if x == 0:
        return 0, 0, 0

    closest_m = 0
    closest_n = 0
    closest_diff = float("inf")

    # Loop through possible values of n and m
    for n in range(0, 17):  # Adjust the range as needed
        m_fs = x * 2**n
        if m_fs < -(2**15) or m_fs > 2**15 - 1:
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
            "Model with ir_version below 4 requires to include initializer in graph input, change ir_version to 7"
        )
        model.ir_version = 7

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    return model


def fp32_nodes(model_input: str | Path | ModelProto) -> dict[str, int]:
    try:
        fp32_nodes_dict = {}
        fp32_model = model_input if isinstance(model_input, onnx.ModelProto) else onnx.load(model_input)
        onnx_model = ONNXModel(fp32_model)

        for node in onnx_model.model.graph.node:
            if node.op_type not in fp32_nodes_dict:
                fp32_nodes_dict[node.op_type] = 0
            fp32_nodes_dict[node.op_type] += 1

        return fp32_nodes_dict

    except Exception:
        return {}


# using data for sub_model to inference
def inference_sub_model_with_data(
    input_model: onnx.ModelProto, start_node_map: dict[str, list[float]], end_node_list: list[str]
) -> list[float]:
    # TODO: Resolve circular references
    from quark.onnx.utils.model_utils import create_infer_session_for_onnx_model

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

    if input_model.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF:
        extractor = onnx.utils.Extractor(input_model)
        sub_model = extractor.extract_model(start_node_tensor, end_node_tensor)
        session = create_infer_session_for_onnx_model(sub_model)
    else:
        sub_model_path = create_tmp_dir(prefix="quark_onnx.submodel.")
        opt_model_output = Path(sub_model_path.name).joinpath("all.onnx").as_posix()
        sub_model_output = Path(sub_model_path.name).joinpath("sub_model.onnx").as_posix()
        onnx.save(input_model, opt_model_output, save_as_external_data=True)
        onnx.utils.extract_model(opt_model_output, sub_model_output, start_node_tensor, end_node_tensor, False)
        session = create_infer_session_for_onnx_model(sub_model_output)
        sub_model_path.cleanup()
    start_tensor_one_batch = {}
    end_tensor_list = []
    for key in start_tensor_map:
        values = start_tensor_map[key]
        for bs in range(len(values)):
            start_tensor_one_batch[key] = values[bs]
            end_tensor_one_tensor = session.run(end_node_tensor, start_tensor_one_batch)
            end_tensor_list.append(end_tensor_one_tensor[0])
    return end_tensor_list


def extract_sub_model(
    input_model: str | Path | ModelProto, start_tensors: list[str], end_tensors: list[str]
) -> onnx.ModelProto:
    if isinstance(input_model, ModelProto):
        model = input_model
        if input_model.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF:
            model = onnx.shape_inference.infer_shapes(input_model)
        extractor = onnx.utils.Extractor(model)
        sub_model = extractor.extract_model(start_tensors, end_tensors)
    else:
        sub_model_path = create_tmp_dir(prefix="quark_onnx.submodel.")
        sub_model_output = Path(sub_model_path.name).joinpath("sub_model.onnx").as_posix()
        onnx.utils.extract_model(input_model, sub_model_output, start_tensors, end_tensors, check_model=False)
        sub_model = onnx.load(sub_model_output)
        sub_model_path.cleanup()
    return sub_model


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
    for i in range(len(onnx_model.model.graph.input)):
        onnx_model.model.graph.input[i].type.tensor_type.shape.dim[0].dim_value = bs
    for i in range(len(onnx_model.model.graph.output)):
        onnx_model.model.graph.output[i].type.tensor_type.shape.dim[0].dim_value = bs
    for i in range(len(onnx_model.model.graph.value_info)):
        if len(onnx_model.model.graph.value_info[i].type.tensor_type.shape.dim) > 1:
            onnx_model.model.graph.value_info[i].type.tensor_type.shape.dim[0].dim_value = bs
    for node in onnx_model.model.graph.node:
        if node.op_type == "Reshape":
            reshape_input_name = node.input[1]
            for tensor in onnx_model.model.graph.initializer:
                if tensor.name == reshape_input_name:
                    tensor_array = onnx.numpy_helper.to_array(tensor)
                    tensor_array_shape = list(tensor_array)
                    tensor_array_shape[0] = bs
                    new_tensor_array = np.array(tensor_array_shape, dtype=np.int64)
                    new_tensor = onnx.numpy_helper.from_array(new_tensor_array, tensor.name)
                    onnx_model.model.graph.initializer.extend([new_tensor])
                    onnx_model.remove_initializer(tensor)
    return onnx_model.model


def infer_custom_op_shape(model: onnx.ModelProto) -> onnx.ModelProto:
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    int_max = 2**31 - 1
    auto_merge = True
    guess_output_rank = True
    verbose = 0
    shape_infer = SymbolicShapeInference(int_max, auto_merge, guess_output_rank, verbose)
    infer_onnx_file = "sym_shape_infer_temp.onnx"
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
    while need_infer:
        for node in model.graph.node:
            if node.op_type in QUANT_OP_TYPES:
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
            elif node.op_type in DEQUANT_OP_TYPES:
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
            elif node.op_type in FN_OP_TYPES:
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
            elif node.op_type == COP_IN_OP_NAME:
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


def skip_node_with_inf_tensor(model: onnx.ModelProto) -> list[str]:
    tensor_to_node_dict = {}
    init_name_to_init_dict = {}
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
            onnx.numpy_helper.to_array(init), -np.inf
        ):
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


def get_shape_from_tensor(tensor: onnx.TensorProto) -> list[int]:
    shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in tensor.type.tensor_type.shape.dim]
    return shape


def convert_fp16_scale_to_fp32(
    input_model: str | Path | ModelProto,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
) -> ModelProto:
    """Convert FP16 tensors to FP32 for selected nodes, inserting Cast nodes at boundaries.

    :param input_model: ONNX model path or ``ModelProto``.
    :param nodes_to_quantize: Node names to include for conversion (whitelist).
    :param nodes_to_exclude: Node names to exclude from conversion (blacklist).
    :return: Model with selected FP16 tensors converted to FP32.
    """
    model = input_model if isinstance(input_model, onnx.ModelProto) else onnx.load(input_model)
    onnx_model = ONNXModel(model)

    input_name_to_nodes = onnx_model.input_name_to_nodes()
    output_name_to_node = onnx_model.output_name_to_node()
    node_by_name = {n.name: n for n in onnx_model.nodes() if n.name}

    # ---------------------------------------------
    # Convert nodes and initializers and value info
    # ---------------------------------------------
    all_node_names = set(node_by_name.keys())

    nodes_to_quantize_clean = [n for n in nodes_to_quantize if n in all_node_names]
    nodes_to_exclude_clean = [n for n in nodes_to_exclude if n in all_node_names]

    # Phase 1 – Identify and convert nodes and initializers
    nodes_for_conversion: set[str] = set()
    inits_for_conversion: set[str] = set()
    for node_name in all_node_names:
        node = node_by_name[node_name]
        if nodes_to_exclude_clean and node_name in nodes_to_exclude_clean:
            continue
        nodes_for_conversion.add(node_name)

        for inp in node.input:
            assert inp, f"Node {node.name} has no input!"
            init = onnx_model.get_initializer(inp)
            if init and init.data_type == onnx.TensorProto.FLOAT16:
                inits_for_conversion.add(inp)

    # Phase 2 – Convert attributes and value information for converted nodes
    for node_name in nodes_for_conversion:
        node = node_by_name[node_name]

        # Convert attributes of the converted nodes
        for attr in node.attribute:
            # For Cast node with to attribute
            if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT
                logger.debug(f"Converting attribute 'to' of node {node.name} from FP16 to FP32.")
            # For Constant node with value attribute
            if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                new_data = onnx.numpy_helper.to_array(attr.t).astype("float32")
                new_tensor = onnx.numpy_helper.from_array(new_data, attr.t.name)
                attr.t.CopyFrom(new_tensor)
                logger.debug(f"Converting attribute 'value' of node {node.name} from FP16 to FP32.")

        # Convert value information at the input tensors of the converted nodes
        for inp in node.input:
            if not inp or onnx_model.is_graph_input(inp):
                continue
            tt = onnx_model.get_tensor_type(inp)
            if tt and tt.elem_type == onnx.TensorProto.FLOAT16:
                tt.elem_type = onnx.TensorProto.FLOAT
                logger.debug(f"Converting value information of node {node.name} from FP16 to FP32.")

    # Phase 3 – Convert initializers
    for init_name in inits_for_conversion:
        old_init = onnx_model.get_initializer(init_name)
        fp32_data = onnx.numpy_helper.to_array(old_init).astype(np.float32)
        new_init = onnx.numpy_helper.from_array(fp32_data, old_init.name)
        onnx_model.remove_initializer(old_init)
        onnx_model.add_initializer(new_init)
        logger.debug(f"Converting initializer {init_name} from FP16 to FP32.")

    # ---------------------------------------------
    # Insert Cast nodes
    # ---------------------------------------------
    # Insert Cast nodes at graph inputs, which should be kept as FP16
    for inp in onnx_model.model.graph.input:
        if inp.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16:
            continue

        consumers = input_name_to_nodes.get(inp.name, [])
        for index, consumer in enumerate(consumers):
            if nodes_to_exclude_clean and consumer.name in nodes_to_exclude_clean:
                continue

            cast_name = f"{inp.name}_Cast_{index}"
            cast_output_name = f"{cast_name}_output"
            onnx_model.replace_node_input(consumer, inp.name, cast_output_name)
            onnx_model.add_node(
                onnx.helper.make_node(
                    "Cast", inputs=[inp.name], outputs=[cast_output_name], name=cast_name, to=onnx.TensorProto.FLOAT
                )
            )
            logger.info(f"Inserted Cast node {cast_name} at the input of node {consumer.name} to convert FP16 to FP32.")

    if not (nodes_to_quantize_clean or nodes_to_exclude_clean):
        # This is the simplest case, where we just need to insert Cast nodes at graph outputs
        for out in onnx_model.model.graph.output:
            if out.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16:
                continue

            producer = output_name_to_node.get(out.name, None)
            if producer:
                cast_name = f"{out.name}_Cast"
                cast_input_name = f"{cast_name}_input"
                onnx_model.replace_node_output(producer, out.name, cast_input_name)
                onnx_model.add_node(
                    onnx.helper.make_node(
                        "Cast",
                        inputs=[cast_input_name],
                        outputs=[out.name],
                        name=cast_name,
                        to=onnx.TensorProto.FLOAT16,
                    )
                )
                logger.info(
                    f"Inserted Cast node {cast_name} at the output of node {producer.name} to convert FP32 to FP16."
                )
    else:
        # Insert Cast nodes at the output of DQs at the boundaries between FP16 and FP32
        for node_name in nodes_for_conversion:
            node = node_by_name[node_name]
            if not (
                nodes_to_quantize_clean and node.name in nodes_to_quantize_clean or node.op_type in DEQUANT_OP_TYPES
            ):
                continue

            for output in node.output:
                tensor_type = onnx_model.get_tensor_type(output)
                if tensor_type and tensor_type.elem_type != onnx.TensorProto.FLOAT16:
                    continue

                consumers = input_name_to_nodes.get(output, [])
                for index, consumer in enumerate(consumers):
                    if nodes_to_exclude_clean and consumer.name not in nodes_to_exclude_clean:
                        continue

                    cast_name = f"{output}_Cast_{index}"
                    cast_output_name = f"{cast_name}_output"
                    onnx_model.replace_node_input(consumer, output, cast_output_name)
                    onnx_model.add_node(
                        onnx.helper.make_node(
                            "Cast",
                            inputs=[output],
                            outputs=[cast_output_name],
                            name=cast_name,
                            to=onnx.TensorProto.FLOAT16,
                        )
                    )
                    logger.info(
                        f"Inserted Cast node {cast_name} at the input of node {consumer.name} to convert FP32 to FP16."
                    )

    onnx_model.topological_sort()
    return onnx_model.model


def insert_quant_nodes_at_boundaries(
    model: ModelProto,
    tensor_quant_overrides: TensorQuantOverridesHelper,
    tensors_range: TensorsData,
    reduce_range: bool = False,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
    extra_options: dict[str, Any] = {},
) -> ModelProto:
    """Insert additional pair of quant nodes (Q/DQ or BFPQDQ or MXQDQ nodes) at the boundary tensors of two different precisions.
    :param model: ONNX model path or ``ModelProto``.
    :param TensorQuantOverridesHelper tensor_quant_overrides: Tensor quantization overrides.
    :param TensorsData tensors_range: Data range for all quantizing tensors.
    :param bool reduce_range: Whether to reduce the range of the quantized tensor.
    :param CalibrationMethod calibrate_method: Calibration method, the default is CalibrationMethod.MinMax.
    :param Dict[str, Any] extra_options: Options for the transformation.
    :return: Model with additional quant node pairs inserted at the boundaries.
    """
    onnx_model = ONNXModel(model)
    output_name_to_node = onnx_model.output_name_to_node()
    input_name_to_nodes = onnx_model.input_name_to_nodes()

    tensor_names: set[str] = set()
    for graph_input in onnx_model.model.graph.input:
        tensor_names.add(graph_input.name)
    for graph_output in onnx_model.model.graph.output:
        tensor_names.add(graph_output.name)
    for initializer in onnx_model.model.graph.initializer:
        tensor_names.add(initializer.name)
    for node in onnx_model.nodes():
        tensor_names.update([name for name in node.input if name])
        tensor_names.update([name for name in node.output if name])

    node_names = {node.name for node in onnx_model.nodes() if node.name}
    nodes_with_mixed_precision = extra_options.get("NodesWithMixedPrecision", [])

    def _unique_name(base: str, existing: set[str]) -> str:
        if base not in existing:
            existing.add(base)
            return base
        index = 1
        while True:
            candidate = f"{base}_{index}"
            if candidate not in existing:
                existing.add(candidate)
                return candidate
            index += 1

    def _attr_to_key(attr: onnx.AttributeProto) -> tuple[str, Any]:
        if attr.type == onnx.AttributeProto.INT:
            value: Any = attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            value = attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            value = attr.s
        # elif attr.type == onnx.AttributeProto.INTS:
        #     value = tuple(attr.ints)
        # elif attr.type == onnx.AttributeProto.FLOATS:
        #     value = tuple(attr.floats)
        # elif attr.type == onnx.AttributeProto.STRINGS:
        #     value = tuple(attr.strings)
        else:
            value = repr(attr)
        return attr.name, value

    def _qtype_from_quant_node(node: NodeProto) -> Any:
        if len(node.input) >= 3 and node.input[2]:
            zp_init = onnx_model.get_initializer(node.input[2])
            if zp_init is not None:
                return zp_init.data_type
        # if node.output and node.output[0]:
        #     tt = onnx_model.get_tensor_type(node.output[0])
        #     if tt:
        #         return tt.elem_type
        return None

    def _signature_from_quant_node(node: NodeProto) -> tuple[str, str, Any]:
        domain = node.domain if node.domain else "ai.onnx"
        return ("quant", f"{domain}::{node.op_type}", _qtype_from_quant_node(node))

    def _signature_from_fn_node(node: NodeProto) -> tuple[str, str, tuple[tuple[str, Any], ...]]:
        domain = node.domain if node.domain else COP_DOMAIN
        attrs = tuple(sorted(_attr_to_key(attr) for attr in node.attribute))
        return ("fn", f"{domain}::{node.op_type}", attrs)

    def _upstream_stage_from_input(
        input_name: str,
    ) -> (
        tuple[
            str,
            tuple[str, str, Any] | tuple[str, str, tuple[tuple[str, Any], ...]],
            tuple[Any, ...],
            str,
        ]
        | None
    ):
        producer = output_name_to_node.get(input_name)

        if producer and producer.op_type in DEQUANT_OP_TYPES:
            quant = output_name_to_node.get(producer.input[0])
            if quant and quant.op_type in QUANT_OP_TYPES:
                signature = _signature_from_quant_node(quant)
                upstream_source = quant.input[0] if len(quant.input) >= 1 and quant.input[0] else input_name
                return "pair", signature, (quant, producer), upstream_source

        if producer and producer.op_type in FN_OP_TYPES:
            signature = _signature_from_fn_node(producer)
            upstream_source = producer.input[0] if len(producer.input) >= 1 and producer.input[0] else input_name
            return "fn", signature, (producer,), upstream_source

        return None

    def _downstream_stage_from_consumer(
        node: NodeProto,
    ) -> (
        tuple[
            str,
            tuple[str, str, Any] | tuple[str, str, tuple[tuple[str, Any], ...]],
            tuple[Any, ...],
            str,
        ]
        | None
    ):
        if node.op_type in QUANT_OP_TYPES:
            dq_consumers = input_name_to_nodes.get(node.output[0], [])
            dq_node = next((consumer for consumer in dq_consumers if consumer.op_type in DEQUANT_OP_TYPES), None)
            if dq_node:
                signature = _signature_from_quant_node(node)
                downstream_target = dq_node.output[0] if len(dq_node.output) >= 1 and dq_node.output[0] else ""
                return "pair", signature, (node, dq_node), downstream_target

        if node.op_type in FN_OP_TYPES:
            signature = _signature_from_fn_node(node)
            downstream_target = node.output[0] if len(node.output) >= 1 and node.output[0] else ""
            return "fn", signature, (node,), downstream_target

        return None

    def _has_tensor_override(tensor_name: str) -> bool:
        return (
            tensor_quant_overrides.has_per_tensor_overrides(tensor_name)
            or tensor_quant_overrides.has_per_channel_overrides(tensor_name)
            if tensor_quant_overrides
            else False
        )

    def _get_input_override_tensor_name(
        input_name: str,
        upstream_source_name: str,
    ) -> str | None:
        candidate_names = [input_name]
        if upstream_source_name and upstream_source_name != input_name:
            candidate_names.append(upstream_source_name)
        for candidate_name in candidate_names:
            if _has_tensor_override(candidate_name):
                return candidate_name
        return None

    def _get_output_override_tensor_name(output_name: str, downstream_target: str = "") -> str | None:
        candidate_names: list[str] = [output_name]
        if downstream_target and downstream_target != output_name:
            candidate_names.append(downstream_target)

        for candidate_name in candidate_names:
            if _has_tensor_override(candidate_name):
                return candidate_name
        return None

    def _insert_stage_between(
        source_tensor: str,
        target_node: NodeProto,
        target_input_name: str,
        stage_kind: str,
        stage_nodes: tuple[Any, ...],
        qparam_tensor_name: str,
        name_scope: str,
    ) -> None:
        if stage_kind == "pair":
            quant_template = stage_nodes[0]
            dequant_template = stage_nodes[1]
            assert quant_template is not None
            assert dequant_template is not None

            new_quant = copy.deepcopy(quant_template)
            new_dequant = copy.deepcopy(dequant_template)

            quant_name_base = f"{name_scope}_additional_{quant_template.op_type}"
            dequant_name_base = f"{name_scope}_additional_{dequant_template.op_type}"
            new_quant.name = _unique_name(quant_name_base, node_names)
            new_dequant.name = _unique_name(dequant_name_base, node_names)

            quant_output = _unique_name(f"{quant_name_base}_output", tensor_names)
            dequant_output = _unique_name(f"{dequant_name_base}_output", tensor_names)

            qtype = _qtype_from_quant_node(quant_template)

            scale_dtype: Any = np.float32
            quant_scale_init = (
                onnx_model.get_initializer(quant_template.input[1])
                if len(quant_template.input) >= 2 and quant_template.input[1]
                else None
            )
            if quant_scale_init is not None:
                scale_dtype = onnx.helper.tensor_dtype_to_np_dtype(quant_scale_init.data_type)
            # dequant_scale_init = (
            #     onnx_model.get_initializer(dequant_template.input[1])
            #     if len(dequant_template.input) >= 2 and dequant_template.input[1]
            #     else None
            # )
            # elif dequant_scale_init is not None:
            #    scale_dtype = onnx.helper.tensor_dtype_to_np_dtype(dequant_scale_init.data_type)

            if qtype is not None:
                scale_zp = _make_scale_zp_initializers(qparam_tensor_name, qtype, scale_dtype)
                if scale_zp is not None and len(new_quant.input) >= 3:
                    new_quant.input[1] = scale_zp[0]
                    new_quant.input[2] = scale_zp[1]
                    if len(new_dequant.input) >= 3:
                        new_dequant.input[1] = scale_zp[0]
                        new_dequant.input[2] = scale_zp[1]

            new_quant.input[0] = source_tensor
            new_quant.output[0] = quant_output
            new_dequant.input[0] = quant_output
            new_dequant.output[0] = dequant_output

            onnx_model.replace_node_input(target_node, target_input_name, dequant_output)
            onnx_model.add_node(new_quant)
            onnx_model.add_node(new_dequant)
            return None

        fn_template = stage_nodes[0]
        assert fn_template is not None

        new_fn = copy.deepcopy(fn_template)
        fn_name_base = f"{name_scope}_additional_{fn_template.op_type}"
        new_fn.name = _unique_name(fn_name_base, node_names)
        fn_output = _unique_name(f"{fn_name_base}_output", tensor_names)
        new_fn.input[0] = source_tensor
        new_fn.output[0] = fn_output

        onnx_model.replace_node_input(target_node, target_input_name, fn_output)
        onnx_model.add_node(new_fn)
        return None

    def _make_scale_zp_initializers(tensor_name: str, qtype: int, scale_dtype: Any) -> tuple[str, str] | None:
        if tensors_range is None or tensor_name not in tensors_range:
            logger.warning(
                f"Skip recomputing boundary scale/zp for tensor {tensor_name} because calibration range is unavailable."
            )
            return None

        td = tensors_range[tensor_name]
        assert hasattr(td, "range_value"), f"Invalid tensor range {td} without range_value"

        rmin, rmax = td.range_value[0], td.range_value[1]
        symmetric = extra_options.get("ActivationSymmetric", qtype in [onnx.TensorProto.INT8, onnx.TensorProto.INT16])
        use_pof2s = extra_options.get("UsePowerOf2Scale", True)

        qmin, qmax = get_qmin_qmax_for_qType(qtype, reduce_range=reduce_range, symmetric=symmetric)
        if qtype in ONNX_FP_QTYPES_LIST:
            zero_point, scale = compute_scale_zp_fp(
                rmin=rmin,
                rmax=rmax,
                qmin=qmin,
                qmax=qmax,
                element_type=qtype,
                method=calibrate_method,
                symmetric=symmetric,
                use_scaling=False,
            )
        else:
            zero_point, scale = compute_scale_zp(
                rmin=rmin,
                rmax=rmax,
                qmin=qmin,
                qmax=qmax,
                element_type=qtype,
                method=calibrate_method,
                symmetric=symmetric,
                use_pof2s=use_pof2s,
            )

        scale_np = np.asarray(scale, dtype=scale_dtype).reshape(())
        zp_np_dtype = onnx.helper.tensor_dtype_to_np_dtype(qtype)
        zp_np = np.asarray(zero_point, dtype=zp_np_dtype).reshape(())
        scale_name = _unique_name(f"{tensor_name}_additional_scale", tensor_names)
        zp_name = _unique_name(f"{tensor_name}_additional_zero_point", tensor_names)

        onnx_model.add_initializer(onnx.numpy_helper.from_array(scale_np, scale_name))
        onnx_model.add_initializer(onnx.numpy_helper.from_array(zp_np, zp_name))
        return scale_name, zp_name

    def _find_template_stage(
        all_node_stage_infos: dict[str, list[dict[str, Any]]], current_node_stage_infos: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Some nodes maybe surround by nodes with mixed precision, we cannot determine the template stage for
        it based on its own stage infos. We need to find the template stage from adjacent nodes.
        """
        for stage_info in current_node_stage_infos:
            for _, infos in all_node_stage_infos.items():
                for info in infos:
                    # The template stage should be from the adjacent node that shares the exact same tensor.
                    if info["override_tensor_name"] == stage_info["override_tensor_name"] and (
                        info["tensor_name"] == stage_info["override_tensor_name"]
                        or info["override_tensor_name"] == stage_info["tensor_name"]
                    ):
                        if info["template_index"] < 0:
                            continue

                        logger.info(
                            f"Found a template stage from the tensor {info['tensor_name']} of node {info['node_name']} "
                            f"for the node {stage_info['node_name']}"
                        )
                        return infos[info["template_index"]]
        return None

    # Phase 1: collect all candidate boundaries and their upstream/downstream stage snapshots.
    node_stage_infos: dict[str, list[dict[str, Any]]] = {}

    for node_index, node in enumerate(list(onnx_model.nodes())):
        if node.op_type in QUANT_OP_TYPES + DEQUANT_OP_TYPES + FN_OP_TYPES:
            continue

        node_key = f"{node.name or node.op_type}_{node_index}"
        node_display_name = node.name or node.op_type
        node_stage_infos[node_key] = []

        for input_index, input_name in enumerate(node.input):
            if input_name:
                upstream_stage = _upstream_stage_from_input(input_name)
                if upstream_stage:
                    upstream_kind, upstream_signature, upstream_nodes, upstream_source_name = upstream_stage

                    # Skip if the upstream source is an initializer.
                    if onnx_model.get_initializer(upstream_source_name) is not None:
                        continue

                    input_override_tensor_name = _get_input_override_tensor_name(input_name, upstream_source_name)
                    node_stage_infos[node_key].append(
                        {
                            "node_name": node_display_name,  # For example, "Conv_0"
                            "tensor_type": "input",  # For example, "input"
                            "tensor_index": input_index,  # For example, 0
                            "tensor_name": input_name,  # For example, "input_0_DequantizeLinear_output"
                            "override_tensor_name": input_override_tensor_name,  # For example, "input_0"
                            "stage_kind": upstream_kind,  # For example, "pair"
                            "stage_signature": upstream_signature,  # For example, ("quant", "ai.onnx::QuantizeLinear", 1)
                            "stage_nodes": upstream_nodes,  # For example, (quant_template, dequant_template)
                            "target_node": node,  # For example, Conv node instance
                            "template_index": -1,  # For example, -1
                        }
                    )

        for output_index, output_name in enumerate(node.output):
            consumers = input_name_to_nodes.get(output_name, [])
            for consumer in consumers:
                downstream_stage = _downstream_stage_from_consumer(consumer)
                if downstream_stage:
                    downstream_kind, downstream_signature, downstream_nodes, downstream_source_name = downstream_stage

                    output_override_tensor_name = _get_output_override_tensor_name(output_name, downstream_source_name)
                    node_stage_infos[node_key].append(
                        {
                            "node_name": node_display_name,
                            "tensor_type": "output",
                            "tensor_index": output_index,
                            "tensor_name": output_name,
                            "override_tensor_name": output_override_tensor_name,
                            "stage_kind": downstream_kind,
                            "stage_signature": downstream_signature,
                            "stage_nodes": downstream_nodes,
                            "target_node": consumer,
                            "template_index": -1,
                        }
                    )

    # Phase 2: determine the template nodes for the boundaries to insert quant nodes.
    inserted_count = 0

    inserted_tensors: set[tuple[str, str, str]] = set()
    for node_key, stage_infos in node_stage_infos.items():
        if not stage_infos:
            continue
        node_name = stage_infos[0]["node_name"]

        """
        # Comment out this to make sure all nodes have target QDQ pairs
        input_signatures = {info["stage_signature"] for info in stage_infos if info["tensor_type"] == "input"}
        output_signatures = {info["stage_signature"] for info in stage_infos if info["tensor_type"] == "output"}
        if input_signatures == output_signatures:
            logger.debug(
                f"Skipping boundary insertion for node {node_name}: input and output activations share the same quantization signature."
            )
            continue
        """

        template_index = -1
        for index, info in enumerate(stage_infos):
            if nodes_with_mixed_precision and node_name not in nodes_with_mixed_precision:
                # Use the first input boundary if no override tensor name exists.
                if not info["override_tensor_name"]:
                    template_index = index
                    break
            else:
                # Use the override tensor name if it exists as the template.
                if info["override_tensor_name"]:
                    template_index = index
                    break

        template_info: dict[str, Any] | None = None
        if template_index >= 0:
            template_info = stage_infos[template_index]
        else:
            template_info = _find_template_stage(node_stage_infos, stage_infos)
        if template_info is None:
            logger.warning(f"No template stage found for node {node_name} to insert quant nodes at the boundaries.")
            continue

        template_kind = template_info["stage_kind"]
        template_signature = template_info["stage_signature"]
        template_nodes = template_info["stage_nodes"]

        for info in stage_infos:
            if info is template_info:
                continue
            if info["stage_signature"] == template_signature:
                continue
            info["template_index"] = template_index  # Update the template index for potential use
            qparam_tensor_name = info["override_tensor_name"] or info["tensor_name"]

            target_node_name = info["target_node"].name or info["target_node"].op_type
            inserted_tensor_key = (node_name, info["tensor_name"], target_node_name)
            if inserted_tensor_key not in inserted_tensors:
                _insert_stage_between(
                    source_tensor=info["tensor_name"],
                    target_node=info["target_node"],
                    target_input_name=info["tensor_name"],
                    stage_kind=template_kind,
                    stage_nodes=template_nodes,
                    qparam_tensor_name=qparam_tensor_name,
                    name_scope=f"{qparam_tensor_name}_{node_name}_{info['tensor_index']}_{target_node_name}",
                )
                inserted_tensors.add(inserted_tensor_key)

                inserted_count += 1

    if inserted_count > 0:
        logger.info(f"Inserted {inserted_count} quant nodes at the boundary tensors of two different precisions.")
        onnx_model.topological_sort()

    return onnx_model.model


def get_eltwise_op(input_model: str | Path | ModelProto) -> list[str]:
    eltwise_op_types = ["Mul", "Add", "Sub", "Div", "Min", "Max"]
    model = input_model if isinstance(input_model, onnx.ModelProto) else onnx.load(input_model)
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


def convert_nparray(qType: Any, arr: np.ndarray[Any, Any]) -> Any:
    onnx_model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Cast", ["X"], ["Y"], to=qType)],
            "qu",
            [helper.make_tensor_value_info("X", onnx_proto.TensorProto.FLOAT, None)],
            [helper.make_tensor_value_info("Y", qType, None)],
        )
    )
    ref = ReferenceEvaluator(onnx_model)
    return ref.run(None, {"X": arr})[0]  # type: ignore


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
                cast_node = onnx.helper.make_node(
                    "Cast", inputs=[input_], outputs=[input_ + "_cast"], to=onnx_proto.TensorProto.BFLOAT16
                )
                add_node_list.append(cast_node)

    input_to_node: dict[str, list[NodeProto]] = {}
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
                    cast_node = onnx.helper.make_node(
                        "Cast", inputs=[output_ + "_cast"], outputs=[output_], to=onnx_proto.TensorProto.FLOAT
                    )
                elif original_data_type == 10:
                    cast_node = onnx.helper.make_node(
                        "Cast", inputs=[output_ + "_cast"], outputs=[output_], to=onnx_proto.TensorProto.FLOAT16
                    )
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


def match_subgraphs(input_model: str | Path | ModelProto, subgraphs: list[tuple[list[str]]]) -> list[str]:
    def _dfs(
        node: NodeProto,
        exclude_nodes_list: list[str],
        start_nodes_list: list[str],
        output2node_dict: dict[str, NodeProto],
        model_input_names_list: list[str],
        visited: list[str],
    ) -> None:
        exclude_nodes_list.append(node.name)
        for inp in node.input:
            if inp in model_input_names_list:
                visited.append(inp)
                return
            if inp in visited:
                return
            if inp in output2node_dict:
                if output2node_dict[inp].name in start_nodes_list:
                    visited.append(inp)
                    exclude_nodes_list.append(output2node_dict[inp].name)
                    return
                else:
                    exclude_nodes_list.append(node.name)
                    visited.append(inp)
                    _dfs(
                        output2node_dict[inp],
                        exclude_nodes_list,
                        start_nodes_list,
                        output2node_dict,
                        model_input_names_list,
                        visited,
                    )

    model = input_model if isinstance(input_model, onnx.ModelProto) else onnx.load(input_model)

    model_input_names_list = [inp.name for inp in model.graph.input]

    name2node_dict = {}
    for node in model.graph.node:
        name2node_dict[node.name] = node

    onnx_model = ONNXModel(model)
    output2node_dict = onnx_model.output_name_to_node()
    visited: list[str] = []

    exclude_nodes_list: list[str] = []
    for subgraph in subgraphs:
        start_nodes_list: list[str] = []
        end_nodes_list: list[str] = []
        start_nodes_list, end_nodes_list = subgraph[0], subgraph[1]  # type: ignore
        exclude_nodes_list.extend(start_nodes_list)
        exclude_nodes_list.extend(end_nodes_list)
        for end_node_name in end_nodes_list:
            father_node = name2node_dict[end_node_name]
            _dfs(father_node, exclude_nodes_list, start_nodes_list, output2node_dict, model_input_names_list, visited)
    for input_name in model_input_names_list:
        if input_name in visited:
            raise ValueError(
                f"Please verify that the value of parameter subgraphs_to_exclude {subgraphs} is valid by ensuring that its start and end nodes form a closed subgraph."
            )
    exclude_nodes_list = list(set(exclude_nodes_list))
    return exclude_nodes_list


def check_model_is_fp16(input_model: str | Path | ModelProto) -> bool:
    fp32_data_type = 1
    fp16_data_type = 10
    model = input_model if isinstance(input_model, onnx.ModelProto) else onnx.load(input_model)
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


def get_all_target_nodes(model: ModelProto, target_node_list: list[str | tuple[list[str]]]) -> list[str]:
    if len(target_node_list) < 1:
        return []

    single_nodes = []
    patterns = []
    subgraphs = []
    for node_or_graph in target_node_list:
        if isinstance(node_or_graph, str):
            if node_or_graph.startswith("^"):
                patterns.append(node_or_graph)
            else:
                single_nodes.append(node_or_graph)
        elif isinstance(node_or_graph, tuple):
            subgraphs.append(node_or_graph)
        else:
            raise TypeError(
                f"Only str and tuple are supported. This {node_or_graph} item with type {type(node_or_graph)} is not supported."
            )
    if subgraphs != []:
        single_nodes.extend(match_subgraphs(model, subgraphs))

    for pattern in patterns:
        if ".*" not in pattern:
            logger.warning(
                f"Your regular expression pattern {pattern} is invalid. Only support the regular expression pattern starting with ^ and contain the .* characters."
            )
            patterns.remove(pattern)

    pattern_match_nodes = []
    for node in model.graph.node:
        node_name = node.name
        for pattern in patterns:
            if re.search(pattern, node_name):
                pattern_match_nodes.append(node_name)
                logger.info(f"Have matched node {node_name} with the regular expression pattern {pattern}.")

    if len(patterns) > 0 and len(pattern_match_nodes) == 0:
        raise ValueError(
            f"Your regular expression pattern {pattern} is wrong, please check and input the correct re pattern with .*"
        )

    all_nodes = set(single_nodes + pattern_match_nodes)
    return list(all_nodes)


def recursive_update(base: dict[str, Any], new: dict[str, Any]) -> None:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            recursive_update(base[k], v)
        else:
            base[k] = v


def get_pre_defined_preprocess_config(pre_defined_template_name: str) -> dict[str, Any]:
    general_preprocess_config = {
        "passes": {
            "onnx_convert_opset_version": {"target_opset_version": 21},
            "onnx_simplify": {"simplify": True},
            "onnx_remove_input_init": {"remove_input_init": True},
            "onnx_copy_bias_init": {"shared_bias_op_types": ["Conv", "ConvTranspose", "Gemm"]},
            "onnx_optimize_with_ort": {"optimize_with_ort": True},
            "onnx_fold_batch_norm": {"fold_batch_norm": True},
            "onnx_fuse_instance_norm": {"fuse_instance_norm": True},
            "onnx_fuse_l2_norm": {"fuse_l2_norm": True},
            "onnx_fuse_gelu": {"fuse_gelu": True},
            "onnx_fuse_layer_norm": {"fuse_layer_norm": True},
        }
    }
    xint8_preprocess_config = {
        "passes": {
            "onnx_convert_opset_version": {"target_opset_version": 21},
            "onnx_simplify": {"simplify": True},
            "onnx_remove_input_init": {"remove_input_init": True},
            "onnx_copy_bias_init": {"shared_bias_op_types": ["Conv", "ConvTranspose", "Gemm"]},
            "onnx_optimize_with_ort": {"optimize_with_ort": True},
            "onnx_fold_batch_norm": {"fold_batch_norm": True},
            "onnx_fuse_instance_norm": {"fuse_instance_norm": True},
            "onnx_fuse_l2_norm": {"fuse_l2_norm": True},
            "onnx_fuse_gelu": {"fuse_gelu": True},
            "onnx_fuse_layer_norm": {"fuse_layer_norm": True},
            "onnx_convert_split_to_slice": {"convert_split_to_slice": True},
            "onnx_convert_bn_to_conv": {"convert_bn_to_conv": True},
            "onnx_convert_reduce_mean_to_global_avg_pool": {"convert_reduce_mean_to_global_avg_pool": True},
            "onnx_split_large_kernel_pool": {"split_large_kernel_pool": True},
        }
    }

    if pre_defined_template_name.lower() in ["a8w8", "a16w8", "bf16", "bfp16"]:
        return general_preprocess_config
    elif pre_defined_template_name.lower() == "xint8":
        return xint8_preprocess_config
    else:
        logger.warning(
            f"The param PreprocessYAML {pre_defined_template_name} is valid. Please choose from xint8, a8w8, a16w8, bf16, bfp16."
        )
        return general_preprocess_config
