#!/usr/bin/env python
# coding: utf-8
#
# Modifications copyright(c) 2023 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from quark.shares.utils.log import ScreenLogger, log_errors
from tqdm import tqdm

import numpy as np
import onnx
from pathlib import Path
from onnxruntime.quantization.calibrate import (CalibraterBase, CalibrationDataCollector, CalibrationDataReader,
                                                CalibrationMethod, MinMaxCalibrater as OrtMinMaxCalibrater,
                                                EntropyCalibrater as OrtEntropyCalibrater, PercentileCalibrater as
                                                OrtPercentileCalibrater)
from onnxruntime.quantization.quant_utils import QuantType
from .quant_utils import (PowerOfTwoMethod, get_tensor_type_from_qType, quantize_data_pof2s, VitisQuantType)
from typing import List, Dict, Any, Union, Optional, Sequence

logger = ScreenLogger(__name__)
calib_quant_type = [
    QuantType.QInt8,
    QuantType.QUInt8,
    VitisQuantType.QInt16,
    VitisQuantType.QUInt16,
    VitisQuantType.QInt32,
    VitisQuantType.QUInt32,
]


class MinMaxCalibrater(OrtMinMaxCalibrater):  # type: ignore
    """
    This method obtains the quantization parameters based on the minimum and maximum values of each tensor.

    :param model_path: Path to the ONNX model to calibrate.
    :param op_types_to_calibrate: List of operator types to calibrate. By default, calibrates all the float32/float16 tensors.
    :param augmented_model_path: Path to save the augmented model. Default is "augmented_model.onnx".
    :param symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is False.
    :param use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is False.
    :param moving_average: Whether to compute the moving average of the minimum and maximum values instead of the global minimum and maximum. Default is False.
    :param averaging_constant: Constant smoothing factor to use when computing the moving average. Default is 0.01. Should be between 0 and 1.
    :raises ValueError: If averaging_constant is not between 0 and 1 when moving_average is True.
    """

    def __init__(
        self,
        model_path: Path,
        op_types_to_calibrate: Union[List[str], None],
        augmented_model_path: str = "augmented_model.onnx",
        symmetric: bool = False,
        use_external_data_format: bool = False,
        moving_average: bool = False,
        averaging_constant: float = 0.01,
    ) -> None:
        super().__init__(
            model_path,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            symmetric=symmetric,
            use_external_data_format=use_external_data_format,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
        )
        self.intermediate_outputs: List[str] = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = {output.name for output in self.model.graph.output}
        self.moving_average = moving_average
        if moving_average and (averaging_constant < 0 or averaging_constant > 1):
            raise ValueError("Invalid averaging constant, which should not be < 0 or > 1.")
        self.averaging_constant = averaging_constant


class EntropyCalibrater(OrtEntropyCalibrater):  # type: ignore
    """
    This method determines the quantization parameters by considering the entropy algorithm of each tensor's distribution.

    :param model_path: Path to the ONNX model to calibrate.
    :param op_types_to_calibrate: List of operator types to calibrate. By default, calibrates all the float32/float16 tensors.
    :param augmented_model_path: Path to save the augmented model. Default is "augmented_model.onnx".
    :param use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is False.
    :param method: Method for calibration. One of ['entropy', 'percentile', 'distribution']. Default is "entropy".
    :param symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is False.
    :param num_bins: Number of bins to create a new histogram for collecting tensor values. Default is 128.
    :param num_quantized_bins: Number of quantized bins. Default is 128.
    """

    def __init__(
        self,
        model_path: Path,
        op_types_to_calibrate: Union[List[str], None],
        augmented_model_path: str = "augmented_model.onnx",
        use_external_data_format: bool = False,
        method: str = "entropy",
        symmetric: bool = False,
        num_bins: int = 128,
        num_quantized_bins: int = 128,
    ) -> None:
        """
        :param model_path: ONNX model to calibrate. It is a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        :param method: A string. One of ['entropy', 'percentile', 'distribution'].
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param num_bins: number of bins to create a new histogram for collecting tensor values.
        :param num_quantized_bins: number of quantized bins. Default 128.
        """
        super().__init__(
            model_path,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            use_external_data_format=use_external_data_format,
            method=method,
            symmetric=symmetric,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
        )


class PercentileCalibrater(OrtPercentileCalibrater):  # type: ignore
    """
    This method calculates quantization parameters using percentiles of the tensor values.

    :param model_path: Path to the ONNX model to calibrate.
    :param op_types_to_calibrate: List of operator types to calibrate. By default, calibrates all the float32/float16 tensors.
    :param augmented_model_path: Path to save the augmented model. Default is "augmented_model.onnx".
    :param use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is False.
    :param method: Method for calibration. One of ['entropy', 'percentile', 'distribution']. Default is "percentile".
    :param symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is False.
    :param num_bins: Number of bins to create a new histogram for collecting tensor values. Default is 2048.
    :param percentile: Percentile value for calibration, a float between [0, 100]. Default is 99.999.
    """

    def __init__(
        self,
        model_path: Path,
        op_types_to_calibrate: Union[List[str], None],
        augmented_model_path: str = "augmented_model.onnx",
        use_external_data_format: bool = False,
        method: str = "percentile",
        symmetric: bool = False,
        num_bins: int = 2048,
        percentile: float = 99.999,
    ):
        """
        :param model_path: ONNX model to calibrate. It is a model path
        :param op_types_to_calibrate: operator types to calibrate. By default, calibrate all the float32/float16 tensors.
        :param augmented_model_path: save augmented model to this path.
        :param use_external_data_format: use external data format to store model which size is >= 2Gb
        :param method: A string. One of ['entropy', 'percentile', 'distribution'].
        :param symmetric: make range of tensor symmetric (central point is 0).
        :param num_quantized_bins: number of quantized bins. Default 128.
        :param percentile: A float number between [0, 100]. Default 99.99.
        """
        super().__init__(
            model_path,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            use_external_data_format=use_external_data_format,
            method=method,
            symmetric=symmetric,
            num_bins=num_bins,
            percentile=percentile,
        )


class PowOfTwoCalibrater(CalibraterBase):  # type: ignore
    """
    This method get the power-of-two quantize parameters for each tensor to minimize the mean-square-loss of quantized values and float values. This takes longer time but usually gets better accuracy.

    :param model: Path to the ONNX model to calibrate.
    :param op_types_to_calibrate: List of operator types to calibrate. By default, calibrates all the float32/float16 tensors.
    :param augmented_model_path: Path to save the augmented model. Default is "augmented_model.onnx".
    :param use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is False.
    :param activation_type: Type of quantization for activations. Default is QuantType.QInt8.
    :param method: Calibration method. Default is PowerOfTwoMethod.MinMSE.
    :param symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is True.
    :param minmse_mode: Mode for the MinMSE method. Default is "All".
    :param percentile: Percentile value for calibration, a float between 0 and 100. Default is 99.999.
    :param quantized_tensor_type: Dictionary specifying the quantized tensor type. Default is an empty dictionary.
    """

    def __init__(
        self,
        model: Path,
        op_types_to_calibrate: Optional[Sequence[str]],
        augmented_model_path: str = "augmented_model.onnx",
        use_external_data_format: bool = False,
        activation_type: Union[QuantType, VitisQuantType] = QuantType.QInt8,
        method: PowerOfTwoMethod = PowerOfTwoMethod.MinMSE,
        symmetric: bool = True,
        minmse_mode: str = "All",
        percentile: float = 99.999,
        quantized_tensor_type: Dict[Any, Any] = {},
    ) -> None:

        super(PowOfTwoCalibrater, self).__init__(model, op_types_to_calibrate, augmented_model_path, symmetric,
                                                 use_external_data_format)
        self.intermediate_outputs: List[str] = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = set(output.name for output in self.model.graph.output)
        self.collector: Optional[PowOfTwoCollector] = None
        self.method = method
        self.symmetric = symmetric
        self.tensors_to_calibrate = None
        self.activation_type = activation_type
        self.use_external_data_format = use_external_data_format
        self.minmse_mode = minmse_mode
        self.percentile = percentile
        self.quantized_tensor_type = quantized_tensor_type

    def augment_graph(self) -> None:
        """
        make all quantization_candidates op type nodes as part of the graph output.
        :return: augmented ONNX model
        """
        model = self.model

        self.tensors_to_calibrate, value_infos = self.select_tensors_to_calibrate(model)
        if self.tensors_to_calibrate is not None:
            for tensor in self.tensors_to_calibrate:
                if tensor not in self.model_original_outputs:
                    model.graph.output.append(value_infos[tensor])
        onnx.save(
            model,
            self.augmented_model_path,
            save_as_external_data=self.use_external_data_format,
        )
        self.augment_model = model

    def clear_collected_data(self) -> None:
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader) -> None:
        while True:
            inputs = data_reader.get_next()
            if not inputs:
                break
            self.intermediate_outputs.append(self.infer_session.run(None, inputs))

        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        output_names = [self.infer_session.get_outputs()[i].name for i in range(len(self.intermediate_outputs[0]))]
        output_dicts_list = [
            dict(zip(output_names, intermediate_output)) for intermediate_output in self.intermediate_outputs
        ]

        merged_dict: Dict[Any, Any] = {}
        for d in output_dicts_list:
            for k, v in d.items():
                merged_dict.setdefault(k, []).append(v)
        if self.tensors_to_calibrate is not None:
            clean_merged_dict: Dict[Any, Any] = dict(
                (i, merged_dict[i]) for i in merged_dict if i in self.tensors_to_calibrate)

        if self.collector is None:
            self.collector = PowOfTwoCollector(activation_type=self.activation_type,
                                               method=self.method,
                                               symmetric=self.symmetric,
                                               minmse_mode=self.minmse_mode,
                                               percentile=self.percentile,
                                               quantized_tensor_type=self.quantized_tensor_type)
        if self.collector is not None:
            self.collector.collect(clean_merged_dict)

        self.clear_collected_data()

    def compute_range(self) -> Any:
        """
        Compute the min-max range of tensor
        :return: dictionary mapping: {tensor name: (min value, max value)}
        """
        if not self.collector:
            raise ValueError("No collector created and can't generate calibration data.")

        return self.collector.compute_collection_result()


class PowOfTwoCollector(CalibrationDataCollector):  # type: ignore
    """
    Collecting PowOfTwoCollector quantize for each tensor. Support MinMSE method.

    :param activation_type: Type of quantization for activations. Default is QuantType.QInt8.
    :param method: Calibration method. Default is PowerOfTwoMethod.MinMSE.
    :param symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is True.
    :param minmse_mode: Mode for the MinMSE method. Default is "All".
    :param percentile: Percentile value for calibration, a float between 0 and 100. Default is 99.999.
    :param quantized_tensor_type: Dictionary specifying the quantized tensor type. Default is an empty dictionary.

    """

    def __init__(self,
                 activation_type: Union[QuantType, VitisQuantType] = QuantType.QInt8,
                 method: PowerOfTwoMethod = PowerOfTwoMethod.MinMSE,
                 symmetric: bool = True,
                 minmse_mode: str = "All",
                 percentile: float = 99.999,
                 quantized_tensor_type: Dict[Any, Any] = {}):
        self.name_to_arr: Dict[Any, Any] = {}
        self.method = method
        self.symmetric = symmetric
        self.minmse_mode = minmse_mode
        self.activation_qType = get_tensor_type_from_qType(activation_type)
        self.percentile = percentile
        self.quantized_tensor_type = quantized_tensor_type

    def check_uniform_length(self, lst: List[Any]) -> bool:
        if isinstance(lst, list) and len(lst) > 2:
            reference_length = len(lst[0])

            for element in lst[1:]:
                if len(element) != reference_length:
                    return False

        return True

    def collect(self, name_to_arr: Dict[Any, Any]) -> None:

        self.name_to_arr = name_to_arr

        return

    def compute_collection_result(self) -> Any:
        if not self.name_to_arr or len(self.name_to_arr) == 0:
            raise ValueError("PowerOfTwoMethod has not been collected. Please run collect() first.")
        logger.info("Finding optimal threshold for each tensor using {} algorithm ...".format(self.method))

        if self.method == PowerOfTwoMethod.MinMSE:
            return self.compute_minmse_range()
        else:
            raise ValueError("Only 'MinMSE' method are supported")

    @log_errors
    def compute_minmse_range(self) -> Dict[Any, Any]:
        thresholds_dict = {}
        if self.minmse_mode == "MostCommon" and self.symmetric:
            logger.info("Use the most common min mse from each batch")
            for tensor, data_arr in tqdm(self.name_to_arr.items(), desc="Computing range", unit="tensor"):
                act_type = self.activation_qType
                method = self.method
                if tensor in self.quantized_tensor_type and self.quantized_tensor_type[tensor] in calib_quant_type:
                    logger.info(
                        f"The type of tensor {tensor} is {self.quantized_tensor_type[tensor]}: using specific tensor precision"
                    )
                    act_type = get_tensor_type_from_qType(self.quantized_tensor_type[tensor])
                scale_list = []
                scale2threshold = {}
                for d in data_arr:
                    rmin_mse, rmax_mse, zp_mse, scale_mse, quantized_data_mse = quantize_data_pof2s(d,
                                                                                                    act_type,
                                                                                                    self.symmetric,
                                                                                                    method=method)
                    scale2threshold[float(scale_mse)] = (rmin_mse, rmax_mse)
                    scale_list.append(scale_mse)
                # get most common pos
                u, indices = np.unique(scale_list, return_inverse=True)
                scale = u[np.argmax(np.bincount(indices))]
                thresholds_dict[tensor] = scale2threshold[scale]

        elif self.minmse_mode == "Percentile":
            logger.info("Use the percentile to calculate min mse, "
                        f"CalibTensorRangeSymmetric: {self.symmetric}, Percentile: {self.percentile}")
            for _, data_arr in (self.name_to_arr.items()):
                if not self.check_uniform_length(data_arr):
                    raise ValueError("The batch size cannot be evenly divided by all data, "
                                     f"Under {self.method} settings, it must be divisible by the total number of data. "
                                     "Please check the batch size configuration.")
                    break
                else:
                    break

            for tensor, data_arr in tqdm(self.name_to_arr.items(), desc="Computing range", unit="tensor"):
                act_type = self.activation_qType
                method = self.method
                if tensor in self.quantized_tensor_type and self.quantized_tensor_type[tensor] in calib_quant_type:
                    logger.info(
                        f"The type of tensor {tensor} is {self.quantized_tensor_type[tensor]}: using specific tensor precision"
                    )
                    act_type = get_tensor_type_from_qType(self.quantized_tensor_type[tensor])
                d = np.array(data_arr).flatten()
                if self.symmetric:
                    lower_limit = -np.percentile(np.abs(d), self.percentile)
                    upper_limit = np.percentile(np.abs(d), self.percentile)
                else:
                    lower_limit = np.percentile(d, (100 - self.percentile) / 2)
                    upper_limit = np.percentile(d, 100 - (100 - self.percentile) / 2)

                d = d[(d >= lower_limit) & (d <= upper_limit)]

                rmin_mse, rmax_mse, _, _, _ = quantize_data_pof2s(d, act_type, self.symmetric, method=method)
                thresholds_dict[tensor] = (rmin_mse, rmax_mse)

        else:
            if self.minmse_mode == "MostCommon":
                logger.warning("Activation asymmetric does not support using 'most common' to calculate min mse")
            if self.minmse_mode != "All":
                logger.warning("Currently MinMSEMode only supports 'All' and 'MostCommon'."
                               f"Does not support {self.minmse_mode}")
            logger.info("Use all calibration data to calculate min mse")
            for _, data_arr in (self.name_to_arr.items()):
                if not self.check_uniform_length(data_arr):
                    raise ValueError("The batch size cannot be evenly divided by all data,"
                                     f"under {self.method} settings, it must be divisible by the total number of data."
                                     "Please check the batch size configuration.")
                    break
                else:
                    break
            for tensor, data_arr in tqdm(self.name_to_arr.items(), desc="Computing range", unit="tensor"):
                act_type = self.activation_qType
                method = self.method
                if tensor in self.quantized_tensor_type and self.quantized_tensor_type[tensor] in calib_quant_type:
                    logger.info(
                        f"The type of tensor {tensor} is {self.quantized_tensor_type[tensor]}: using specific tensor precision"
                    )
                    act_type = get_tensor_type_from_qType(self.quantized_tensor_type[tensor])
                d = np.array(data_arr).flatten()
                rmin_mse, rmax_mse, _, _, _ = quantize_data_pof2s(d, act_type, self.symmetric, method=method)
                thresholds_dict[tensor] = (rmin_mse, rmax_mse)
        return thresholds_dict


def create_calibrator_power_of_two(
    model: Path,
    op_types_to_calibrate: List[str],
    augmented_model_path: str = "augmented_model.onnx",
    activation_type: Union[VitisQuantType, QuantType] = QuantType.QInt8,
    method: PowerOfTwoMethod = PowerOfTwoMethod.NonOverflow,
    use_external_data_format: bool = False,
    execution_providers: Union[List[str], None] = ['CPUExecutionProvider'],
    quantized_tensor_type: Dict[Any, Any] = {},
    extra_options: Dict[str, Any] = {},
) -> Any:
    """
    Create a calibrator for power-of-two quantization.

    :param model: Path to the ONNX model to calibrate.
    :param op_types_to_calibrate: List of operator types to calibrate.
    :param augmented_model_path: Path to save the augmented ONNX model.
    :param activation_type: Type of quantization for activations.
    :param method: Calibration method to use.
    :param use_external_data_format: Whether to use external data format for large models.
    :param execution_providers: List of execution providers for ONNX Runtime.
    :param quantized_tensor_type: Dictionary specifying the quantized tensor type.
    :param extra_options: Additional options for calibrator configuration.
    :return: Initialized calibrator object.
    """
    calibrator = None

    # default settings for min-max algorithm
    method = method
    symmetric = True if "symmetric" not in extra_options else extra_options["symmetric"]
    moving_average = False if "moving_average" not in extra_options else extra_options["moving_average"]
    averaging_constant = 0.01 if "averaging_constant" not in extra_options else extra_options["averaging_constant"]
    minmse_mode = 'All' if "minmse_mode" not in extra_options else extra_options["minmse_mode"]
    percentile = 99.999 if "percentile" not in extra_options else extra_options["percentile"]

    activation_type = QuantType.QInt8 if activation_type not in calib_quant_type else activation_type
    if method == PowerOfTwoMethod.NonOverflow:
        calibrator = MinMaxCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
        )
    elif method == PowerOfTwoMethod.MinMSE:
        calibrator = PowOfTwoCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            activation_type=activation_type,
            method=method,
            symmetric=symmetric,
            minmse_mode=minmse_mode,
            percentile=percentile,
            quantized_tensor_type=quantized_tensor_type,
        )

    if calibrator:
        calibrator.augment_graph()
        calibrator.execution_providers = execution_providers
        calibrator.create_inference_session()
        return calibrator


@log_errors
def create_calibrator_float_scale(
        model: Path,
        op_types_to_calibrate: Union[List[str], None],
        augmented_model_path: str = "augmented_model.onnx",
        calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
        use_external_data_format: bool = False,
        execution_providers: Union[List[str], None] = ['CPUExecutionProvider'],
        extra_options: Dict[str, Any] = {},  # noqa: B006
) -> Any:
    """
    Create a calibrator for floating-point scale quantization.

    :param model: Path to the ONNX model to calibrate.
    :param op_types_to_calibrate: List of operator types to calibrate. If None, all float32/float16 tensors are calibrated.
    :param augmented_model_path: Path to save the augmented ONNX model.
    :param calibrate_method: Calibration method to use (MinMax, Entropy, Percentile, or Distribution).
    :param use_external_data_format: Whether to use external data format for large models.
    :param execution_providers: List of execution providers for ONNX Runtime.
    :param extra_options: Additional options for calibrator configuration.
    :return: Initialized calibrator object.
    """
    calibrator = None
    if calibrate_method == CalibrationMethod.MinMax:
        # default settings for min-max algorithm
        symmetric = False if "symmetric" not in extra_options else extra_options["symmetric"]
        moving_average = False if "moving_average" not in extra_options else extra_options["moving_average"]
        averaging_constant = 0.01 if "averaging_constant" not in extra_options else extra_options["averaging_constant"]
        calibrator = MinMaxCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
        )
    elif calibrate_method == CalibrationMethod.Entropy:
        # default settings for entropy algorithm
        num_bins = 128 if "num_bins" not in extra_options else extra_options["num_bins"]
        num_quantized_bins = 128 if "num_quantized_bins" not in extra_options else extra_options["num_quantized_bins"]
        symmetric = False if "symmetric" not in extra_options else extra_options["symmetric"]
        calibrator = EntropyCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
        )
    elif calibrate_method == CalibrationMethod.Percentile:
        # default settings for percentile algorithm
        num_bins = 2048 if "num_bins" not in extra_options else extra_options["num_bins"]
        percentile = 99.999 if "percentile" not in extra_options else extra_options["percentile"]
        symmetric = True if "symmetric" not in extra_options else extra_options["symmetric"]
        calibrator = PercentileCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            num_bins=num_bins,
            percentile=percentile,
        )

    elif calibrate_method == CalibrationMethod.Distribution:
        # default settings for percentile algorithm
        num_bins = 2048 if "num_bins" not in extra_options else extra_options["num_bins"]
        scenario = "same" if "scenario" not in extra_options else extra_options["scenario"]

        from onnxruntime.quantization.calibrate import DistributionCalibrater
        calibrator = DistributionCalibrater(
            model,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            num_bins=num_bins,
            scenario=scenario,
        )

    if calibrator:
        calibrator.augment_graph()
        calibrator.execution_providers = execution_providers
        calibrator.create_inference_session()
        return calibrator

    raise ValueError(f"Unsupported calibration method {calibrate_method}")
