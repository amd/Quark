#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import shutil
import copy
import os
import random
import time
import logging
import onnxruntime as ort
import numpy as np
from typing import Union, Optional, Any, Dict, Tuple

from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType
from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.api import ModelQuantizer
from quark.onnx.quant_utils import PowerOfTwoMethod
from quark.onnx.operators.custom_ops import get_library_path

Level1_config_keys = [
    'calibrate_method',
    'quant_format',
    'activation_type',
    'weight_type',
    'input_nodes',
    'output_nodes',
    'op_types_to_quantize',
    'nodes_to_quantize',
    'nodes_to_exclude',
    'specific_tensor_precision',
    'execution_providers',
    'per_channel',
    'reduce_range',
    'optimize_model',
    'use_dynamic_quant',
    'use_external_data_format',
    'convert_fp16_to_fp32',
    'convert_nchw_to_nhwc',
    'include_sq',
    'include_cle',
    'include_auto_mp',
    'include_fast_ft',
    'enable_npu_cnn',
    'enable_npu_transformer',
    'debug_mode',
    'print_summary',
    'ignore_warnings',
    'log_severity_level',
    'extra_options',
]
Level2_config_keys = [
    'ActivationSymmetric',
    'WeightSymmetric',
    'UseUnsignedReLU',
    'QuantizeBias',
    'Int32Bias',
    'RemoveInputInit',
    'SimplifyModel',
    'EnableSubgraph',
    'ForceQuantizeNoInputCheck',
    'MatMulConstBOnly',
    'AddQDQPairToWeight',
    'OpTypesToExcludeOutputQuantization',
    'DedicatedQDQPair',
    'QDQOpTypePerChannelSupportToAxis',
    'UseQDQVitisCustomOps',
    'CalibTensorRangeSymmetric',
    'CalibMovingAverage',
    'CalibMovingAverageConstant',
    'Percentile',
    'RandomDataReaderInputDataRange',
    'Int16Scale',
    'MinMSEMode',
    'ConvertBNToConv',
    'ConvertReduceMeanToGlobalAvgPool',
    'SplitLargeKernelPool',
    'ConvertSplitToSlice',
    'FuseInstanceNorm',
    'FuseL2Norm',
    'FuseLayerNorm',
    'ConvertClipToRelu',
    'SimulateDPU',
    'ConvertLeakyReluToDPUVersion',
    'ConvertSigmoidToHardSigmoid',
    'ConvertHardSigmoidToDPUVersion',
    'ConvertAvgPoolToDPUVersion',
    'ConvertReduceMeanToDPUVersion',
    'ConvertSoftmaxToDPUVersion',
    'NPULimitationCheck',
    'AdjustShiftCut',
    'AdjustShiftBias',
    'AdjustShiftRead',
    'AdjustShiftWrite',
    'AdjustHardSigmoid',
    'AdjustShiftSwish',
    'AlignConcat',
    'AlignPool',
    'AlignPad',
    'AlignSlice',
    'AlignEltwiseQuantType',
    'ReplaceClip6Relu',
    'CLESteps',
    'CLETotalLayerDiffThreshold',
    'CLEScaleAppendBias',
    'FastFinetune',
    'SmoothAlpha',
    'RemoveQDQConvRelu',
    'RemoveQDQConvLeakyRelu',
    'RemoveQDQConvPRelu',
    'RemoveQDQInstanceNorm',
    'FoldBatchNorm',
    'FixShapes',
    'MixedPrecisionTensor',
    'AutoMixprecision',
    'FoldRelu',
    'CalibDataSize',
    'SaveTensorHistFig',
    'WeightsOnly',
    'BFPAttributes',
]
Level3_config_keys = [
    'DataSize', 'FixedSeed', 'BatchSize', 'NumIterations', 'LearningRate', 'OptimAlgorithm', 'OptimDevice', 'EarlyStop',
    'FixedSeed', 'NumBatches', 'LRAdjust', 'TargetOpType', 'SelectiveUpdate', 'UpdataBias', 'OutputQDQ', 'DropRatio',
    'LogPeriod'
]


def l2_metric(base_input: Any, ref_input: Any) -> Any:
    """
    Calculate the L2 metric between baseline and reference inputs.

    Args:
        base_input: Baseline input as a numpy array of float32.
        ref_input: Reference input as a numpy array of float32.

    Returns:
        The L2 metric as a float32 value.

    Note:
        Only np.ndarray datatype is accepted as input.
    """
    return np.mean(np.square(base_input - ref_input)).astype(float)


def l1_metric(base_input: Any, ref_input: Any) -> Any:
    """
    Calculate the L1 metric between baseline and reference inputs.

    Args:
        base_input: Baseline input as a numpy array of float32.
        ref_input: Reference input as a numpy array of float32.

    Returns:
        The L1 metric as a float32 value.

    Note:
        Only np.ndarray datatype is accepted as input.
    """
    return np.mean(np.abs(base_input - ref_input)).astype(float)


def cos_metric(base_input: Any, ref_input: Any) -> Any:
    """
    Calculate the cosine metric between baseline and reference inputs.

    Args:
        base_input: Baseline input as a numpy array of float32.
        ref_input: Reference input as a numpy array of float32.

    Returns:
        The cosine metric as a float32 value. Value range: [0.0, 1.0]

    Note:
        Only np.ndarray datatype is accepted as input.
    """
    v1 = base_input.reshape(-1)
    v2 = ref_input.reshape(-1)
    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def psnr_metric(base_input: Any, ref_input: Any) -> Any:
    """
    Calculate the psnr metric between baseline and reference inputs.

    Args:
        base_input: Baseline input as a numpy array of float32.
        ref_input: Reference input as a numpy array of float32.

    Returns:
        The psnr metric as a float32 value.

    Note:
        Only np.ndarray datatype is accepted as input.
    """
    mse_value = np.mean((base_input / 1.0 - ref_input / 1.0)**2)
    if mse_value < 1e-10:
        return 100
    psnr_value = 20 * np.log10(255 / np.sqrt(mse_value))
    return psnr_value


def ssim_metric(base_input: Any, ref_input: Any) -> Any:
    """
    Calculate the ssim metric between baseline and reference inputs.

    Args:
        base_input: Baseline input as a numpy array of float32.
        ref_input: Reference input as a numpy array of float32.

    Returns:
        The ssim metric as a float32 value.

    Note:
        Only np.ndarray datatype is accepted as input.
    """
    y_true = ref_input
    y_pred = base_input
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_ture = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_ture)
    std_pred = np.sqrt(var_pred)
    R = 255
    c1 = np.square(0.01 * R)
    c2 = np.square(0.03 * R)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true**2 + u_pred**2 + c1) * (var_pred + var_ture + c2)
    return ssim / denom


def split_config_levels(input_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    raw_config = copy.deepcopy(input_config)
    temp_level1_space_config: Dict[str, Any] = {}
    temp_level2_space_config: Dict[str, Any] = {}
    temp_level3_space_config: Dict[str, Any] = {}
    #  search space split into 3 levels
    if 'extra_options' in raw_config.keys():
        if 'FastFinetune' in raw_config['extra_options'].keys():
            temp_level3_space_config = copy.deepcopy(raw_config['extra_options']['FastFinetune'])
            del raw_config['extra_options']['FastFinetune']
            if len(raw_config['extra_options']) > 0:
                temp_level2_space_config = copy.deepcopy(raw_config['extra_options'])
            del raw_config['extra_options']
            temp_level1_space_config = copy.deepcopy(raw_config)
        else:
            # judge is the dict is empty
            if len(raw_config['extra_options']) > 0:
                temp_level2_space_config = copy.deepcopy(raw_config['extra_options'])
                del raw_config['extra_options']
            temp_level1_space_config = copy.deepcopy(raw_config)
    else:
        temp_level1_space_config = copy.deepcopy(raw_config)
    return temp_level1_space_config, temp_level2_space_config, temp_level3_space_config


def buildin_eval_func(onnx_path: str, data_reader: Any, save_path: str = '', save_prefix: str = "iter_x_") -> str:
    """
    Buildin evalation function using data_reader

    Args:
        onnx_path: onnx model path that will excute evalution, it can be  either float porint or quantized onnx model
        data_reader: user defined data_reader
        save_path: path used to save the output result
        save_prefix: prefix string used to name the saved output

    Note: Data reader here should be defined as dataloader.Because the raw data reader is iterator, it's
           not convient for evaluation.
    """
    # TODO check the onnx
    if 'ROCMExecutionProvider' in ort.get_available_providers():
        device = 'ROCM'
        providers = ['ROCMExecutionProvider']
    elif 'CUDAExecutionProvider' in ort.get_available_providers():
        device = 'CUDA'
        providers = ['CUDAExecutionProvider']
    else:
        device = 'CPU'
        providers = ['CPUExecutionProvider']
    so = ort.SessionOptions()
    so.register_custom_ops_library(get_library_path(device))
    ort_session = ort.InferenceSession(onnx_path, so, providers=providers)

    for excute_idx, data in enumerate(data_reader):
        output = ort_session.run(None, data)
        excute_idx += 1
        if save_path is not None:
            temp_save_path = os.path.join(save_path, str(save_prefix) + str(excute_idx) + ".npy")
            output = np.concatenate([item.reshape(-1) for item in output])
            np.save(temp_save_path, output)
    return save_path


def logger_config(log_path: str = './auto_search.log', logging_name: str = 'auto search') -> logging.Logger:
    logger = logging.getLogger(logging_name)

    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


METRICS = {
    "L2": l2_metric,
    "L1": l1_metric,
    "cos": cos_metric,
    "psnr": psnr_metric,
    "ssim": ssim_metric,
}


class AutoSearchConfig:
    search_space: Dict[str, Any] = {
        "calibrate_method": [
            PowerOfTwoMethod.MinMSE, PowerOfTwoMethod.NonOverflow, CalibrationMethod.MinMax, CalibrationMethod.Entropy,
            CalibrationMethod.Percentile
        ],
        "activation_type": [QuantType.QInt8, QuantType.QInt16],
        "weight_type": [QuantType.QInt8, QuantType.QInt16],
        "include_cle": [True, False],
        "include_auto_mp": [False, True],
        "include_fast_ft": [False, True],
        "include_sq": [False, True],
        "extra_options": {
            "ActivationSymmetric": [True, False],
            "WeightSymmetric": [True, False],
            "CalibMovingAverage": [True, False],
            "CalibMovingAverageConstant": [0.01, 0.001],
            "Percentile": [99.99, 99.999],
            "SmoothAlpha": [0.5, 0.6],
            'FastFinetune': {
                'DataSize': [500, 1000],
                'NumIterations': [100, 1000],
                'OptimAlgorithm': ['adaround', 'adaquant'],
                'LearningRate': [0.01, 0.001, 0.0001],
            }
        }
    }
    search_metric: str = "L2"
    search_algo: str = "grid_search"  # candidates: "grid_search", "random"
    search_evaluator = None
    search_metric_tolerance: float = 2.00
    search_cache_dir: str = "./"
    search_output_dir: str = "./"
    search_log_path: str = "./auto_search.log"

    search_stop_condition: dict[str, Any] = {
        "find_n_candidates": -1,
        "find_best_candidate": -1,
        "iteration_limit": 10000,
        "time_limit": 36000,  # unit: second
    }


class AssembleIdxs():
    """
    List all the combination of one list.
    Example:
            input_idxs: [[1,2,], [3,4]]
            output: [[1,3], [1,4], [2,3], [2,4]]

    Args:
        values_idxs

    Note:
        Only list the item in the input[i] list.
    """

    def __init__(self, values_idxs: Any) -> None:
        """
        Initilize the input idxs for listing combination.

    Args:
        values_idxs: input list.
    """
        self.values_idxs = values_idxs
        self.values_length: int = len(values_idxs)
        self.result: list[Union[int, list[int]]] = []

    def search_forward(self, item_forward: list[Any]) -> None:
        """
        Recresively find the next item until the last one.

        Args:
            item_forward: searched item collection.
        """
        item_forward_length = len(item_forward)
        if item_forward_length == self.values_length:
            self.result.append(copy.deepcopy(item_forward))
        else:
            for item in self.values_idxs[item_forward_length]:
                item_forward.append(item)
                self.search_forward(item_forward)
                item_forward.pop()

    def run(self, ) -> list[Union[int, list[int]]]:
        """
         Excute the assemble process and return the result
        """
        self.search_forward([])
        return self.result


class SearchSpace():
    """
    Build the all possible search space from the input.
    # TODO remove the invalid config generated by the search space
    # TODO give the config priority
    # TODO validate the space dict right

    Args:
        config: config which includes the search space defined by the list

    Note:
        Because the search space is in difference levels, so we need to split the level of the search space.
    """

    def __init__(self, conf: Dict[str, Any]) -> None:
        self.raw_search_space: dict[str, Any] = conf

        self.all_configs: list[dict[str, str]] = []
        self.all_spaces: list[Any] = []

        self.level1_space: list[Any] = []
        self.level2_space: list[Any] = []
        self.level3_space: list[Any] = []
        self.level1_space_config: Dict[str, Any] = {}
        self.level2_space_config: Dict[str, Any] = {}
        self.level3_space_config: Dict[str, Any] = {}

        #  search space split into 3 levels
        self.level1_space_config, self.level2_space_config, self.level3_space_config = split_config_levels(
            self.raw_search_space)

        # validate the config
        if len(self.level1_space_config) > 0:
            for level1_key in self.level1_space_config.keys():
                if level1_key not in Level1_config_keys:
                    print(
                        f"{level1_key} is not supported yet or your input config is wrong! Please check it! And this setting will be ignored!"
                    )
        if len(self.level2_space_config) > 0:
            for level2_key in self.level2_space_config.keys():
                if level2_key not in Level2_config_keys:
                    print(
                        f"{level2_key} is not supported yet or your input config is wrong! Please check it! And this setting will be ignored!"
                    )
        if len(self.level3_space_config) > 0:
            for level3_key in self.level3_space_config.keys():
                if level3_key not in Level3_config_keys:
                    print(
                        f"{level3_key} is not supported yet or your input config is wrong! Please check it! And this setting will be ignored!"
                    )

    def build_search_space(self, space_dict: Dict[str, Any]) -> list[Union[int, list[int]]]:
        values = space_dict.values()
        values_lengths = [[i for i in range(len(item))] for item in values]
        assemble_idxs_ins = AssembleIdxs(values_lengths)
        space_result = assemble_idxs_ins.run()
        del assemble_idxs_ins
        return space_result

    def three_level_spaces(self, ) -> list[Any]:
        """
        According to the user defined search space, list all the possible configs.
        There several situation we need to tell it apart(splited spaces):
        leve11 + level2
        level1 + level3
        level1 + level3
        level1
        """

        # level1 + level2
        if (self.level1_space_config != {}) and (self.level2_space_config != {}) and \
                (self.level3_space_config == {}):
            self.level1_space = self.build_search_space(self.level1_space_config)
            self.level2_space = self.build_search_space(self.level2_space_config)
            simple_joint_space = [self.level1_space, self.level2_space]
            assemble_idxs_ins = AssembleIdxs(simple_joint_space)
            space_result = assemble_idxs_ins.run()
            del assemble_idxs_ins
            self.all_spaces = copy.deepcopy(space_result)
        # level1 + level3
        elif (self.level1_space_config != {}) and (self.level2_space_config == {}) and \
                (self.level3_space_config != {}):
            self.level1_space = self.build_search_space(self.level1_space_config)
            self.level3_space = self.build_search_space(self.level3_space_config)
            simple_joint_space = [self.level1_space, self.level3_space]
            assemble_idxs_ins = AssembleIdxs(simple_joint_space)
            space_result = assemble_idxs_ins.run()
            del assemble_idxs_ins
            self.all_spaces = copy.deepcopy(space_result)
        # level1 + level2 + level3
        elif (self.level1_space_config != {}) and (self.level2_space_config != {}) and \
                (self.level3_space_config != {}):
            self.level1_space = self.build_search_space(self.level1_space_config)
            self.level2_space = self.build_search_space(self.level2_space_config)
            self.level3_space = self.build_search_space(self.level3_space_config)
            simple_joint_space = [self.level1_space, self.level2_space, self.level3_space]
            assemble_idxs_ins = AssembleIdxs(simple_joint_space)
            space_result = assemble_idxs_ins.run()
            del assemble_idxs_ins
            self.all_spaces = copy.deepcopy(space_result)
        # level1
        elif (self.level1_space_config != {}) and (self.level2_space_config == {}) and \
                (self.level3_space_config == {}):
            self.level1_space = self.build_search_space(self.level1_space_config)
            self.all_spaces = self.level1_space

        return self.all_spaces

    def get_all_configs(self, ) -> list[Any]:
        # make all space to into configs
        self.three_level_spaces()
        # level1 + level2 serch space
        if (self.level1_space_config != {}) and (self.level2_space_config != {}) and \
                (self.level3_space_config == {}):

            level1_keys = list(self.level1_space_config.keys())
            level2_keys = list(self.level2_space_config.keys())
            for one_space in self.all_spaces:
                temp_level1_space = one_space[0]
                temp_level2_space = one_space[1]
                temp_level1_vals = [
                    self.level1_space_config[level1_keys[level1_idx]][temp_level1_space[level1_idx]]
                    for level1_idx in range(len(temp_level1_space))
                ]
                temp_level1_config = {key_item: val_item for key_item, val_item in zip(level1_keys, temp_level1_vals)}
                temp_level2_vals = [
                    self.level2_space_config[level2_keys[level2_idx]][temp_level2_space[level2_idx]]
                    for level2_idx in range(len(temp_level2_space))
                ]
                temp_level2_config = {key_item: val_item for key_item, val_item in zip(level2_keys, temp_level2_vals)}

                temp_level1_config['extra_options'] = temp_level2_config
                self.all_configs.append(copy.deepcopy(temp_level1_config))
        # level1 + level3 search space
        elif (self.level1_space_config != {}) and (self.level2_space_config == {}) and \
                (self.level3_space_config != {}):
            level1_keys = list(self.level1_space_config.keys())
            level3_keys = list(self.level3_space_config.keys())
            for one_space in self.all_spaces:
                temp_level1_space = one_space[0]
                temp_level3_space = one_space[1]
                temp_level1_vals = [
                    self.level1_space_config[level1_keys[level1_idx]][temp_level1_space[level1_idx]]
                    for level1_idx in range(len(temp_level1_space))
                ]
                temp_level1_config = {key_item: val_item for key_item, val_item in zip(level1_keys, temp_level1_vals)}
                temp_level3_vals = [
                    self.level3_space_config[level3_keys[level3_idx]][temp_level3_space[level3_idx]]
                    for level3_idx in range(len(temp_level3_space))
                ]
                temp_level3_config = {key_item: val_item for key_item, val_item in zip(level3_keys, temp_level3_vals)}

                temp_level1_config['extra_options'] = {"FastFinetune": temp_level3_config}
                self.all_configs.append(copy.deepcopy(temp_level1_config))
        # level1 + level2 + level3 search space
        elif (self.level1_space_config != {}) and (self.level2_space_config != {}) and \
                (self.level2_space_config != {}):
            level1_keys = list(self.level1_space_config.keys())
            level2_keys = list(self.level2_space_config.keys())
            level3_keys = list(self.level3_space_config.keys())
            for one_space in self.all_spaces:
                temp_level1_space = one_space[0]
                temp_level2_space = one_space[1]
                temp_level3_space = one_space[2]
                temp_level1_vals = [
                    self.level1_space_config[level1_keys[level1_idx]][temp_level1_space[level1_idx]]
                    for level1_idx in range(len(temp_level1_space))
                ]
                temp_level1_config = {key_item: val_item for key_item, val_item in zip(level1_keys, temp_level1_vals)}

                temp_level2_vals = [
                    self.level2_space_config[level2_keys[level2_idx]][temp_level2_space[level2_idx]]
                    for level2_idx in range(len(temp_level2_space))
                ]
                temp_level2_config = {key_item: val_item for key_item, val_item in zip(level2_keys, temp_level2_vals)}

                temp_level3_vals = [
                    self.level3_space_config[level3_keys[level3_idx]][temp_level3_space[level3_idx]]
                    for level3_idx in range(len(temp_level3_space))
                ]
                temp_level3_config = {key_item: val_item for key_item, val_item in zip(level3_keys, temp_level3_vals)}

                temp_level2_config['FastFinetune'] = temp_level3_config
                temp_level1_config['extra_options'] = temp_level2_config
                self.all_configs.append(copy.deepcopy(temp_level1_config))
        # level1 search space
        elif (self.level1_space_config != {}) and (self.level2_space_config == {}) and \
                (self.level2_space_config == {}):
            level1_keys = list(self.level1_space_config.keys())

            for one_space in self.all_spaces:
                temp_level1_space = one_space
                temp_level1_vals = [
                    self.level1_space_config[level1_keys[level1_idx]][temp_level1_space[level1_idx]]
                    for level1_idx in range(len(temp_level1_space))
                ]
                temp_level1_config = {key_item: val_item for key_item, val_item in zip(level1_keys, temp_level1_vals)}

                # print(temp_level1_config)
                self.all_configs.append(copy.deepcopy(temp_level1_config))
        else:
            print("wrong config setting!")

        return self.all_configs

    def verify_one_config_base(self, verify_item: str, base_item: str, base_config: Dict[str, Any],
                               verify_standard: Any) -> bool:
        valid_flag = True
        if base_item not in base_config.keys():
            valid_flag = False
            print(f"Please set {base_item} to be {verify_standard}, when you use {verify_item}! ")
        elif base_item in base_config.keys() and base_config[base_item] != verify_standard:
            valid_flag = False
            print(f"Please set {base_item} to be {verify_standard}, when you use {verify_item}! ")
        return valid_flag

    def verify_one_config(self, verify_item: str, verify_config: Dict[str, Any], base_item: str,
                          base_config: Dict[str, Any], verify_standard: Any) -> Dict[str, Any]:
        if verify_item in verify_config.keys():
            if base_item not in base_config.keys():
                print(f"Please set {base_item} to be {verify_standard}, when you use {verify_item}! ")
                del verify_config[verify_item]
            elif base_item in base_config.keys() and base_config[base_item] != verify_standard:
                print(f"Please set {base_item} to be {verify_standard}, when you use {verify_item}! ")
                del verify_config[verify_item]
        return verify_config

    def remove_invalid_configs(self, input_all_coinfgs: list[Dict[str, Any]]) -> list[Any]:
        valid_configs = []
        for _, item_config in enumerate(input_all_coinfgs):
            item_config_l1, item_config_l2, item_config_l3 = split_config_levels(item_config)
            # level1 pass
            # level1 + level2
            if len(item_config_l1) > 0 and len(item_config_l2) > 0 and len(item_config_l3) == 0:
                # verify "CalibMovingAverage" and "CalibMovingAverageConstant"
                item_config_l2 = self.verify_one_config("CalibMovingAverage", item_config_l2, "calibrate_method",
                                                        item_config_l1, CalibrationMethod.MinMax)
                item_config_l2 = self.verify_one_config("CalibMovingAverageConstant", item_config_l2,
                                                        "calibrate_method", item_config_l1, CalibrationMethod.MinMax)
                item_config_l2 = self.verify_one_config("CalibMovingAverageConstant", item_config_l2,
                                                        "CalibMovingAverageConstant", item_config_l2, True)
                # verify "Percentile"
                item_config_l2 = self.verify_one_config('Percentile', item_config_l2, 'calibrate_method',
                                                        item_config_l1, CalibrationMethod.Percentile)
                # verify "SmoothAlpha"
                item_config_l2 = self.verify_one_config("SmoothAlpha", item_config_l2, "include_sq", item_config_l1,
                                                        True)
                item_config_l1['extra_options'] = item_config_l2
            # level1 + level3
            elif len(item_config_l1) > 0 and len(item_config_l2) == 0 and len(item_config_l3) > 0:
                # verify FastFinetune
                item_config_l3_flag = self.verify_one_config_base("FastFinetune", "include_fast_ft", item_config_l1,
                                                                  True)
                if item_config_l3_flag:
                    item_config_l1['extra_options'] = {'FastFinetune': item_config_l3}
            # level1 + level2 + level3
            elif len(item_config_l1) > 0 and len(item_config_l2) > 0 and len(item_config_l3) > 0:
                # verify "CalibMovingAverage" and "CalibMovingAverageConstant"
                item_config_l2 = self.verify_one_config("CalibMovingAverage", item_config_l2, "calibrate_method",
                                                        item_config_l1, CalibrationMethod.MinMax)
                item_config_l2 = self.verify_one_config("CalibMovingAverageConstant", item_config_l2,
                                                        "calibrate_method", item_config_l1, CalibrationMethod.MinMax)
                item_config_l2 = self.verify_one_config("CalibMovingAverageConstant", item_config_l2,
                                                        "CalibMovingAverageConstant", item_config_l2, True)
                # verify "Percentile"
                item_config_l2 = self.verify_one_config('Percentile', item_config_l2, 'calibrate_method',
                                                        item_config_l1, CalibrationMethod.Percentile)
                # verify "SmoothAlpha"
                item_config_l2 = self.verify_one_config("SmoothAlpha", item_config_l2, "include_sq", item_config_l1,
                                                        True)
                # verify FastFinetune
                item_config_l3_flag = self.verify_one_config_base("FastFinetune", "include_fast_ft", item_config_l1,
                                                                  True)

                if len(item_config_l2) > 0:
                    item_config_l1['extra_options'] = item_config_l2
                if len(item_config_l2) > 0 and item_config_l3_flag:
                    item_config_l1['extra_options']['FastFinetune'] = item_config_l3
                elif len(item_config_l2) == 0 and item_config_l3_flag:
                    item_config_l1['extra_options'] = {'FastFinetune': item_config_l3}

            if item_config_l1 not in valid_configs:
                valid_configs.append(item_config_l1)

        print(f"The invalid configs ratio is {len(self.all_configs) - len(valid_configs)} / {len(self.all_configs)}")

        return valid_configs


class AutoSearch():

    def __init__(self,
                 config: Config,
                 auto_search_config: AutoSearchConfig,
                 model_input: str,
                 model_output: str,
                 eval_dataloader: Any = None,
                 calibration_data_reader: Union[CalibrationDataReader, None, Any] = None,
                 calibration_data_path: Optional[str] = None) -> None:
        # basic settings
        self.config = config
        self.config_backup = copy.deepcopy(config)
        self.auto_search_config = auto_search_config
        self.model_input = model_input
        self.model_output = model_output
        self.calibration_data_reader = calibration_data_reader
        self.calibration_data_path = calibration_data_path
        self.eval_dataloader = eval_dataloader if eval_dataloader is not None else calibration_data_reader
        self.searched_space: Dict[int, float] = {}

        self.cache_dir = self.auto_search_config.search_cache_dir
        self.output_dir = self.auto_search_config.search_output_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.logger = logger_config(os.path.join(self.auto_search_config.search_output_dir, "auto_search.log"),
                                    logging_name="auto_search log")

        if self.auto_search_config.search_evaluator is None:
            metric_str = self.auto_search_config.search_metric
            if metric_str in METRICS.keys():
                self.metric = METRICS[self.auto_search_config.search_metric]
            else:
                self.metric = METRICS["L2"]
                self.logger.error(
                    f"search metric:{metric_str} is not supported yet! you can change your metric setting or set it in user defined evaluator!"
                )

        self.evaluator = self.build_evaluator()
        self.quantizer = self.build_quantize_instance()
        self.all_configs = self.build_all_configs(search_space_config=self.auto_search_config.search_space)

        # search stop conditions initilization
        self.iterations = 0
        self.time_consumed = 0.0
        self.candidates: list[Dict[str, Any]] = []
        self.best_candidate = None

    def build_evaluator(self, ) -> Any:
        if self.auto_search_config.search_evaluator is not None:
            return self.auto_search_config.search_evaluator
        else:
            # using self.calibration_data_reader to inference float and quantized model, and cal the L2 distance
            return buildin_eval_func

    def build_quantize_instance(self, ) -> ModelQuantizer:
        return ModelQuantizer(self.config)

    def build_all_configs(self, search_space_config: Dict[str, Any]) -> list[Any]:
        search_space = SearchSpace(search_space_config)
        all_configs = search_space.get_all_configs()
        del search_space
        return all_configs

    def sampler(self, input_idxs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if self.auto_search_config.search_algo == "grid_search":
            return input_idxs
        elif self.auto_search_config.search_algo == "random":
            return random.sample(input_idxs, len(input_idxs))
        else:
            self.logger.warning(
                f"{self.auto_search_config.search_algo} is not supported yet! return grid_search method!")
            return input_idxs

    def runner(self, one_config: Dict[str, Any], data_reader: Any) -> None:

        # step 1: set the search config to the self.config
        temp_level1_config, temp_level2_config, temp_level3_config = split_config_levels(one_config)
        if (temp_level1_config != {}) and (temp_level2_config != {}) and \
                (temp_level3_config == {}):
            level1_keys = list(temp_level1_config.keys())
            level2_keys = list(temp_level2_config.keys())
            # set level2 config
            for temp_level2_key in level2_keys:
                self.config.global_quant_config.extra_options[temp_level2_key] = temp_level2_config[temp_level2_key]
            # set level1 config
            for temp_level1_key in level1_keys:
                self.config.global_quant_config.__setattr__(temp_level1_key, temp_level1_config[temp_level1_key])
        elif (temp_level1_config != {}) and (temp_level2_config == {}) and \
                (temp_level3_config != {}):
            level1_keys = list(temp_level1_config.keys())
            # set level1 config
            for temp_level1_key in level1_keys:
                self.config.global_quant_config.__setattr__(temp_level1_key, temp_level1_config[temp_level1_key])
            # set level3 config
            self.config.global_quant_config.extra_options = {"FastFinetune": temp_level3_config}

        elif (temp_level1_config != {}) and (temp_level2_config != {}) and \
                (temp_level3_config != {}):
            level1_keys = list(temp_level1_config.keys())
            level2_keys = list(temp_level2_config.keys())
            # set level2 config
            for temp_level2_key in level2_keys:
                self.config.global_quant_config.extra_options[temp_level2_key] = temp_level2_config[temp_level2_key]
            # set level1 config
            all_config_atrrs = dir(self.config.global_quant_config)
            for temp_level1_key in level1_keys:
                if temp_level1_key in all_config_atrrs:
                    self.config.global_quant_config.__setattr__(temp_level1_key, temp_level1_config[temp_level1_key])
            # set level3 config
            self.config.global_quant_config.extra_options["FastFinetune"] = temp_level3_config
        elif (temp_level1_config != {}) and (temp_level2_config == {}) and \
                (temp_level3_config == {}):
            all_config_atrrs = dir(self.config.global_quant_config)
            level1_keys = list(temp_level1_config.keys())
            for temp_level1_key in level1_keys:
                if temp_level1_key in all_config_atrrs:
                    self.config.global_quant_config.__setattr__(temp_level1_key, temp_level1_config[temp_level1_key])

        else:
            self.logger.error("Your search space if invalid!")

        # step2: change the config and run
        quantize_start_time = time.time()
        self.quantizer.config = self.config.global_quant_config
        temp_output_path = os.path.join(self.cache_dir, f"iter_{self.iterations}.onnx")
        self.quantizer.quantize_model(self.model_input, temp_output_path, data_reader)
        quantize_end_time = time.time()

        quantize_time_consumed = quantize_end_time - quantize_start_time
        self.time_consumed += quantize_time_consumed

        # step 3: evalute the quantized model and decide if remove it and log it
        if self.auto_search_config.search_evaluator is None:
            prefix = "model_output"
            quantized_model_res_path = os.path.join(self.cache_dir, "quantized_" + prefix)
            if not os.path.exists(quantized_model_res_path):
                os.makedirs(quantized_model_res_path)
            self.quantized_model_output = self.evaluator(temp_output_path,
                                                         self.eval_dataloader,
                                                         save_path=quantized_model_res_path,
                                                         save_prefix=prefix)
        else:
            self.quantized_model_output = self.evaluator(temp_output_path)

        self.logger.info(f"config_index:{self.iterations}")
        self.logger.info(f"config: {one_config}")
        self.logger.info(f"quantized time consumed:{quantize_time_consumed}s")

        # step 4: calculate the metric difference
        if self.auto_search_config.search_evaluator is None:
            diffs = []
            fp_output_files = os.listdir(self.base_model_output)
            for file in fp_output_files:
                fp_file = os.path.join(self.base_model_output, file)
                quantized_file = os.path.join(self.quantized_model_output, file)
                fp_output = np.load(fp_file)
                quantized_output = np.load(quantized_file)
                diff_item = self.metric(fp_output, quantized_output)
                diffs.append(diff_item)
            # remove the temporay quantized output
            shutil.rmtree(self.quantized_model_output)
            diff = np.mean(diffs)
            metric_name = self.auto_search_config.search_metric if self.auto_search_config.search_metric is not None else "L2"
        else:
            diff = self.base_model_output - self.quantized_model_output
            metric_name = "customer definded"

        self.logger.info(
            f"{metric_name} distance is:{diff}, with tolerance:{self.auto_search_config.search_metric_tolerance}")
        self.searched_space[self.iterations] = diff

        # step 5: judge the tolerance
        if diff <= self.auto_search_config.search_metric_tolerance:
            save_output_path = os.path.join(self.output_dir, f"iter_{self.iterations}.onnx")
            shutil.move(temp_output_path, save_output_path)
            temp_one_candidate = {
                "config_index": self.iterations,
                "difference": diff,
                "model_path": save_output_path,
                "config": one_config
            }
            self.candidates.append(temp_one_candidate)
        else:
            os.remove(temp_output_path)

        # step 6: rebase the config
        self.config = self.config_backup

    def search_model(self, ) -> list[Dict[str, Any]]:
        # step 1: get all configs
        all_configs = self.all_configs

        #  step 2: get the baseline metric
        if self.auto_search_config.search_evaluator is None:
            prefix = "model_output"
            fp_model_res_path = os.path.join(self.cache_dir, "fp_" + prefix)
            if not os.path.exists(fp_model_res_path):
                os.makedirs(fp_model_res_path)
            self.base_model_output = self.evaluator(self.model_input,
                                                    self.eval_dataloader,
                                                    save_path=fp_model_res_path,
                                                    save_prefix=prefix)
        else:
            self.base_model_output = self.evaluator(self.model_input)

        # step 3: use sampler to sample the configs
        all_configs = self.sampler(all_configs)

        stop_flag = False

        # stop condiation settings
        if "find_n_candidates" in self.auto_search_config.search_stop_condition.keys():
            find_n_candidates = self.auto_search_config.search_stop_condition["find_n_candidates"]
        else:
            find_n_candidates = -1
        if "iteration_limit" in self.auto_search_config.search_stop_condition.keys():
            iteration_limit = self.auto_search_config.search_stop_condition["iteration_limit"]
        else:
            iteration_limit = -1
        if "time_limit" in self.auto_search_config.search_stop_condition.keys():
            time_limit = self.auto_search_config.search_stop_condition["time_limit"]

        while not stop_flag:
            # set the initial data reader and run the config
            data_reader_temp = copy.deepcopy(self.calibration_data_reader)
            self.runner(all_configs[self.iterations], data_reader=data_reader_temp)
            self.iterations += 1

            # stop condition list
            if self.iterations == len(all_configs):
                stop_flag = True
            if find_n_candidates != -1 and len(self.candidates) >= find_n_candidates:
                stop_flag = True
            if iteration_limit != -1 and self.iterations >= iteration_limit:
                stop_flag = True
            if time_limit != -1 and self.time_consumed >= time_limit:
                stop_flag = True

        if self.auto_search_config.search_evaluator is None:
            shutil.rmtree(fp_model_res_path)

        self.logger.info("----------------------------------------------------------------")
        self.logger.info("--------------------Sorted searhed space------------------------")
        self.logger.info("----------------------------------------------------------------")
        sorted_diffs = sorted(self.searched_space.items(), key=lambda kv: (kv[1], kv[0]))
        if self.auto_search_config.search_evaluator is None:
            metric_name = self.auto_search_config.search_metric if self.auto_search_config.search_metric is not None else "L2"
        else:
            metric_name = "customer definded"
        for item in sorted_diffs:
            self.logger.info(f"config idex:{item[0]}, {metric_name} distance:{item[1]}")
        self.logger.info("----------------------------------------------------------------")
        self.logger.info("----------All candidates meeting the target---------------------")
        self.logger.info("----------------------------------------------------------------")
        sorted_iter_nums = [sorted_diffs[i][0] for i in range(len(sorted_diffs))]
        candidates_nums = {}
        for j in range(len(self.candidates)):
            temp_key = self.candidates[j]["config_index"]
            candidates_nums[temp_key] = j
        if len(candidates_nums) == 0:
            self.logger.info(
                "There is no config that meet the tolerance! You can choose a good one from the searched space or you can reset the search space to search aggin!"
            )
        else:
            for i in range(len(sorted_iter_nums)):
                if sorted_iter_nums[i] in list(candidates_nums.keys()):
                    self.logger.info(f"{self.candidates[candidates_nums[sorted_iter_nums[i]]]}")
        return self.candidates
