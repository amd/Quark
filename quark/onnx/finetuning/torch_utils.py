#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Union, Tuple, List, Dict, Any, Optional
from numpy.typing import NDArray
import numpy
import random

import torch
from torch.utils.data import Dataset

import onnx
from onnxruntime.quantization import CalibrationDataReader

from .create_torch.create_model import TorchModel
from .train_torch.train_model import ModelOptimizer
from .train_torch.train_model_param import TrainParameters

from quark.shares.utils.log import ScreenLogger, log_errors

logger = ScreenLogger(__name__)


def setup_seed(seed: int) -> None:
    """
    Set the seed for random functions
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_onnx_to_torch(
    onnx_model: onnx.ModelProto,
    float_weight: Optional[NDArray[Any]] = None,
    float_bias: Optional[NDArray[Any]] = None,
) -> torch.nn.Module:
    """
    Convert a onnx model to torch module. Since the onnx model is always a quantized one,
    which has a folded QuantizeLinear in the weight tensor's QDQ.
    In order to obtain the original float weight without loss for the quantize wrapper,
    an additional float weight needs to be feed in.
    :param onnx_model: instance of onnx model
    :param float_weight: float weight
    :param float_bias: float bias
    :return: a torch nn.Module instance
    """

    torch_model = TorchModel(onnx_model)

    if float_weight is not None:
        torch_model.set_weight(float_weight)

    if float_bias is not None:
        torch_model.set_bias(float_bias)

    return torch_model


def convert_torch_to_onnx(torch_model: torch.nn.Module, input_data: Union[NDArray[Any],
                                                                          List[NDArray[Any]]]) -> onnx.ModelProto:
    """
    Convert a torch model to onnx model, do not support models bigger than 2GB
    :param torch_model: instance of torch model
    :param input_data: numpy array for single input or list for multiple inputs
    :return: the onnx model instance
    """
    import tempfile
    from pathlib import Path

    if isinstance(input_data, (list, tuple)):
        dummy_input: Union[torch.Tensor, Tuple[Any, ...]] = ()
        for data in input_data:
            dummy_input = dummy_input + (torch.tensor(data), )
    else:
        dummy_input = torch.tensor(input_data)

    torch_model = torch_model.to(torch.device("cpu"))

    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        model_path = Path(quant_tmp_dir).joinpath("exported.onnx").as_posix()
        torch.onnx.export(torch_model,
                          dummy_input,
                          model_path,
                          verbose=False,
                          opset_version=13,
                          input_names=["inputs"],
                          output_names=["outputs"])
        onnx_model = onnx.load(model_path)

    return onnx_model


def save_torch_model(torch_model: torch.nn.Module,
                     model_path: str,
                     input_data: Union[None, NDArray[Any], List[NDArray[Any]]] = None) -> None:
    """
    Save a torch model to file
    :param torch_model: instance of torch model
    :param model path: the path to save
    :param input data: the input numpy array data for jit.trace
    """
    # torch.save(torch_model, torch_model_path)
    # torch.save(torch_model.state_dict(), model_path)

    if input_data is None:
        scripted_model = torch.jit.script(torch_model)
        torch.jit.save(scripted_model, model_path)
    else:
        if isinstance(input_data, (list, tuple)):
            dummy_input: Union[torch.Tensor, Tuple[Any, ...]] = ()
            for data in input_data:
                dummy_input = dummy_input + (torch.tensor(data), )
        else:
            dummy_input = torch.tensor(input_data)

        traced_model = torch.jit.trace(torch_model, dummy_input)
        torch.jit.save(traced_model, model_path)


class CachedDataset(Dataset):  # type: ignore
    """
    Cache data from calibration data reader of onnxruntime-based quantizer.
    """

    @log_errors
    def __init__(self, data_reader: CalibrationDataReader) -> None:
        """
        Cache data from calibration data reader, will have the same batch size.
        """

        self._data_cache: List[Any] = []

        while True:
            # This will be a dict: key is input name, value is numpy array
            input_dict = data_reader.get_next()
            if not input_dict:
                break

            # An element in the cache represents a mini-batch (may have multiple inputs)
            tensors: Tuple[Any, ...] = ()
            for arr in input_dict.values():
                tensors = tensors + (torch.tensor(arr), )
            self._data_cache.append(tensors)

        self._num_batches: int = len(self._data_cache)

        if self._num_batches == 0:
            raise ValueError("No data in the input data reader (self._num_batches=0). Please check the CachedDataset.")
        else:
            logger.info("Cached {} batches data from data reader".format(self._num_batches))

    def __len__(self) -> int:
        return self._num_batches

    def __getitem__(self, index: int) -> Any:
        return self._data_cache[index]


def parse_options_to_params(extra_options: Dict[str, Any]) -> TrainParameters:
    """
    Get train parameters from extra options
    """
    train_params = TrainParameters()

    if 'FastFinetune' not in extra_options:
        logger.warning("Not found extra options for FastFinetune, will use default parameters")
        return train_params
    elif not isinstance(extra_options['FastFinetune'], Dict):
        logger.warning(f"Invalid extra options {extra_options['FastFinetune']} for FastFinetune")
        return train_params

    if 'DataSize' in extra_options['FastFinetune']:
        train_params.data_size = extra_options['FastFinetune']['DataSize']
    if 'FixedSeed' in extra_options['FastFinetune']:
        train_params.fixed_seed = extra_options['FastFinetune']['FixedSeed']

    # For ordinary applications
    if 'BatchSize' in extra_options['FastFinetune']:
        train_params.batch_size = extra_options['FastFinetune']['BatchSize']
    if 'NumBatches' in extra_options['FastFinetune']:
        train_params.num_batches = extra_options['FastFinetune']['NumBatches']
    if 'NumIterations' in extra_options['FastFinetune']:
        train_params.num_iterations = extra_options['FastFinetune']['NumIterations']
    if 'LearningRate' in extra_options['FastFinetune']:
        train_params.lr = extra_options['FastFinetune']['LearningRate']
    if 'OptimAlgorithm' in extra_options['FastFinetune']:
        train_params.algorithm = extra_options['FastFinetune']['OptimAlgorithm'].lower()
    if 'OptimDevice' in extra_options['FastFinetune']:
        train_params.device = extra_options['FastFinetune']['OptimDevice'].lower()

    # For advanced applications
    if 'LRAdjust' in extra_options['FastFinetune']:
        train_params.lr_adjust = extra_options['FastFinetune']['LRAdjust']
    # if 'SelectiveUpdate' in extra_options['FastFinetune']:
    #    train_params.selective_update = extra_options['FastFinetune'][
    #        'SelectiveUpdate']
    if 'EarlyStop' in extra_options['FastFinetune']:
        train_params.early_stop = extra_options['FastFinetune']['EarlyStop']
    if 'UpdateBias' in extra_options['FastFinetune']:
        train_params.update_bias = extra_options['FastFinetune']['UpdateBias']
    if 'RegParam' in extra_options['FastFinetune']:
        train_params.reg_param = extra_options['FastFinetune']['RegParam']
    if 'BetaRange' in extra_options['FastFinetune']:
        train_params.beta_range = extra_options['FastFinetune']['BetaRange']
    if 'WarmStart' in extra_options['FastFinetune']:
        train_params.warm_start = extra_options['FastFinetune']['WarmStart']
    if 'DropRatio' in extra_options['FastFinetune']:
        train_params.drop_ratio = extra_options['FastFinetune']['DropRatio']

    if 'LogPeriod' in extra_options['FastFinetune']:
        train_params.log_period = extra_options['FastFinetune']['LogPeriod']
    else:
        train_params.log_period = train_params.num_iterations / 10

    # default lr for adaquant and adaround is different
    if train_params.algorithm == 'adaquant' and 'LearningRate' not in extra_options['FastFinetune']:
        train_params.lr = 0.00001
    if train_params.algorithm == 'adaround' and 'LearningRate' not in extra_options['FastFinetune']:
        train_params.lr = 0.1

    return train_params


def train_torch_module_api(quant_module: torch.nn.Module, inp_data_quant: Union[NDArray[Any], List[NDArray[Any]]],
                           inp_data_float: Union[NDArray[Any], List[NDArray[Any]]],
                           out_data_float: Union[NDArray[Any], List[NDArray[Any]]], extra_options: Any) -> Any:
    """
    Call torch training classes for adaround or adaquant
    """

    train_params = parse_options_to_params(extra_options)

    # If the user specified using GPU but no GPU is available, will downgrade to CPU
    if not torch.cuda.is_available() and not train_params.device.startswith('cpu'):
        logger.warning(f"The torch training will run on CPU instead of {train_params.device}")
        train_params.device = "cpu"

    ModelOptimizer.run(quant_module, inp_data_quant, inp_data_float, out_data_float, train_params)

    if train_params.algorithm == 'adaquant' and train_params.update_bias:
        return quant_module.get_weight(), quant_module.get_bias()
    else:
        return quant_module.get_weight(), None


def optimize_module(quant_model: onnx.ModelProto, float_weight: NDArray[Any], float_bias: Optional[NDArray[Any]],
                    inp_data_quant: Union[NDArray[Any], List[NDArray[Any]]], inp_data_float: Union[NDArray[Any],
                                                                                                   List[NDArray[Any]]],
                    out_data_float: Union[NDArray[Any], List[NDArray[Any]]], extra_options: Any) -> Any:
    """
    Optimize the onnx module with fast finetune algorithms by torch optimizer
    """

    torch_module = convert_onnx_to_torch(quant_model, float_weight, float_bias)

    quant_params = train_torch_module_api(torch_module, inp_data_quant, inp_data_float, out_data_float, extra_options)

    return quant_params
