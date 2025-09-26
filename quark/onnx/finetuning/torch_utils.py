#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
import onnx
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from quark.shares.utils.log import ScreenLogger

from .create_torch.create_model import TorchModel
from .train_torch.train_model import ModelOptimizer
from .train_torch.train_model_param import TrainParameters

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
    float_weight: NDArray[Any] | None = None,
    float_bias: NDArray[Any] | None = None,
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


def parse_options_to_params(extra_options: dict[str, Any]) -> TrainParameters:
    """
    Get train parameters from extra options
    """
    train_params = TrainParameters()

    if "FastFinetune" not in extra_options:
        logger.warning("Not found extra options for FastFinetune, will use default parameters")
        return train_params
    elif not isinstance(extra_options["FastFinetune"], dict):
        logger.warning(f"Invalid extra options {extra_options['FastFinetune']} for FastFinetune")
        return train_params

    if "DataSize" in extra_options["FastFinetune"]:
        train_params.data_size = extra_options["FastFinetune"]["DataSize"]
    if "FixedSeed" in extra_options["FastFinetune"]:
        train_params.fixed_seed = extra_options["FastFinetune"]["FixedSeed"]

    # For ordinary applications
    if "BatchSize" in extra_options["FastFinetune"]:
        train_params.batch_size = extra_options["FastFinetune"]["BatchSize"]
    if "NumBatches" in extra_options["FastFinetune"]:
        train_params.num_batches = extra_options["FastFinetune"]["NumBatches"]
    if "NumIterations" in extra_options["FastFinetune"]:
        train_params.num_iterations = extra_options["FastFinetune"]["NumIterations"]
    if "LearningRate" in extra_options["FastFinetune"]:
        train_params.lr = extra_options["FastFinetune"]["LearningRate"]
    if "OptimAlgorithm" in extra_options["FastFinetune"]:
        train_params.algorithm = extra_options["FastFinetune"]["OptimAlgorithm"].lower()
    if "OptimDevice" in extra_options["FastFinetune"]:
        train_params.device = extra_options["FastFinetune"]["OptimDevice"].lower()

    # For advanced applications
    if "LRAdjust" in extra_options["FastFinetune"]:
        train_params.lr_adjust = extra_options["FastFinetune"]["LRAdjust"]
    # if 'SelectiveUpdate' in extra_options['FastFinetune']:
    #    train_params.selective_update = extra_options['FastFinetune'][
    #        'SelectiveUpdate']
    if "EarlyStop" in extra_options["FastFinetune"]:
        train_params.early_stop = extra_options["FastFinetune"]["EarlyStop"]
    if "UpdateBias" in extra_options["FastFinetune"]:
        train_params.update_bias = extra_options["FastFinetune"]["UpdateBias"]
    if "RegParam" in extra_options["FastFinetune"]:
        train_params.reg_param = extra_options["FastFinetune"]["RegParam"]
    if "BetaRange" in extra_options["FastFinetune"]:
        train_params.beta_range = extra_options["FastFinetune"]["BetaRange"]
    if "WarmStart" in extra_options["FastFinetune"]:
        train_params.warm_start = extra_options["FastFinetune"]["WarmStart"]
    if "DropRatio" in extra_options["FastFinetune"]:
        train_params.drop_ratio = extra_options["FastFinetune"]["DropRatio"]

    if "LogPeriod" in extra_options["FastFinetune"]:
        train_params.log_period = extra_options["FastFinetune"]["LogPeriod"]
    else:
        train_params.log_period = train_params.num_iterations / 10

    # default lr for adaquant and adaround is different
    if train_params.algorithm == "adaquant" and "LearningRate" not in extra_options["FastFinetune"]:
        train_params.lr = 0.00001
    if train_params.algorithm == "adaround" and "LearningRate" not in extra_options["FastFinetune"]:
        train_params.lr = 0.1

    return train_params


class TrainDataset(Dataset[Any]):  # type: ignore
    """
    Dataset for training, which can load a mini-batch only at each time.
    """

    def __init__(self, inp_data_quant: list[Any], inp_data_float: list[Any], out_data_float: list[Any]) -> None:
        self._inp_data_quant_files = inp_data_quant
        self._inp_data_float_files = inp_data_float
        self._out_data_float_files = out_data_float

    def __len__(self) -> int:
        return len(self._inp_data_quant_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp_data_quant_tensor = torch.from_numpy(numpy.load(self._inp_data_quant_files[index])).squeeze(0)
        inp_data_float_tensor = torch.from_numpy(numpy.load(self._inp_data_float_files[index])).squeeze(0)
        out_data_float_tensor = torch.from_numpy(numpy.load(self._out_data_float_files[index])).squeeze(0)
        return inp_data_quant_tensor, inp_data_float_tensor, out_data_float_tensor


def train_torch_module_api(
    quant_module: torch.nn.Module,
    inp_data_quant: Union[NDArray[Any], list[Any]],
    inp_data_float: Union[NDArray[Any], list[Any]],
    out_data_float: Union[NDArray[Any], list[Any]],
    train_params: TrainParameters,
) -> Any:
    """
    Call torch training classes for adaround or adaquant
    """
    if isinstance(inp_data_quant, list) and len(inp_data_quant) > 0 and isinstance(inp_data_quant[0], str):
        train_dataset = TrainDataset(inp_data_quant, inp_data_float, out_data_float)  # type: ignore
        ModelOptimizer.run_with_dataset(quant_module, train_dataset, train_params)
    else:
        ModelOptimizer.run(quant_module, inp_data_quant, inp_data_float, out_data_float, train_params)

    if train_params.algorithm == "adaquant" and train_params.update_bias:
        return quant_module.get_weight(), quant_module.get_bias()
    else:
        return quant_module.get_weight(), None


def optimize_module(
    quant_model: onnx.ModelProto,
    float_weight: NDArray[Any],
    float_bias: NDArray[Any] | None,
    inp_data_quant: Union[NDArray[Any], list[Any]],
    inp_data_float: Union[NDArray[Any], list[Any]],
    out_data_float: Union[NDArray[Any], list[Any]],
    extra_options: Any,
) -> Any:
    """
    Optimize the onnx module with fast finetune algorithms by torch optimizer
    """

    torch_module = convert_onnx_to_torch(quant_model, float_weight, float_bias)

    train_params = parse_options_to_params(extra_options)

    return train_torch_module_api(torch_module, inp_data_quant, inp_data_float, out_data_float, train_params)
