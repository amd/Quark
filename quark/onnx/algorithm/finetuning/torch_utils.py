#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import random
from collections.abc import Iterator
from typing import Any

import numpy
import onnx
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from quark.common.utils.log import ScreenLogger

from .create_torch.create_model import TorchModel
from .onnx_subgraph import Subgraph
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
    if "NumWorkers" in extra_options["FastFinetune"]:
        train_params.num_workers = extra_options["FastFinetune"]["NumWorkers"]
    if "PinMemory" in extra_options["FastFinetune"]:
        train_params.pin_memory = extra_options["FastFinetune"]["PinMemory"]

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


class DaliLoaderWrapper:
    """
    Wrapper class for DALI data loader to provide PyTorch-compatible interface.
    Enables iteration and length calculation for DALI iterators in PyTorch workflows.
    """

    def __init__(self, dali_iterator: Any, total_samples: int, batch_size: int) -> None:
        self.loader = dali_iterator
        self.total_samples = total_samples
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Any]:
        return iter(self.loader)

    def __len__(self) -> int:
        return self.total_samples


def build_nv_gds_dataloader(
    inp_data_quant: list[Any],
    inp_data_float: list[Any],
    out_data_float: list[Any],
    train_params: TrainParameters,
    num_threads: int = 4,
    device_id: int = 0,
    shuffle: bool = True,
) -> DaliLoaderWrapper:
    """
    Build a DALI DataLoader using GPU Direct Storage (GDS) to read three sets of .npy files
    from a single data_dir, separated by file prefixes:
        - Quantized input: files starting with "q_input_data"
        - Float input: files starting with "f_input_data"
        - Float output: files starting with "f_output_data"
    The loader guarantees alignment and optional synchronized shuffling.
    """

    from tempfile import NamedTemporaryFile

    import nvidia.dali.fn as fn  # type: ignore
    import nvidia.dali.types as types  # type: ignore
    from nvidia.dali.pipeline import pipeline_def  # type: ignore
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy  # type: ignore

    # Create temporary .txt files for DALI
    def create_file_list_txt(files: list[str]) -> str:
        """
        Create a temporary text file containing a list of file paths.

        :param files: A list of file paths to write to the temporary file
        :return: The path to the created temporary file
        """
        with NamedTemporaryFile(mode="w+", delete=False) as tmp:  # pragma: no cover
            for f in files:
                tmp.write(f + "\n")
            tmp.flush()
        return tmp.name

    quant_list_txt = create_file_list_txt(inp_data_quant)
    float_in_list_txt = create_file_list_txt(inp_data_float)
    float_out_list_txt = create_file_list_txt(out_data_float)

    @pipeline_def
    def numpy_gds_pipeline(
        quant_file_list: str,
        float_in_file_list: str,
        float_out_file_list: str,
        shard_id: int = 0,
        num_shards: int = 1,
        shuffle: bool = False,
    ) -> tuple[Any, Any, Any]:
        """
        DALI pipeline definition for reading three sets of numpy files from GPU using GPU Direct Storage.

        :param quant_file_list: Path to text file containing list of quantized input .npy files
        :param float_in_file_list: Path to text file containing list of float input .npy files
        :param float_out_file_list: Path to text file containing list of float output .npy files
        :param shard_id: Shard ID for distributed training (default: 0)
        :param num_shards: Total number of shards for distributed training (default: 1)
        :return: Tuple of three tensors (inp_quant, inp_float, out_float) read from GPU and cast to FLOAT type
        """
        seed = random.randint(1, 1705472343)
        inp_quant = fn.readers.numpy(
            device="gpu",
            file_list=quant_file_list,
            random_shuffle=shuffle,
            shard_id=shard_id,
            num_shards=num_shards,
            name="QuantReader",
            seed=seed,
        )
        inp_quant = fn.cast(inp_quant, dtype=types.FLOAT)

        inp_float = fn.readers.numpy(
            device="gpu",
            file_list=float_in_file_list,
            random_shuffle=shuffle,
            shard_id=shard_id,
            num_shards=num_shards,
            name="FloatInputReader",
            seed=seed,
        )
        inp_float = fn.cast(inp_float, dtype=types.FLOAT)

        out_float = fn.readers.numpy(
            device="gpu",
            file_list=float_out_file_list,
            random_shuffle=shuffle,
            shard_id=shard_id,
            num_shards=num_shards,
            name="FloatOutputReader",
            seed=seed,
        )
        out_float = fn.cast(out_float, dtype=types.FLOAT)

        return inp_quant, inp_float, out_float

    # Build pipeline
    pipe = numpy_gds_pipeline(
        quant_file_list=quant_list_txt,
        float_in_file_list=float_in_list_txt,
        float_out_file_list=float_out_list_txt,
        batch_size=train_params.batch_size,
        num_threads=num_threads,
        device_id=device_id,
        shuffle=shuffle,
    )
    pipe.build()

    # Wrap pipeline for PyTorch
    loader = DALIGenericIterator(
        pipelines=pipe,
        output_map=["inp_quant", "inp_float", "out_float"],
        dynamic_shape=False,
        last_batch_policy=LastBatchPolicy.DROP,  # replace fill_last_batch
        auto_reset=True,  # auto reset for each epoch
    )

    total_samples = len(inp_data_quant)
    loader = DaliLoaderWrapper(loader, total_samples, train_params.batch_size)

    return loader


def train_torch_module_api(
    quant_module: torch.nn.Module,
    inp_data_quant: NDArray[Any] | list[Any],
    inp_data_float: NDArray[Any] | list[Any],
    out_data_float: NDArray[Any] | list[Any],
    train_params: TrainParameters,
    gds_info: dict[str, Any] = {"use_gds": False},
) -> Any:
    """
    Call torch training classes for adaround or adaquant
    """
    if isinstance(inp_data_quant, list) and len(inp_data_quant) > 0 and isinstance(inp_data_quant[0], str):
        if gds_info["use_gds"]:
            train_dataset = build_nv_gds_dataloader(inp_data_quant, inp_data_float, out_data_float, train_params)
        else:
            train_dataset = TrainDataset(inp_data_quant, inp_data_float, out_data_float)  # type: ignore
        ModelOptimizer.run_with_dataset(quant_module, train_dataset, train_params, gds_info)
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
    inp_data_quant: NDArray[Any] | list[Any],
    inp_data_float: NDArray[Any] | list[Any],
    out_data_float: NDArray[Any] | list[Any],
    extra_options: Any,
    gds_info: dict[str, Any] = {"use_gds": False},
) -> Any:
    """
    Optimize the onnx module with fast finetune algorithms by torch optimizer
    """

    torch_module = convert_onnx_to_torch(quant_model, float_weight, float_bias)

    train_params = parse_options_to_params(extra_options)

    return train_torch_module_api(torch_module, inp_data_quant, inp_data_float, out_data_float, train_params, gds_info)


def estimate_memory(model: torch.nn.Module, dummy_input: Any) -> float:
    # Step 1: calculate the model parameters memory usage
    total_params = sum(p.numel() for p in model.parameters())
    param_memory = total_params * 4 / (1024**2)

    # Step 2: calculate activation memory usage for each layer
    total_activation_memory = 0

    def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> None:
        nonlocal total_activation_memory
        total_activation_memory += output.numel() * 4 / (1024**2)

    # Register hook to calcuate activation memory usage
    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Execute forward once
    model(dummy_input)

    # remove all hooks
    for hook in hooks:
        hook.remove()

    # Step 3: calculate the optimizer memory usage
    optimizer_memory = 0
    for param in model.parameters():
        # for Adam，each parameters has three parts(momentum, variance, and the parameters）
        optimizer_memory += 3 * param.numel() * 4 / (1024**2)

    # all estimated memory usage
    total_memory = param_memory + total_activation_memory + optimizer_memory
    return total_memory


def estimate_largest_mem_layer(sg: Subgraph) -> list[int]:
    layers_estimated_peak_memory = []
    sg_mem_opt_level = sg.mem_opt_level
    sg.mem_opt_level = 1
    for i, module in enumerate(sg.subgraph_qmodel_list):
        batch_f_input_data, _ = map(numpy.array, sg.get_f_input_output_data(i, data_size=1))
        one_batch_f_input_data = torch.from_numpy(batch_f_input_data[0])
        f_weight = numpy.array(sg.f_weight_list[i])
        f_bias = None if sg.f_bias_list[i] is None else numpy.array(sg.f_bias_list[i]).reshape(-1)

        torch_module = convert_onnx_to_torch(module, f_weight, f_bias)
        estimated_peak_memory = estimate_memory(torch_module, one_batch_f_input_data)
        layers_estimated_peak_memory.append(estimated_peak_memory)

    max_index_mem = numpy.argmax(layers_estimated_peak_memory)
    selected_fastft_layers = [int(max_index_mem)]

    sg.mem_opt_level = sg_mem_opt_level

    return selected_fastft_layers
