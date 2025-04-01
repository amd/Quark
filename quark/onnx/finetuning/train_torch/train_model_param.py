#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Union, Tuple, Any

from torch.utils.data import DataLoader


class TrainParameters:
    """
    Configuration parameters for AdaRound and AdaQuant algorithms.

    The AdaRound is referenced from the following paper:
    "Markus Nagel et al., Up or Down? Adaptive Rounding for Post-Training Quantization,
    arXiv:2004.10568, 2020."

    The AdaQuant is referenced from the following paper:
    "Itay Hubara et al., Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming,
    arXiv:2006.10518, 2020."
    """

    def __init__(
        self,
        data_loader: Union[DataLoader[Any], None] = None,
        data_size: Union[int, None] = None,
        fixed_seed: Union[int, None] = None,
        num_batches: int = 1,
        num_iterations: int = 1000,
        batch_size: int = 1,
        initial_lr: float = 0.1,
        optim_algo: str = "adaround",
        optim_device: str = "cpu",
        lr_adjust: Any = (),
        selective_update: bool = False,
        early_stop: bool = False,
        log_period: Union[float, int] = 100,
        update_bias: bool = True,
        reg_param: float = 0.01,
        beta_range: Tuple[int, int] = (20, 2),
        warm_start: float = 0.2,
        drop_ratio: float = 0.5,
        block_recon: bool = False,
        dummy_path: str = '',
    ) -> None:
        """
        :param data_loader: Data loader for torch.
        :param data_size: Data samples number in the data loader.
        :param fixed_seed: A seed for torch or numpy's random functions, which is used to reproduce results.

        :param num_batches: Number of mini-batches in a iteration.
        :param num_iterations: Number of iterations to train each layer.
        :param batch_size: Batch size.
        :param initial_lr: Optimizer's initial learning rate.
        :param optim_algo: The algorithm used by the optimizer, now supported "adaround" and "adaquant".
        :param optim_device: Device the optimizer running on, 'cpu', 'cuda' or 'cuda:{id}'.

        :param lr_adjust: Layers have large error need large lr, if greater than a threshold then apply a new lr.
        :param selective_update: If the metric of a layer has decreased after optimization, do not update weight and bias.
        :param early_stop: If the current iteration's loss is worse than the previous best loss, early stop.
        :param log_period: How many iterations should the optimizer prints logs once.
        :param update_bias: If the module has bias, optimize bias or not. It is valid for adaquant.
        :param reg_param: Regularization parameter, trading off between rounding loss vs reconstruction loss for adaround.
        :param beta_range: Start and stop beta parameter for annealing of rounding loss (start_beta, end_beta).
        :param warm_start: Warm up period, during which rounding loss has zero effect.
        :param drop_ratio: Fetch quantized model's input data with probability of drop_ratio, 0 means from float model's.
        :param block_recon: Use block-wise reconstruction or layer-wise reconstruction.
        :param dummy_path: Dummy input path, used to store the input data or intermediate results for each module.
        """
        self.data_loader = data_loader
        self.data_size = data_size
        self.fixed_seed = fixed_seed

        self.num_batches = num_batches
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.lr = initial_lr
        self.algorithm = optim_algo
        self.device = optim_device

        self.lr_adjust = lr_adjust
        self.selective_update = selective_update
        self.early_stop = early_stop
        self.log_period = log_period
        self.update_bias = update_bias
        self.reg_param = reg_param
        self.beta_range = beta_range
        self.warm_start = warm_start
        self.drop_ratio = drop_ratio
        self.block_recon = block_recon
        self.dummy_path = dummy_path
