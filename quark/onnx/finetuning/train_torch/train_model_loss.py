#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Tuple, Any
import torch

from .train_model_param import TrainParameters

from quark.onnx.finetuning.create_torch.base_qdq_quantizers import AdaroundConstants
import numpy as np


class TrainLoss:
    """
    Calculates the Reconstruction loss, and Rounding loss
    This class is referenced from the AdaRound algorithm proposed in the following paper:
    "Markus Nagel et al., Up or Down? Adaptive Rounding for Post-Training Quantization,
    arXiv:2004.10568, 2020."
    """

    @staticmethod
    def calc_recon_loss(quant_output: torch.Tensor, float_output: torch.Tensor) -> Any:
        """
        Calculate Reconstruction Loss using Squared Frobenius Norm
        :param quant_output: Activation output from quantized wrapper module
        :param float_output: Activation output from original float module
        :return: Reconstruction loss
        """
        recon_loss = (torch.norm(quant_output - float_output, p="fro", dim=1)**2).mean()

        return recon_loss

    @classmethod
    def calc_round_loss(cls, alpha: torch.Tensor, params: TrainParameters, cur_iter: int) -> Any:
        """
        Calculate Rounding Loss (This is for AdaRound optimization to learn weight rounding)
        :param alpha: Parameter 'alpha' to be optimized
        :param params: Optimization parameters for AdaRound
        :param cur_iter: Current iteration
        :return: Rounding loss
        """
        if cur_iter < params.num_iterations * params.warm_start:
            round_loss = torch.tensor(0.0)
        else:
            h_alpha = torch.clamp(
                torch.sigmoid(alpha) * (AdaroundConstants.ZETA - AdaroundConstants.GAMMA) + AdaroundConstants.GAMMA, 0,
                1)

            beta = cls._calculate_beta(params.num_iterations, cur_iter, params.beta_range, params.warm_start)

            reg_term = torch.add(1, -(torch.add(2 * h_alpha, -1).abs()).pow(beta)).sum()

            round_loss = params.reg_param * reg_term

        return round_loss

    @staticmethod
    def _calculate_beta(max_iter: int, cur_iter: int, beta_range: Tuple[float, float], warm_start: float) -> Any:
        """
        Calculate beta parameter used in regularization function using cosine decay
        :param max_iter: Total maximum number of iterations
        :param cur_iter: Current iteration
        :param beta_range: Range for beta decay (start_beta, end_beta)
        :param warm_start: Warm up period, during which rounding loss has zero effect
        :return: Parameter beta
        """
        assert cur_iter < max_iter, 'Current iteration should be less than total maximum number of iterations.'

        start_beta, end_beta = beta_range

        warm_start_end_iter = warm_start * max_iter

        rel_iter = (cur_iter - warm_start_end_iter) / (max_iter - warm_start_end_iter)

        beta = end_beta + 0.5 * (start_beta - end_beta) * (1 + np.cos(rel_iter * np.pi))

        return beta
