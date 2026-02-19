#
# Modifications copyright(c) 2025 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023 潘其威(William)
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import copy
import math
import time
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.blockwise_tuning.blockwise_utils import block_forward
from quark.torch.algorithm.common import BaseHessianAlgorithm, BaseHessianProcessor, RestoreOriginalWeights
from quark.torch.algorithm.utils.module import get_device
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.quantization.config.config import GPTAQConfig
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

logger = ScreenLogger(__name__)

__all__ = ["GptaqProcessor"]

CPU = torch.device("cpu")
META = torch.device("meta")


class Gptaq(BaseHessianAlgorithm):
    """
    Handles the core Gptaq logic, an advanced post-training quantization algorithm. Implemented as proposed in https://arxiv.org/pdf/2504.02692
    """

    def __init__(self, layer: nn.Module) -> None:
        super().__init__(layer)
        # Referred to as `∆(XX.T)` in GPTAQ paper
        self.delta_X_Xt: torch.Tensor | None = torch.zeros(
            (self.columns, self.columns), device=self.device, dtype=torch.float32
        )

    def add_batch_quantized(self, quant_inp: torch.Tensor, out: torch.Tensor, name: str) -> None:
        assert self.H is not None
        quant_inp = quant_inp.float()

        if len(quant_inp.shape) == 2:
            quant_inp = quant_inp.unsqueeze(0)
        batch_size = quant_inp.shape[0]

        if isinstance(self.layer, nn.Linear):
            if len(quant_inp.shape) == 3:
                quant_inp = quant_inp.reshape((-1, quant_inp.shape[-1]))
            quant_inp = quant_inp.t()

        # H = \tilde{X} @ \tilde{X}^T
        self.nsamples += batch_size
        self.H *= (self.nsamples - batch_size) / self.nsamples
        self.delta_X_Xt *= (self.nsamples - batch_size) / (self.nsamples)

        quant_inp = math.sqrt(2 / self.nsamples) * quant_inp
        self.H += quant_inp.matmul(quant_inp.t())

        self.q_input = quant_inp
        self.batch_size = batch_size

    def add_batch_nonquantized(self, original_input: torch.Tensor, out: torch.Tensor, name: str) -> None:
        assert self.q_input is not None
        assert self.batch_size is not None
        original_input = original_input.float()

        if len(original_input.shape) == 2:
            original_input = original_input.unsqueeze(0)

        if isinstance(self.layer, nn.Linear):
            if len(original_input.shape) == 3:
                original_input = original_input.reshape((-1, original_input.shape[-1]))
            original_input = original_input.t()
        original_input = math.sqrt(2 / self.nsamples) * original_input
        delta_input = original_input - self.q_input

        self.delta_X_Xt += delta_input.matmul(self.q_input.t())

    def _get_quantizer(
        self,
        group_size: int,
        i1: int,
        i: int,
        static_groups: bool,
        actorder: bool,
        perm: torch.Tensor,
        W: torch.Tensor,
        quantizers: list[ScaledFakeQuantize],
        now_group: int,
        scale: list[torch.Tensor],
        zero: list[torch.Tensor],
    ) -> ScaledFakeQuantize:
        quantizer = quantizers[0]
        if group_size is not None and group_size > 0:
            if not static_groups:
                col_idx = i1 + i
                group_idx = col_idx // group_size
                if group_idx != now_group:
                    # finalize previous group's params
                    if now_group != -1:
                        scale.append(quantizer.scale)
                        zero.append(quantizer.zero_point)
                    # observe the new group
                    group_start = group_idx * group_size
                    group_end = min((group_idx + 1) * group_size, self.columns)
                    quantizer = quantizers[0]
                    quantizer.observer.reset_min_max_vals()
                    quantizer.observe(W[:, group_start:group_end])
                    now_group = group_idx
            else:
                idx = i1 + i
                if actorder:
                    idx = perm[idx]
                quantizer = quantizers[idx // group_size]
        return quantizer

    def quantize(
        self, blocksize: int, percdamp: float, alpha: float, group_size: int, actorder: bool, static_groups: bool
    ) -> None:
        assert self.H is not None

        per_group = group_size is not None and group_size > 0

        if get_device(self.layer) == META:  # get from cpu dict
            W = self.layer._hf_hook.weights_map["weight"].data.to(self.layer._hf_hook.execution_device)
        else:
            W = self.layer.weight.data.clone()

        orig_dtype = W.dtype
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.delta_X_Xt[:, dead] = 0

        scale: list[torch.Tensor] = []
        zero: list[torch.Tensor] = []

        quantizers = []
        if per_group:
            if static_groups:
                # only pergroup group_size > 0 need static_group
                # if not static, we will create quantizer for pergroup (groupsize > 0) in the following codes.
                for i in range(0, self.columns, group_size):
                    quantizer = copy.deepcopy(self.quantizer)  # TODO: this is very slow as well!
                    quantizer.observe(W[:, i : (i + group_size)])
                    quantizers.append(quantizer)
                    scale.append(quantizer.scale)
                    zero.append(quantizer.zero_point)
            else:
                quantizers.append(self.quantizer)
        else:
            # per-tensor, per-channel, and group_size = -1 cases.
            self.quantizer.observe(W)
            quantizers.append(self.quantizer)
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero_point)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            self.delta_X_Xt = self.delta_X_Xt[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            perm = None

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        L = Hinv.T
        P = alpha * ((self.delta_X_Xt @ L).triu_(diagonal=1)) @ L.T
        del self.delta_X_Xt

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            L1T = L.T[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            now_group = -1
            quantizer = quantizers[0]

            for i in range(count):
                w = W1[:, i]
                d = L1T[i, i]

                quantizer = self._get_quantizer(
                    group_size, i1, i, static_groups, actorder, perm, W, quantizers, now_group, scale, zero
                )

                q = quantizer.fake_quantize_with_qparams(
                    w.unsqueeze(1), scale=quantizer.scale, zero_point=quantizer.zero_point
                ).squeeze(1)
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(L1T[i, i:].unsqueeze(0)) - w.unsqueeze(1).matmul(
                    P1[i, i:].unsqueeze(0)
                )
                Err1[:, i] = err1
                # if dynamic per-group, push the last group's params
                if group_size is not None and group_size > 0 and not static_groups and now_group != -1:
                    scale.append(quantizer.scale)
                    zero.append(quantizer.zero_point)

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(L.T[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])

        if torch.cuda.is_available():
            torch.cuda.synchronize(Q.device)

        logger.info(f"duration: {(time.time() - tick)}")

        group_size_for_order = group_size if per_group else self.columns
        if static_groups and actorder:
            g_idx = perm // group_size_for_order

            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if get_device(self.layer) == META:
            # Directly replace weight in dict with qweight
            self.layer._hf_hook.weights_map["weight"].data = Q.reshape(self.layer.weight.shape).to(orig_dtype).to("cpu")
        else:
            self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)

        per_group = group_size is not None and group_size > 0
        if per_group:
            self.layer._weight_quantizer.scale = torch.cat([s.view(-1, 1) for s in scale], dim=1)
            self.layer._weight_quantizer.zero_point = torch.cat([z.view(-1, 1) for z in zero], dim=1)
        elif group_size == -1:
            self.layer._weight_quantizer.scale = self.quantizer.scale
            self.layer._weight_quantizer.zero_point = self.quantizer.zero_point

    def free(self) -> None:
        self.H = None
        self.delta_X_Xt = None
        self.q_input = None
        clear_memory()


class GptaqProcessor(BaseHessianProcessor):
    def __init__(self, model: nn.Module, quant_algo_config: GPTAQConfig, data_loader: DataLoader[torch.Tensor]) -> None:
        super().__init__(model, quant_algo_config, data_loader)
        self.percdamp = quant_algo_config.damp_percent
        self.alpha = quant_algo_config.alpha
        self._use_original_weight = True

    def _get_algorithm_instance(self, layer: nn.Module) -> Gptaq:
        return Gptaq(layer)

    def _quantize_layer(self, algo_instance: Gptaq, layer: nn.Module, group_size: int) -> None:
        algo_instance.quantize(
            self.block_size, self.percdamp, self.alpha, group_size, self.act_order, self.static_groups
        )

    def _collect_statistics(
        self,
        layer: nn.Module,
        grouped_inner_layers: dict[str, nn.Module],
        algo_instances: dict[str, Gptaq],
        layer_inputs: list[torch.Tensor],
        orig_layer_inputs: list[torch.Tensor],
        num_batches: int,
        current_layer_device: torch.device,
    ) -> None:
        # Process one sample at a time to ensure the quantized input from each sample's
        # quantized forward pass is available for the corresponding G calculation
        for batch_idx in tqdm(range(num_batches), desc="Collecting GPTAQ statistics"):
            # block_forward expectes List[torch.Tensor] as input.
            # tensor shape: [1, seq_length, dim]
            batch_input = [layer_inputs[batch_idx]]
            orig_batch_input = [orig_layer_inputs[batch_idx]]

            # define hook to collect H = \tilde{X} @ \tilde{X}^T
            def add_batch_quantized_hook(
                name: str,
            ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
                def hook(module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                    algo_instances[name].add_batch_quantized(input[0].data, output.data, name)

                return hook

            # define hook to collect G = X @ \tilde{X}^T
            def add_batch_nonquantized_hook(
                name: str,
            ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
                def hook(module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                    algo_instances[name].add_batch_nonquantized(input[0].data, output.data, name)

                return hook

            hook_handles_H = []
            for name in grouped_inner_layers:
                hook_handles_H.append(grouped_inner_layers[name].register_forward_hook(add_batch_quantized_hook(name)))

            # calculate H = \tilde{X} @ \tilde{X}^T
            _ = block_forward(
                layer=layer,
                module_kwargs=self.module_kwargs,
                num_batches=1,
                device=current_layer_device,
                layer_inputs=batch_input,  # type: ignore[arg-type]
                fp_layer_outputs=[],
                cache_examples_on_gpu=True,
            )

            for hook in hook_handles_H:
                hook.remove()

            hook_handles_G = []
            for name in grouped_inner_layers:
                hook_handles_G.append(
                    grouped_inner_layers[name].register_forward_hook(add_batch_nonquantized_hook(name))
                )

            with RestoreOriginalWeights(layer):
                # calculate ∆(XX.T) = (\tilde{X} - X)X^T
                _ = block_forward(
                    layer=layer,
                    module_kwargs=self.module_kwargs,
                    num_batches=1,
                    device=current_layer_device,
                    layer_inputs=orig_batch_input,  # type: ignore[arg-type]
                    fp_layer_outputs=[],
                    cache_examples_on_gpu=True,
                )

            for hook in hook_handles_G:
                hook.remove()
