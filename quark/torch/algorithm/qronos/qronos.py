#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023 潘其威(William)
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import math
import time
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.blockwise_tuning.blockwise_utils import block_forward
from quark.torch.algorithm.common import BaseHessianAlgorithm, BaseHessianProcessor, RestoreOriginalWeights
from quark.torch.algorithm.utils.module import get_device
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.quantization.config.config import QronosConfig
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

logger = ScreenLogger(__name__)

__all__ = ["QronosProcessor"]

CPU = torch.device("cpu")
META = torch.device("meta")


class Qronos(BaseHessianAlgorithm):
    """
    Handles the core Qronos logic, an advanced post-training quantization algorithm. Implemented as proposed in https://arxiv.org/pdf/2505.11695
    """

    def __init__(self, layer: nn.Module) -> None:
        super().__init__(layer)
        self.G: torch.Tensor | None = torch.zeros((self.columns, self.columns), device=self.device, dtype=torch.float32)

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
        self.H += (quant_inp.matmul(quant_inp.t())) / self.nsamples
        self.q_input = quant_inp

    def add_batch_nonquantized(self, inp: torch.Tensor, out: torch.Tensor, name: str) -> None:
        assert self.G is not None
        assert self.q_input is not None
        inp = inp.float()

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        batch_size = inp.shape[0]

        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        # G = X @ \tilde{X}^T
        self.G *= (self.nsamples - batch_size) / self.nsamples
        self.G += (inp.matmul(self.q_input.t())) / self.nsamples
        self.q_input = None

    def quantize(
        self, blocksize: int, alpha: float, beta: float, group_size: int, actorder: bool, static_groups: bool
    ) -> None:
        assert self.H is not None
        assert self.G is not None

        per_group = group_size is not None and group_size > 0

        if get_device(self.layer) == META:  # get from cpu dict
            W = self.layer._hf_hook.weights_map["weight"].data.to(self.layer._hf_hook.execution_device)
        else:
            W = self.layer.weight.data.clone()

        orig_dtype = W.dtype
        W = W.float()

        tick = time.time()

        H = self.H
        G = self.G
        del self.H, self.G
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        quantizers, scale, zero = self._setup_quantizers(W, group_size, static_groups)
        W, H, perm, invperm = self._apply_activation_order(W, H, actorder)

        if actorder:
            G = G[perm][:, perm]

        qronos_inner_kwargs = {
            "H": H,
            "G": G,
            "W": W,
            "perm": perm,
            "quantizers": quantizers,
            "actorder": actorder,
            "group_size": group_size,
            "columns": self.columns,
            "blocksize": blocksize,
            "static_groups": static_groups,
            "alpha": alpha,
            "beta": beta,
        }

        Q, _, _, _ = self.qronos_inner(**qronos_inner_kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize(Q.device)

        logger.info(f"duration: {(time.time() - tick)}")

        group_size_for_order = group_size if per_group else self.columns
        if static_groups and actorder:
            assert perm is not None  # for mypy
            g_idx = perm // group_size_for_order

            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        self._set_weight(Q, orig_dtype)
        self._update_layer_quantizer(scale, zero, group_size)

    def qronos_inner(
        self,
        H: torch.Tensor,
        G: torch.Tensor,
        W: torch.Tensor,
        perm: torch.Tensor | None,
        quantizers: list[ScaledFakeQuantize],
        actorder: bool,
        group_size: int,
        columns: int,
        blocksize: int,
        static_groups: bool,
        alpha: float,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None, list[torch.Tensor] | None]:
        Hinv = H.clone()
        damp = alpha * self.power_iteration(H, 30)
        diag = torch.arange(columns, device=W.device)
        Hinv[diag, diag] += damp
        Hinv = torch.linalg.cholesky(Hinv)
        Hinv = torch.cholesky_inverse(Hinv)

        inputs = {
            "H": H,
            "Hinv": Hinv,
            "G": G,
            "W": W,
            "perm": perm,
            "quantizers": quantizers,
            "actorder": actorder,
            "group_size": group_size,
            "columns": columns,
            "blocksize": blocksize,
            "static_groups": static_groups,
            "alpha_damp": damp,
            "beta": beta,
        }

        Q, Losses, scale, zero_point = self.qronos_inner_eager(self, **inputs)  # type: ignore[arg-type]

        return Q, Losses, scale, zero_point

    @staticmethod
    def qronos_inner_eager(
        self: Qronos,
        H: torch.Tensor,
        Hinv: torch.Tensor,
        G: torch.Tensor,
        W: torch.Tensor,
        perm: torch.Tensor | None,
        quantizers: list[ScaledFakeQuantize],
        actorder: bool,
        group_size: int,
        columns: int,
        blocksize: int,
        static_groups: bool,
        alpha_damp: float,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        device = W.device
        columns = W.shape[-1]
        Q = torch.zeros_like(W)

        # Carry out Qronos update rule - we want to find argmin_q ( 1/2 * || X * w - tilde{X} * q ||^2 )
        # This is approximately solved in two sequential steps instead since it is NP hard optimisation.
        # 1) calculate q_1 = QuantScheme( (G_{1,>=1} * w - H_{1, >=2} * w^{0}_{>=2}) /  H_{1,1} )
        # 2) update w^{1}_{>=2} = (H_{>=2, >=2})^-1 * (G_{>=2,>=1} * w - H_{>=2, 1} * q_1)

        if group_size is not None and group_size > 0:
            first_idx = perm[0] if actorder else 0  # type: ignore[index]
            quantizer = quantizers[first_idx // group_size]
        else:
            quantizer = quantizers[0]

        # Extract 1/H_{1,1} and H_{1, >=2}
        Dhi_0 = 0 if H[0, 0] == 0 else 1.0 / H[0, 0]
        H_0 = H[0, :].clone()
        H_0[0] = 0

        # Qronos 1) calculate q_1 = QuantScheme( (G_{1,>=1} * w - H_{1, >=2} * w^{0}_{>=2}) /  H_{1,1} )
        # G_{1,>=1} * w / H_{1,1}
        Gw = W.matmul(G[:, 0] * Dhi_0)

        # H_{1, >=2} * w^{0}_{>=2} / H_{1,1}
        Hv = W.matmul(H_0 * Dhi_0)

        # (G_{1,>=1} * w - H_{1, >=2} * w_{>=2}) /  H_{1,1}
        q_arg = Gw - Hv

        # q_1 = QuantScheme( (G_{1,>=1} * w - H_{1, >=2} * w^{0}_{>=2}) /  H_{1,1} )
        q_0 = quantizer.fake_quantize_with_qparams(
            q_arg.unsqueeze(1), scale=quantizer.scale, zero_point=quantizer.zero_point
        ).squeeze(1)
        Q[:, 0] = q_0

        del H_0, Dhi_0

        # Sherman-Morrison-Woodbury update for the inverse Hessian after first col
        A = Hinv[1:, 1:]
        c = Hinv[0, 0]
        b = Hinv[1:, [0]]
        A -= (b.matmul(b.T)) / c

        Hinv = A  # (H_{>=2, >=2})^-1
        del A, b, c

        # Qronos 2) update w_{>=2} =  (H_{>=2, >=2})^-1 * (G_{>=2,>=1} * w - H_{>=2, 1} * q_1)
        diag_damp = torch.diag(torch.full(size=(columns,), fill_value=alpha_damp, device=device))
        G_damp = G + diag_damp

        # ( H_{>=2, >=2} )^-1 * G_{>=2,>=1} * w
        Gw = W.matmul(G_damp[:, 1:] @ Hinv)

        # ( H_{>=2, >=2}) ^-1 * H_{>=2, 1} * q_1
        Hq = q_0.unsqueeze(1).matmul(H[:1, 1:] @ Hinv)

        # update w^{1}_{>=2} = ( H_{>=2, >=2} )^-1 * ( G_{>=2,>=1} * w - H_{>=2, 1} * q_1 )
        W[:, 1:] = Gw - Hq

        del G, H, G_damp

        Losses = torch.zeros_like(W)
        Q1 = torch.zeros_like(W[:, :blocksize])
        Err1 = torch.zeros_like(W[:, :blocksize])
        Losses1 = torch.zeros_like(W[:, :blocksize])

        scale: list[torch.Tensor] = []
        zero: list[torch.Tensor] = []
        now_idx = 1

        # re-calculate cholesky decomposition using a fairly large constant beta for stabilisation
        # gives us the updated L matrix for GPTQ algo (H^-1 = LL^T), it will be 1 dim smaller than original H_inv
        L = torch.linalg.cholesky(Hinv * beta, upper=True) / math.sqrt(beta)
        del Hinv

        # GPTQ loop to calculate Q[:, 1:] using the error diffused W[:, 1:]
        for i1 in range(1, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1

            W1 = W[:, i1:i2]

            if i1 + blocksize > columns:
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)

            # index with -1 because of the Sherman-Morrison-Woodbury update
            Hinv1 = L[i1 - 1 : i2 - 1, i1 - 1 : i2 - 1]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                quantizer = self._get_quantizer(
                    group_size, static_groups, quantizers, W, i1, i, now_idx, scale, zero, actorder, perm
                )

                q = quantizer.fake_quantize_with_qparams(
                    w.unsqueeze(1), scale=quantizer.scale, zero_point=quantizer.zero_point
                ).squeeze(1)

                Q1[:, i] = q

                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))

                Err1[:, i] = err1

            Q[:, i1:i2] = Q1

            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(L[i1 - 1 : i2 - 1, i2 - 1 :])

        return Q, Losses, scale, zero

    def _get_quantizer(
        self,
        group_size: int | None,
        static_groups: bool,
        quantizers: list[ScaledFakeQuantize],
        W: torch.Tensor,
        i1: int,
        i: int,
        now_idx: int,
        scale: list[torch.Tensor],
        zero: list[torch.Tensor],
        actorder: bool,
        perm: torch.Tensor | None,
    ) -> ScaledFakeQuantize:
        quantizer = quantizers[0]
        if group_size is not None and group_size > 0:
            if not static_groups:
                if (i1 + i) % group_size == 0:
                    quantizer = quantizers[0]

                    quantizer.observer.reset_min_max_vals()
                    quantizer.observe(W[:, (i1 + i) : (i1 + i + group_size)])
                if ((i1 + i) // group_size) - now_idx == -1:
                    scale.append(quantizer.scale)
                    zero.append(quantizer.zero_point)
                    now_idx += 1
            else:
                idx = i1 + i
                if actorder:
                    idx = perm[idx]  # type: ignore
                quantizer = quantizers[idx // group_size]
        return quantizer

    @staticmethod
    def power_iteration(H: torch.Tensor, num_iterations: int, eps: float = 1e-12) -> float:
        """
        Power iteration to compute the largest eigenvalue of the Hessian.
        Used to determine an 'optimal' dampening factor
        """
        # NOTE: this implementation doesn't necessitate convergence to the dominant eigenvector but should be a good approximation nonetheless.
        # This computation is done on CPU for better deterministic results
        # Please check issue #3832 for more details
        torch_default_device = torch.tensor([0]).device
        b_k_device = CPU if torch_default_device == CPU else H.device
        b_k = torch.ones(H.shape[1], device=b_k_device)
        c_k = H.max().abs()  # get absmax
        H_k = (H / c_k).to(b_k.device)
        for _ in range(num_iterations):
            b_k1 = torch.mv(H_k, b_k)  # H*b_k
            b_k1_norm = torch.norm(b_k1)  # ||H*b_k||
            b_k = b_k1 / (b_k1_norm + eps)  # b_{k+1} = H*b_k / ( ||H*b_k|| + epsilon )

        # λ_max ~= b_k^T · (H · b_k) when b_k is normalised; this is a simplification of the rayleigh quotient since ||b_k|| = 1
        max_eigenval = torch.dot(b_k, torch.mv(H_k, b_k)) * c_k
        return max_eigenval.item()

    def free(self) -> None:
        self.H = None
        self.G = None
        clear_memory()


class QronosProcessor(BaseHessianProcessor):
    def __init__(
        self, model: nn.Module, quant_algo_config: QronosConfig, data_loader: DataLoader[torch.Tensor]
    ) -> None:
        super().__init__(model, quant_algo_config, data_loader)
        self.alpha = quant_algo_config.alpha
        self.beta = quant_algo_config.beta
        self._use_original_weight = True

    def _get_algorithm_instance(self, layer: nn.Module) -> Qronos:
        return Qronos(layer)

    def _quantize_layer(self, algo_instance: Qronos, layer: nn.Module, group_size: int) -> None:
        algo_instance.quantize(self.block_size, self.alpha, self.beta, group_size, self.act_order, self.static_groups)

    def _collect_statistics(
        self,
        layer: nn.Module,
        grouped_inner_layers: dict[str, nn.Module],
        algo_instances: dict[str, Qronos],
        layer_inputs: list[torch.Tensor],
        orig_layer_inputs: list[torch.Tensor],
        num_batches: int,
        current_layer_device: torch.device,
    ) -> None:
        for batch_idx in range(num_batches):
            batch_input = [layer_inputs[batch_idx]]
            orig_batch_input = [orig_layer_inputs[batch_idx]]

            def add_batch_quantized_hook(
                name: str,
            ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
                def hook(module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                    algo_instances[name].add_batch_quantized(input[0].data, output.data, name)

                return hook

            def add_batch_nonquantized_hook(
                name: str,
            ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
                def hook(module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                    algo_instances[name].add_batch_nonquantized(input[0].data, output.data, name)

                return hook

            hook_handles_H = []
            for name in grouped_inner_layers:
                hook_handles_H.append(grouped_inner_layers[name].register_forward_hook(add_batch_quantized_hook(name)))

            block_forward(
                layer, self.module_kwargs, 1, current_layer_device, batch_input, [], cache_examples_on_gpu=True
            )

            for hook in hook_handles_H:
                hook.remove()

            hook_handles_G = []
            for name in grouped_inner_layers:
                hook_handles_G.append(
                    grouped_inner_layers[name].register_forward_hook(add_batch_nonquantized_hook(name))
                )

            with RestoreOriginalWeights(layer):
                block_forward(
                    layer, self.module_kwargs, 1, current_layer_device, orig_batch_input, [], cache_examples_on_gpu=True
                )

            for hook in hook_handles_G:
                hook.remove()

    @staticmethod
    def register_original_weights(layer: nn.Module) -> None:
        for submodule in layer.modules():
            if hasattr(submodule, "weight"):
                submodule.register_buffer("weight_orig", submodule.weight.detach().clone())

    @staticmethod
    def delete_original_weight_buffer(layer: nn.Module) -> None:
        for submodule in layer.modules():
            if hasattr(submodule, "weight_orig"):
                delattr(submodule, "weight_orig")
