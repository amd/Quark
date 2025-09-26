#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023 潘其威(William)
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import copy
import fnmatch
import math
import time
from functools import partial
from types import TracebackType
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.blockwise_tuning.blockwise_utils import block_forward
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import get_device, get_named_quant_linears, move_to_device
from quark.torch.algorithm.utils.prepare import init_blockwise_algo, init_device_map, reset_model_kv_cache
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.quantization.config.config import QronosConfig, QuantizationSpec
from quark.torch.quantization.config.type import QSchemeType
from quark.torch.quantization.observer.observer import PerChannelMinMaxObserver
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, ScaledFakeQuantize

logger = ScreenLogger(__name__)

__all__ = ["QronosProcessor"]

CPU = torch.device("cpu")
META = torch.device("meta")


class Qronos:
    """
    Handles the core Qronos logic, an advanced post-training quantization algorithm. Implemented as proposed in https://arxiv.org/pdf/2505.11695
    """

    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.device = self.layer.weight.device
        self.columns = layer.weight.data.shape[1]
        if self.device == META:
            self.device = self.layer._hf_hook.execution_device
        self.q_input: torch.Tensor | None = None
        self.H: torch.Tensor | None = torch.zeros((self.columns, self.columns), device=self.device, dtype=torch.float32)
        self.G: torch.Tensor | None = torch.zeros((self.columns, self.columns), device=self.device, dtype=torch.float32)
        self.nsamples = 0
        self.original_qspec = self.layer._weight_quantizer.observer.qspec

        # for per group minmaxobserver: group_size > 1 and group_size == -1
        if self.original_qspec.qscheme == QSchemeType.per_group:
            self.adjusted_qspec = QuantizationSpec(
                dtype=self.original_qspec.dtype,
                qscheme=QSchemeType.per_channel,
                observer_cls=PerChannelMinMaxObserver,
                symmetric=self.original_qspec.symmetric,
                scale_type=self.original_qspec.scale_type,
                round_method=self.original_qspec.round_method,  # useless for perchannel
                ch_axis=0,
                is_dynamic=self.original_qspec.is_dynamic,
                mx_element_dtype=self.original_qspec.mx_element_dtype,
                scale_format=self.original_qspec.scale_format,
                scale_calculation_mode=self.original_qspec.scale_calculation_mode,
            )
            self.quantizer = FakeQuantizeBase.get_fake_quantize(self.adjusted_qspec, self.device)
        # pertensor & perchannel
        else:
            self.quantizer = layer._weight_quantizer

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

    def qronos_quantize(
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

                    scale.append(quantizer.scale)
                    zero.append(quantizer.zero_point)
                    quantizers.append(quantizer)
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
            G = G[perm][:, perm]
            invperm = torch.argsort(perm)
        else:
            perm = None

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
            g_idx = perm // group_size_for_order

            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if get_device(self.layer) == META:
            # Directly replace weight in dict with qweight
            self.layer._hf_hook.weights_map["weight"].data = Q.reshape(self.layer.weight.shape).to(orig_dtype).to("cpu")
        else:
            self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)

        # scale and zero of perchannel, pertensor have been added to buffer
        # but per_group (any groupsize) need be added
        if group_size is not None:
            if group_size > 0:  # if not static, scale and zero_point need to be reordered when using quantization
                self.layer._weight_quantizer.scale = torch.cat([s.view(-1, 1) for s in scale], dim=1)
                self.layer._weight_quantizer.zero_point = torch.cat([z.view(-1, 1) for z in zero], dim=1)
            else:  # when group size == -1, static_group does not work
                self.layer._weight_quantizer.scale = self.quantizer.scale
                self.layer._weight_quantizer.zero_point = self.quantizer.zero_point

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

        Q, Losses, scale, zero_point = self.qronos_inner_eager(**inputs)  # type: ignore[arg-type]

        return Q, Losses, scale, zero_point

    @staticmethod
    def qronos_inner_eager(
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

        scale = []
        zero = []
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
                else:
                    quantizer = quantizers[0]

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

    @staticmethod
    def power_iteration(H: torch.Tensor, num_iterations: int, eps: float = 1e-12) -> float:
        """
        Power iteration to compute the largest eigenvalue of the Hessian.
        Used to determine an 'optimal' dampening factor
        """
        # NOTE: this implementation doesn't necessitate convergence to the dominant eigenvector but should be a good approximation nonetheless.
        b_k = torch.rand(H.shape[1], device=H.device)
        for _ in range(num_iterations):
            b_k1 = torch.mv(H, b_k)  # H*b_k
            b_k1_norm = torch.norm(b_k1)  # ||H*b_k||
            b_k = b_k1 / (b_k1_norm + eps)  # b_{k+1} = H*b_k / ( ||H*b_k|| + epsilon )

        # λ_max ~= b_k^T · (H · b_k) when b_k is normalised; this is a simplification of the rayleigh quotient since ||b_k|| = 1
        max_eigenval = torch.dot(b_k, torch.mv(H, b_k))
        return max_eigenval.item()

    def free(self) -> None:
        self.H = None
        self.G = None
        clear_memory()


class QronosProcessor(BaseAlgoProcessor):
    def __init__(
        self, model: nn.Module, quant_algo_config: QronosConfig, data_loader: DataLoader[torch.Tensor]
    ) -> None:
        self.model = model
        self.block_size = quant_algo_config.block_size
        self.alpha = quant_algo_config.alpha
        self.beta = quant_algo_config.beta
        self.act_order = quant_algo_config.desc_act
        self.static_groups = quant_algo_config.static_groups
        self.inside_layer_modules = quant_algo_config.inside_layer_modules
        self.model_decoder_layers = quant_algo_config.model_decoder_layers
        self.data_loader = data_loader
        self.device_map = init_device_map(self.model)
        self.modules, self.module_kwargs, self.inps = init_blockwise_algo(
            self.model, self.model_decoder_layers, self.data_loader
        )

    def apply(self) -> None:
        num_batches = len(self.inps)
        layer_inputs = [inp.clone() for inp in self.inps]
        orig_layer_inputs = [inp.clone() for inp in self.inps]
        layer_outputs: list[torch.Tensor] = []
        orig_layer_outputs: list[torch.Tensor] = []
        forward_pass_use_cache = reset_model_kv_cache(self.model, use_cache=False)

        for i in tqdm(range(len(self.modules)), desc="Applying Qronos on layers"):
            logger.info(f"Start quantizing layer {i + 1}/{len(self.modules)}")
            self.register_original_weights(self.modules[i])
            layer = self.modules[i]

            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to_device(layer, self.device_map[f"{self.model_decoder_layers}.{i}"])
                force_layer_back_to_cpu = True
            current_layer_device = (
                get_device(layer) if not get_device(layer) == META else layer._hf_hook.execution_device
            )

            # all_inner_layer_modules.keys: ['self_attn.k_proj', 'feed_forward.experts.0.gate_proj', 'feed_forward.experts.1.gate_proj', ...]
            all_inner_layer_modules = get_named_quant_linears(layer)
            assert self.inside_layer_modules is not None
            inner_layer_module_names: list[str] = self.inside_layer_modules

            # inner_layer_module_names: ['self_attn.k_proj', 'mlp.down_proj', ...]
            for layer_name in inner_layer_module_names:
                # handle both dense and moe layers, e.g. feed_forward.experts.*.down_proj are grouped into mlp.down_proj
                matched_names = fnmatch.filter(all_inner_layer_modules.keys(), "*" + layer_name)
                grouped_inner_layers = {
                    inner_layer: all_inner_layer_modules[inner_layer]
                    for inner_layer in matched_names
                    if getattr(all_inner_layer_modules[inner_layer], "_weight_quantizer", None) is not None
                }

                qronos = {}
                for inner_layer in grouped_inner_layers:
                    qronos[inner_layer] = Qronos(grouped_inner_layers[inner_layer])

                # Process one sample at a time to ensure the quantized input from each sample's
                # quantized forward pass is available for the corresponding G calculation
                for batch_idx in range(num_batches):
                    # block_forward expectes List[torch.Tensor] as input.
                    # tensor shape: [1, seq_length, dim]
                    batch_input = [layer_inputs[batch_idx]]
                    orig_batch_input = [orig_layer_inputs[batch_idx]]

                    # define hook to collect H = \tilde{X} @ \tilde{X}^T
                    def add_batch_hook_quantized(
                        module: nn.Module, inp: torch.Tensor, out: torch.Tensor, name: str
                    ) -> None:
                        qronos[name].add_batch_quantized(inp[0].data, out.data, name)

                    # define hook to collect G = X @ \tilde{X}^T
                    def add_batch_hook_nonquantized(
                        module: nn.Module, inp: torch.Tensor, out: torch.Tensor, name: str
                    ) -> None:
                        qronos[name].add_batch_nonquantized(inp[0].data, out.data, name)

                    hook_handles_H = []
                    for name in grouped_inner_layers:
                        hook_handles_H.append(
                            grouped_inner_layers[name].register_forward_hook(
                                partial(add_batch_hook_quantized, name=name)
                            )
                        )

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
                            grouped_inner_layers[name].register_forward_hook(
                                partial(add_batch_hook_nonquantized, name=name)
                            )
                        )

                    with RestoreOriginalWeights(layer):
                        # calculate G = X @ \tilde{X}^T
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

                # apply qronos to each inner layer e.g. mlp_down_proj
                for name in grouped_inner_layers:
                    logger.info(f"Quantizing {name} in layer {i + 1}/{len(self.modules)}...")
                    qronos[name].qronos_quantize(
                        blocksize=self.block_size,
                        alpha=self.alpha,
                        beta=self.beta,
                        group_size=grouped_inner_layers[name]._weight_quantizer.group_size,
                        actorder=self.act_order,
                        static_groups=self.static_groups,
                    )
                    qronos[name].free()

            # get whole decoder layer output
            layer_outputs = block_forward(
                layer,
                self.module_kwargs,
                num_batches,
                current_layer_device,
                layer_inputs,
                layer_outputs,
                cache_examples_on_gpu=True,
            )

            with RestoreOriginalWeights(layer):
                orig_layer_outputs = block_forward(
                    layer,
                    self.module_kwargs,
                    num_batches,
                    current_layer_device,
                    orig_layer_inputs,
                    orig_layer_outputs,
                    cache_examples_on_gpu=True,
                )

            if get_device(layer) != META:
                # if meta, scale and zero point are in execution_device, and weight is in meta, can't change.
                layer = move_to_device(layer, CPU if force_layer_back_to_cpu else current_layer_device)

            del layer
            del qronos
            del layer_inputs
            del orig_layer_inputs
            self.delete_original_weight_buffer(self.modules[i])
            clear_memory()

            orig_layer_inputs, layer_inputs = orig_layer_outputs, layer_outputs
            orig_layer_outputs, layer_outputs = [], []

        reset_model_kv_cache(self.model, use_cache=forward_pass_use_cache)

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


class RestoreOriginalWeights:
    """
    Used to temporarily switch layers to use their original (unquantized) weights.
    Useful for collection of cross-correlation matrices that depend on non-quantized values i.e. G = X @ tilde{X}^T
    """

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.quantized_weight_states = []  # type: ignore[var-annotated]

    def __enter__(self) -> RestoreOriginalWeights:
        for submodule in self.module.modules():
            if hasattr(submodule, "weight") and hasattr(submodule, "weight_orig"):
                self._swap_to_unquantized_weights(submodule)

        return self

    def _swap_to_unquantized_weights(self, module: nn.Module) -> None:
        quantized_weight_state = {"module": module, "current_weight": module.weight.data}
        self.quantized_weight_states.append(quantized_weight_state)
        module.weight.data = module.weight_orig.data

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: TracebackType | None
    ) -> None:
        for state in self.quantized_weight_states:
            module = state["module"]
            module.weight.data = state["current_weight"]

        self.quantized_weight_states.clear()
